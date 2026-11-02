from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
import threading
import time
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from src.optimization.cache import InterpretationCache, interpretation_key
from src.optimization.llm import LLMProviderError
import src.optimization.api as api
from src.web.app import app


RESPONSE = {
    "source": "llm",
    "interpretation": {"summary": "Teste"},
    "quality_evaluation": {"passed": True, "score": 7, "maximum": 7},
}


class InterpretationCacheTests(unittest.TestCase):
    def test_hit_is_fast_and_response_cannot_mutate_cached_data(self) -> None:
        cache = InterpretationCache()
        generate = Mock(return_value=deepcopy(RESPONSE))
        first = cache.get_or_generate("a", generate)
        first["interpretation"]["summary"] = "changed"
        second = cache.get_or_generate("a", generate)
        self.assertFalse(first["cache"]["hit"])
        self.assertTrue(second["cache"]["hit"])
        self.assertEqual(second["interpretation"]["summary"], "Teste")
        generate.assert_called_once()

    def test_ttl_expires_from_generation_completion(self) -> None:
        clock = Mock(return_value=0)
        cache = InterpretationCache(ttl_seconds=10, clock=clock)
        generate = Mock(return_value=RESPONSE)
        cache.get_or_generate("a", generate)
        clock.return_value = 9
        self.assertTrue(cache.get_or_generate("a", generate)["cache"]["hit"])
        clock.return_value = 10
        self.assertFalse(cache.get_or_generate("a", generate)["cache"]["hit"])
        self.assertEqual(generate.call_count, 2)

    def test_lru_capacity_is_bounded(self) -> None:
        cache = InterpretationCache(max_entries=2)
        generate = Mock(return_value=RESPONSE)
        for key in ("a", "b", "a", "c"):
            cache.get_or_generate(key, generate)
        self.assertTrue(cache.get_or_generate("a", generate)["cache"]["hit"])
        self.assertFalse(cache.get_or_generate("b", generate)["cache"]["hit"])

    def test_errors_local_and_failed_quality_are_not_cached(self) -> None:
        for response in (
            {"source": "local", "quality_evaluation": {"passed": True}},
            {"source": "llm", "quality_evaluation": {"passed": False}},
        ):
            cache = InterpretationCache()
            generate = Mock(return_value=response)
            for _ in range(2):
                self.assertFalse(cache.get_or_generate("a", generate)["cache"]["stored"])
            self.assertEqual(generate.call_count, 2)
        cache = InterpretationCache()
        generate = Mock(side_effect=[LLMProviderError("offline"), RESPONSE])
        with self.assertRaises(LLMProviderError):
            cache.get_or_generate("a", generate)
        self.assertFalse(cache.get_or_generate("a", generate)["cache"]["hit"])

    def test_concurrent_identical_requests_generate_once(self) -> None:
        cache = InterpretationCache()
        barrier = threading.Barrier(4)
        generate = Mock(side_effect=lambda: (time.sleep(0.05), RESPONSE)[1])

        def request(_: int) -> dict:
            barrier.wait(timeout=5)
            return cache.get_or_generate("a", generate)

        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(request, range(4)))
        generate.assert_called_once()
        self.assertEqual(sum(result["cache"]["hit"] for result in responses), 3)

    def test_key_changes_with_metrics_question_provider_and_prompt(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            initial = interpretation_key({"recall": 0.5}, "Não melhorou?")
            self.assertEqual(initial, interpretation_key({"recall": 0.5}, " Não melhorou? "))
            self.assertNotEqual(initial, interpretation_key({"recall": 0.6}, "Não melhorou?"))
            self.assertNotEqual(initial, interpretation_key({"recall": 0.5}, "Melhorou?"))
            for setting in (
                "LLM_MODEL", "LLM_BASE_URL", "LLM_PROVIDER", "LLM_API_KEY",
                "LLM_TEMPERATURE", "LLM_NUM_PREDICT", "LLM_TIMEOUT_SECONDS",
            ):
                with patch.dict(os.environ, {setting: "changed"}):
                    self.assertNotEqual(initial, interpretation_key({"recall": 0.5}, "Não melhorou?"))
            with patch("src.optimization.cache.PROMPT_REVISION", "new"):
                self.assertNotEqual(initial, interpretation_key({"recall": 0.5}, "Não melhorou?"))

    def test_api_reuses_response_and_invalidates_changed_metrics(self) -> None:
        with patch.object(api, "INTERPRETATION_CACHE", InterpretationCache()), patch.object(
            api, "_load_study", return_value={}
        ), patch.object(
            api, "build_interpretation_context", return_value={"recall": 0.5}
        ) as context, patch.object(
            api, "generate_interpretation", return_value=RESPONSE
        ) as generate, TestClient(app) as client:
            payload = {"estudo": "pcos", "pergunta": "Qual recall?"}
            first = client.post("/api/interpretacao", json=payload)
            second = client.post("/api/interpretacao", json=payload)
            self.assertEqual(first.status_code, 200)
            self.assertFalse(first.json()["cache"]["hit"])
            self.assertTrue(second.json()["cache"]["hit"])
            generate.assert_called_once()
            context.return_value = {"recall": 0.7}
            self.assertFalse(client.post("/api/interpretacao", json=payload).json()["cache"]["hit"])
            self.assertEqual(generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
