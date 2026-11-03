from io import BytesIO
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from fastapi.testclient import TestClient

from src.logistics.genetic import GAConfig, optimize_routes
from src.logistics.scenario import DEMO_SCENARIO
from src.optimization.cache import InterpretationCache
from src.optimization.llm import (
    LLMConfigurationError, LLMProviderError, _settings, request_chat_completion,
)
from src.web.app import app


class CloudTransportTests(unittest.TestCase):
    def test_closed_gemini_error_body_preserves_original_http_failure(self) -> None:
        body = BytesIO(b'{"error":{"status":"UNAVAILABLE"}}')
        body.close()
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai_compatible",
            "LLM_MODEL": "gemini-3.1-flash-lite",
            "LLM_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai",
            "LLM_API_KEY": "secret-test-key",
        }, clear=True), patch(
            "src.optimization.llm._read_completion_response",
            side_effect=HTTPError("https://generativelanguage.googleapis.com", 503, "", {}, body),
        ), self.assertRaisesRegex(LLMProviderError, "HTTP 503"):
            request_chat_completion("system", "synthetic probe")

    def test_gemini_error_exposes_only_documented_status_not_provider_body(self) -> None:
        for body, expected in (
            ({"error": {"status": "RESOURCE_EXHAUSTED", "message": "secret-test-key"}}, True),
            ({"error": {"status": "secret-test-key", "message": "secret-test-key"}}, False),
            ({"error": "secret-test-key"}, False),
            (["secret-test-key"], False),
        ):
            with self.subTest(body=body), patch.dict(os.environ, {
                "LLM_PROVIDER": "openai_compatible",
                "LLM_MODEL": "gemini-3.1-flash-lite",
                "LLM_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai",
                "LLM_API_KEY": "secret-test-key",
            }, clear=True), patch(
                "src.optimization.llm.urlopen",
                side_effect=HTTPError(
                    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                    429, "Too many requests", {}, BytesIO(json.dumps(body).encode()),
                ),
            ):
                with self.assertRaises(LLMProviderError) as error:
                    request_chat_completion("system", "synthetic probe")
                self.assertEqual("RESOURCE_EXHAUSTED" in str(error.exception), expected)
                self.assertNotIn("secret-test-key", str(error.exception))

    def test_both_module_apis_use_qwen_and_reuse_validated_cache(self) -> None:
        self.assert_module_cloud_cache(
            "https://ap-southeast-1.maas.qwencloudapi.com/compatible-mode/v1",
            "qwen-flash", {"LLM_ENABLE_THINKING": "false"},
            {"enable_thinking": False},
        )

    def test_both_module_apis_use_gemini_and_reuse_validated_cache(self) -> None:
        self.assert_module_cloud_cache(
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-3.1-flash-lite", {"LLM_REASONING_EFFORT": "minimal"},
            {"reasoning_effort": "minimal"},
        )

    def assert_module_cloud_cache(
        self, base_url: str, model: str, options: dict[str, str],
        expected_options: dict[str, str | bool],
    ) -> None:
        plan = optimize_routes(
            DEMO_SCENARIO, GAConfig(population_size=4, generations=1, tournament_size=2)
        )
        context = {
            "study": "PCOS", "positive_class": "PCOS", "test_samples": 20,
            "models": {"Modelo": {
                "baseline_test_metrics": dict(recall=.5, precision=.6, f1=.5, accuracy=.7),
                "optimized_test_metrics": dict(recall=.6, precision=.6, f1=.6, accuracy=.7),
            }},
        }
        diagnostic = {
            "summary": "As métricas do modelo exigem validação externa.",
            "findings": ["O recall aumentou nos dados de teste."],
            "limitations": ["Dados de um único holdout."],
            "disclaimer": "Não substitui avaliação clínica.",
        }
        logistics = {
            "period": "Simulação", "summary": "Plano demonstrativo.",
            "improvements": ["Validar distâncias reais."],
            "caveats": ["Sem trânsito real."], "disclaimer": "Confirme com a coordenação.",
        }
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai_compatible", "LLM_MODEL": model,
            "LLM_BASE_URL": base_url, **options,
            "LLM_API_KEY": "private-test-key",
        }, clear=True), patch(
            "src.optimization.api.INTERPRETATION_CACHE", InterpretationCache()
        ), patch(
            "src.logistics.llm.ROUTE_CONTENT_CACHE",
            InterpretationCache(is_cacheable=lambda r: r.get("source") == "llm"),
        ), patch(
            "src.optimization.api._load_study", return_value={}
        ), patch(
            "src.optimization.api.build_interpretation_context", return_value=context
        ), patch(
            "src.logistics.api.load_route_plan", return_value=plan
        ), patch(
            "src.logistics.api.route_history_summary", return_value={}
        ), patch("src.optimization.llm.urlopen") as provider, TestClient(app) as client:
            for path, payload, content in (
                ("/api/interpretacao", {"estudo": "pcos"}, diagnostic),
                ("/api/rotas/assistente", {"action": "daily_report"}, logistics),
            ):
                provider.return_value.__enter__.return_value.read.return_value = json.dumps({
                    "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
                }).encode()
                first = client.post(path, json=payload)
                second = client.post(path, json=payload)
                self.assertEqual(first.status_code, 200, first.text)
                self.assertEqual(second.status_code, 200, second.text)
                self.assertEqual(first.json()["source"], "llm")
                self.assertFalse(first.json()["cache"]["hit"])
                self.assertTrue(second.json()["cache"]["hit"])
                request = provider.call_args.args[0]
                self.assertEqual(
                    request.full_url,
                    base_url + "/chat/completions",
                )
                body = json.loads(request.data)
                for name, value in expected_options.items():
                    self.assertEqual(body[name], value)
                self.assertNotIn("private-test-key", request.data.decode())
            self.assertEqual(provider.call_count, 2)

    def test_remote_endpoint_requires_https_and_exact_local_hostname(self) -> None:
        for base in (
            "http://api.openai.com/v1",
            "https://localhost.evil.example/v1",
            "https://api.example/v1?api_key=secret",
        ):
            with self.subTest(base=base), patch.dict(os.environ, {
                "LLM_PROVIDER": "openai_compatible", "LLM_MODEL": "test",
                "LLM_BASE_URL": base,
            }, clear=True), self.assertRaises(LLMConfigurationError):
                _settings()

    def test_cloud_auth_json_and_token_limit(self) -> None:
        for base in ("https://api.openai.com/v1", "https://api.groq.com/openai/v1"):
            with self.subTest(base=base), patch.dict(os.environ, {
                "LLM_PROVIDER": "openai_compatible", "LLM_MODEL": "test",
                "LLM_BASE_URL": base, "LLM_API_KEY": "private-test-key",
            }, clear=True), patch("src.optimization.llm.urlopen") as provider:
                provider.return_value.__enter__.return_value.read.return_value = (
                    b'{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}'
                )
                self.assertEqual(request_chat_completion(
                    "system", "aggregate metrics only", json_mode=True, max_tokens=280,
                ), "{}")
            request = provider.call_args.args[0]
            self.assertEqual(request.full_url, base + "/chat/completions")
            self.assertEqual(request.get_header("Authorization"), "Bearer private-test-key")
            payload = json.loads(request.data)
            self.assertEqual(payload["max_tokens"], 280)
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertNotIn("enable_thinking", payload)
            self.assertNotIn("reasoning_effort", payload)

    def test_invalid_reasoning_effort_fails_before_network(self) -> None:
        for extra in (
            {"LLM_REASONING_EFFORT": "invalid"},
            {"LLM_REASONING_EFFORT": "none", "LLM_ENABLE_THINKING": "false"},
            {"LLM_REASONING_EFFORT": "none", "LLM_PROVIDER": "ollama"},
        ):
            with self.subTest(extra=extra), patch.dict(os.environ, {
                "LLM_PROVIDER": "openai_compatible", "LLM_MODEL": "test",
                "LLM_API_KEY": "fake-test-key", **extra,
            }, clear=True), patch("src.optimization.llm.urlopen") as provider:
                with self.assertRaisesRegex(LLMConfigurationError, "LLM_REASONING_EFFORT"):
                    request_chat_completion("system", "user")
                provider.assert_not_called()

    def test_invalid_thinking_option_fails_before_network(self) -> None:
        for flag, provider_name in (("invalid", "openai_compatible"), ("false", "ollama")):
            with self.subTest(flag=flag, provider=provider_name), patch.dict(os.environ, {
                "LLM_PROVIDER": provider_name, "LLM_MODEL": "test",
                "LLM_API_KEY": "fake-test-key", "LLM_ENABLE_THINKING": flag,
            }, clear=True), patch("src.optimization.llm.urlopen") as provider:
                with self.assertRaisesRegex(LLMConfigurationError, "LLM_ENABLE_THINKING"):
                    request_chat_completion("system", "user")
                provider.assert_not_called()

    def test_truncation_and_rate_limits_are_explicit_not_cached_success(self) -> None:
        with patch.dict(os.environ, {
            "LLM_MODEL": "test", "LLM_API_KEY": "private-test-key",
        }, clear=True), patch("src.optimization.llm.urlopen") as provider:
            provider.return_value.__enter__.return_value.read.return_value = (
                b'{"choices":[{"message":{"content":"{}"},"finish_reason":"length"}]}'
            )
            with self.assertRaisesRegex(LLMProviderError, "limite de 280"):
                request_chat_completion("system", "user", max_tokens=280)
            provider.side_effect = HTTPError(
                "https://api.openai.com/v1/chat/completions", 429, "Too many",
                {}, BytesIO(b"private-test-key"),
            )
            with self.assertRaisesRegex(LLMProviderError, "429") as error:
                request_chat_completion("system", "user")
            self.assertNotIn("private-test-key", str(error.exception))

    def test_service_unavailable_is_actionable_and_does_not_expose_response(self) -> None:
        with patch.dict(os.environ, {
            "LLM_MODEL": "test", "LLM_API_KEY": "private-test-key",
        }, clear=True), patch("src.optimization.llm.time.sleep"), patch(
            "src.optimization.llm.urlopen"
        ) as provider:
            provider.side_effect = HTTPError(
                "https://api.openai.com/v1/chat/completions", 503,
                "Service Unavailable", {}, BytesIO(b"private-test-key"),
            )
            with self.assertRaisesRegex(
                LLMProviderError, "temporariamente indisponível ou sobrecarregado"
            ) as error:
                request_chat_completion("system", "aggregate metrics only")
            self.assertIn("tente novamente", str(error.exception))
            self.assertNotIn("private-test-key", str(error.exception))
            self.assertEqual(provider.call_count, 3)

    def test_remote_503_recovers_with_bounded_backoff(self) -> None:
        with patch.dict(os.environ, {
            "LLM_MODEL": "test", "LLM_API_KEY": "private-test-key",
        }, clear=True), patch("src.optimization.llm.urlopen") as provider, patch(
            "src.optimization.llm.time.sleep"
        ) as sleep, patch(
            "src.optimization.llm.random.uniform", return_value=0.1
        ), self.assertLogs("src.optimization.llm", level="WARNING") as logs:
            errors = [
                HTTPError("https://api.openai.com/v1/chat/completions", 503,
                          "Service Unavailable", {}, BytesIO(b"private-test-key"))
                for _ in range(2)
            ]
            response = provider.return_value
            response.__enter__.return_value.read.return_value = (
                b'{"choices":[{"message":{"content":"recovered"}}]}'
            )
            provider.side_effect = [*errors, response]
            self.assertEqual(request_chat_completion("system", "user"), "recovered")
            self.assertEqual(provider.call_count, 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.1, 2.1])
            self.assertTrue(all(error.fp.closed for error in errors))
            self.assertNotIn("private-test-key", "\n".join(logs.output))

    def test_remote_503_does_not_retry_beyond_timeout_budget(self) -> None:
        with patch.dict(os.environ, {
            "LLM_MODEL": "test", "LLM_API_KEY": "private-test-key",
            "LLM_TIMEOUT_SECONDS": "1",
        }, clear=True), patch(
            "src.optimization.llm.urlopen",
            side_effect=HTTPError("https://api.openai.com/v1/chat/completions", 503,
                                  "Service Unavailable", {}, BytesIO()),
        ) as provider, patch("src.optimization.llm.time.sleep") as sleep:
            with self.assertRaisesRegex(LLMProviderError, "503"):
                request_chat_completion("system", "user")
            provider.assert_called_once()
            sleep.assert_not_called()

    def test_remote_retry_uses_only_remaining_timeout(self) -> None:
        for elapsed_after_sleep in (3, 10):
            with self.subTest(elapsed=elapsed_after_sleep), patch.dict(os.environ, {
                "LLM_MODEL": "test", "LLM_API_KEY": "private-test-key",
                "LLM_TIMEOUT_SECONDS": "10",
            }, clear=True), patch("src.optimization.llm.urlopen") as provider, patch(
                "src.optimization.llm.time.sleep"
            ), patch(
                "src.optimization.llm.random.uniform", return_value=0
            ), patch(
                "src.optimization.llm.time.monotonic",
                side_effect=[0, 0, 2, elapsed_after_sleep],
            ):
                response = provider.return_value
                response.__enter__.return_value.read.return_value = (
                    b'{"choices":[{"message":{"content":"recovered"}}]}'
                )
                provider.side_effect = [
                    HTTPError("https://api.openai.com/v1/chat/completions", 503,
                              "Service Unavailable", {}, BytesIO()),
                    response,
                ]
                if elapsed_after_sleep == 10:
                    with self.assertRaisesRegex(LLMProviderError, "503"):
                        request_chat_completion("system", "user")
                    provider.assert_called_once()
                else:
                    self.assertEqual(request_chat_completion("system", "user"), "recovered")
                    self.assertEqual(
                        [call.kwargs["timeout"] for call in provider.call_args_list],
                        [10, 7],
                    )

    def test_non_transient_errors_and_local_503_are_not_retried(self) -> None:
        for provider_name, code in (
            ("openai_compatible", 400), ("openai_compatible", 401),
            ("openai_compatible", 403), ("openai_compatible", 404),
            ("openai_compatible", 429), ("ollama", 503),
        ):
            with self.subTest(provider=provider_name, code=code), patch.dict(os.environ, {
                "LLM_PROVIDER": provider_name, "LLM_MODEL": "test",
                "LLM_API_KEY": "private-test-key",
            }, clear=True), patch(
                "src.optimization.llm.urlopen",
                side_effect=HTTPError("http://localhost", code, "Error", {}, BytesIO()),
            ) as provider, patch("src.optimization.llm.time.sleep") as sleep:
                with self.assertRaisesRegex(LLMProviderError, str(code)):
                    request_chat_completion("system", "user")
                provider.assert_called_once()
                sleep.assert_not_called()

    def test_gemini_model_not_found_is_actionable_and_does_not_expose_response(self) -> None:
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai_compatible",
            "LLM_MODEL": "gemini-3.1-flash-lite",
            "LLM_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai",
            "LLM_API_KEY": "private-test-key",
        }, clear=True), patch("src.optimization.llm.urlopen") as provider:
            provider.side_effect = HTTPError(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                404, "Not found", {}, BytesIO(b"private-test-key"),
            )
            with self.assertRaisesRegex(LLMProviderError, "gemini-3.1-flash-lite") as error:
                request_chat_completion("system", "aggregate metrics only")
            self.assertIn("GET /models", str(error.exception))
            self.assertNotIn("private-test-key", str(error.exception))


if __name__ == "__main__":
    unittest.main()
