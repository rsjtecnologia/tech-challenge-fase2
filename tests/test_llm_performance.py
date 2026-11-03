from copy import deepcopy
import json
import os
import unittest
from unittest.mock import patch

from src.logistics import llm as routes
from src.logistics.genetic import GAConfig, optimize_routes
from src.logistics.scenario import DEMO_SCENARIO
from src.optimization.cache import InterpretationCache
from src.optimization.llm import (
    LLMConfigurationError,
    LLMProviderError,
    generate_interpretation,
    request_chat_completion,
)


class RouteInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = optimize_routes(
            DEMO_SCENARIO, GAConfig(population_size=4, generations=1, tournament_size=2)
        )

    def setUp(self) -> None:
        cache = InterpretationCache(
            is_cacheable=lambda response: response.get("source") == "llm"
        )
        self.cache_patch = patch.object(routes, "ROUTE_CONTENT_CACHE", cache)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

    def test_prompt_budgets_and_no_duplicate_paths(self) -> None:
        for action in ("instructions", "daily_report", "question"):
            context = routes._prompt_context(self.plan, action, None)
            size = len(json.dumps(context, ensure_ascii=False, separators=(",", ":")))
            self.assertLess(size, 9000 if action == "question" else 1200)
            self.assertNotIn('"path"', json.dumps(context))
            self.assertNotIn('"latitude"', json.dumps(context))

    def test_municipality_retrieval_preserves_verified_assignment(self) -> None:
        plan = deepcopy(self.plan)
        stop = plan["comparison"]["optimized"]["routes"][0]["stops"][0]
        stop["name"] = "Polo simulado — Campinas/SP"
        context = routes._prompt_context(
            plan, "question", None, "Qual veículo atende Campinas?"
        )
        self.assertNotIn("routes", context)
        self.assertEqual(len(context["matched_destinations"]), 1)
        self.assertEqual(
            context["matched_destinations"][0]["vehicle"],
            plan["comparison"]["optimized"]["routes"][0]["vehicle_name"],
        )
        self.assertIn("globais", context["retrieval_scope"])
        self.assertIn("routes", routes._prompt_context(
            plan, "question", None, "Compare os veículos."
        ))

    def test_llm_does_not_recreate_or_change_driver_stops(self) -> None:
        response = {
            "summary": "Plano simulado.",
            "operational_checks": ["Confirmar os horários."],
            "disclaimer": "Confirme com a coordenação.",
            "vehicle_instructions": [{"vehicle": "inventado"}],
        }
        with patch.object(routes, "request_chat_completion", return_value=json.dumps(response)) as provider:
            result = routes.generate_route_content(self.plan, "instructions")
        expected = routes._local_route_content(self.plan, "instructions", None, None)["content"]
        self.assertEqual(result["content"]["vehicle_instructions"], expected["vehicle_instructions"])
        self.assertTrue(result["verified_plan_fields"])
        self.assertNotIn("vehicle_instructions", provider.call_args.kwargs["response_schema"]["properties"])
        self.assertTrue(provider.call_args.kwargs["json_mode"])
        self.assertEqual(provider.call_args.kwargs["max_tokens"], 240)

    def test_report_metrics_are_exact_and_results_are_cached(self) -> None:
        response = {
            "period": "Período inventado",
            "summary": "Resumo da simulação.",
            "key_metrics": {"distancia_otimizada_km": -999},
            "improvements": ["Medir tempos reais."],
            "caveats": ["Distância em linha reta."],
            "disclaimer": "Confirme com a coordenação.",
        }
        with patch.object(routes, "request_chat_completion", return_value=json.dumps(response)) as provider:
            first = routes.generate_route_content(self.plan, "daily_report")
            second = routes.generate_route_content(self.plan, "daily_report")
            self.assertFalse(first["cache"]["hit"])
            self.assertTrue(second["cache"]["hit"])
            provider.assert_called_once()
            self.assertEqual(provider.call_args.kwargs["max_tokens"], 280)
            self.assertEqual(
                second["content"]["key_metrics"]["distancia_otimizada_km"],
                self.plan["comparison"]["optimized"]["total_distance_km"],
            )
            changed = deepcopy(self.plan)
            changed["scenario_id"] += "-changed"
            routes.generate_route_content(changed, "daily_report")
            self.assertEqual(provider.call_count, 2)
            with patch.dict(os.environ, {"LLM_MODEL": "other-model"}):
                routes.generate_route_content(self.plan, "daily_report")
            self.assertEqual(provider.call_count, 3)

    def test_errors_and_invalid_configuration_do_not_become_cached_success(self) -> None:
        for error in (
            LLMProviderError("offline"), LLMConfigurationError("invalid settings"),
        ):
            with patch.object(routes, "request_chat_completion", side_effect=error) as provider:
                for _ in range(2):
                    with self.assertRaises(type(error)):
                        routes.generate_route_content(self.plan, "daily_report")
                self.assertEqual(provider.call_count, 2)

    def test_history_changes_invalidate_weekly_cache(self) -> None:
        response = {
            "period": "Sete dias", "summary": "Simulações.",
            "improvements": [], "caveats": [], "disclaimer": "Confirme.",
        }
        with patch.object(routes, "request_chat_completion", return_value=json.dumps(response)) as provider:
            for count in (1, 1, 2):
                routes.generate_route_content(
                    self.plan, "weekly_report", history={"run_count": count}
                )
        self.assertEqual(provider.call_count, 2)


class SharedProviderBudgetTests(unittest.TestCase):
    def test_clinical_prompts_request_short_answers_with_smaller_token_budgets(self) -> None:
        metrics = dict(recall=0.8, precision=0.9, f1=0.85, accuracy=0.9)
        context = {
            "study": "PCOS", "positive_class": "PCOS", "test_samples": 20,
            "models": {"Modelo": {
                "baseline_test_metrics": metrics, "optimized_test_metrics": metrics,
            }},
        }
        response = {
            "summary": "O recall do modelo exige validação.",
            "findings": ["Valide as métricas em dados independentes."],
            "limitations": ["Holdout único."],
            "disclaimer": "Não substitui avaliação clínica.",
        }
        for question, budget in ((None, 280), ("Compare recall e precisão.", 240)):
            with self.subTest(question=question), patch(
                "src.optimization.llm.request_chat_completion",
                return_value=json.dumps(response),
            ) as provider:
                result = generate_interpretation(context, question=question)
                self.assertEqual(provider.call_args.kwargs["max_tokens"], budget)
                self.assertTrue(result["quality_evaluation"]["passed"])
                self.assertIn("recall", provider.call_args.args[1])
                self.assertIn("precision", provider.call_args.args[1])

    def test_badge_summary_uses_verified_metrics_not_llm_arithmetic(self) -> None:
        metrics = {"recall": 0.5833, "precision": 0.9, "f1": 0.7, "accuracy": 0.8}
        context = {
            "study": "PCOS", "positive_class": "PCOS", "test_samples": 20,
            "models": {"Regressao Logistica": {
                "baseline_test_metrics": metrics,
                "optimized_test_metrics": metrics,
            }},
        }
        response = {
            "summary": "Precisão 100%, afirmação errada.",
            "findings": ["Recall: valide em amostra independente."],
            "limitations": ["Holdout único."],
            "disclaimer": "Não substitui avaliação clínica.",
        }
        with patch("src.optimization.llm.request_chat_completion", return_value=json.dumps(response)):
            result = generate_interpretation(
                context, question="Qual modelo teve o maior recall após a otimização?"
            )
        self.assertEqual(result["interpretation"]["summary"], "Maior recall otimizado: Regressao Logistica; 58.33%.")
        self.assertTrue(result["verified_summary"])
        self.assertTrue(result["quality_evaluation"]["passed"])

    def test_native_schema_and_task_budget_preserve_explicit_overrides(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        for override, expected in (({}, 280), ({"LLM_NUM_PREDICT": "600"}, 600)):
            with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", **override}, clear=True), patch(
                "src.optimization.llm.urlopen"
            ) as provider:
                provider.return_value.__enter__.return_value.read.return_value = (
                    b'{"message":{"content":"{}"}}'
                )
                request_chat_completion("system", "user", max_tokens=280, response_schema=schema)
            payload = json.loads(provider.call_args.args[0].data)
            self.assertEqual(payload["options"]["num_predict"], expected)
            self.assertEqual(payload["format"], schema)

    def test_openai_compatible_completion_is_also_bounded(self) -> None:
        with patch.dict(os.environ, {
            "LLM_MODEL": "test-model", "LLM_BASE_URL": "http://localhost:11434/v1",
        }, clear=True), patch(
            "src.optimization.llm.urlopen"
        ) as provider:
            provider.return_value.__enter__.return_value.read.return_value = (
                b'{"choices":[{"message":{"content":"{}"}}]}'
            )
            request_chat_completion("system", "user", json_mode=True, max_tokens=360)
        payload = json.loads(provider.call_args.args[0].data)
        self.assertEqual(payload["max_tokens"], 360)
        self.assertEqual(payload["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
