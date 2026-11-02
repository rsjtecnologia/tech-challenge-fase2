from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from fastapi.testclient import TestClient

from src.optimization.experiments import (
    MODEL_NAMES,
    SEARCH_SPACES,
    _classifier,
    load_dataset,
    run_optimization,
)
from src.optimization.genetic import GAConfig, GeneticOptimizer
from src.optimization.llm import (
    LLMConfigurationError,
    LLMProviderError,
    _example_answer,
    _settings,
    build_interpretation_context,
    evaluate_interpretation,
    generate_interpretation,
    request_chat_completion,
    validate_configuration,
)
from src.web.app import app
import src.optimization.api as phase2_api


class GeneticOptimizerTests(unittest.TestCase):
    def test_runs_deterministically_and_returns_required_metrics(self) -> None:
        def evaluate(params: dict[str, object]) -> dict[str, float]:
            value = float(params["value"])
            return {
                "accuracy": value,
                "recall": value,
                "f1": value,
            }

        config = GAConfig(
            name="unit",
            population_size=4,
            generations=2,
            mutation_rate=0.5,
            cv_folds=2,
            tournament_size=2,
        )
        first = GeneticOptimizer({"value": (0.1, 0.5, 0.9)}, config, evaluate).run()
        second = GeneticOptimizer({"value": (0.1, 0.5, 0.9)}, config, evaluate).run()

        self.assertEqual(first.best_params, second.best_params)
        self.assertEqual(first.best_fitness, second.best_fitness)
        self.assertEqual(first.best_params, {"value": 0.9})
        self.assertEqual(len(first.history), 2)
        self.assertGreaterEqual(first.evaluations, 1)
        self.assertLessEqual(first.evaluations, 12)

    def test_rejects_invalid_configurations_and_incomplete_metrics(self) -> None:
        with self.assertRaises(ValueError):
            GAConfig("invalid", 1, 1, 0.1)
        optimizer = GeneticOptimizer(
            {"x": (1,)},
            GAConfig("unit", 2, 1, 0.1, tournament_size=1),
            lambda _: {"accuracy": 1.0},
        )
        with self.assertRaisesRegex(ValueError, "recall"):
            optimizer.run()

    def test_all_existing_models_have_search_spaces_and_factories(self) -> None:
        self.assertEqual(set(MODEL_NAMES), set(SEARCH_SPACES))
        for model_name in MODEL_NAMES:
            estimator = _classifier(model_name, {}, "breast_cancer")
            self.assertTrue(hasattr(estimator, "fit"))


class DatasetPreparationTests(unittest.TestCase):
    def test_phase_one_datasets_are_reproduced(self) -> None:
        for dataset, expected_rows, expected_columns in (
            ("breast_cancer", 569, 30),
            ("pcos", 541, 9),
        ):
            with self.subTest(dataset=dataset):
                features, target, _ = load_dataset(dataset)
                self.assertEqual(len(features), expected_rows)
                self.assertEqual(features.shape[1], expected_columns)
                self.assertEqual(len(target), expected_rows)
                self.assertTrue(set(target.unique()).issubset({0, 1}))
                self.assertFalse(
                    set(features.columns).intersection(
                        {"id", "Unnamed: 32", "Sl. No", "Patient File No."}
                    )
                )

    def test_unknown_dataset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "desconhecido"):
            load_dataset("unknown")

    def test_all_models_optimize_both_datasets_end_to_end(self) -> None:
        quick_config = GAConfig(
            name="test",
            population_size=2,
            generations=1,
            mutation_rate=0.2,
            cv_folds=2,
            tournament_size=2,
        )
        result = run_optimization(configs=(quick_config,))

        self.assertEqual(set(result["studies"]), {"breast_cancer", "pcos"})
        for study in result["studies"].values():
            self.assertEqual(set(study["models"]), set(MODEL_NAMES))
            for model in study["models"].values():
                self.assertEqual(
                    set(model["baseline"]["test_metrics"]),
                    {
                        "accuracy",
                        "recall",
                        "f1",
                        "precision",
                        "balanced_accuracy",
                    },
                )
                self.assertEqual(
                    len(model["experiments"]),
                    1,
                )


class LLMIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.study = {
            "label": "Estudo de teste",
            "positive_class": "classe positiva",
            "split": {"train": 80, "test": 20},
            "patient_rows": [{"name": "must not be sent"}],
            "models": {
                "Modelo": {
                    "baseline": {
                        "test_metrics": {
                            "recall": 0.5,
                            "precision": 0.6,
                            "f1": 0.55,
                            "accuracy": 0.7,
                        }
                    },
                    "optimized": {
                        "test_metrics": {
                            "recall": 0.4,
                            "precision": 0.65,
                            "f1": 0.62,
                            "accuracy": 0.72,
                        },
                        "cv_metrics": {"recall": 0.55},
                        "selected_experiment": "teste",
                    },
                }
            },
        }

    def test_context_contains_only_aggregate_metrics(self) -> None:
        context = build_interpretation_context(self.study)
        self.assertNotIn("patient_rows", context)
        self.assertNotIn("must not be sent", json.dumps(context))
        self.assertEqual(context["test_samples"], 20)
        self.assertEqual(
            context["future_textual_data_readiness"]["status"],
            "not_enabled",
        )

    def test_quality_rubric_checks_medical_disclaimer_and_certainty(self) -> None:
        response = {
            "summary": "O modelo apresentou métricas diferentes no teste.",
            "findings": [
                "O recall aumentou; valide o resultado em dados independentes."
            ],
            "limitations": ["O teste contém uma amostra limitada."],
            "disclaimer": "Isto não substitui avaliação clínica.",
        }
        evaluation = evaluate_interpretation(response)
        self.assertTrue(evaluation["passed"])
        response["summary"] = "O modelo garante um diagnóstico definitivo."
        self.assertFalse(evaluate_interpretation(response)["passed"])

    def test_missing_provider_uses_transparent_local_interpretation(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = generate_interpretation(
                build_interpretation_context(self.study)
            )
        self.assertEqual(result["source"], "local")
        self.assertTrue(result["quality_evaluation"]["passed"])
        interpretation = result["interpretation"]
        self.assertIn("holdout", interpretation["summary"])
        self.assertTrue(
            any("recall caiu" in finding for finding in interpretation["findings"])
        )
        self.assertTrue(
            any("validação externa" in limitation for limitation in interpretation["limitations"])
        )
        self.assertIn("não substitui", interpretation["disclaimer"])

    def test_provider_errors_are_not_hidden_by_local_fallback(self) -> None:
        with patch(
            "src.optimization.llm.request_chat_completion",
            side_effect=LLMProviderError("provedor indisponível"),
        ):
            with self.assertRaisesRegex(LLMProviderError, "indisponível"):
                generate_interpretation(
                    build_interpretation_context(self.study)
                )

    def test_question_is_grounded_in_aggregate_metrics_and_preserves_negation(self) -> None:
        response = {
            "summary": "O recall do modelo não melhorou: caiu de 50% para 40%.",
            "findings": ["A precisão passou de 60% para 65%; revise os resultados."],
            "limitations": ["Valide em dados independentes."],
            "disclaimer": "Não substitui avaliação clínica.",
        }
        with patch(
            "src.optimization.llm.request_chat_completion",
            return_value=json.dumps(response),
        ) as provider:
            result = generate_interpretation(
                build_interpretation_context(self.study),
                question="  Por que o recall não melhorou?  ",
            )
        self.assertEqual(result["source"], "llm")
        system, prompt = provider.call_args.args
        self.assertIn("conteúdo não confiável", system)
        self.assertIn("Não analise pacientes", system)
        payload = json.loads(prompt[prompt.index("{"):])
        self.assertEqual(payload["question"], "Por que o recall não melhorou?")
        self.assertEqual(
            payload["aggregate_metrics"]["models"]["Modelo"]["optimized"],
            [40.0, 65.0, 62.0, 72.0],
        )
        self.assertNotIn("must not be sent", prompt)
        self.assertEqual(result["interpretation"], response)

    def test_question_without_llm_fails_instead_of_answering_by_rules(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(LLMConfigurationError, "exigem uma LLM"):
                generate_interpretation(
                    build_interpretation_context(self.study),
                    question="Qual modelo melhorou?",
                )

    def test_question_provider_failure_remains_explicit(self) -> None:
        with patch(
            "src.optimization.llm.request_chat_completion",
            side_effect=LLMProviderError("indisponível"),
        ), self.assertRaisesRegex(LLMProviderError, "indisponível"):
            generate_interpretation(
                build_interpretation_context(self.study), question="Qual o recall?"
            )

    def test_all_badge_questions_have_verified_evidence_for_both_studies(self) -> None:
        questions = (
            "Qual modelo teve o maior recall após a otimização?",
            "Em quais modelos o recall caiu após a otimização?",
            "Qual a diferença entre recall e precisão neste estudo?",
            "Compare o F1 da Regressao Logistica antes e depois da otimização.",
        )
        context = build_interpretation_context(self.study)
        context["models"]["Regressao Logistica"] = context["models"].pop("Modelo")
        for study in ("Mama", "PCOS"):
            context["study"] = study
            for question in questions:
                with self.subTest(study=study, question=question):
                    self.assertIsNotNone(_example_answer(context, question))
        self.assertIn("40.00%", _example_answer(context, questions[0]))
        self.assertIn("50.00% para 40.00%", _example_answer(context, questions[1]))
        self.assertIn("+7.00 pontos", _example_answer(context, questions[3]))
        self.assertIsNone(_example_answer(context, "Outra pergunta"))

    def test_badge_prompt_uses_verified_result_instead_of_all_metrics(self) -> None:
        response = {
            "summary": "O maior recall do modelo foi 40%.",
            "findings": ["Revise o recall."],
            "limitations": ["Validação externa necessária."],
            "disclaimer": "Não substitui avaliação clínica.",
        }
        with patch(
            "src.optimization.llm.request_chat_completion",
            return_value=json.dumps(response),
        ) as provider:
            generate_interpretation(
                build_interpretation_context(self.study),
                question="Qual modelo teve o maior recall após a otimização?",
            )
        prompt = provider.call_args.args[1]
        payload = json.loads(prompt[prompt.index("{"):])
        self.assertIn("40.00%", payload["verified_result"])
        self.assertIn("validação externa", payload["limitation"])
        self.assertNotIn("aggregate_metrics", payload)

    def test_ollama_provider_uses_local_defaults_without_api_key(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "ollama"},
            clear=True,
        ):
            self.assertTrue(validate_configuration())
            base_url, model, api_key, timeout = _settings()
        self.assertEqual(base_url, "http://localhost:11434")
        self.assertEqual(model, "llama3.2:latest")
        self.assertIsNone(api_key)
        self.assertEqual(timeout, 60)

    def test_ollama_native_payload_and_response(self) -> None:
        with patch.dict(
            os.environ, {"LLM_PROVIDER": "ollama"}, clear=True
        ), patch("src.optimization.llm.urlopen") as provider:
            provider.return_value.__enter__.return_value.read.return_value = (
                b'{"message":{"content":"response"},"done_reason":"stop",'
                b'"eval_count":10,"prompt_eval_count":20}'
            )
            self.assertEqual(
                request_chat_completion("system", "user", json_mode=True),
                "response",
            )
        request = provider.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(
            payload["options"], {"temperature": 0.1, "num_predict": 500}
        )
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["keep_alive"], "30m")
        self.assertEqual(payload["format"], "json")
        self.assertIsNone(request.get_header("Authorization"))

    def test_ollama_accepts_legacy_v1_url_but_rejects_remote_hosts(self) -> None:
        for base in ("http://localhost:11434/v1", "http://127.0.0.1:11434"):
            with patch.dict(
                os.environ, {"LLM_PROVIDER": "ollama", "LLM_BASE_URL": base},
                clear=True,
            ):
                self.assertEqual(_settings()[0], base.removesuffix("/v1"))
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "ollama", "LLM_BASE_URL": "https://localhost.example.com"},
            clear=True,
        ):
            with self.assertRaises(LLMConfigurationError):
                _settings()

    def test_ollama_options_can_be_overridden(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "ollama", "LLM_TEMPERATURE": "0.3", "LLM_NUM_PREDICT": "800"},
            clear=True,
        ), patch("src.optimization.llm.urlopen") as provider:
            provider.return_value.__enter__.return_value.read.return_value = (
                b'{"message":{"content":"response"}}'
            )
            request_chat_completion("system", "user")
        self.assertEqual(
            json.loads(provider.call_args.args[0].data)["options"],
            {"temperature": 0.3, "num_predict": 800},
        )

    def test_ollama_truncated_response_is_explicit(self) -> None:
        with patch.dict(
            os.environ, {"LLM_PROVIDER": "ollama"}, clear=True
        ), patch("src.optimization.llm.urlopen") as provider:
            provider.return_value.__enter__.return_value.read.return_value = (
                b'{"message":{"content":"partial"},"done_reason":"length"}'
            )
            with self.assertRaisesRegex(LLMProviderError, "500 tokens"):
                request_chat_completion("system", "user")

    def test_invalid_generation_settings_are_explicit(self) -> None:
        for settings in (
            {"LLM_TEMPERATURE": "nan"},
            {"LLM_TEMPERATURE": "invalid"},
            {"LLM_NUM_PREDICT": "0"},
            {"LLM_NUM_PREDICT": "1.5"},
        ):
            with self.subTest(settings=settings), patch.dict(
                os.environ, {"LLM_PROVIDER": "ollama", **settings}, clear=True
            ):
                with self.assertRaises(LLMConfigurationError):
                    request_chat_completion("system", "user")

    def test_ollama_timeout_can_be_overridden(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "ollama", "LLM_TIMEOUT_SECONDS": "900"},
            clear=True,
        ):
            self.assertEqual(_settings()[3], 900)

    def test_timeout_reports_duration_and_configuration_guidance(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "ollama", "LLM_TIMEOUT_SECONDS": "120"},
            clear=True,
        ), patch(
            "src.optimization.llm.urlopen",
            side_effect=TimeoutError(),
        ) as provider_request:
            with self.assertRaisesRegex(
                LLMProviderError, "120s.*LLM_TIMEOUT_SECONDS"
            ):
                generate_interpretation(
                    build_interpretation_context(self.study)
                )
        self.assertEqual(provider_request.call_args.kwargs["timeout"], 120)

    def test_unavailable_ollama_returns_setup_instructions(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "ollama"},
            clear=True,
        ), patch(
            "src.optimization.llm.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with self.assertRaisesRegex(
                LLMProviderError,
                r"ollama pull llama3\.2:latest",
            ):
                generate_interpretation(
                    build_interpretation_context(self.study)
                )

    def test_missing_provider_configuration_keeps_local_mode_available(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(validate_configuration())

    def test_openai_compatible_response_is_parsed_and_evaluated(self) -> None:
        interpretation = {
            "summary": "As métricas variam entre os modelos avaliados.",
            "findings": [
                "O recall mede a identificação da classe positiva; valide em dados independentes."
            ],
            "limitations": ["Os dados representam uma amostra de teste."],
            "disclaimer": "O resultado não substitui avaliação clínica.",
        }
        response_body = json.dumps(
            {
                "choices": [
                    {"message": {"content": json.dumps(interpretation)}}
                ]
            }
        ).encode("utf-8")

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return response_body

        with patch.dict(
            os.environ,
            {
                "LLM_BASE_URL": "http://localhost:11434/v1",
                "LLM_MODEL": "local-test-model",
                "LLM_API_KEY": "",
            },
            clear=True,
        ), patch(
            "src.optimization.llm.urlopen", return_value=FakeResponse()
        ) as provider_request:
            result = generate_interpretation(
                build_interpretation_context(self.study)
            )
        sent_request = provider_request.call_args.args[0]
        self.assertEqual(
            json.loads(sent_request.data)["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(provider_request.call_args.kwargs["timeout"], 45)
        self.assertEqual(result["interpretation"], interpretation)
        self.assertTrue(result["quality_evaluation"]["passed"])
        self.assertEqual(result["source"], "llm")


class Phase2APITests(unittest.TestCase):
    def test_liveness_endpoint(self) -> None:
        with TestClient(app) as client:
            response = client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_both_dashboard_pages_expose_the_interpretation_control(self) -> None:
        with TestClient(app) as client:
            breast_page = client.get("/")
            pcos_page = client.get("/pcos")
        for response in (breast_page, pcos_page):
            self.assertEqual(response.status_code, 200)
            self.assertIn("interpretButton", response.text)
            self.assertIn("geneticComparison", response.text)
            self.assertIn('id="interpretationComparison"', response.text)
            self.assertIn(
                'id="interpretationComparisonSection" hidden',
                response.text,
            )
            self.assertIn("/js/interpretation.js", response.text)
            self.assertIn('id="interpretationQuestionForm"', response.text)
            self.assertIn('maxlength="500"', response.text)
            self.assertIn('class="question-badge"', response.text)
            self.assertEqual(response.text.count('data-question='), 4)
            self.assertNotIn("Pergunte sobre os resultados deste estudo (PLN com LLM)", response.text)
            self.assertIn("não inclua nomes, sintomas ou dados de pacientes", response.text)
            self.assertLess(
                response.text.index('id="interpretationComparison"'),
                response.text.index('id="geneticComparison"'),
            )

    def test_interpretation_requires_generated_results(self) -> None:
        with patch.object(
            phase2_api,
            "RESULTS_PATH",
            Path(tempfile.gettempdir()) / "missing-phase2-results.json",
        ), TestClient(app) as client:
            response = client.post(
                "/api/interpretacao",
                json={"estudo": "breast_cancer"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("src.optimization.cli", response.json()["detail"])

    def test_questions_are_validated_and_forwarded_for_both_studies(self) -> None:
        for study in ("breast_cancer", "pcos"):
            with self.subTest(study=study), patch.object(
                phase2_api, "_load_study", return_value={}
            ), patch.object(
                phase2_api, "build_interpretation_context", return_value={"study": study}
            ), patch.object(
                phase2_api, "generate_interpretation", return_value={"source": "llm"}
            ) as generate, TestClient(app) as client:
                result = client.post("/api/interpretacao", json={
                    "estudo": study, "pergunta": "  O recall não melhorou?  "
                })
                self.assertEqual(result.status_code, 200)
                generate.assert_called_once_with(
                    {"study": study}, question="O recall não melhorou?"
                )
                for question in ("", "   ", "x" * 501, 42):
                    response = client.post("/api/interpretacao", json={
                        "estudo": study, "pergunta": question
                    })
                    self.assertEqual(response.status_code, 422)

    def test_question_configuration_and_provider_errors_return_explicit_status(self) -> None:
        for error, status in (
            (LLMConfigurationError("Configure Ollama"), 503),
            (LLMProviderError("Ollama indisponível"), 502),
        ):
            with self.subTest(status=status), patch.object(
                phase2_api, "_load_study", return_value={}
            ), patch.object(
                phase2_api, "build_interpretation_context", return_value={}
            ), patch.object(
                phase2_api, "generate_interpretation", side_effect=error
            ), TestClient(app) as client:
                response = client.post("/api/interpretacao", json={
                    "estudo": "breast_cancer", "pergunta": "Qual o recall?"
                })
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["detail"], str(error))

    def test_interpretation_returns_local_result_without_provider_configuration(self) -> None:
        result_payload = {
            "studies": {
                "breast_cancer": {
                    "label": "Estudo",
                    "positive_class": "positivo",
                    "split": {"train": 80, "test": 20},
                    "models": {
                        "Modelo": {
                            "baseline": {
                                "test_metrics": {
                                    "recall": 0.5,
                                    "precision": 0.6,
                                    "f1": 0.55,
                                    "accuracy": 0.7,
                                }
                            },
                            "optimized": {
                                "test_metrics": {
                                    "recall": 0.6,
                                    "precision": 0.65,
                                    "f1": 0.62,
                                    "accuracy": 0.72,
                                },
                                "cv_metrics": {"recall": 0.55},
                                "selected_experiment": "teste",
                            },
                        }
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps(result_payload), encoding="utf-8")
            with patch.object(phase2_api, "RESULTS_PATH", path), patch.dict(
                os.environ, {}, clear=True
            ), TestClient(app) as client:
                response = client.post(
                    "/api/interpretacao",
                    json={"estudo": "breast_cancer"},
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source"], "local")
        self.assertIn("findings", response.json()["interpretation"])

    def test_optimization_endpoint_returns_baseline_and_experiment_results(self) -> None:
        result_payload = {
            "fitness": {"formula": "recall + f1 + accuracy"},
            "studies": {
                "breast_cancer": {
                    "dataset": "breast_cancer",
                    "label": "Estudo",
                    "positive_class": "positivo",
                    "models": {
                        "Modelo": {
                            "baseline": {"test_metrics": {"recall": 0.5}},
                            "optimized": {
                                "selected_experiment": "ampla",
                                "hyperparameters": {"C": 1.0},
                                "test_metrics": {"recall": 0.6},
                            },
                            "experiments": [
                                {
                                    "name": "ampla",
                                    "best_fitness": 0.6,
                                    "best_cv_metrics": {"recall": 0.6},
                                    "unique_candidates_evaluated": 3,
                                    "config": {"population_size": 4},
                                }
                            ],
                        }
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps(result_payload), encoding="utf-8")
            with patch.object(
                phase2_api, "RESULTS_PATH", path
            ), TestClient(app) as client:
                response = client.get("/api/otimizacao/breast_cancer")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        model = body["models"]["Modelo"]
        self.assertEqual(model["baseline_metrics"]["recall"], 0.5)
        self.assertEqual(model["optimized_metrics"]["recall"], 0.6)
        self.assertEqual(len(model["experiments"]), 1)


if __name__ == "__main__":
    unittest.main()
