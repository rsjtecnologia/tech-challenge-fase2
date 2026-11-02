from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.logistics import service
from src.logistics.genetic import GAConfig, decode_chromosome, optimize_routes
from src.logistics.llm import generate_route_content
from src.logistics.scenario import DEMO_SCENARIO
from src.optimization.llm import LLMConfigurationError, LLMNotConfiguredError, LLMProviderError
from src.web.app import app


class RouteOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GAConfig(
            population_size=20,
            generations=30,
            mutation_rate=0.2,
            tournament_size=3,
            seed=21,
        )

    def test_genetic_vrp_serves_every_delivery_with_hard_constraints(self) -> None:
        result = optimize_routes(DEMO_SCENARIO, self.config)
        optimized = result["comparison"]["optimized"]
        stops = [
            stop
            for route in optimized["routes"]
            for stop in route["stops"]
        ]
        served_ids = [stop["delivery_id"] for stop in stops]
        self.assertEqual(len(served_ids), len(set(served_ids)))
        self.assertEqual(set(served_ids), {item.id for item in DEMO_SCENARIO.deliveries})
        self.assertEqual(optimized["unserved_delivery_ids"], [])
        self.assertEqual(
            optimized["served_delivery_count"],
            len(DEMO_SCENARIO.deliveries),
        )
        self.assertGreater(optimized["estimated_vehicle_minutes"], 0)
        self.assertEqual(
            optimized["total_load_kg"],
            sum(delivery.demand_kg for delivery in DEMO_SCENARIO.deliveries),
        )
        self.assertGreater(optimized["fleet_utilization_percent"], 0)
        self.assertLess(
            optimized["total_distance_km"],
            result["comparison"]["baseline"]["total_distance_km"],
        )

        vehicles = {vehicle.id: vehicle for vehicle in DEMO_SCENARIO.vehicles}
        deliveries = {delivery.id: delivery for delivery in DEMO_SCENARIO.deliveries}
        for route in optimized["routes"]:
            vehicle = vehicles[route["vehicle_id"]]
            self.assertLessEqual(route["load_kg"], vehicle.capacity_kg)
            self.assertLessEqual(route["distance_km"], vehicle.max_distance_km)
            for stop in route["stops"]:
                delivery = deliveries[stop["delivery_id"]]
                if delivery.requires_refrigeration:
                    self.assertTrue(vehicle.refrigerated)

        self.assertEqual(
            result["algorithm"]["config"]["seed"],
            self.config.seed,
        )
        self.assertEqual(len(result["algorithm"]["history"]), 30)

    def test_reproducible_genome_is_a_permutation(self) -> None:
        first = optimize_routes(DEMO_SCENARIO, self.config)
        second = optimize_routes(DEMO_SCENARIO, self.config)
        self.assertEqual(
            first["comparison"]["optimized"]["fitness_cost"],
            second["comparison"]["optimized"]["fitness_cost"],
        )
        self.assertEqual(
            first["comparison"]["optimized"]["total_distance_km"],
            second["comparison"]["optimized"]["total_distance_km"],
        )

    def test_decoder_rejects_non_permutation_chromosome(self) -> None:
        with self.assertRaisesRegex(ValueError, "exatamente uma vez"):
            decode_chromosome(("entrega-01",), DEMO_SCENARIO)

    def test_priority_changes_the_cost_of_the_same_delivery_delay(self) -> None:
        delivery = replace(
            DEMO_SCENARIO.deliveries[0],
            deadline_minute=8 * 60 + 1,
        )
        critical_scenario = replace(
            DEMO_SCENARIO,
            vehicles=(DEMO_SCENARIO.vehicles[0],),
            deliveries=(replace(delivery, priority="critica"),),
        )
        regular_scenario = replace(
            critical_scenario,
            deliveries=(replace(delivery, priority="regular"),),
        )

        critical = decode_chromosome(
            (delivery.id,),
            critical_scenario,
        )
        regular = decode_chromosome(
            (delivery.id,),
            regular_scenario,
        )

        self.assertGreater(critical.total_weighted_lateness, 0)
        self.assertAlmostEqual(
            critical.total_weighted_lateness,
            regular.total_weighted_lateness * 8,
            places=1,
        )
        self.assertGreater(critical.fitness, regular.fitness)

    def test_invalid_ga_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GAConfig(population_size=2)
        with self.assertRaises(ValueError):
            GAConfig(mutation_rate=1.1)

    def test_saved_plan_is_atomic_and_can_be_loaded(self) -> None:
        result = optimize_routes(DEMO_SCENARIO, self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.json"
            with patch.object(service, "RESULTS_PATH", path):
                service.save_route_plan(result)
                loaded = service.load_route_plan()
            self.assertEqual(
                loaded["comparison"]["optimized"]["total_distance_km"],
                result["comparison"]["optimized"]["total_distance_km"],
            )
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_old_saved_plan_is_enriched_with_resource_metrics(self) -> None:
        result = optimize_routes(DEMO_SCENARIO, self.config)
        for evaluation in result["comparison"].values():
            if not isinstance(evaluation, dict):
                continue
            evaluation.pop("estimated_vehicle_minutes", None)
            evaluation.pop("vehicles_used", None)
            evaluation.pop("total_load_kg", None)
            evaluation.pop("fleet_utilization_percent", None)
            for route in evaluation["routes"]:
                route.pop("speed_kmh", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.json"
            with patch.object(service, "RESULTS_PATH", path):
                service.save_route_plan(result)
                loaded = service.load_route_plan()
        optimized = loaded["comparison"]["optimized"]
        self.assertGreater(optimized["estimated_vehicle_minutes"], 0)
        self.assertEqual(
            optimized["vehicles_used"],
            len(optimized["routes"]),
        )

    def test_saved_plan_from_previous_scenario_version_is_rejected(self) -> None:
        result = optimize_routes(DEMO_SCENARIO, self.config)
        result["scenario_id"] = "demo-sp-sintetico"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.json"
            with patch.object(service, "RESULTS_PATH", path):
                service.save_route_plan(result)
                with self.assertRaisesRegex(FileNotFoundError, "versão anterior"):
                    service.load_route_plan()

    def test_route_history_aggregates_only_last_seven_days(self) -> None:
        now = datetime.now(timezone.utc)
        recent_first = service._history_record(self.plan_for_history())
        recent_first["recorded_at"] = (now - timedelta(days=6)).isoformat()
        recent_first["optimized"]["distance_km"] = 270.0
        recent_first["baseline"]["distance_km"] = 300.0
        recent_second = deepcopy(recent_first)
        recent_second["run_id"] = "recent-second"
        recent_second["recorded_at"] = (now - timedelta(days=1)).isoformat()
        recent_second["optimized"]["distance_km"] = 210.0
        recent_second["baseline"]["distance_km"] = 300.0
        old_run = deepcopy(recent_first)
        old_run["run_id"] = "old-run"
        old_run["recorded_at"] = (now - timedelta(days=9)).isoformat()

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                service,
                "HISTORY_PATH",
                Path(directory) / "history.json",
            ):
                service._save_history([recent_first, recent_second, old_run])
                summary = service.route_history_summary(now=now)

        self.assertEqual(summary["run_count"], 2)
        self.assertTrue(summary["trend_available"])
        self.assertEqual(summary["mean_optimized_distance_km"], 240.0)
        self.assertEqual(summary["mean_baseline_distance_km"], 300.0)
        self.assertEqual(
            summary["mean_estimated_vehicle_minutes_saved"],
            round(
                (
                    recent_first["baseline"]["estimated_vehicle_minutes"]
                    - recent_first["optimized"]["estimated_vehicle_minutes"]
                    + recent_second["baseline"]["estimated_vehicle_minutes"]
                    - recent_second["optimized"]["estimated_vehicle_minutes"]
                )
                / 2,
                2,
            ),
        )
        self.assertEqual(len(summary["recent_runs"]), 2)

    def plan_for_history(self) -> dict:
        return optimize_routes(
            DEMO_SCENARIO,
            GAConfig(population_size=8, generations=2, tournament_size=2),
        )


class RouteLLMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = optimize_routes(
            DEMO_SCENARIO,
            GAConfig(population_size=12, generations=12),
        )

    def test_llm_receives_only_scenario_and_route_data(self) -> None:
        response = json.dumps(
            {
                "answer": "O plano possui cinco veículos.",
                "supporting_facts": ["O cenário disponibiliza cinco veículos."],
                "missing_information": ["Trânsito em tempo real não foi fornecido."],
                "disclaimer": "Confirme com a coordenação.",
            }
        )
        with patch(
            "src.logistics.llm.request_chat_completion",
            return_value=response,
        ) as client:
            result = generate_route_content(
                self.plan,
                "question",
                "Quantos veículos existem?",
            )
        prompt = client.call_args.args[1]
        self.assertIn(DEMO_SCENARIO.id, prompt)
        self.assertNotIn("patient", prompt.casefold())
        self.assertEqual(result["content"]["answer"], "O plano possui cinco veículos.")

    def test_prompts_exist_for_instructions_daily_and_weekly_reports(self) -> None:
        responses = {
            "instructions": {
                "summary": "Plano demonstrativo.",
                "vehicle_instructions": [
                    {
                        "vehicle": "Veículo A",
                        "ordered_stops": ["Unidade fictícia"],
                        "notes": ["Confirmar a entrega."],
                    }
                ],
                "operational_checks": ["Confirmar o horário."],
                "disclaimer": "Confirme com a coordenação.",
            },
            "daily_report": {
                "period": "Plano diário simulado",
                "summary": "Resumo ilustrativo.",
                "key_metrics": {"distance_km": 30},
                "improvements": ["Medir distâncias reais."],
                "caveats": ["Uma única simulação."],
                "disclaimer": "Não representa entregas realizadas.",
            },
            "weekly_report": {
                "period": "Plano semanal simulado",
                "summary": "Resumo ilustrativo.",
                "key_metrics": {"distance_km": 30},
                "improvements": ["Coletar dados por dia."],
                "caveats": ["Sem histórico semanal."],
                "disclaimer": "Não representa uma semana real.",
            },
        }
        for action, payload in responses.items():
            with self.subTest(action=action), patch(
                "src.logistics.llm.request_chat_completion",
                return_value=json.dumps(payload),
            ):
                result = generate_route_content(self.plan, action)
                self.assertEqual(result["action"], action)
                self.assertEqual(result["source"], "llm")

    def test_unconfigured_llm_uses_grounded_local_assistant(self) -> None:
        with patch(
            "src.logistics.llm.request_chat_completion",
            side_effect=LLMNotConfiguredError("Configure LLM_MODEL."),
        ):
            question = generate_route_content(
                self.plan,
                "question",
                "Qual veículo leva os itens refrigerados?",
            )
            instructions = generate_route_content(self.plan, "instructions")
            report = generate_route_content(self.plan, "weekly_report")

        self.assertEqual(question["source"], "local")
        self.assertIn("refrigerado", question["content"]["answer"])
        self.assertIn(
            "Veículo refrigerado A",
            " ".join(question["content"]["supporting_facts"]),
        )
        self.assertEqual(instructions["source"], "local")
        self.assertTrue(instructions["content"]["vehicle_instructions"])
        self.assertEqual(report["source"], "local")
        self.assertIn("1 simulação", report["content"]["period"])

    def test_weekly_report_uses_history_and_only_claims_observed_patterns(self) -> None:
        history = {
            "run_count": 2,
            "minimum_runs_for_trend": 2,
            "trend_available": True,
            "first_run_at": "2026-10-10T09:00:00+00:00",
            "last_run_at": "2026-10-15T09:00:00+00:00",
            "mean_optimized_distance_km": 240.0,
            "mean_baseline_distance_km": 300.0,
            "mean_late_delivery_count": 0,
            "mean_weighted_lateness": 0,
            "recent_runs": [
                {
                    "optimized_distance_km": 270.0,
                    "baseline_distance_km": 300.0,
                    "optimized_estimated_vehicle_minutes": 500,
                    "late_delivery_count": 0,
                    "weighted_lateness": 0,
                },
                {
                    "optimized_distance_km": 210.0,
                    "baseline_distance_km": 300.0,
                    "optimized_estimated_vehicle_minutes": 440,
                    "late_delivery_count": 0,
                    "weighted_lateness": 0,
                },
            ],
        }
        with patch(
            "src.logistics.llm.request_chat_completion",
            side_effect=LLMNotConfiguredError("Configure LLM_MODEL."),
        ):
            result = generate_route_content(
                self.plan,
                "weekly_report",
                history=history,
            )
        self.assertIn("2 simulações", result["content"]["period"])
        self.assertTrue(
            result["content"]["key_metrics"][
                "tendencia_entre_simulacoes_disponivel"
            ]
        )
        self.assertIn("caiu 60.000 km", result["content"]["improvements"][0])
        self.assertIn("60.00", result["content"]["improvements"][1])
        self.assertIn("entregas reais", result["content"]["caveats"][0])

    def test_llm_weekly_prompt_receives_aggregated_history(self) -> None:
        history = {
            "run_count": 2,
            "window_days": 7,
            "trend_available": True,
            "recent_runs": [],
        }
        response = json.dumps(
            {
                "period": "Últimos sete dias",
                "summary": "Duas simulações comparáveis.",
                "key_metrics": {"run_count": 2},
                "improvements": ["Testar mais sementes."],
                "caveats": ["Não representa operação real."],
                "disclaimer": "Resultado sintético.",
            }
        )
        with patch(
            "src.logistics.llm.request_chat_completion",
            return_value=response,
        ) as client:
            generate_route_content(
                self.plan,
                "weekly_report",
                history=history,
            )
        self.assertIn("simulation_history_last_7_days", client.call_args.args[1])
        self.assertIn('"run_count":2', client.call_args.args[1])

    def test_invalid_llm_response_is_reported(self) -> None:
        with patch(
            "src.logistics.llm.request_chat_completion",
            return_value='{"answer":"no schema"}',
        ), self.assertRaises(LLMProviderError):
            generate_route_content(self.plan, "question", "Pergunta?")

    def test_questions_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            generate_route_content(self.plan, "question")
        with self.assertRaises(ValueError):
            generate_route_content(self.plan, "question", "x" * 1001)


class RouteAPITests(unittest.TestCase):
    def test_route_page_and_synthetic_scenario_endpoints(self) -> None:
        with TestClient(app) as client:
            page = client.get("/rotas")
            scenario = client.get("/api/rotas/cenario")
        self.assertEqual(page.status_code, 200)
        self.assertIn("routeMap", page.text)
        self.assertIn('id="routeVehicleSelect"', page.text)
        self.assertIn("Todos os veículos", page.text)
        self.assertIn('id="routeItemTypeSelect"', page.text)
        self.assertIn("Destinos com medicamentos", page.text)
        self.assertIn("Destinos com insumos", page.text)
        self.assertIn('id="routeSeed"', page.text)
        self.assertEqual(scenario.status_code, 200)
        self.assertTrue(scenario.json()["synthetic"])
        self.assertEqual(len(scenario.json()["deliveries"]), 18)
        self.assertEqual(len(scenario.json()["vehicles"]), 5)
        latitudes = [delivery["latitude"] for delivery in scenario.json()["deliveries"]]
        longitudes = [delivery["longitude"] for delivery in scenario.json()["deliveries"]]
        self.assertGreater(max(latitudes) - min(latitudes), 0.25)
        self.assertGreater(max(longitudes) - min(longitudes), 0.30)

    def test_optimization_endpoint_saves_a_valid_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                service, "RESULTS_PATH", Path(directory) / "latest.json"
            ), patch.object(
                service, "HISTORY_PATH", Path(directory) / "history.json"
            ), TestClient(app) as client:
                response = client.post(
                    "/api/rotas/otimizar",
                    json={
                        "population_size": 12,
                        "generations": 10,
                        "tournament_size": 3,
                        "seed": 4,
                    },
                )
                saved = client.get("/api/rotas/resultado")
                history = client.get("/api/rotas/historico")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["comparison"]["optimized"]["served_delivery_count"],
            len(DEMO_SCENARIO.deliveries),
        )
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json()["run_count"], 1)

    def test_route_history_endpoint_reports_saved_simulations(self) -> None:
        result = optimize_routes(
            DEMO_SCENARIO,
            GAConfig(population_size=8, generations=2, tournament_size=2),
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                service, "RESULTS_PATH", Path(directory) / "latest.json"
            ), patch.object(
                service, "HISTORY_PATH", Path(directory) / "history.json"
            ), TestClient(app) as client:
                service.save_route_plan(result)
                response = client.get("/api/rotas/historico")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_count"], 1)
        self.assertFalse(response.json()["trend_available"])

    def test_api_rejects_invalid_tournament_and_handles_missing_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                service, "RESULTS_PATH", Path(directory) / "missing.json"
            ), TestClient(app) as client:
                invalid = client.post(
                    "/api/rotas/otimizar",
                    json={
                        "population_size": 8,
                        "generations": 1,
                        "tournament_size": 9,
                    },
                )
                missing = client.get("/api/rotas/resultado")
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(missing.status_code, 404)

    def test_route_assistant_requires_saved_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                service, "RESULTS_PATH", Path(directory) / "missing.json"
            ), TestClient(app) as client:
                response = client.post(
                    "/api/rotas/assistente",
                    json={"action": "daily_report"},
                )
        self.assertEqual(response.status_code, 404)

    def test_assistant_endpoint_falls_back_when_llm_is_not_configured(self) -> None:
        result = optimize_routes(
            DEMO_SCENARIO,
            GAConfig(population_size=8, generations=2, tournament_size=2),
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                service, "RESULTS_PATH", Path(directory) / "latest.json"
            ), patch.object(
                service, "HISTORY_PATH", Path(directory) / "history.json"
            ), patch(
                "src.logistics.llm.request_chat_completion",
                side_effect=LLMNotConfiguredError("Configure LLM_MODEL."),
            ), TestClient(app) as client:
                service.save_route_plan(result)
                response = client.post(
                    "/api/rotas/assistente",
                    json={
                        "action": "question",
                        "question": "Qual veículo leva os itens refrigerados?",
                    },
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source"], "local")


if __name__ == "__main__":
    unittest.main()
