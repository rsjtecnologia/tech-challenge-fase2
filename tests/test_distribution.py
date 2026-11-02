import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.logistics import distribution, service
from src.logistics.distribution import build_distribution_scenario
from src.logistics.genetic import GAConfig, decode_chromosome, optimize_routes
from src.logistics.llm import _prompt_context
from src.logistics.scenario import DEMO_SCENARIO
from src.web.app import app


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inventory = self.root / "inventory_data.csv"
        self.write_inventory(stock=0)

    def tearDown(self):
        self.temp.cleanup()

    def write_inventory(self, stock):
        with self.inventory.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "Item_Type", "Item_Name", "Avg_Usage_Per_Day",
                "Current_Stock", "Min_Required", "Restock_Lead_Time",
            ])
            writer.writerow(["Consumable", "Gloves", 100, stock, 200, 5])
            writer.writerow(["Equipment", "Ventilator", 10, 50, 10, 5])

    def test_quantities_load_and_priority_follow_simulated_stock(self):
        scenario = build_distribution_scenario(DEMO_SCENARIO, self.inventory)
        manifests = scenario.data_source["distribution_manifest"]
        self.assertEqual(len(manifests), 18)
        self.assertEqual({line["category"] for m in manifests for line in m["lines"]},
                         {"medicamento", "insumo"})
        self.assertTrue(all(d.priority == "critica" for d in scenario.deliveries))
        for delivery, manifest in zip(scenario.deliveries, manifests):
            self.assertEqual(delivery.demand_kg, manifest["load_kg"])
            self.assertAlmostEqual(
                delivery.demand_kg,
                sum(line["quantity"] * line["unit_weight_kg"] for line in manifest["lines"]),
            )
            for line in manifest["lines"]:
                self.assertGreater(line["quantity"], 0)
                self.assertLessEqual(line["quantity"], 30)
                self.assertEqual(line["stock_units"], 0)
                self.assertTrue(line["simulated"])
                self.assertEqual(line["reference_item"], "Gloves")
        self.assertIn("fictício", manifests[0]["lines"][0]["name"])
        self.assertEqual(scenario.deliveries[0].latitude, DEMO_SCENARIO.deliveries[0].latitude)
        self.assertEqual(
            scenario.id, build_distribution_scenario(DEMO_SCENARIO, self.inventory).id
        )
        self.write_inventory(stock=10000)
        replenished = build_distribution_scenario(DEMO_SCENARIO, self.inventory)
        self.assertNotEqual(scenario.id, replenished.id)
        self.assertTrue(all(d.priority == "regular" for d in replenished.deliveries))
        self.assertLess(scenario.deliveries[0].deadline_minute,
                        replenished.deliveries[0].deadline_minute)

    def test_genetic_plan_respects_load_refrigeration_and_exposes_llm_stock(self):
        scenario = build_distribution_scenario(DEMO_SCENARIO, self.inventory)
        plan = optimize_routes(
            scenario, GAConfig(population_size=8, generations=4, tournament_size=2)
        )
        self.assertEqual(plan["comparison"]["optimized"]["served_delivery_count"], 18)
        for route in plan["comparison"]["optimized"]["routes"]:
            self.assertLessEqual(route["load_kg"], route["capacity_kg"])
            vehicle = next(v for v in scenario.vehicles if v.id == route["vehicle_id"])
            for stop in route["stops"]:
                delivery = next(d for d in scenario.deliveries if d.id == stop["delivery_id"])
                if delivery.requires_refrigeration:
                    self.assertTrue(vehicle.refrigerated)
        context = _prompt_context(plan, "question", None, "Onde o estoque está crítico?")
        self.assertTrue(context["simulated_distribution"]["destinations"])
        self.assertIn("fictícia", context["simulated_distribution"]["notice"])
        self.assertEqual(
            context["simulated_distribution"]["destinations"][0]["lines"][0]["stock_units"], 0
        )

    def test_missing_inventory_is_an_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "Inventário Hospital Supply Chain ausente"):
            build_distribution_scenario(DEMO_SCENARIO, self.root / "missing.csv")

    def test_cnes_metadata_matches_distribution_without_changing_base(self):
        facilities = [{
            "delivery_id": delivery.id,
            "medicine_demand_kg": 99,
            "supply_demand_kg": 99,
            "priority_score": 1,
            "demand_is_simulated": False,
        } for delivery in DEMO_SCENARIO.deliveries]
        base = replace(DEMO_SCENARIO, data_source={
            "kind": "cnes_sp", "facility_aggregates": facilities,
        })
        scenario = build_distribution_scenario(base, self.inventory)
        self.assertEqual(scenario.data_source["geography_source"], "cnes_sp")
        for facility, delivery in zip(
            scenario.data_source["facility_aggregates"], scenario.deliveries
        ):
            self.assertAlmostEqual(
                facility["medicine_demand_kg"] + facility["supply_demand_kg"],
                delivery.demand_kg,
            )
            self.assertEqual(facility["priority_score"], 9)
            self.assertTrue(facility["demand_is_simulated"])
            self.assertTrue(facility["priority_is_simulated"])
        self.assertEqual(base.data_source["facility_aggregates"][0]["medicine_demand_kg"], 99)
        self.assertFalse(base.data_source["facility_aggregates"][0]["demand_is_simulated"])

    def test_coverage_below_restock_lead_time_gives_high_priority(self):
        self.write_inventory(stock=400)
        scenario = build_distribution_scenario(DEMO_SCENARIO, self.inventory)
        self.assertTrue(all(d.priority == "alta" for d in scenario.deliveries))
        line = scenario.data_source["distribution_manifest"][0]["lines"][0]
        self.assertEqual(line["stock_units"], 20)
        self.assertEqual(line["daily_usage_units"], 5)
        self.assertEqual(line["stock_coverage_days"], 4)
        self.assertEqual(line["restock_lead_time_days"], 5)

    def test_stock_priority_changes_lateness_penalty_in_fitness(self):
        scenario = build_distribution_scenario(DEMO_SCENARIO, self.inventory)
        delivery = replace(scenario.deliveries[0], deadline_minute=0)
        critical = replace(scenario, deliveries=(delivery,))
        regular = replace(critical, deliveries=(replace(delivery, priority="regular"),))
        critical_result = decode_chromosome((delivery.id,), critical)
        regular_result = decode_chromosome((delivery.id,), regular)
        self.assertAlmostEqual(
            critical_result.total_weighted_lateness,
            regular_result.total_weighted_lateness * 8,
            delta=0.05,
        )
        self.assertGreater(critical_result.fitness, regular_result.fitness)

    def test_geography_uses_cnes_only_when_available_and_does_not_hide_bad_csv(self):
        with patch.object(distribution, "CNES_DATASET_DIR", self.root), patch.object(
            distribution, "KAGGLE_INVENTORY_PATH", self.inventory
        ), patch.object(
            distribution, "load_olist_scenario", return_value=DEMO_SCENARIO
        ) as olist, patch.object(
            distribution, "load_cnes_scenario", side_effect=ValueError("CNES inválido")
        ) as cnes:
            result = distribution.load_distribution_scenario()
            self.assertEqual(result.data_source["geography_source"], "demo")
            olist.assert_called_once()
            cnes.assert_not_called()
            (self.root / distribution.CNES_FILENAME).touch()
            with self.assertRaisesRegex(ValueError, "CNES inválido"):
                distribution.load_distribution_scenario()

    def test_api_source_plan_and_history_are_separate(self):
        scenario = build_distribution_scenario(DEMO_SCENARIO, self.inventory)
        with patch.object(service, "load_distribution_scenario", return_value=scenario), patch.object(
            service, "RESULTS_PATH", self.root / "latest.json"
        ), patch.object(service, "HISTORY_PATH", self.root / "history.json"), TestClient(app) as client:
            payload = client.get("/api/rotas/cenario?source=hybrid_sp")
            self.assertEqual(payload.status_code, 200)
            self.assertEqual(len(payload.json()["data_source"]["distribution_manifest"]), 18)
            response = client.post("/api/rotas/otimizar", json={
                "source": "hybrid_sp", "population_size": 8,
                "generations": 3, "tournament_size": 2,
            })
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(client.get("/api/rotas/historico?source=hybrid_sp").json()["run_count"], 1)
            self.assertEqual(client.get("/api/rotas/historico").json()["run_count"], 0)
            self.assertTrue((self.root / "latest_hybrid_sp.json").is_file())
            self.assertIn('value="hybrid_sp" selected', client.get("/rotas").text)


if __name__ == "__main__":
    unittest.main()
