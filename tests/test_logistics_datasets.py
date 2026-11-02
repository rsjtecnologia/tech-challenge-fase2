import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.logistics import service
from src.logistics.cnes_dataset import CNES_FILENAME, _load_cached as _load_cnes_cached, load_cnes_scenario
from src.logistics.datasets import FILES, _load_cached, load_olist_scenario
from src.logistics.genetic import GAConfig, optimize_routes
from src.logistics.item_types import classify_medical_item_type
from src.logistics.llm import _route_context
from src.logistics.scenario import DEMO_SCENARIO
from src.optimization.llm import LLMNotConfiguredError
from src.web.app import app


def write_dataset(root: Path) -> None:
    rows = {
        "customers": [
            ["private-capital", "1000", "São Paulo", "SP"],
            ["private-campinas", "13000", "Campinas", "SP"],
            ["private-santos", "11000", "Santos", "SP"],
            ["private-ribeirao", "14000", "Ribeirão Preto", "SP"],
            ["private-rio", "20000", "Rio de Janeiro", "RJ"],
            ["private-missing", "99999", "Sem Coordenadas", "SP"],
        ],
        "geolocation": [
            ["1000", "-23.55", "-46.63", "SP"],
            ["13000", "-22.9", "-47.1", "SP"],
            ["13000", "-22.8", "-47.0", "SP"],
            ["13000", "28", "-47", "SP"],
            ["13000", "-22.9", "-61", "SP"],
            ["13000", "NaN", "-47", "SP"],
            ["11000", "-23.95", "-46.33", "SP"],
            ["14000", "-21.17", "-47.81", "SP"],
            ["20000", "-22.9", "-43.2", "RJ"],
        ],
        "orders": [
            ["order-capital", "private-capital", "delivered", "2018-01-01 10:00:00"],
            ["order-campinas", "private-campinas", "delivered", "2018-01-02 10:00:00"],
            ["order-campinas-2", "private-campinas", "delivered", "2018-01-03 10:00:00"],
            ["order-santos", "private-santos", "delivered", "2018-01-04 10:00:00"],
            ["order-ribeirao", "private-ribeirao", "delivered", "2018-01-05 10:00:00"],
            ["order-missing", "private-missing", "delivered", "2018-01-06 10:00:00"],
            ["order-rio", "private-rio", "delivered", "2018-01-06 10:00:00"],
            ["order-cancelled", "private-santos", "canceled", "2018-01-07 10:00:00"],
        ],
        "items": [
            ["order-capital", "private-product"],
            ["order-campinas", "private-product"],
            ["order-campinas", "private-product"],
            ["order-campinas-2", "private-product"],
            ["order-santos", "private-unknown-weight"],
            ["order-ribeirao", "private-product"],
            ["order-rio", "private-product"],
        ],
        "products": [
            ["private-product", "2000", "medicamentos"],
            ["private-unknown-weight", "", "beleza_saude"],
        ],
    }
    for label, (name, columns) in FILES.items():
        with (root / name).open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(columns)
            writer.writerows(rows[label])


class OlistDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        write_dataset(self.root)
        _load_cached.cache_clear()
        _load_cnes_cached.cache_clear()

    def tearDown(self) -> None:
        _load_cached.cache_clear()
        _load_cnes_cached.cache_clear()
        self.directory.cleanup()

    def test_sp_filter_medians_aggregation_provenance_and_no_individual_ids(self) -> None:
        scenario = load_olist_scenario(self.root)
        data = scenario.data_source
        self.assertEqual(data["delivered_orders_sp"], 6)
        self.assertEqual(data["discarded_geolocation_rows"], 3)
        self.assertEqual(data["municipalities_without_coordinates"], 1)
        self.assertEqual(len(scenario.deliveries), 3)
        self.assertEqual(scenario.deliveries[0].name, "Polo simulado — Campinas/SP")
        self.assertAlmostEqual(scenario.deliveries[0].latitude, -22.85)
        self.assertAlmostEqual(scenario.deliveries[0].longitude, -47.05)
        campinas = data["municipal_aggregates"][0]
        self.assertEqual(campinas["historical_delivered_orders"], 2)
        self.assertEqual(campinas["historical_items"], 3)
        self.assertEqual(campinas["observed_product_weight_kg"], 6.0)
        self.assertEqual(campinas["historical_medicine_items"], 3)
        self.assertEqual(campinas["historical_supply_items"], 0)
        self.assertEqual(campinas["historical_unclassified_items"], 0)
        santos = next(row for row in data["municipal_aggregates"] if row["municipality"] == "santos")
        self.assertEqual(santos["items_with_known_weight"], 0)
        self.assertEqual(santos["historical_medicine_items"], 0)
        self.assertEqual(santos["historical_supply_items"], 0)
        self.assertEqual(santos["historical_health_beauty_items"], 1)
        self.assertEqual(santos["historical_unclassified_items"], 0)
        self.assertTrue(scenario.synthetic)
        self.assertEqual(scenario.deliveries[0].demand_kg, 12)
        self.assertEqual(len(data["files_sha256"]), 5)
        self.assertNotIn("private-", json.dumps(data))
        self.assertNotIn("Rio", str(scenario.deliveries))
        self.assertEqual(data["purchase_period"]["end"], "2018-01-06T10:00:00")

    def test_medical_category_classifier_requires_explicit_category_terms(self) -> None:
        self.assertEqual(classify_medical_item_type("medicamentos"), "medicamento")
        self.assertEqual(classify_medical_item_type("Farmácia"), "medicamento")
        self.assertEqual(classify_medical_item_type("insumos médicos"), "insumo")
        self.assertEqual(classify_medical_item_type("material hospitalar"), "insumo")
        self.assertEqual(
            classify_medical_item_type("beleza_saude"),
            "saude_beleza_nao_confirmado",
        )
        self.assertEqual(classify_medical_item_type("utilidades_domesticas"), "nao_classificado")

    def test_category_counts_preserve_mixed_items_and_missing_products(self) -> None:
        with (self.root / FILES["products"][0]).open(
            "a", newline="", encoding="utf-8"
        ) as output:
            csv.writer(output).writerows([
                ["supply-product", "500", "material hospitalar"],
                ["other-product", "1000", "utilidades_domesticas"],
            ])
        with (self.root / FILES["items"][0]).open(
            "a", newline="", encoding="utf-8"
        ) as output:
            csv.writer(output).writerows([
                ["order-campinas", "supply-product"],
                ["order-campinas", "private-unknown-weight"],
                ["order-campinas", "other-product"],
                ["order-campinas", "missing-product"],
            ])
        scenario = load_olist_scenario(self.root)
        campinas = scenario.data_source["municipal_aggregates"][0]
        self.assertEqual(campinas["historical_items"], 7)
        self.assertEqual(campinas["historical_medicine_items"], 3)
        self.assertEqual(campinas["historical_supply_items"], 1)
        self.assertEqual(campinas["historical_health_beauty_items"], 1)
        self.assertEqual(campinas["historical_unclassified_items"], 2)
        self.assertEqual(campinas["items_with_known_weight"], 5)
        self.assertEqual(campinas["observed_product_weight_kg"], 7.5)

    def test_cnes_scenario_uses_facility_coordinates_and_labels_simulated_demand(self) -> None:
        rows = [
            ["CO_CNES", "NO_FANTASIA", "DS_TIPO_UNIDADE", "NO_MUNICIPIO",
             "CO_UF", "NU_LATITUDE", "NU_LONGITUDE"]
        ]
        for index in range(22):
            rows.append([
                str(1000 + index),
                f"Unidade {index}",
                "Hospital Geral" if index == 0 else "Unidade Básica de Saúde",
                f"Município {index}",
                "35",
                str(-23.55 - index * 0.03),
                str(-46.63 - index * 0.02),
            ])
        rows.extend([
            ["9998", "Unidade RJ", "Hospital", "Rio de Janeiro", "33", "-22.9", "-43.2"],
            ["9999", "Coordenada inválida", "Hospital", "São Paulo", "35", "0", "0"],
        ])
        with (self.root / CNES_FILENAME).open("w", newline="", encoding="utf-8") as output:
            csv.writer(output).writerows(rows)

        scenario = load_cnes_scenario(self.root)

        self.assertEqual(len(scenario.deliveries), 18)
        self.assertEqual(scenario.data_source["kind"], "cnes_sp")
        self.assertEqual(scenario.data_source["establishments_sp_with_coordinates"], 22)
        self.assertTrue(scenario.data_source["demand_is_simulated"])
        self.assertEqual(scenario.deliveries[0].latitude, -23.55)
        self.assertIn("Hospital Geral", scenario.data_source["facility_aggregates"][0]["facility_type"])
        self.assertTrue(scenario.data_source["facility_aggregates"][0]["demand_is_simulated"])
        self.assertIn("medicamento simulado", scenario.deliveries[0].items)
        self.assertIn("insumo médico simulado", scenario.deliveries[0].items)

    def test_cnes_scenario_uses_optional_joined_demand_and_reports_missing_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "Dataset CNES ausente"):
            load_cnes_scenario(self.root)
        path = self.root / CNES_FILENAME
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "cnes_id", "name", "facility_type", "uf", "latitude", "longitude",
                "priority_score", "medicine_kg", "supply_kg",
            ])
            writer.writerows([
                ["1", "Hospital A", "Hospital", "SP", "-23.55", "-46.63", "10", "4.5", "1.5"],
                ["2", "UBS B", "UBS", "SP", "-23.60", "-46.70", "7", "0", "2"],
            ])

        scenario = load_cnes_scenario(self.root)
        aggregates = scenario.data_source["facility_aggregates"]
        self.assertFalse(scenario.data_source["demand_is_simulated"])
        self.assertFalse(aggregates[0]["demand_is_simulated"])
        self.assertEqual(aggregates[0]["medicine_demand_kg"], 4.5)
        self.assertEqual(aggregates[0]["supply_demand_kg"], 1.5)

    def test_cnes_scenario_adds_kaggle_inventory_as_unlinked_reference_profile(self) -> None:
        cnes_path = self.root / CNES_FILENAME
        with cnes_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "cnes_id", "name", "facility_type", "uf", "latitude", "longitude",
            ])
            writer.writerows([
                ["1", "Hospital A", "Hospital", "SP", "-23.55", "-46.63"],
                ["2", "UBS B", "UBS", "SP", "-23.60", "-46.70"],
            ])
        kaggle_dir = self.root / "hospital_supply_chain"
        kaggle_dir.mkdir()
        inventory_path = kaggle_dir / "inventory_data.csv"
        with inventory_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "Date", "Item_ID", "Item_Type", "Item_Name", "Current_Stock",
                "Min_Required", "Avg_Usage_Per_Day", "Restock_Lead_Time",
            ])
            writer.writerows([
                ["2024-01-01", "10", "Consumable", "Surgical Mask", "150", "20", "50", "4"],
                ["2024-01-02", "11", "Consumable", "Surgical Mask", "130", "20", "70", "6"],
                ["2024-01-02", "12", "Equipment", "Ventilator", "4", "1", "2", "10"],
            ])

        scenario = load_cnes_scenario(self.root)
        profile = scenario.data_source["kaggle_inventory_profile"]

        self.assertEqual(profile["records"], 3)
        self.assertEqual(profile["license"].split(" ")[0], "MIT")
        self.assertIn("inventory_data.csv", scenario.data_source["files_sha256"])
        masks = next(item for item in profile["items"] if item["item_name"] == "Surgical Mask")
        self.assertEqual(masks["records"], 2)
        self.assertEqual(masks["average_usage_per_day"], 60)
        self.assertIn("sem CNES", profile["scope"])
        self.assertTrue(all(item["demand_is_simulated"] for item in scenario.data_source["facility_aggregates"]))
        plan = optimize_routes(
            scenario,
            GAConfig(population_size=8, generations=2, tournament_size=2),
        )
        llm_context = _route_context(plan)
        self.assertEqual(
            llm_context["inventory_reference"]["dataset"],
            "Hospital Supply Chain — Kaggle",
        )
        self.assertIn("sem CNES", llm_context["inventory_reference"]["limitation"])

    def test_api_exposes_cnes_source_and_keeps_its_plans_separate(self) -> None:
        path = self.root / CNES_FILENAME
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "cnes_id", "name", "facility_type", "uf", "latitude", "longitude",
            ])
            writer.writerows([
                ["cnes-1", "Hospital A", "Hospital", "SP", "-23.55", "-46.63"],
                ["cnes-2", "UBS B", "UBS", "SP", "-23.60", "-46.70"],
            ])
        scenario = load_cnes_scenario(self.root)
        with patch.object(service, "load_cnes_scenario", return_value=scenario), patch.object(
            service, "RESULTS_PATH", self.root / "latest.json"
        ), patch.object(service, "HISTORY_PATH", self.root / "history.json"), TestClient(app) as client:
            response = client.get("/api/rotas/cenario?source=cnes_sp")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["data_source"]["kind"], "cnes_sp")
            self.assertEqual(client.get("/api/rotas/resultado?source=cnes_sp").status_code, 404)
            result = client.post("/api/rotas/otimizar", json={
                "source": "cnes_sp",
                "population_size": 8,
                "generations": 3,
                "tournament_size": 2,
            })
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(
                client.get("/api/rotas/resultado?source=cnes_sp").json()["scenario_id"],
                scenario.id,
            )
            self.assertEqual(client.get("/api/rotas/historico?source=cnes_sp").json()["run_count"], 1)
            self.assertEqual(client.get("/api/rotas/historico?source=olist_sp").json()["run_count"], 0)

    def test_deterministic_load_cache_and_input_change_invalidate_plan_id(self) -> None:
        first = load_olist_scenario(self.root)
        self.assertIs(first, load_olist_scenario(self.root))
        product_path = self.root / FILES["products"][0]
        product_path.write_text(
            product_path.read_text().replace("2000", "30000"), encoding="utf-8"
        )
        second = load_olist_scenario(self.root)
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(second.data_source["municipal_aggregates"][0]["observed_product_weight_kg"], 90)

    def test_missing_or_malformed_files_fail_explicitly(self) -> None:
        with patch.object(Path, "resolve", side_effect=PermissionError("acesso negado")):
            with self.assertRaisesRegex(ValueError, "Não foi possível ler dataset_rotasBR"):
                load_olist_scenario(self.root)
        path = self.root / FILES["customers"][0]
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Arquivos ausentes"):
            load_olist_scenario(self.root)
        path.write_text("wrong,column\n1,2\n")
        with self.assertRaisesRegex(ValueError, "Colunas inválidas"):
            load_olist_scenario(self.root)

    def test_no_valid_depot_or_too_few_destinations_fail(self) -> None:
        path = self.root / FILES["geolocation"][0]
        text = path.read_text()
        path.write_text(text.replace("-23.55", "0"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "depósito"):
            load_olist_scenario(self.root)
        path.write_text(text.replace("-23.95", "0").replace("-21.17", "0"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dois municípios"):
            load_olist_scenario(self.root)

    def test_invalid_dates_and_duplicate_ids_are_not_silently_accepted(self) -> None:
        path = self.root / FILES["orders"][0]
        text = path.read_text()
        path.write_text(text.replace("2018-01-01 10:00:00", "not-a-date"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "data de compra"):
            load_olist_scenario(self.root)
        path.write_text(text.replace("order-campinas-2", "order-campinas"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "order_id"):
            load_olist_scenario(self.root)

    def test_genetic_vrp_uses_dataset_coordinates_and_hard_constraints(self) -> None:
        scenario = load_olist_scenario(self.root)
        result = optimize_routes(scenario, GAConfig(population_size=8, generations=4, tournament_size=2))
        optimized = result["comparison"]["optimized"]
        self.assertEqual(optimized["served_delivery_count"], 3)
        self.assertEqual(optimized["unserved_delivery_ids"], [])
        self.assertEqual(result["data_source"]["kind"], "olist_sp")
        for route in optimized["routes"]:
            self.assertLessEqual(route["load_kg"], route["capacity_kg"])
            self.assertLessEqual(route["distance_km"], route["max_distance_km"])
            vehicle = next(v for v in scenario.vehicles if v.id == route["vehicle_id"])
            for stop in route["stops"]:
                delivery = next(d for d in scenario.deliveries if d.id == stop["delivery_id"])
                self.assertEqual(stop["latitude"], delivery.latitude)
                if delivery.requires_refrigeration:
                    self.assertTrue(vehicle.refrigerated)
        llm_context = _route_context(result)
        self.assertEqual(llm_context["data_source"]["state"], "SP")
        self.assertNotIn("private-", json.dumps(llm_context))

    def test_api_source_plan_history_assistant_and_demo_are_isolated(self) -> None:
        scenario = load_olist_scenario(self.root)
        with patch.object(service, "load_olist_scenario", return_value=scenario), patch.object(
            service, "RESULTS_PATH", self.root / "latest.json"
        ), patch.object(
            service, "HISTORY_PATH", self.root / "history.json"
        ), patch(
            "src.logistics.llm.request_chat_completion",
            side_effect=LLMNotConfiguredError("no model"),
        ), TestClient(app) as client:
            payload = client.get("/api/rotas/cenario?source=olist_sp")
            self.assertEqual(payload.status_code, 200)
            self.assertEqual(payload.json()["id"], scenario.id)
            self.assertEqual(client.get("/api/rotas/resultado?source=olist_sp").status_code, 404)
            result = client.post("/api/rotas/otimizar", json={
                "source": "olist_sp", "population_size": 8, "generations": 3, "tournament_size": 2,
            })
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(client.get("/api/rotas/resultado?source=olist_sp").json()["scenario_id"], scenario.id)
            self.assertEqual(client.get("/api/rotas/historico?source=olist_sp").json()["run_count"], 1)
            self.assertEqual(client.get("/api/rotas/historico").json()["run_count"], 0)
            self.assertEqual(client.get("/api/rotas/resultado").status_code, 404)
            assistant = client.post("/api/rotas/assistente", json={"source": "olist_sp", "action": "daily_report"})
            self.assertEqual(assistant.status_code, 200, assistant.text)
            self.assertEqual(assistant.json()["source"], "local")
            self.assertEqual(client.get("/api/rotas/cenario").json()["id"], DEMO_SCENARIO.id)
            with patch.object(service, "load_olist_scenario", return_value=replace(scenario, id="changed")):
                self.assertEqual(client.get("/api/rotas/resultado?source=olist_sp").status_code, 404)

    def test_api_missing_dataset_and_invalid_source_return_errors(self) -> None:
        with patch.object(
            service, "load_olist_scenario", side_effect=ValueError("Dataset ausente")
        ), TestClient(app) as client:
            for path in ("cenario", "resultado", "historico"):
                response = client.get(f"/api/rotas/{path}?source=olist_sp")
                self.assertEqual(response.status_code, 503)
                self.assertIn("Dataset ausente", response.json()["detail"])
            self.assertEqual(client.post("/api/rotas/otimizar", json={"source": "olist_sp"}).status_code, 503)
            self.assertEqual(client.get("/api/rotas/cenario?source=RJ").status_code, 422)
            self.assertEqual(client.post("/api/rotas/otimizar", json={"source": "RJ"}).status_code, 422)
            page = client.get("/rotas")
            self.assertIn('id="scenarioSource"', page.text)
            self.assertIn('value="olist_sp"', page.text)
            self.assertIn('value="hybrid_sp" selected', page.text)
            self.assertIn('value="cnes_sp"', page.text)
            self.assertIn('id="datasetDetails"', page.text)


if __name__ == "__main__":
    unittest.main()
