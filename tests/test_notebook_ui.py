from pathlib import Path
import unittest
from unittest.mock import patch

from IPython.display import HTML, Markdown
import nbformat
import ipywidgets as widgets

from src.optimization import notebook_ui


ROOT = Path(__file__).resolve().parents[1]


class NotebookInterfaceTests(unittest.TestCase):
    @staticmethod
    def _route_scenario() -> dict:
        return {
            "name": "Olist — cenário médico simulado",
            "description": "Agregado de municípios.",
            "data_source": {
                "kind": "olist_sp",
                "delivered_orders_sp": 120,
                "municipalities_with_coordinates": 25,
                "selected_municipalities": 1,
                "municipal_aggregates": [{
                    "delivery_id": "d1",
                    "historical_delivered_orders": 12,
                    "historical_items": 16,
                    "historical_medicine_items": 3,
                    "historical_supply_items": 0,
                    "historical_health_beauty_items": 1,
                    "items_with_known_weight": 14,
                    "observed_product_weight_kg": 2.3,
                }],
            },
            "depot": {
                "name": "Depósito", "latitude": -23.5, "longitude": -46.6,
            },
            "deliveries": [{
                "id": "d1",
                "name": "Destino <um>",
                "priority": "alta",
                "demand_kg": 5,
                "items": ["medicamento prioritário"],
                "deadline_time": "10:00",
                "requires_refrigeration": False,
            }],
        }

    @staticmethod
    def _route_plan() -> dict:
        route = {
            "vehicle_name": "Veículo 1",
            "distance_km": 12.5,
            "load_kg": 5,
            "capacity_kg": 50,
            "path": [
                {"latitude": -23.5, "longitude": -46.6},
                {"latitude": -23.6, "longitude": -46.7},
                {"latitude": -23.5, "longitude": -46.6},
            ],
            "stops": [{
                "name": "Destino <um>",
                "priority": "alta",
                "demand_kg": 5,
                "arrival_time": "08:20",
                "lateness_minutes": 0,
                "items": ["insumo médico simulado"],
                "latitude": -23.6,
                "longitude": -46.7,
            }],
        }
        baseline = {
            "routes": [route],
            "total_distance_km": 15,
            "served_delivery_count": 1,
            "delivery_count": 1,
            "late_delivery_count": 1,
            "vehicles_used": 1,
            "estimated_vehicle_minutes": 20,
            "total_load_kg": 5,
            "fleet_utilization_percent": 10,
            "unserved_delivery_ids": [],
        }
        optimized = {
            **baseline,
            "routes": [route],
            "total_distance_km": 12.5,
            "late_delivery_count": 0,
        }
        return {
            "depot": {"latitude": -23.5, "longitude": -46.6, "name": "Depósito"},
            "comparison": {
                "baseline": baseline,
                "optimized": optimized,
                "distance_saved_km": 2.5,
                "distance_saved_percent": 16.67,
            },
        }

    def test_notebook_ui_is_appended_after_unchanged_phase_one_cells(self) -> None:
        notebooks = (
            ("notebooks/breast-cancer-wisconsin-data/08_export_web.ipynb", "breast_cancer"),
            ("notebooks/polycystic-ovary-syndrome-pcos/05_export_web.ipynb", "pcos"),
        )
        for relative_path, study in notebooks:
            with self.subTest(notebook=relative_path):
                notebook = nbformat.read(ROOT / relative_path, as_version=4)
                section = next(
                    index for index, cell in enumerate(notebook.cells)
                    if cell.cell_type == "markdown"
                    and cell.source.startswith("## Fase 2 — Interpretação")
                )
                self.assertGreater(section, 0)
                self.assertEqual(len(notebook.cells) - section, 3)
                self.assertIn("display_phase2_interpretation", notebook.cells[-1].source)
                self.assertIn(f'"{study}"', notebook.cells[-1].source)
                nbformat.validate(notebook)

    def test_examples_only_fill_question_and_explicit_button_submits(self) -> None:
        comparison = {
            "models": {
                "Modelo <A>": {
                    "selected_experiment": "Exp 1",
                    "baseline_metrics": {
                        "recall": .5, "precision": .6, "f1": .55, "accuracy": .7,
                    },
                    "optimized_metrics": {
                        "recall": .7, "precision": .65, "f1": .63, "accuracy": .72,
                    },
                },
            },
        }
        response = {
            "study": "PCOS",
            "source": "llm",
            "interpretation": {
                "summary": "Métricas agregadas.",
                "findings": ["Recall subiu."],
                "limitations": ["Holdout único."],
                "disclaimer": "Não substitui avaliação clínica.",
            },
            "quality_evaluation": {"score": 7, "maximum": 7},
        }
        with patch.object(notebook_ui, "display") as display_mock, patch.object(
            notebook_ui, "_api_request",
            side_effect=[comparison, response],
        ) as api_request:
            ui = notebook_ui.display_phase2_interpretation("pcos")
            self.assertEqual(len(ui["examples"]), 4)
            example = ui["examples"][0]
            example._click_handlers.callbacks[0](example)
            self.assertEqual(ui["question"].value, notebook_ui.EXAMPLES[0])
            self.assertEqual(api_request.call_count, 1)
            ui["ask_button"]._click_handlers.callbacks[0](ui["ask_button"])
            self.assertEqual(api_request.call_count, 2)
            self.assertEqual(
                api_request.call_args.args,
                (
                    "http://127.0.0.1:8020",
                    "/api/interpretacao",
                    {"estudo": "pcos", "pergunta": notebook_ui.EXAMPLES[0]},
                ),
            )
            self.assertTrue(all(
                control.disabled is False
                for control in (
                    ui["question"], ui["ask_button"], ui["generate_button"],
                    *ui["examples"],
                )
            ))
        rendered_html = "\n".join(
            item.args[0].data for item in display_mock.call_args_list
            if item.args and isinstance(item.args[0], HTML)
        )
        self.assertIn("20.00 p.p.", rendered_html)
        self.assertIn("&lt;A&gt;", rendered_html)
        self.assertIn("Métricas agregadas.", rendered_html)
        self.assertTrue(any(isinstance(item.args[0], Markdown) for item in display_mock.call_args_list))

    def test_general_interpretation_sends_no_question_to_local_api(self) -> None:
        comparison = {"models": {}}
        response = {
            "source": "local",
            "interpretation": {
                "summary": "Recall e F1 agregados.",
                "findings": ["Métrica observada."],
                "limitations": ["Amostra limitada."],
                "disclaimer": "Não substitui avaliação clínica.",
            },
        }
        with patch.object(notebook_ui, "display"), patch.object(
            notebook_ui, "_api_request",
            side_effect=[comparison, response],
        ) as api_request:
            ui = notebook_ui.display_phase2_interpretation("breast_cancer")
            ui["generate_button"]._click_handlers.callbacks[0](ui["generate_button"])
        self.assertEqual(
            api_request.call_args.args,
            ("http://127.0.0.1:8020", "/api/interpretacao", {"estudo": "breast_cancer"}),
        )

    def test_only_loopback_api_urls_are_accepted(self) -> None:
        for url in (
            "https://api.example.com",
            "http://localhost.evil.example:8020",
            "http://127.0.0.1:8020/?key=secret",
            "http://user@localhost:8020",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                notebook_ui.display_phase2_interpretation("breast_cancer", url)

    def test_notebook_api_port_can_be_configured_from_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {notebook_ui.API_BASE_URL_ENV: "http://127.0.0.1:8021"},
        ), patch.object(notebook_ui, "display"), patch.object(
            notebook_ui, "_api_request", return_value={"models": {}}
        ) as api_request:
            notebook_ui.display_phase2_interpretation("breast_cancer")

        self.assertEqual(api_request.call_args.args[0], "http://127.0.0.1:8021")

    def test_widgets_are_real_notebook_controls(self) -> None:
        comparison = {"models": {}}
        with patch.object(notebook_ui, "display"), patch.object(
            notebook_ui, "_api_request", return_value=comparison
        ):
            ui = notebook_ui.display_phase2_interpretation("breast_cancer")
        self.assertIsInstance(ui["question"], widgets.Text)
        self.assertTrue(all(isinstance(item, widgets.Button) for item in ui["examples"]))
        self.assertEqual(len(ui["examples"]), 4)

    def test_route_notebook_examples_fill_without_sending_and_question_is_explicit(self) -> None:
        scenario = self._route_scenario()
        plan = self._route_plan()
        history = {"run_count": 1, "window_days": 7, "trend_available": False}
        answer = {
            "source": "local",
            "content": {
                "answer": "O plano atende uma entrega.",
                "supporting_facts": ["Uma parada simulada."],
                "missing_information": [],
                "disclaimer": "Simulação acadêmica.",
            },
        }
        with patch.object(notebook_ui, "display"), patch.object(
            notebook_ui, "_api_request",
            side_effect=[scenario, plan, history, answer],
        ) as api_request:
            ui = notebook_ui.display_route_notebook()
            self.assertEqual(api_request.call_count, 3)
            button = ui["examples"][0]
            button._click_handlers.callbacks[0](button)
            self.assertEqual(
                ui["question"].value,
                "Qual veículo leva os itens refrigerados?",
            )
            self.assertEqual(api_request.call_count, 3)
            ui["ask_button"]._click_handlers.callbacks[0](ui["ask_button"])

        self.assertEqual(
            api_request.call_args.args,
            (
                "http://127.0.0.1:8020",
                "/api/rotas/assistente",
                {
                    "source": "olist_sp",
                    "action": "question",
                    "question": "Qual veículo leva os itens refrigerados?",
                },
            ),
        )
        self.assertEqual(api_request.call_args.kwargs, {"timeout": 300})
        self.assertFalse(ui["ask_button"].disabled)

    def test_route_notebook_optimization_uses_selected_parameters(self) -> None:
        scenario = self._route_scenario()
        plan = self._route_plan()
        history = {"run_count": 1, "window_days": 7, "trend_available": False}
        with patch.object(notebook_ui, "display"), patch.object(
            notebook_ui, "_api_request",
            side_effect=[scenario, plan, history, plan, history],
        ) as api_request:
            ui = notebook_ui.display_route_notebook()
            ui["population"].value = 48
            ui["generations"].value = 60
            ui["mutation"].value = 0.22
            ui["seed"].value = 7
            ui["optimize_button"]._click_handlers.callbacks[0](ui["optimize_button"])

        self.assertEqual(api_request.call_args.args[1], "/api/rotas/historico?source=olist_sp")
        optimization_call = api_request.call_args_list[3]
        self.assertEqual(
            optimization_call.args[1:],
            (
                "/api/rotas/otimizar",
                {
                    "source": "olist_sp",
                    "population_size": 48,
                    "generations": 60,
                    "mutation_rate": 0.22,
                    "tournament_size": 4,
                    "seed": 7,
                },
            ),
        )
        self.assertEqual(optimization_call.kwargs, {"timeout": 600})

    def test_route_notebook_offers_cnes_as_a_separate_data_source(self) -> None:
        scenario = self._route_scenario()
        with patch.object(notebook_ui, "display"), patch.object(
            notebook_ui, "_api_request", return_value=scenario,
        ) as api_request:
            ui = notebook_ui.display_route_notebook()
            self.assertIn(("CNES — estabelecimentos de saúde SP (CSV local)", "cnes_sp"), ui["source"].options)
            ui["source"].value = "cnes_sp"

        self.assertEqual(
            api_request.call_args_list[3].args[1],
            "/api/rotas/cenario?source=cnes_sp",
        )

    def test_route_notebook_renders_escaped_map_and_comparison(self) -> None:
        metrics, route_map = notebook_ui._route_result_html(
            self._route_plan(), "optimized", self._route_scenario()
        )
        self.assertIn("12.500 km", metrics)
        self.assertIn("16.67%", metrics)
        self.assertIn("Destino &lt;um&gt;", metrics)
        self.assertIn("Mapa esquemático", route_map)
        self.assertIn("Itens simulados: insumo médico simulado", route_map)
        self.assertIn("Central de distribuição", route_map)
        self.assertIn("Destinos com medicamentos", metrics)
        self.assertIn("3 item(ns) Olist", metrics)
        self.assertIn("1 simulado(s)", metrics)
        self.assertIn("saúde/beleza (não confirmado como item médico)", metrics)
        self.assertNotIn("<um>", route_map)

    def test_hybrid_notebook_renders_medical_quantities_and_stock(self) -> None:
        scenario = self._route_scenario()
        delivery = scenario["deliveries"][0]
        scenario["data_source"].update({
            "kind": "hybrid_sp",
            "geography_source": "olist_sp",
            "demand_source": "Associação fictícia",
            "distribution_method": {"priority_rule": "Crítica quando abaixo do mínimo"},
            "distribution_manifest": [{
                "delivery_id": delivery["id"],
                "lines": [{
                    "category": "medicamento", "name": "Kit <médico>",
                    "quantity": 10, "load_kg": 0.5, "stock_units": 0,
                    "minimum_stock_units": 5, "daily_usage_units": 2,
                    "stock_coverage_days": 0, "restock_lead_time_days": 5,
                }],
            }],
        })
        plan = self._route_plan()
        metrics, _ = notebook_ui._route_result_html(plan, "optimized", scenario)
        self.assertIn("10 kits", metrics)
        self.assertIn("0.50 kg", metrics)
        with patch.object(notebook_ui, "display") as display_mock, patch.object(
            notebook_ui, "_api_request",
            side_effect=lambda base, path: (
                scenario if "/cenario?" in path
                else plan if "/resultado?" in path else {"run_count": 0}
            ),
        ):
            ui = notebook_ui.display_route_notebook()
            ui["source"].value = "hybrid_sp"
        html = "\n".join(
            call.args[0].data for call in display_mock.call_args_list
            if call.args and isinstance(call.args[0], HTML)
        )
        self.assertIn("estoque 0, mínimo 5", html)
        self.assertIn("Kit &lt;médico&gt;", html)
        self.assertIn("Associação fictícia", html)

    def test_route_notebook_file_is_valid_and_does_not_optimize_on_open(self) -> None:
        notebook = nbformat.read(
            ROOT / "notebooks/medical-routes/01_otimizacao_rotas.ipynb",
            as_version=4,
        )
        nbformat.validate(notebook)
        self.assertIn(
            "display_route_notebook",
            notebook.cells[-1].source,
        )
        self.assertNotIn("/api/rotas/otimizar", notebook.cells[-1].source)


if __name__ == "__main__":
    unittest.main()
