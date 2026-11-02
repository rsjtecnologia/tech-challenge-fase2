"""Interactive notebook widgets for Phase 2 interpretation and route planning."""

from __future__ import annotations

from html import escape
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from IPython.display import HTML, Markdown, display
import ipywidgets as widgets

from src.logistics.item_types import classify_medical_item_type


STUDY_LABELS = {
    "breast_cancer": "Câncer de mama",
    "pcos": "Síndrome do ovário policístico (PCOS)",
}
METRICS = (
    ("recall", "Recall"),
    ("precision", "Precisão"),
    ("f1", "F1"),
    ("accuracy", "Acurácia"),
)
EXAMPLES = (
    "Qual modelo teve o maior recall após a otimização?",
    "Em quais modelos o recall caiu após a otimização?",
    "Qual a diferença entre recall e precisão neste estudo?",
    "Compare o F1 da Regressao Logistica antes e depois da otimização.",
)
DEFAULT_API_BASE_URL = "http://127.0.0.1:8020"
API_BASE_URL_ENV = "TECH_CHALLENGE_API_BASE_URL"


class _APIRequestError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _local_api_url(api_base_url: str) -> str:
    parsed = urlsplit(api_base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError(
            "O notebook envia somente para a API local em http://127.0.0.1:<porta>."
        )
    return f"http://{parsed.netloc}"


def _configured_api_url(api_base_url: str | None) -> str:
    if api_base_url is None:
        api_base_url = os.environ.get(API_BASE_URL_ENV, DEFAULT_API_BASE_URL)
    return _local_api_url(api_base_url)


def _api_request(
    api_base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 15,
) -> dict[str, Any]:
    base_url = _local_api_url(api_base_url)
    if not path.startswith("/api/"):
        raise ValueError("A rota deve pertencer à API local do projeto.")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        try:
            error_body = json.loads(error.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            error_body = {}
        finally:
            error.close()
        detail = error_body.get("detail")
        message = detail if isinstance(detail, str) else f"API local respondeu HTTP {error.code}."
        raise _APIRequestError(message, error.code) from error
    except (TimeoutError, URLError) as error:
        raise RuntimeError(
            "Não foi possível concluir a chamada à API local. Verifique se o "
            "dashboard está ativo em localhost:8020 e mantenha o terminal aberto."
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("A API local retornou uma resposta inválida.") from error
    if not isinstance(result, dict):
        raise RuntimeError("A API local retornou um formato inesperado.")
    return result


def _comparison_html(data: dict[str, Any]) -> str:
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise RuntimeError(
            "Resultados genéticos ausentes. Execute antes "
            "`python -m src.optimization.cli --study <estudo>`."
        )
    rows = []
    for name, result in models.items():
        if not isinstance(result, dict):
            raise RuntimeError("Formato inválido nos resultados de otimização.")
        baseline = result.get("baseline_metrics")
        optimized = result.get("optimized_metrics")
        if not isinstance(baseline, dict) or not isinstance(optimized, dict):
            raise RuntimeError("Métricas baseline/otimizadas não encontradas.")
        experiment = escape(str(result.get("selected_experiment", "—")))
        model_name = escape(str(name))
        for metric, label in METRICS:
            try:
                before = float(baseline[metric])
                after = float(optimized[metric])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Métrica {label} ausente ou inválida no modelo {model_name}."
                ) from error
            delta = (after - before) * 100
            rows.append(
                "<tr>"
                f"<td>{model_name}</td><td>{experiment}</td><td>{label}</td>"
                f"<td>{before * 100:.2f}%</td><td>{after * 100:.2f}%</td>"
                f"<td>{delta:+.2f} p.p.</td></tr>"
            )
    return (
        '<div style="overflow-x:auto"><table><thead><tr>'
        "<th>Modelo</th><th>Experimento GA selecionado</th><th>Métrica</th>"
        "<th>Fase 1</th><th>Fase 2</th><th>Δ</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _answer_html(
    response: dict[str, Any], study: str, question: str | None
) -> str:
    interpretation = response.get("interpretation")
    if not isinstance(interpretation, dict):
        raise RuntimeError("Resposta da API não contém a interpretação esperada.")
    sections = [
        f"<p><strong>{escape(STUDY_LABELS[study])}</strong></p>"
    ]
    if question:
        sections.append(f"<p><strong>Pergunta:</strong> {escape(question)}</p>")
    if response.get("source") == "llm":
        sections.append("<p><strong>Origem:</strong> resposta da LLM</p>")
    elif response.get("source") == "local":
        sections.append("<p><strong>Origem:</strong> interpretação local por regras</p>")
    cache = response.get("cache")
    if isinstance(cache, dict) and cache.get("hit"):
        sections.append("<p>Resposta reutilizada do cache local.</p>")
    quality = response.get("quality_evaluation")
    if isinstance(quality, dict) and "score" in quality:
        sections.append(
            f"<p>Avaliação estrutural: {escape(str(quality['score']))}/"
            f"{escape(str(quality.get('maximum', '?')))}</p>"
        )
    if response.get("verified_summary"):
        sections.append(
            "<p>Resumo calculado diretamente das métricas; explicação complementar pela LLM.</p>"
        )
    for key, label in (
        ("summary", "Resumo"),
        ("findings", "Principais observações"),
        ("limitations", "Limitações"),
        ("disclaimer", "Aviso"),
    ):
        value = interpretation.get(key)
        if isinstance(value, list):
            items = "".join(f"<li>{escape(str(item))}</li>" for item in value)
            sections.append(f"<h4>{label}</h4><ul>{items}</ul>")
        elif isinstance(value, str):
            sections.append(f"<h4>{label}</h4><p>{escape(value)}</p>")
    return "<section>" + "".join(sections) + "</section>"


def display_phase2_interpretation(
    study: str,
    api_base_url: str | None = None,
) -> dict[str, Any]:
    """Display Phase 2 metrics, example badges, and general/question LLM actions."""
    if study not in STUDY_LABELS:
        raise ValueError(f"Estudo desconhecido: {study}")
    base_url = _configured_api_url(api_base_url)
    display(Markdown(
        "### Interpretação em linguagem natural\n"
        "Explicação agregada dos experimentos da Fase 2; não analisa pacientes."
    ))

    comparison_output = widgets.Output()
    with comparison_output:
        try:
            comparison = _api_request(base_url, f"/api/otimizacao/{study}")
            display(HTML(_comparison_html(comparison)))
        except RuntimeError as error:
            display(HTML(f"<p><strong>Comparação indisponível:</strong> {escape(str(error))}</p>"))

    question = widgets.Text(
        description="",
        placeholder="Digite sua pergunta ou escolha um exemplo abaixo",
        layout=widgets.Layout(width="100%"),
    )
    generate_button = widgets.Button(
        description="Gerar interpretação",
        button_style="primary",
    )
    ask_button = widgets.Button(
        description="Perguntar à LLM",
        button_style="primary",
    )
    examples = tuple(
        widgets.Button(description=label, tooltip=example)
        for label, example in zip(
            ("Maior recall", "Quedas de recall", "Recall × precisão", "F1 antes × depois"),
            EXAMPLES,
            strict=True,
        )
    )
    answer_output = widgets.Output()
    controls = [question, generate_button, ask_button, *examples]

    def submit_question(_: Any = None) -> None:
        text = question.value.strip()
        if not text:
            with answer_output:
                answer_output.clear_output()
                display(HTML("<p>Digite uma pergunta sobre as métricas agregadas.</p>"))
            return
        submit(text)

    def submit(text: str | None) -> None:
        for control in controls:
            control.disabled = True
        try:
            payload: dict[str, Any] = {"estudo": study}
            if text is not None:
                payload["pergunta"] = text
            with answer_output:
                answer_output.clear_output()
                display(HTML("<p>Consultando a LLM configurada no dashboard…</p>"))
            response = _api_request(base_url, "/api/interpretacao", payload)
            rendered = _answer_html(response, study, text)
            with answer_output:
                answer_output.clear_output()
                display(HTML(rendered))
        except RuntimeError as error:
            with answer_output:
                answer_output.clear_output()
                display(HTML(f"<p><strong>Não foi possível gerar a interpretação:</strong> {escape(str(error))}</p>"))
        finally:
            for control in controls:
                control.disabled = False

    def select_example(button: widgets.Button) -> None:
        question.value = button.tooltip

    for example_button in examples:
        example_button.on_click(select_example)
    generate_button.on_click(lambda _: submit(None))
    ask_button.on_click(submit_question)
    question.on_submit(submit_question)

    display(
        widgets.VBox([
            widgets.HTML("<h3>Gerar interpretação</h3>"),
            comparison_output,
            widgets.HBox([question, ask_button]),
            widgets.HBox(examples),
            widgets.HTML(
                "<p>Escolha um exemplo para preencher o campo, edite se desejar "
                "e clique em Perguntar à LLM ou pressione Enter.</p>"
                "<details><summary>Privacidade e uso da LLM</summary>"
                "<p>Somente métricas agregadas do estudo e a pergunta seguem para "
                "a LLM configurada no dashboard. Não informe nomes, prontuários, "
                "sintomas ou dados identificáveis de pacientes.</p></details>"
                "<p>Gere a explicação geral ou pergunte sobre as métricas agregadas. "
                "Os resultados da otimização genética precisam estar disponíveis; "
                "sem LLM, apenas a explicação geral por regras funciona.</p>"
            ),
            widgets.HBox([generate_button]),
            answer_output,
        ])
    )
    return {
        "question": question,
        "generate_button": generate_button,
        "ask_button": ask_button,
        "examples": examples,
        "comparison_output": comparison_output,
        "answer_output": answer_output,
    }


def _route_map_html(
    plan: dict[str, Any], solution: dict[str, Any]
) -> str:
    width, height, padding = 760, 430, 38
    routes = solution.get("routes")
    if not isinstance(routes, list) or not routes:
        return "<p>O plano não contém rotas viáveis para desenhar.</p>"
    points = [
        point
        for route in routes
        for point in route.get("path", [])
        if isinstance(point, dict)
    ]
    if not points:
        return "<p>Coordenadas de rota não disponíveis no plano.</p>"
    latitudes = [float(point["latitude"]) for point in points]
    longitudes = [float(point["longitude"]) for point in points]
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)
    lat_range = max(max_lat - min_lat, 1e-8)
    lon_range = max(max_lon - min_lon, 1e-8)
    colors = ("#60a5fa", "#7dd3fc", "#bfdbfe", "#38bdf8", "#ffffff")

    def project(point: dict[str, Any]) -> tuple[float, float]:
        x = padding + (
            (float(point["longitude"]) - min_lon) / lon_range
        ) * (width - 2 * padding)
        y = height - padding - (
            (float(point["latitude"]) - min_lat) / lat_range
        ) * (height - 2 * padding)
        return x, y

    drawings = []
    for index, route in enumerate(routes):
        color = colors[index % len(colors)]
        path_points = [
            project(point)
            for point in route.get("path", [])
            if isinstance(point, dict)
        ]
        if not path_points:
            continue
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in path_points)
        dash = ' stroke-dasharray="8 5"' if index % 2 else ""
        drawings.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" '
            f'stroke-width="3"{dash}/>'
        )
        for stop_index, stop in enumerate(route.get("stops", []), start=1):
            x, y = project(stop)
            items = stop.get("items")
            item_text = ", ".join(map(str, items)) if isinstance(items, list) else "Não informado"
            details = (
                f"{stop.get('name', stop.get('delivery_id', 'Destino'))}\n"
                f"Itens simulados: {item_text}\n"
                f"Prioridade: {stop.get('priority', '—')} · "
                f"Carga simulada: {float(stop.get('demand_kg', 0)):.2f} kg\n"
                f"Chegada: {stop.get('arrival_time', '—')} · "
                f"Atraso estimado: {float(stop.get('lateness_minutes', 0)):.0f} min"
            )
            title = escape(details)
            drawings.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" '
                f'stroke="#061a33" stroke-width="2"><title>{title}</title></circle>'
                f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" fill="#ffffff" '
                f'font-size="12">{stop_index}</text>'
            )
    depot = plan.get("depot")
    if isinstance(depot, dict):
        x, y = project(depot)
        drawings.append(
            f'<rect x="{x - 7:.1f}" y="{y - 7:.1f}" width="14" height="14" '
            'fill="#fbbf24" stroke="#061a33" stroke-width="2">'
            f'<title>{escape(str(depot.get("name", "Depósito")))}</title></rect>'
        )
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin:4px 12px 4px 0">'
        f'<i style="display:inline-block;width:14px;height:4px;background:{colors[index % len(colors)]}"></i>'
        f'{escape(str(route.get("vehicle_name", route.get("vehicle_id", "Veículo"))))}</span>'
        for index, route in enumerate(routes)
    )
    return (
        '<div style="overflow-x:auto"><svg role="img" aria-label="Mapa esquemático '
        'das rotas estimadas" viewBox="0 0 760 430" style="width:100%;min-width:420px;'
        'background:#0b294b;border-radius:12px" xmlns="http://www.w3.org/2000/svg">'
        + "".join(drawings)
        + "</svg></div><p>"
        + legend
        + '<span>■ Central de distribuição</span></p><p>Selecione ou passe o '
        "cursor sobre uma parada para conferir os itens e as restrições. "
        "Mapa esquemático; linhas retas entre coordenadas, sem ruas ou trânsito.</p>"
    )


def _route_result_html(
    plan: dict[str, Any], view: str, scenario: dict[str, Any] | None = None
) -> tuple[str, str]:
    comparison = plan.get("comparison")
    if not isinstance(comparison, dict) or not {"baseline", "optimized"}.issubset(comparison):
        raise RuntimeError("Plano sem comparação baseline/otimizado válida.")
    optimized = comparison["optimized"]
    baseline = comparison["baseline"]
    current = comparison[view]
    summary_rows = (
        ("Distância estimada", "total_distance_km", "km", 3),
        ("Entregas atendidas", "served_delivery_count", "", 0),
        ("Entregas no plano", "delivery_count", "", 0),
        ("Entregas atrasadas", "late_delivery_count", "", 0),
        ("Veículos utilizados", "vehicles_used", "", 0),
        ("Tempo estimado", "estimated_vehicle_minutes", "min", 2),
        ("Carga total simulada", "total_load_kg", "kg", 2),
        ("Utilização da frota", "fleet_utilization_percent", "%", 2),
    )
    rows = []
    for label, key, unit, decimals in summary_rows:
        value = current.get(key, 0)
        formatted = f"{float(value):,.{decimals}f} {unit}".strip() if decimals else f"{value} {unit}".strip()
        rows.append(f"<tr><th>{label}</th><td>{escape(formatted)}</td></tr>")
    if scenario:
        deliveries = scenario.get("deliveries", [])
        aggregates = scenario.get("data_source", {}).get("municipal_aggregates", [])
        for item_type, label, historical_field in (
            ("medicamento", "Destinos com medicamentos", "historical_medicine_items"),
            ("insumo", "Destinos com insumos", "historical_supply_items"),
            (
                "saude_beleza_nao_confirmado",
                "Olist: saúde/beleza (não confirmado como item médico)",
                "historical_health_beauty_items",
            ),
        ):
            simulated_ids = {
                str(delivery.get("id"))
                for delivery in deliveries
                if any(
                    classify_medical_item_type(item) == item_type
                    for item in delivery.get("items", [])
                )
            }
            historical_ids = {
                str(aggregate.get("delivery_id"))
                for aggregate in aggregates
                if int(aggregate.get(historical_field, 0)) > 0
            }
            historical_item_count = sum(
                int(aggregate.get(historical_field, 0))
                for aggregate in aggregates
            )
            value = (
                f"{len(simulated_ids | historical_ids)} destino(s) · "
                f"{len(simulated_ids)} simulado(s) · "
                f"{historical_item_count} item(ns) Olist"
            )
            rows.append(
                f"<tr><th>{label}</th><td>{escape(value)}</td></tr>"
            )
        manifests = scenario.get("data_source", {}).get("distribution_manifest", [])
        for category, label in (
            ("medicamento", "Medicamentos simulados"),
            ("insumo", "Insumos simulados"),
        ):
            lines = [
                line for manifest in manifests for line in manifest["lines"]
                if line["category"] == category
            ]
            if lines:
                quantity = sum(line["quantity"] for line in lines)
                load = sum(line["load_kg"] for line in lines)
                rows.append(
                    f"<tr><th>{label}</th><td>{quantity} kits · {load:.2f} kg</td></tr>"
                )
    metrics_html = (
        "<h4>Resumo da solução "
        + ("genética" if view == "optimized" else "baseline")
        + "</h4><table><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    saved = float(comparison.get("distance_saved_km", 0))
    percent = float(comparison.get("distance_saved_percent", 0))
    comparison_html = (
        "<p><strong>Comparação do algoritmo:</strong> "
        f"baseline {float(baseline['total_distance_km']):.3f} km → "
        f"otimizado {float(optimized['total_distance_km']):.3f} km; "
        f"diferença estimada {saved:.3f} km ({percent:.2f}%). "
        "Estimativa da simulação, não economia realizada.</p>"
    )
    route_rows = []
    for route in current.get("routes", []):
        for sequence, stop in enumerate(route.get("stops", []), start=1):
            route_rows.append(
                "<tr>"
                f"<td>{escape(str(route.get('vehicle_name', route.get('vehicle_id', ''))))}</td>"
                f"<td>{sequence}</td><td>{escape(str(stop.get('name', '')))}</td>"
                f"<td>{escape(str(stop.get('priority', '')))}</td>"
                f"<td>{float(stop.get('demand_kg', 0)):.2f} kg</td>"
                f"<td>{escape(str(stop.get('arrival_time', '—')))}</td>"
                f"<td>{float(stop.get('lateness_minutes', 0)):.1f} min</td>"
                "</tr>"
            )
    routes_html = (
        "<h4>Sequência de paradas e horários estimados</h4>"
        "<div style='overflow-x:auto'><table><thead><tr><th>Veículo</th>"
        "<th>Ordem</th><th>Destino</th><th>Prioridade</th><th>Carga simulada</th>"
        "<th>Chegada estimada</th><th>Atraso estimado</th></tr></thead><tbody>"
        + "".join(route_rows)
        + "</tbody></table></div>"
    )
    if current.get("unserved_delivery_ids"):
        unserved = ", ".join(map(str, current["unserved_delivery_ids"]))
        routes_html += f"<p>Destinos não atendidos no plano: {escape(unserved)}</p>"
    return metrics_html + comparison_html + routes_html, _route_map_html(plan, current)


def _route_assistant_html(response: dict[str, Any], action: str) -> str:
    content = response.get("content")
    if not isinstance(content, dict):
        raise RuntimeError("A API retornou um conteúdo do assistente inválido.")
    origin = (
        "Resposta gerada pela LLM configurada."
        if response.get("source") == "llm"
        else "Resposta local, baseada no plano (sem LLM)."
    )
    sections = [f"<p><strong>{origin}</strong></p>"]
    if response.get("cache", {}).get("hit"):
        sections.append("<p>Resposta reutilizada do cache deste plano.</p>")
    if action == "instructions":
        sections.append(f"<p>{escape(str(content.get('summary', '')))}</p>")
        for vehicle in content.get("vehicle_instructions", []):
            stops = "".join(
                f"<li>{escape(str(stop))}</li>"
                for stop in vehicle.get("ordered_stops", [])
            )
            notes = vehicle.get("notes", "")
            if isinstance(notes, list):
                notes = " ".join(map(str, notes))
            sections.append(
                f"<h4>{escape(str(vehicle.get('vehicle', 'Veículo')))}</h4>"
                f"<ol>{stops}</ol><p>{escape(str(notes))}</p>"
            )
        checks = "".join(
            f"<li>{escape(str(item))}</li>"
            for item in content.get("operational_checks", [])
        )
        sections.append(f"<h4>Verificações operacionais</h4><ul>{checks}</ul>")
    elif action in ("daily_report", "weekly_report"):
        sections.append(f"<p><strong>{escape(str(content.get('period', '')))}</strong></p>")
        sections.append(f"<p>{escape(str(content.get('summary', '')))}</p>")
        metrics = "".join(
            f"<li><strong>{escape(str(key))}:</strong> "
            f"{escape(str(value))}</li>"
            for key, value in content.get("key_metrics", {}).items()
        )
        sections.append(f"<h4>Métricas verificadas pelo Python</h4><ul>{metrics}</ul>")
        for key, label in (("improvements", "Melhorias propostas"), ("caveats", "Limitações")):
            items = "".join(
                f"<li>{escape(str(item))}</li>" for item in content.get(key, [])
            )
            sections.append(f"<h4>{label}</h4><ul>{items}</ul>")
    else:
        sections.append(f"<p>{escape(str(content.get('answer', '')))}</p>")
        for key, label in (
            ("supporting_facts", "Dados que sustentam a resposta"),
            ("missing_information", "Informações não disponíveis"),
        ):
            items = "".join(
                f"<li>{escape(str(item))}</li>" for item in content.get(key, [])
            )
            sections.append(f"<h4>{label}</h4><ul>{items}</ul>")
    if content.get("disclaimer"):
        sections.append(f"<p><strong>{escape(str(content['disclaimer']))}</strong></p>")
    return "<section>" + "".join(sections) + "</section>"


def display_route_notebook(
    api_base_url: str | None = None,
) -> dict[str, Any]:
    """Display route scenario, GA optimization, map and assistant in a notebook."""
    base_url = _configured_api_url(api_base_url)
    display(Markdown(
        "## Fase 2 — Otimização de rotas médicas (VRP)\n\n"
        "Demonstração acadêmica: coordenadas e rotas são estimativas em linha reta. "
        "No Olist, destinos e volumes são agregados de e-commerce; frota, cargas, "
        "prioridades e prazos médicos são simulados. CNES usa coordenadas reais "
        "do CSV local, mas demanda médica permanece simulada sem dados específicos. "
        "Não representa operação real."
    ))
    source = widgets.Dropdown(
        options=(
            ("Olist — municípios do estado de São Paulo", "olist_sp"),
            ("Distribuição híbrida — medicamentos e insumos", "hybrid_sp"),
            ("CNES — estabelecimentos de saúde SP (CSV local)", "cnes_sp"),
            ("Cenário fictício de demonstração", "demo"),
        ),
        value="olist_sp",
        description="Fonte:",
        layout=widgets.Layout(width="480px"),
    )
    population = widgets.IntSlider(
        value=32, min=8, max=128, step=8, description="População:",
        continuous_update=False, readout=True,
    )
    generations = widgets.IntSlider(
        value=40, min=10, max=160, step=10, description="Gerações:",
        continuous_update=False, readout=True,
    )
    mutation = widgets.FloatSlider(
        value=0.18, min=0.0, max=0.8, step=0.02, description="Mutação:",
        continuous_update=False, readout_format=".2f",
    )
    tournament = widgets.IntSlider(
        value=4, min=2, max=16, step=1, description="Torneio:",
        continuous_update=False,
    )
    seed = widgets.IntText(value=42, description="Semente:")
    optimize_button = widgets.Button(
        description="Otimizar rotas (GA)", button_style="primary"
    )
    view = widgets.ToggleButtons(
        options=(("Otimizado", "optimized"), ("Baseline", "baseline")),
        value="optimized",
        description="Exibir:",
    )
    scenario_output = widgets.Output()
    metrics_output = widgets.Output()
    map_output = widgets.Output()
    history_output = widgets.Output()
    assistant_output = widgets.Output()
    question = widgets.Text(
        placeholder="Pergunte sobre veículos, entregas, distâncias e restrições",
        layout=widgets.Layout(width="100%"),
    )
    ask_button = widgets.Button(description="Perguntar")
    examples = tuple(
        widgets.Button(description=label, tooltip=prompt)
        for label, prompt in zip(
            ("Itens refrigerados", "Distância e redução", "Entregas e atrasos", "Veículos e cargas"),
            (
                "Qual veículo leva os itens refrigerados?",
                "Qual a distância total e a redução estimada em relação à rota base?",
                "Quantas entregas foram atendidas e quantas estão atrasadas no plano otimizado?",
                "Quantos veículos são utilizados e qual a carga de cada um?",
            ),
            strict=True,
        )
    )
    action_buttons = {
        action: widgets.Button(description=label)
        for action, label in (
            ("instructions", "Instruções para motoristas"),
            ("daily_report", "Relatório diário"),
            ("weekly_report", "Relatório semanal"),
        )
    }
    state: dict[str, Any] = {"scenario": None, "plan": None, "history": None}
    async_controls = [
        source, population, generations, mutation, tournament, seed, optimize_button,
        view, question, ask_button, *examples, *action_buttons.values(),
    ]

    def render_scenario() -> None:
        state["plan"] = None
        scenario_output.clear_output()
        with scenario_output:
            try:
                state["scenario"] = _api_request(
                    base_url, f"/api/rotas/cenario?source={source.value}"
                )
                details = state["scenario"]
                data = details.get("data_source") or {}
                description = escape(str(details.get("description", "")))
                label = escape(str(details.get("name", "")))
                display(HTML(f"<h3>{label}</h3><p>{description}</p>"))
                if data.get("kind") == "olist_sp":
                    display(HTML(
                        "<p><strong>Base agregada:</strong> "
                        f"{int(data.get('delivered_orders_sp', 0)):,} pedidos históricos "
                        f"em {int(data.get('municipalities_with_coordinates', 0))} municípios; "
                        f"{int(data.get('selected_municipalities', 0))} destinos selecionados. "
                        "Demanda/medicamentos são simulados, não derivados do volume como carga médica."
                        "</p>"
                    ))
                    profile = data.get("kaggle_inventory_profile")
                    if profile:
                        leaders = sorted(
                            profile.get("items", []),
                            key=lambda item: item.get("average_usage_per_day", 0),
                            reverse=True,
                        )[:3]
                        profile_items = "; ".join(
                            f"{escape(str(item.get('item_name', '')))} "
                            f"({escape(str(item.get('item_type', '')))}): "
                            f"{float(item.get('average_usage_per_day', 0)):.2f}/dia"
                            for item in leaders
                        )
                        display(HTML(
                            "<p><strong>Perfil de referência Kaggle (não associado "
                            "à demanda Olist/CNES):</strong> "
                            f"{int(profile.get('records', 0))} registros; "
                            f"maiores médias: {profile_items}. "
                            "Não representa demanda por unidade e não altera kg da rota."
                            "</p>"
                        ))
                elif data.get("kind") == "cnes_sp":
                    profile = data.get("kaggle_inventory_profile")
                    profile_text = (
                        f" Perfil Hospital Supply Chain/Kaggle: {int(profile.get('records', 0))} "
                        f"registros de inventário em {len(profile.get('items', []))} "
                        "combinações tipo/item, apenas referência global sem ligação a CNES."
                        if profile else " Perfil Kaggle opcional não fornecido."
                    )
                    if profile:
                        leaders = sorted(
                            profile.get("items", []),
                            key=lambda item: item.get("average_usage_per_day", 0),
                            reverse=True,
                        )[:3]
                        profile_text += " Maiores médias/dia: " + "; ".join(
                            f"{escape(str(item.get('item_name', '')))} "
                            f"({escape(str(item.get('item_type', '')))}): "
                            f"{float(item.get('average_usage_per_day', 0)):.2f}"
                            for item in leaders
                        ) + ". Esses valores não são demanda das unidades CNES."
                    display(HTML(
                        "<p><strong>Base geográfica:</strong> "
                        f"{int(data.get('establishments_sp_with_coordinates', 0))} "
                        "estabelecimentos CNES em SP com coordenadas válidas; "
                        f"{int(data.get('selected_establishments', 0))} destinos selecionados. "
                        f"{escape(str(data.get('demand_source', '')))}. "
                        "Itens, prioridades e cargas são simulados quando não informados no CSV."
                        f"{escape(profile_text)}"
                        "</p>"
                    ))
                elif data.get("kind") == "hybrid_sp":
                    display(HTML(
                        "<p><strong>Distribuição médica simulada:</strong> "
                        f"geografia {escape(str(data.get('geography_source', '')))}; "
                        f"{escape(str(data.get('demand_source', '')))} "
                        f"{escape(str(data['distribution_method']['priority_rule']))}</p>"
                    ))
                aggregates = {
                    item.get("delivery_id"): item
                    for item in (
                        data.get("municipal_aggregates", [])
                        or data.get("facility_aggregates", [])
                    )
                }
                delivery_rows = []
                distribution = {
                    manifest["delivery_id"]: manifest
                    for manifest in data.get("distribution_manifest", [])
                }
                for delivery in details.get("deliveries", []):
                    aggregate = aggregates.get(delivery.get("id"), {})
                    manifest = distribution.get(delivery.get("id"))
                    if manifest:
                        historical = "; ".join(
                            f"{line['name']}: {line['quantity']} kits; "
                            f"estoque {line['stock_units']}, mínimo {line['minimum_stock_units']}; "
                            f"uso {line['daily_usage_units']}/dia; "
                            f"cobertura {line['stock_coverage_days']} dias; "
                            f"reposição {line['restock_lead_time_days']} dias (simulado)"
                            for line in manifest["lines"]
                        )
                    elif data.get("kind") == "cnes_sp" and aggregate:
                        historical = (
                            f"{escape(str(aggregate.get('facility_type', '')))}; "
                            f"prioridade {int(aggregate.get('priority_score', 0))}/10; "
                            f"{float(aggregate.get('medicine_demand_kg', 0)):.2f} kg medicamento; "
                            f"{float(aggregate.get('supply_demand_kg', 0)):.2f} kg insumo "
                            f"({'simulados' if aggregate.get('demand_is_simulated') else 'do CSV'})"
                        )
                    else:
                        historical = (
                            f"{int(aggregate.get('historical_delivered_orders', 0)):,} pedidos; "
                            f"{int(aggregate.get('historical_items', 0)):,} itens; "
                            f"{float(aggregate.get('observed_product_weight_kg', 0)):.2f} kg "
                            f"históricos observados ({int(aggregate.get('items_with_known_weight', 0))} "
                            "itens com peso conhecido)"
                            if aggregate else "—"
                        )
                    destination_name = str(delivery.get("name", ""))
                    if aggregate.get("cnes_id"):
                        destination_name += f" · CNES {aggregate['cnes_id']}"
                    delivery_rows.append(
                        "<tr>"
                        f"<td>{escape(destination_name)}</td>"
                        f"<td>{escape(str(delivery.get('priority', '')))}</td>"
                        f"<td>{float(delivery.get('demand_kg', 0)):.2f} kg (simulado)</td>"
                        f"<td>{escape(str(delivery.get('deadline_time', '—')))}</td>"
                        f"<td>{'Sim' if delivery.get('requires_refrigeration') else 'Não'}</td>"
                        f"<td>{escape(historical)}</td>"
                        "</tr>"
                    )
                display(HTML(
                    "<h4>Destinos e restrições simuladas</h4>"
                    "<div style='overflow-x:auto'><table><thead><tr><th>Destino</th>"
                    "<th>Prioridade</th><th>Carga</th><th>Prazo</th><th>Cadeia fria</th>"
                    "<th>Dados de origem/demanda</th>"
                    "</tr></thead><tbody>"
                    + "".join(delivery_rows)
                    + "</tbody></table></div>"
                ))
                if not state["plan"]:
                    try:
                        state["plan"] = _api_request(
                            base_url, f"/api/rotas/resultado?source={source.value}"
                        )
                    except _APIRequestError as error:
                        if error.status_code != 404:
                            raise
                        state["plan"] = None
                        state["plan_missing"] = True
                    else:
                        state["plan_missing"] = False
            except (RuntimeError, KeyError, TypeError, ValueError) as error:
                state["scenario"] = None
                state["plan"] = None
                state["plan_missing"] = False
                display(HTML(
                    f"<p><strong>Cenário indisponível:</strong> {escape(str(error))}</p>"
                ))
        render_plan()
        render_history()

    def render_plan() -> None:
        metrics_output.clear_output()
        map_output.clear_output()
        if not state["plan"]:
            with metrics_output:
                message = (
                    "Não há plano salvo para esta fonte. Ajuste os parâmetros e "
                    "clique em Otimizar rotas (GA)."
                    if state.get("plan_missing")
                    else "Sem plano nesta fonte. Ajuste os parâmetros e clique em Otimizar rotas (GA)."
                )
                display(HTML(f"<p>{escape(message)}</p>"))
            return
        try:
            metrics, route_map = _route_result_html(
                state["plan"], view.value, state["scenario"]
            )
            with metrics_output:
                display(HTML(metrics))
            with map_output:
                display(HTML(route_map))
        except (RuntimeError, KeyError, TypeError, ValueError) as error:
            with metrics_output:
                display(HTML(f"<p><strong>Plano inválido:</strong> {escape(str(error))}</p>"))

    def render_history() -> None:
        history_output.clear_output()
        with history_output:
            try:
                history = _api_request(
                    base_url, f"/api/rotas/historico?source={source.value}"
                )
                state["history"] = history
                count = int(history.get("run_count", 0))
                if count == 0:
                    message = "Ainda não há simulações registradas nesta fonte."
                else:
                    message = (
                        f"{count} simulações locais nos últimos "
                        f"{history.get('window_days', 7)} dias. "
                        "Histórico de simulações, não de entregas reais."
                    )
                    if history.get("trend_available"):
                        message += (
                            f" Distância otimizada média: "
                            f"{history.get('mean_optimized_distance_km')} km; "
                            f"atrasos médios: {history.get('mean_late_delivery_count')}."
                        )
                display(HTML(f"<p>{escape(message)}</p>"))
            except RuntimeError as error:
                state["history"] = None
                display(HTML(f"<p>Histórico indisponível: {escape(str(error))}</p>"))

    def run_action(action: str, text: str | None = None) -> None:
        if not state["plan"]:
            with assistant_output:
                assistant_output.clear_output()
                display(HTML("<p>Otimize um plano para esta fonte antes de consultar o assistente.</p>"))
            return
        for control in async_controls:
            control.disabled = True
        assistant_output.clear_output()
        with assistant_output:
            display(HTML("<p>Consultando o assistente com o plano agregado desta simulação…</p>"))
        payload: dict[str, Any] = {"source": source.value, "action": action}
        if text is not None:
            payload["question"] = text
        try:
            response = _api_request(
                base_url, "/api/rotas/assistente", payload, timeout=300
            )
            rendered = _route_assistant_html(response, action)
        except RuntimeError as error:
            rendered = (
                f"<p><strong>Não foi possível gerar a resposta:</strong> "
                f"{escape(str(error))}</p>"
            )
        finally:
            for control in async_controls:
                control.disabled = False
        with assistant_output:
            assistant_output.clear_output()
            display(HTML(rendered))

    def optimize(_: Any = None) -> None:
        for control in async_controls:
            control.disabled = True
        metrics_output.clear_output()
        with metrics_output:
            display(HTML(
                "<p>Executando algoritmo genético local; o tempo depende do "
                "número de destinos e gerações selecionado…</p>"
            ))
        try:
            if tournament.value > population.value:
                raise ValueError("O tamanho do torneio não pode exceder a população.")
            plan = _api_request(
                base_url,
                "/api/rotas/otimizar",
                {
                    "source": source.value,
                    "population_size": population.value,
                    "generations": generations.value,
                    "mutation_rate": mutation.value,
                    "tournament_size": tournament.value,
                    "seed": seed.value,
                },
                timeout=600,
            )
            state["plan"] = plan
            render_plan()
            render_history()
        except (RuntimeError, ValueError) as error:
            state["plan"] = None
            metrics_output.clear_output()
            with metrics_output:
                display(HTML(
                    f"<p><strong>Não foi possível otimizar:</strong> {escape(str(error))}</p>"
                ))
        finally:
            for control in async_controls:
                control.disabled = False

    def select_example(button: widgets.Button) -> None:
        question.value = button.tooltip

    def ask_question(_: Any) -> None:
        if question.value.strip():
            run_action("question", question.value.strip())
            return
        assistant_output.clear_output()
        with assistant_output:
            display(HTML("<p>Digite uma pergunta ou escolha um exemplo antes de enviar.</p>"))

    source.observe(lambda change: render_scenario() if change["name"] == "value" else None, names="value")
    view.observe(lambda change: render_plan() if change["name"] == "value" else None, names="value")
    optimize_button.on_click(optimize)
    for action, button in action_buttons.items():
        button.on_click(lambda _, selected=action: run_action(selected))
    for button in examples:
        button.on_click(select_example)
    ask_button.on_click(ask_question)
    question.on_submit(ask_question)

    render_scenario()
    display(widgets.VBox([
        widgets.HTML("<h3>Otimização de rotas via algoritmo genético</h3>"),
        widgets.HBox([source]),
        widgets.HBox([population, generations]),
        widgets.HBox([mutation, tournament, seed]),
        widgets.HBox([optimize_button, view]),
        scenario_output,
        metrics_output,
        map_output,
        widgets.HTML("<h3>Histórico local de simulações (últimos sete dias)</h3>"),
        history_output,
        widgets.HTML("<h3>Assistente de logística</h3>"),
        widgets.HBox(list(action_buttons.values())),
        widgets.HBox([question, ask_button]),
        widgets.HBox(examples),
        widgets.HTML(
            "<p>Escolha um exemplo para preencher a pergunta; revise e envie pelo "
            "botão ou Enter. O assistente local responde a perguntas com fatos do "
            "plano; uma LLM configurada pode ampliar as interpretações. Instruções "
            "e relatórios também têm resposta local.</p>"
            "<details><summary>Privacidade e uso da LLM</summary>"
            "<p>São enviados apenas dados agregados/simulados do plano e sua pergunta "
            "ao provedor configurado. Não informe nomes, endereços individuais, "
            "prontuários ou dados identificáveis. O dataset Olist descreve comércio "
            "eletrônico, não uma operação médica.</p></details>"
        ),
        assistant_output,
    ]))
    return {
        "source": source,
        "population": population,
        "generations": generations,
        "mutation": mutation,
        "tournament": tournament,
        "seed": seed,
        "optimize_button": optimize_button,
        "view": view,
        "question": question,
        "ask_button": ask_button,
        "examples": examples,
        "action_buttons": action_buttons,
        "scenario_output": scenario_output,
        "metrics_output": metrics_output,
        "map_output": map_output,
        "history_output": history_output,
        "assistant_output": assistant_output,
        "state": state,
    }
