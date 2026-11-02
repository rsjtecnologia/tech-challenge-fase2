"""Local orchestration and persistence for synthetic route optimization."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4
from typing import Literal

from src.logistics.datasets import load_olist_scenario
from src.logistics.cnes_dataset import load_cnes_scenario
from src.logistics.distribution import load_distribution_scenario
from src.logistics.genetic import GAConfig, optimize_routes
from src.logistics.scenario import DEMO_SCENARIO, Scenario


LOGGER = logging.getLogger(__name__)
RESULTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "db"
    / "outputs"
    / "phase2-logistics"
    / "latest_route_plan.json"
)
HISTORY_PATH = RESULTS_PATH.with_name("route_plan_history.json")
HISTORY_WINDOW_DAYS = 7
HISTORY_LIMIT = 500
ScenarioSource = Literal["demo", "olist_sp", "cnes_sp", "hybrid_sp"]


def get_scenario(source: ScenarioSource = "demo") -> Scenario:
    if source == "hybrid_sp":
        return load_distribution_scenario()
    if source == "olist_sp":
        return load_olist_scenario()
    if source == "cnes_sp":
        return load_cnes_scenario()
    if source != "demo":
        raise ValueError("Fonte de cenário inválida.")
    return DEMO_SCENARIO


def _source_path(path: Path, source: ScenarioSource) -> Path:
    return path.with_name(f"{path.stem}_{source}{path.suffix}") if source != "demo" else path


def scenario_payload(source: ScenarioSource = "demo") -> dict[str, Any]:
    scenario = get_scenario(source)
    return {
        "id": scenario.id,
        "name": scenario.name,
        "description": scenario.description,
        "synthetic": scenario.synthetic,
        "data_source": scenario.data_source,
        "depot": {
            "id": scenario.depot.id,
            "name": scenario.depot.name,
            "latitude": scenario.depot.latitude,
            "longitude": scenario.depot.longitude,
        },
        "vehicles": [
            {
                "id": vehicle.id,
                "name": vehicle.name,
                "capacity_kg": vehicle.capacity_kg,
                "max_distance_km": vehicle.max_distance_km,
                "speed_kmh": vehicle.speed_kmh,
                "refrigerated": vehicle.refrigerated,
                "start_time": f"{vehicle.start_minute // 60:02d}:{vehicle.start_minute % 60:02d}",
            }
            for vehicle in scenario.vehicles
        ],
        "deliveries": [
            {
                "id": delivery.id,
                "name": delivery.name,
                "latitude": delivery.latitude,
                "longitude": delivery.longitude,
                "demand_kg": delivery.demand_kg,
                "priority": delivery.priority,
                "deadline_minute": delivery.deadline_minute,
                "deadline_time": f"{delivery.deadline_minute // 60:02d}:{delivery.deadline_minute % 60:02d}",
                "requires_refrigeration": delivery.requires_refrigeration,
                "service_minutes": delivery.service_minutes,
                "items": list(delivery.items),
            }
            for delivery in scenario.deliveries
        ],
    }


def save_route_plan(result: dict[str, Any], source: ScenarioSource = "demo") -> Path:
    result_path = _source_path(RESULTS_PATH, source)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=result_path.parent,
            delete=False,
            suffix=".tmp",
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(result, output_file, ensure_ascii=False, indent=2)
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.replace(result_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    LOGGER.info(
        "route_plan_saved path=%s distance_km=%.3f unserved=%d",
        result_path,
        result["comparison"]["optimized"]["total_distance_km"],
        len(result["comparison"]["optimized"]["unserved_delivery_ids"]),
    )
    return result_path


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            suffix=".tmp",
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _history_record(result: dict[str, Any]) -> dict[str, Any]:
    _ensure_plan_metrics(result)
    comparison = result["comparison"]
    baseline = comparison["baseline"]
    optimized = comparison["optimized"]
    return {
        "run_id": str(uuid4()),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_generated_at": result.get("generated_at"),
        "scenario_id": result["scenario_id"],
        "algorithm_config": result["algorithm"]["config"],
        "elapsed_seconds": result["elapsed_seconds"],
        "baseline": {
            "distance_km": baseline["total_distance_km"],
            "estimated_vehicle_minutes": baseline["estimated_vehicle_minutes"],
            "vehicles_used": baseline["vehicles_used"],
            "total_load_kg": baseline["total_load_kg"],
            "fleet_utilization_percent": baseline["fleet_utilization_percent"],
            "served_delivery_count": baseline["served_delivery_count"],
            "delivery_count": baseline["delivery_count"],
            "late_delivery_count": baseline["late_delivery_count"],
            "weighted_lateness": baseline["total_weighted_lateness"],
        },
        "optimized": {
            "distance_km": optimized["total_distance_km"],
            "estimated_vehicle_minutes": optimized["estimated_vehicle_minutes"],
            "vehicles_used": optimized["vehicles_used"],
            "total_load_kg": optimized["total_load_kg"],
            "fleet_utilization_percent": optimized["fleet_utilization_percent"],
            "served_delivery_count": optimized["served_delivery_count"],
            "delivery_count": optimized["delivery_count"],
            "late_delivery_count": optimized["late_delivery_count"],
            "weighted_lateness": optimized["total_weighted_lateness"],
        },
        "distance_saved_km": comparison["distance_saved_km"],
        "distance_saved_percent": comparison["distance_saved_percent"],
    }


def _ensure_plan_metrics(result: dict[str, Any]) -> None:
    vehicle_specs = {vehicle["id"]: vehicle for vehicle in result.get("vehicles", [])}
    for label in ("baseline", "optimized"):
        evaluation = result["comparison"][label]
        routes = evaluation.get("routes", [])
        if "estimated_vehicle_minutes" not in evaluation:
            evaluation["estimated_vehicle_minutes"] = round(
                sum(
                    route["distance_km"]
                    / route.get(
                        "speed_kmh",
                        vehicle_specs[route["vehicle_id"]]["speed_kmh"],
                    )
                    * 60
                    + sum(stop["service_minutes"] for stop in route["stops"])
                    for route in routes
                ),
                2,
            )
        evaluation.setdefault("vehicles_used", len(routes))
        evaluation.setdefault(
            "total_load_kg",
            round(sum(route["load_kg"] for route in routes), 2),
        )
        if "fleet_utilization_percent" not in evaluation:
            used_capacity = sum(route["capacity_kg"] for route in routes)
            evaluation["fleet_utilization_percent"] = round(
                100 * evaluation["total_load_kg"] / used_capacity
                if used_capacity
                else 0,
                2,
            )


def _read_history(source: ScenarioSource = "demo") -> list[dict[str, Any]]:
    history_path = _source_path(HISTORY_PATH, source)
    if not history_path.is_file():
        return []
    try:
        with history_path.open(encoding="utf-8") as history_file:
            history = json.load(history_file)
    except json.JSONDecodeError as error:
        raise ValueError("O histórico de rotas não contém JSON válido.") from error
    if (
        not isinstance(history, dict)
        or history.get("schema_version") != 1
        or not isinstance(history.get("runs"), list)
        or not all(isinstance(run, dict) for run in history["runs"])
    ):
        raise ValueError("O arquivo de histórico de rotas possui formato inválido.")
    return history["runs"]


def _save_history(runs: list[dict[str, Any]], source: ScenarioSource = "demo") -> None:
    _atomic_json_write(
        _source_path(HISTORY_PATH, source),
        {"schema_version": 1, "runs": runs[-HISTORY_LIMIT:]},
    )


def append_route_history(
    result: dict[str, Any], source: ScenarioSource = "demo"
) -> dict[str, Any]:
    record = _history_record(result)
    runs = _read_history(source)
    runs.append(record)
    _save_history(runs, source)
    LOGGER.info(
        "route_history_recorded run_id=%s scenario=%s distance_km=%.3f",
        record["run_id"],
        record["scenario_id"],
        record["optimized"]["distance_km"],
    )
    return record


def route_history_summary(
    now: datetime | None = None,
    current_plan: dict[str, Any] | None = None,
    source: ScenarioSource = "demo",
) -> dict[str, Any]:
    """Return aggregate metrics for recorded synthetic runs in the last 7 days."""
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    start_time = current_time - timedelta(days=HISTORY_WINDOW_DAYS)

    scenario_id = get_scenario(source).id
    if current_plan and current_plan.get("scenario_id") != scenario_id:
        raise ValueError("O plano e a fonte do histórico pertencem a cenários diferentes.")
    runs = _read_history(source)
    if current_plan and not any(
        run.get("scenario_id") == current_plan.get("scenario_id")
        and run.get("source_generated_at") == current_plan.get("generated_at")
        for run in runs
    ):
        legacy_record = _history_record(current_plan)
        legacy_record["run_id"] = f"saved-plan:{current_plan['generated_at']}"
        legacy_record["recorded_at"] = current_plan["generated_at"]
        runs.append(legacy_record)
        _save_history(runs, source)

    recent_runs = []
    for run in runs:
        if run.get("scenario_id") != scenario_id:
            continue
        try:
            recorded_at = datetime.fromisoformat(
                str(run["recorded_at"]).replace("Z", "+00:00")
            )
        except (KeyError, ValueError):
            raise ValueError("O histórico contém uma data de execução inválida.")
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=timezone.utc)
        if start_time <= recorded_at <= current_time:
            recent_runs.append(run)

    recent_runs.sort(key=lambda run: run["recorded_at"])
    optimized_distances = [
        float(run["optimized"]["distance_km"]) for run in recent_runs
    ]
    baseline_distances = [
        float(run["baseline"]["distance_km"]) for run in recent_runs
    ]
    lateness_counts = [
        int(run["optimized"]["late_delivery_count"]) for run in recent_runs
    ]
    weighted_lateness = [
        float(run["optimized"]["weighted_lateness"]) for run in recent_runs
    ]
    vehicle_minutes = [
        float(run["optimized"]["estimated_vehicle_minutes"])
        for run in recent_runs
    ]
    baseline_vehicle_minutes = [
        float(run["baseline"]["estimated_vehicle_minutes"])
        for run in recent_runs
    ]
    vehicles_used = [
        int(run["optimized"]["vehicles_used"]) for run in recent_runs
    ]
    fleet_utilization = [
        float(run["optimized"]["fleet_utilization_percent"])
        for run in recent_runs
    ]
    distance_saved_percent = [
        float(run["distance_saved_percent"]) for run in recent_runs
    ]
    count = len(recent_runs)
    return {
        "scenario_id": scenario_id,
        "window_days": HISTORY_WINDOW_DAYS,
        "run_count": count,
        "minimum_runs_for_trend": 2,
        "trend_available": count >= 2,
        "first_run_at": recent_runs[0]["recorded_at"] if recent_runs else None,
        "last_run_at": recent_runs[-1]["recorded_at"] if recent_runs else None,
        "mean_optimized_distance_km": (
            round(sum(optimized_distances) / count, 3) if count else None
        ),
        "mean_baseline_distance_km": (
            round(sum(baseline_distances) / count, 3) if count else None
        ),
        "mean_distance_saved_percent": (
            round(sum(distance_saved_percent) / count, 2) if count else None
        ),
        "mean_late_delivery_count": (
            round(sum(lateness_counts) / count, 2) if count else None
        ),
        "mean_weighted_lateness": (
            round(sum(weighted_lateness) / count, 2) if count else None
        ),
        "mean_estimated_vehicle_minutes": (
            round(sum(vehicle_minutes) / count, 2) if count else None
        ),
        "mean_baseline_estimated_vehicle_minutes": (
            round(sum(baseline_vehicle_minutes) / count, 2) if count else None
        ),
        "mean_estimated_vehicle_minutes_saved": (
            round(
                sum(
                    baseline - optimized
                    for baseline, optimized in zip(
                        baseline_vehicle_minutes,
                        vehicle_minutes,
                        strict=True,
                    )
                )
                / count,
                2,
            )
            if count
            else None
        ),
        "mean_vehicles_used": (
            round(sum(vehicles_used) / count, 2) if count else None
        ),
        "mean_fleet_utilization_percent": (
            round(sum(fleet_utilization) / count, 2) if count else None
        ),
        "recent_runs": [
            {
                "recorded_at": run["recorded_at"],
                "seed": run["algorithm_config"]["seed"],
                "population_size": run["algorithm_config"]["population_size"],
                "generations": run["algorithm_config"]["generations"],
                "optimized_distance_km": run["optimized"]["distance_km"],
                "baseline_distance_km": run["baseline"]["distance_km"],
                "optimized_estimated_vehicle_minutes": run["optimized"][
                    "estimated_vehicle_minutes"
                ],
                "baseline_estimated_vehicle_minutes": run["baseline"][
                    "estimated_vehicle_minutes"
                ],
                "optimized_vehicles_used": run["optimized"]["vehicles_used"],
                "optimized_fleet_utilization_percent": run["optimized"][
                    "fleet_utilization_percent"
                ],
                "distance_saved_percent": run["distance_saved_percent"],
                "late_delivery_count": run["optimized"]["late_delivery_count"],
                "weighted_lateness": run["optimized"]["weighted_lateness"],
            }
            for run in recent_runs[-30:]
        ],
        "notice": (
            "Histórico de simulações locais; não representa entregas realizadas."
        ),
    }


def load_route_plan(source: ScenarioSource = "demo") -> dict[str, Any]:
    scenario = get_scenario(source)
    result_path = _source_path(RESULTS_PATH, source)
    if not result_path.is_file():
        raise FileNotFoundError(
            "Ainda não existem rotas otimizadas. Execute a otimização primeiro."
        )
    try:
        with result_path.open(encoding="utf-8") as result_file:
            result = json.load(result_file)
    except json.JSONDecodeError as error:
        raise ValueError("O arquivo de rotas salvo não contém JSON válido.") from error
    if not isinstance(result, dict) or "comparison" not in result:
        raise ValueError("O arquivo de rotas salvo possui um formato inválido.")
    if result.get("scenario_id") != scenario.id:
        raise FileNotFoundError(
            "O plano salvo pertence a uma versão anterior do cenário. "
            "Execute novamente a otimização."
        )
    _ensure_plan_metrics(result)
    return result


def run_route_optimization(
    config: GAConfig = GAConfig(), source: ScenarioSource = "demo"
) -> dict[str, Any]:
    result = optimize_routes(get_scenario(source), config)
    save_route_plan(result, source)
    append_route_history(result, source)
    return result
