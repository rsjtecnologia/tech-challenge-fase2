"""Genetic algorithm and constrained multi-vehicle route decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
import random
import time
from typing import Any

from src.logistics.scenario import (
    PRIORITY_WEIGHTS,
    Delivery,
    Scenario,
    Vehicle,
)


LOGGER = logging.getLogger(__name__)
EARTH_RADIUS_KM = 6371.0088
UNSERVED_PENALTY = 1_000_000.0
AUTONOMY_PENALTY = 10_000.0
CAPACITY_PENALTY = 10_000.0
SEED = 42


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 64
    generations: int = 160
    mutation_rate: float = 0.18
    tournament_size: int = 4
    seed: int = SEED

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("A população deve ter pelo menos 4 indivíduos.")
        if self.generations < 1:
            raise ValueError("A quantidade de gerações deve ser positiva.")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("A taxa de mutação deve estar entre 0 e 1.")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("O torneio deve caber dentro da população.")


@dataclass(frozen=True)
class RouteEvaluation:
    routes: list[dict[str, Any]]
    total_distance_km: float
    total_weighted_lateness: float
    unserved_delivery_ids: list[str]
    fitness: float


def haversine_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Great-circle distance in km; demo estimates, not road distances."""
    lat_a, lat_b = math.radians(latitude_a), math.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, value)))


def _distance_between(first: dict[str, float], second: dict[str, float]) -> float:
    return haversine_km(
        first["latitude"],
        first["longitude"],
        second["latitude"],
        second["longitude"],
    )


def _location(point: Any) -> dict[str, float]:
    return {"latitude": point.latitude, "longitude": point.longitude}


def _simulate_route(
    vehicle: Vehicle,
    route: list[Delivery],
    depot: Any,
) -> dict[str, Any]:
    current = _location(depot)
    current_minute = vehicle.start_minute
    distance = 0.0
    load_kg = 0.0
    weighted_lateness = 0.0
    late_minutes_total = 0.0
    stops: list[dict[str, Any]] = []

    for delivery in route:
        destination = _location(delivery)
        leg_distance = _distance_between(current, destination)
        distance += leg_distance
        current_minute += leg_distance / vehicle.speed_kmh * 60
        lateness = max(0.0, current_minute - delivery.deadline_minute)
        weighted_lateness += lateness * PRIORITY_WEIGHTS[delivery.priority]
        late_minutes_total += lateness
        stops.append(
            {
                "delivery_id": delivery.id,
                "name": delivery.name,
                "latitude": delivery.latitude,
                "longitude": delivery.longitude,
                "priority": delivery.priority,
                "demand_kg": delivery.demand_kg,
                "deadline_minute": delivery.deadline_minute,
                "arrival_minute": round(current_minute, 2),
                "arrival_time": _format_minute(current_minute),
                "lateness_minutes": round(lateness, 2),
                "service_minutes": delivery.service_minutes,
                "items": list(delivery.items),
            }
        )
        current_minute += delivery.service_minutes
        load_kg += delivery.demand_kg
        current = destination

    if route:
        distance += _distance_between(current, _location(depot))

    return {
        "vehicle_id": vehicle.id,
        "vehicle_name": vehicle.name,
        "speed_kmh": vehicle.speed_kmh,
        "capacity_kg": vehicle.capacity_kg,
        "max_distance_km": vehicle.max_distance_km,
        "load_kg": round(load_kg, 2),
        "remaining_capacity_kg": round(vehicle.capacity_kg - load_kg, 2),
        "distance_km": round(distance, 3),
        "remaining_range_km": round(vehicle.max_distance_km - distance, 3),
        "late_minutes": round(late_minutes_total, 2),
        "weighted_lateness": round(weighted_lateness, 2),
        "stops": stops,
        "path": [
            {
                "id": depot.id,
                "name": depot.name,
                "latitude": depot.latitude,
                "longitude": depot.longitude,
                "kind": "depot",
            },
            *[
                {
                    "id": delivery.id,
                    "name": delivery.name,
                    "latitude": delivery.latitude,
                    "longitude": delivery.longitude,
                    "kind": "delivery",
                }
                for delivery in route
            ],
            {
                "id": depot.id,
                "name": depot.name,
                "latitude": depot.latitude,
                "longitude": depot.longitude,
                "kind": "depot",
            },
        ]
        if route
        else [],
    }


def _route_cost(route: dict[str, Any]) -> float:
    return (
        route["distance_km"]
        + route["weighted_lateness"]
        + max(0.0, -route["remaining_range_km"]) * AUTONOMY_PENALTY
        + max(0.0, -route["remaining_capacity_kg"]) * CAPACITY_PENALTY
    )


def decode_chromosome(
    chromosome: tuple[str, ...],
    scenario: Scenario,
) -> RouteEvaluation:
    """Assign a giant-tour permutation to feasible vehicle routes by insertion."""
    deliveries = {delivery.id: delivery for delivery in scenario.deliveries}
    expected = set(deliveries)
    if len(chromosome) != len(expected) or set(chromosome) != expected:
        raise ValueError("O cromossomo precisa conter cada entrega exatamente uma vez.")

    routes: dict[str, list[Delivery]] = {vehicle.id: [] for vehicle in scenario.vehicles}
    vehicles = {vehicle.id: vehicle for vehicle in scenario.vehicles}
    unserved: list[str] = []

    for delivery_id in chromosome:
        delivery = deliveries[delivery_id]
        candidates: list[tuple[float, str, int, dict[str, Any]]] = []
        for vehicle in scenario.vehicles:
            assigned = routes[vehicle.id]
            if delivery.requires_refrigeration and not vehicle.refrigerated:
                continue
            current_load = sum(stop.demand_kg for stop in assigned)
            if current_load + delivery.demand_kg > vehicle.capacity_kg:
                continue
            current_eval = _simulate_route(vehicle, assigned, scenario.depot)
            current_cost = _route_cost(current_eval)
            for insertion_index in range(len(assigned) + 1):
                candidate_route = (
                    assigned[:insertion_index]
                    + [delivery]
                    + assigned[insertion_index:]
                )
                candidate_eval = _simulate_route(
                    vehicle,
                    candidate_route,
                    scenario.depot,
                )
                if candidate_eval["distance_km"] > vehicle.max_distance_km:
                    continue
                cost_delta = _route_cost(candidate_eval) - current_cost
                candidates.append(
                    (cost_delta, vehicle.id, insertion_index, candidate_eval)
                )
        if not candidates:
            unserved.append(delivery_id)
            continue
        _, vehicle_id, insertion_index, _ = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        routes[vehicle_id].insert(insertion_index, delivery)

    evaluated_routes = [
        _simulate_route(vehicle, routes[vehicle.id], scenario.depot)
        for vehicle in scenario.vehicles
        if routes[vehicle.id]
    ]
    total_distance = sum(route["distance_km"] for route in evaluated_routes)
    weighted_lateness = sum(
        route["weighted_lateness"] for route in evaluated_routes
    )
    fitness = (
        total_distance
        + weighted_lateness
        + len(unserved) * UNSERVED_PENALTY
    )
    return RouteEvaluation(
        routes=evaluated_routes,
        total_distance_km=round(total_distance, 3),
        total_weighted_lateness=round(weighted_lateness, 2),
        unserved_delivery_ids=unserved,
        fitness=round(fitness, 3),
    )


def _format_minute(value: float) -> str:
    minute = int(round(value))
    return f"{(minute // 60) % 24:02d}:{minute % 60:02d}"


def _order_crossover(
    first: tuple[str, ...],
    second: tuple[str, ...],
    rng: random.Random,
) -> tuple[str, ...]:
    """Order crossover (OX): preserve a segment and the other parent's order."""
    left, right = sorted(rng.sample(range(len(first)), 2))
    child: list[str | None] = [None] * len(first)
    child[left : right + 1] = first[left : right + 1]
    remaining = [gene for gene in second if gene not in child]
    fill_positions = list(range(right + 1, len(first))) + list(range(left))
    for position, gene in zip(fill_positions, remaining, strict=True):
        child[position] = gene
    return tuple(gene for gene in child if gene is not None)


def _mutate(chromosome: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
    """Reverse a random subroute to explore neighborhood orderings."""
    left, right = sorted(rng.sample(range(len(chromosome)), 2))
    return (
        chromosome[:left]
        + tuple(reversed(chromosome[left : right + 1]))
        + chromosome[right + 1 :]
    )


def _baseline_order(scenario: Scenario) -> tuple[str, ...]:
    """Priority-first assignment baseline with no population search."""
    ordered = sorted(
        scenario.deliveries,
        key=lambda delivery: (
            not delivery.requires_refrigeration,
            -PRIORITY_WEIGHTS[delivery.priority],
            delivery.deadline_minute,
            delivery.id,
        ),
    )
    return tuple(delivery.id for delivery in ordered)


def optimize_routes(
    scenario: Scenario,
    config: GAConfig = GAConfig(),
) -> dict[str, Any]:
    if len(scenario.deliveries) < 2:
        raise ValueError("O algoritmo precisa de pelo menos duas entregas.")
    rng = random.Random(config.seed)
    baseline_chromosome = _baseline_order(scenario)
    baseline = decode_chromosome(baseline_chromosome, scenario)
    if baseline.unserved_delivery_ids:
        raise ValueError(
            "O cenário não é viável pelo baseline de referência: "
            f"{baseline.unserved_delivery_ids}"
        )
    cache: dict[tuple[str, ...], RouteEvaluation] = {}

    def evaluate(chromosome: tuple[str, ...]) -> RouteEvaluation:
        if chromosome not in cache:
            cache[chromosome] = decode_chromosome(chromosome, scenario)
        return cache[chromosome]

    population = [baseline_chromosome]
    while len(population) < config.population_size:
        candidate = list(baseline_chromosome)
        rng.shuffle(candidate)
        population.append(tuple(candidate))

    history: list[dict[str, float]] = []
    started = time.perf_counter()
    LOGGER.info(
        "route_optimization_start scenario=%s vehicles=%d deliveries=%d "
        "population=%d generations=%d mutation=%.3f",
        scenario.id,
        len(scenario.vehicles),
        len(scenario.deliveries),
        config.population_size,
        config.generations,
        config.mutation_rate,
    )

    def select() -> tuple[str, ...]:
        candidates = rng.sample(
            population,
            k=min(config.tournament_size, len(population)),
        )
        return min(candidates, key=lambda item: evaluate(item).fitness)

    for generation in range(config.generations):
        ranked = sorted(population, key=lambda item: evaluate(item).fitness)
        history.append(
            {
                "generation": generation,
                "best_fitness": evaluate(ranked[0]).fitness,
                "mean_fitness": round(
                    sum(evaluate(item).fitness for item in ranked) / len(ranked),
                    3,
                ),
            }
        )
        next_population = [ranked[0]]
        while len(next_population) < config.population_size:
            child = _order_crossover(select(), select(), rng)
            if rng.random() < config.mutation_rate:
                child = _mutate(child, rng)
            next_population.append(child)
        population = next_population

    best_chromosome = min(population, key=lambda item: evaluate(item).fitness)
    optimized = evaluate(best_chromosome)
    if optimized.unserved_delivery_ids:
        raise ValueError(
            "O algoritmo genético não encontrou uma solução viável para todas "
            f"as entregas: {optimized.unserved_delivery_ids}"
        )
    elapsed = round(time.perf_counter() - started, 3)
    LOGGER.info(
        "route_optimization_complete scenario=%s baseline_km=%.3f "
        "optimized_km=%.3f baseline_cost=%.3f optimized_cost=%.3f "
        "unserved=%d evaluated=%d duration_seconds=%.3f",
        scenario.id,
        baseline.total_distance_km,
        optimized.total_distance_km,
        baseline.fitness,
        optimized.fitness,
        len(optimized.unserved_delivery_ids),
        len(cache),
        elapsed,
    )

    return {
        "schema_version": 1,
        "scenario_id": scenario.id,
        "scenario_name": scenario.name,
        "scenario_description": scenario.description,
        "synthetic": scenario.synthetic,
        "data_source": scenario.data_source,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm": {
            "name": "Genetic Algorithm — Order Crossover + Reverse Mutation",
            "chromosome_encoding": "permutation of unique delivery IDs",
            "selection": "tournament",
            "crossover": "order crossover (OX)",
            "mutation": "random subroute inversion",
            "elitism": True,
            "seed": config.seed,
            "config": asdict(config),
            "evaluated_chromosomes": len(cache),
            "history": history,
        },
        "objective": {
            "formula": (
                "distance_km + priority_weight * late_minutes + "
                "1_000_000 per unserved delivery"
            ),
            "priority_weights": PRIORITY_WEIGHTS,
            "distance_model": "Haversine straight-line approximation",
        },
        "depot": asdict(scenario.depot),
        "vehicles": [asdict(vehicle) for vehicle in scenario.vehicles],
        "deliveries": [
            {
                **asdict(delivery),
                "items": list(delivery.items),
                "deadline_time": _format_minute(delivery.deadline_minute),
            }
            for delivery in scenario.deliveries
        ],
        "comparison": {
            "baseline": _serialize_evaluation(baseline, scenario),
            "optimized": _serialize_evaluation(optimized, scenario),
            "distance_saved_km": round(
                baseline.total_distance_km - optimized.total_distance_km,
                3,
            ),
            "distance_saved_percent": round(
                (
                    100
                    * (baseline.total_distance_km - optimized.total_distance_km)
                    / baseline.total_distance_km
                )
                if baseline.total_distance_km
                else 0.0,
                2,
            ),
        },
        "elapsed_seconds": elapsed,
    }


def _serialize_evaluation(
    evaluation: RouteEvaluation,
    scenario: Scenario,
) -> dict[str, Any]:
    delivery_count = len(scenario.deliveries)
    served = delivery_count - len(evaluation.unserved_delivery_ids)
    total_priority = sum(
        PRIORITY_WEIGHTS[delivery.priority] for delivery in scenario.deliveries
    )
    delivered_priority = sum(
        PRIORITY_WEIGHTS[delivery.priority]
        for route in evaluation.routes
        for stop in route["stops"]
        for delivery in scenario.deliveries
        if delivery.id == stop["delivery_id"]
    )
    estimated_vehicle_minutes = sum(
        route["distance_km"] / route["speed_kmh"] * 60
        + sum(stop["service_minutes"] for stop in route["stops"])
        for route in evaluation.routes
    )
    total_load_kg = sum(route["load_kg"] for route in evaluation.routes)
    used_capacity_kg = sum(route["capacity_kg"] for route in evaluation.routes)
    return {
        "routes": evaluation.routes,
        "total_distance_km": evaluation.total_distance_km,
        "estimated_vehicle_minutes": round(estimated_vehicle_minutes, 2),
        "vehicles_used": len(evaluation.routes),
        "total_load_kg": round(total_load_kg, 2),
        "fleet_utilization_percent": round(
            100 * total_load_kg / used_capacity_kg if used_capacity_kg else 0,
            2,
        ),
        "total_weighted_lateness": evaluation.total_weighted_lateness,
        "late_delivery_count": sum(
            stop["lateness_minutes"] > 0
            for route in evaluation.routes
            for stop in route["stops"]
        ),
        "unserved_delivery_ids": evaluation.unserved_delivery_ids,
        "served_delivery_count": served,
        "delivery_count": delivery_count,
        "served_priority_percent": round(
            100 * delivered_priority / total_priority if total_priority else 100,
            2,
        ),
        "fitness_cost": evaluation.fitness,
    }
