"""Genetic search for discrete scikit-learn hyperparameter spaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any, Callable


@dataclass(frozen=True)
class GAConfig:
    name: str
    population_size: int
    generations: int
    mutation_rate: float
    cv_folds: int = 3
    tournament_size: int = 3

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size deve ser pelo menos 2.")
        if self.generations < 1:
            raise ValueError("generations deve ser pelo menos 1.")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate deve estar entre 0 e 1.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds deve ser pelo menos 2.")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size deve estar entre 1 e population_size.")


EXPERIMENT_CONFIGS = (
    GAConfig(
        name="exploracao_conservadora",
        population_size=6,
        generations=3,
        mutation_rate=0.15,
    ),
    GAConfig(
        name="exploracao_ampla",
        population_size=10,
        generations=3,
        mutation_rate=0.30,
    ),
    GAConfig(
        name="refinamento",
        population_size=8,
        generations=5,
        mutation_rate=0.10,
    ),
)


@dataclass(frozen=True)
class Individual:
    genes: dict[str, Any]
    fitness: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class GeneticSearchResult:
    config: dict[str, Any]
    best_params: dict[str, Any]
    best_fitness: float
    best_metrics: dict[str, float]
    history: list[dict[str, float]]
    evaluations: int


MetricEvaluator = Callable[[dict[str, Any]], dict[str, float]]


class GeneticOptimizer:
    """Optimize a bounded discrete parameter space using a genetic algorithm."""

    def __init__(
        self,
        search_space: dict[str, tuple[Any, ...]],
        config: GAConfig,
        evaluator: MetricEvaluator,
        seed: int = 42,
    ) -> None:
        if not search_space:
            raise ValueError("search_space não pode ser vazio.")
        if any(not values for values in search_space.values()):
            raise ValueError("Cada gene precisa ter ao menos um valor possível.")
        self.search_space = search_space
        self.config = config
        self.evaluator = evaluator
        self.rng = random.Random(seed)
        self._fitness_cache: dict[tuple[tuple[str, str], ...], Individual] = {}

    def _key(self, genes: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        return tuple((name, repr(genes[name])) for name in sorted(genes))

    def _evaluate(self, genes: dict[str, Any]) -> Individual:
        key = self._key(genes)
        if key not in self._fitness_cache:
            metrics = self.evaluator(dict(genes))
            required = {"accuracy", "recall", "f1"}
            missing = required.difference(metrics)
            if missing:
                raise ValueError(f"O avaliador não retornou as métricas: {sorted(missing)}")
            fitness = (
                0.50 * metrics["recall"]
                + 0.30 * metrics["f1"]
                + 0.20 * metrics["accuracy"]
            )
            self._fitness_cache[key] = Individual(dict(genes), fitness, metrics)
        return self._fitness_cache[key]

    def _random_genes(self) -> dict[str, Any]:
        return {
            name: self.rng.choice(values)
            for name, values in self.search_space.items()
        }

    def _select(self, population: list[Individual]) -> Individual:
        candidates = self.rng.sample(
            population,
            k=min(self.config.tournament_size, len(population)),
        )
        return max(candidates, key=lambda individual: individual.fitness)

    def _offspring(self, first: Individual, second: Individual) -> dict[str, Any]:
        child = {
            name: self.rng.choice((first.genes[name], second.genes[name]))
            for name in self.search_space
        }
        for name, choices in self.search_space.items():
            if self.rng.random() < self.config.mutation_rate:
                alternatives = [value for value in choices if value != child[name]]
                if alternatives:
                    child[name] = self.rng.choice(alternatives)
        return child

    def run(self) -> GeneticSearchResult:
        population = [
            self._evaluate(self._random_genes())
            for _ in range(self.config.population_size)
        ]
        history: list[dict[str, float]] = []

        for generation in range(self.config.generations):
            population.sort(key=lambda individual: individual.fitness, reverse=True)
            history.append(
                {
                    "generation": float(generation),
                    "best_fitness": population[0].fitness,
                    "mean_fitness": sum(item.fitness for item in population)
                    / len(population),
                }
            )
            next_population = [population[0]]
            while len(next_population) < self.config.population_size:
                parent_a = self._select(population)
                parent_b = self._select(population)
                next_population.append(
                    self._evaluate(self._offspring(parent_a, parent_b))
                )
            population = next_population

        best = max(population, key=lambda individual: individual.fitness)
        return GeneticSearchResult(
            config=asdict(self.config),
            best_params=best.genes,
            best_fitness=best.fitness,
            best_metrics=best.metrics,
            history=history,
            evaluations=len(self._fitness_cache),
        )
