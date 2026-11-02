"""Command-line entry point for Phase 2 experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

from src.optimization.experiments import (
    DATASETS,
    MODEL_NAMES,
    configure_logging,
    run_optimization,
    save_results,
)
from src.optimization.genetic import EXPERIMENT_CONFIGS
from src.optimization.llm import (
    build_interpretation_context,
    generate_interpretation,
    validate_configuration,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "db"
    / "outputs"
    / "phase2-genetic"
    / "optimization_results.json"
)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa três experimentos de algoritmo genético nos modelos."
    )
    parser.add_argument(
        "--study",
        choices=(*DATASETS, "all"),
        default="all",
        help="Estudo a otimizar (padrão: all).",
    )
    parser.add_argument(
        "--model",
        choices=(*MODEL_NAMES, "all"),
        default="all",
        help="Modelo a otimizar (padrão: todos).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destino dos resultados JSON.",
    )
    parser.add_argument(
        "--interpret",
        action="store_true",
        help="Também solicita interpretações agregadas a um endpoint LLM.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    log_path = args.output.parent / "optimization.log"
    configure_logging(log_path)
    studies = DATASETS if args.study == "all" else (args.study,)
    models = MODEL_NAMES if args.model == "all" else (args.model,)

    llm_configured = validate_configuration() if args.interpret else False
    if args.interpret and not llm_configured:
        LOGGER.info(
            "llm_not_configured using deterministic local interpretation"
        )

    results = run_optimization(
        datasets=studies,
        model_names=models,
        configs=EXPERIMENT_CONFIGS,
    )
    save_results(results, args.output)
    LOGGER.info("results_saved path=%s", args.output.resolve())

    if args.interpret:
        interpretations: dict[str, Any] = {}
        for study_name, study in results["studies"].items():
            response = generate_interpretation(
                build_interpretation_context(study)
            )
            interpretations[study_name] = response
            LOGGER.info(
                "llm_interpretation_complete study=%s quality=%s/%s",
                study_name,
                response["quality_evaluation"]["score"],
                response["quality_evaluation"]["maximum"],
            )
        interpretation_path = args.output.with_name(
            "llm_interpretations.json"
        )
        save_results(interpretations, interpretation_path)
        LOGGER.info("llm_results_saved path=%s", interpretation_path.resolve())

    return 0


if __name__ == "__main__":
    sys.exit(main())
