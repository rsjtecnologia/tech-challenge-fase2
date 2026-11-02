"""API endpoints for aggregate LLM interpretation."""

from pathlib import Path
import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.optimization.cache import InterpretationCache, interpretation_key
from src.optimization.experiments import REPOSITORY_ROOT
from src.optimization.llm import (
    LLMConfigurationError,
    LLMProviderError,
    build_interpretation_context,
    generate_interpretation,
)


router = APIRouter(prefix="/api", tags=["Fase 2"])
INTERPRETATION_CACHE = InterpretationCache()
RESULTS_PATH = (
    REPOSITORY_ROOT
    / "src"
    / "db"
    / "outputs"
    / "phase2-genetic"
    / "optimization_results.json"
)
STUDY_NAMES = {
    "breast_cancer": "Câncer de mama",
    "pcos": "Síndrome do ovário policístico",
}


class InterpretationRequest(BaseModel):
    estudo: Literal["breast_cancer", "pcos"]
    pergunta: str | None = Field(default=None, min_length=1, max_length=500)

    @field_validator("pergunta")
    @classmethod
    def validate_question(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("A pergunta não pode conter somente espaços.")
        return value


def _load_study(study_name: str) -> dict:
    if not RESULTS_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "Resultados da Fase 2 indisponíveis. Execute "
                "`python -m src.optimization.cli` primeiro."
            ),
        )
    try:
        with RESULTS_PATH.open(encoding="utf-8") as results_file:
            results = json.load(results_file)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=503,
            detail="O arquivo de resultados da Fase 2 está inválido.",
        ) from error
    if not isinstance(results, dict) or not isinstance(
        results.get("studies"), dict
    ):
        raise HTTPException(
            status_code=503,
            detail="O arquivo de resultados da Fase 2 não corresponde ao esquema.",
        )
    study = results["studies"].get(study_name)
    if study is None:
        raise HTTPException(
            status_code=404,
            detail=f"Resultados não encontrados para {STUDY_NAMES[study_name]}.",
        )
    if not isinstance(study, dict):
        raise HTTPException(
            status_code=503,
            detail=f"Resultados inválidos para {STUDY_NAMES[study_name]}.",
        )
    return {**study, "fitness": results.get("fitness", {})}


@router.get("/otimizacao/{study_name}")
def optimization_results(
    study_name: Literal["breast_cancer", "pcos"],
) -> dict:
    """Return aggregate baseline/tuned metrics and all three GA experiments."""
    study = _load_study(study_name)
    models = {}
    for name, result in study["models"].items():
        models[name] = {
            "selected_experiment": result["optimized"]["selected_experiment"],
            "hyperparameters": result["optimized"]["hyperparameters"],
            "baseline_metrics": result["baseline"]["test_metrics"],
            "optimized_metrics": result["optimized"]["test_metrics"],
            "experiments": [
                {
                    "name": experiment["name"],
                    "best_fitness": experiment["best_fitness"],
                    "best_cv_metrics": experiment["best_cv_metrics"],
                    "unique_candidates_evaluated": experiment[
                        "unique_candidates_evaluated"
                    ],
                    "config": experiment["config"],
                }
                for experiment in result["experiments"]
            ],
        }
    return {
        "study": study["dataset"],
        "label": study["label"],
        "positive_class": study["positive_class"],
        "fitness": study["fitness"],
        "models": models,
    }


@router.post("/interpretacao")
def interpret_results(request: InterpretationRequest) -> dict:
    """Generate a plain-language explanation from aggregate experiment metrics."""
    study = _load_study(request.estudo)
    context = build_interpretation_context(study)
    try:
        return INTERPRETATION_CACHE.get_or_generate(
            interpretation_key(context, request.pergunta),
            lambda: generate_interpretation(context, question=request.pergunta),
        )
    except LLMConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except LLMProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
