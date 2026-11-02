"""Local API for hospital delivery route planning."""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.logistics.genetic import GAConfig
from src.logistics.llm import generate_route_content
from src.logistics.service import (
    load_route_plan,
    run_route_optimization,
    route_history_summary,
    scenario_payload,
    ScenarioSource,
)
from src.optimization.llm import LLMConfigurationError, LLMProviderError


router = APIRouter(prefix="/api/rotas", tags=["Otimização de rotas"])


class OptimizationRequest(BaseModel):
    source: ScenarioSource = "demo"
    population_size: int = Field(default=64, ge=4, le=300)
    generations: int = Field(default=160, ge=1, le=500)
    mutation_rate: float = Field(default=0.18, ge=0, le=1)
    tournament_size: int = Field(default=4, ge=2, le=300)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class RouteContentRequest(BaseModel):
    source: ScenarioSource = "demo"
    action: Literal["instructions", "daily_report", "weekly_report", "question"]
    question: str | None = Field(default=None, max_length=1000)


@router.get("/cenario")
def get_scenario(source: ScenarioSource = "demo") -> dict:
    try:
        return scenario_payload(source)
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/otimizar")
def optimize(request: OptimizationRequest) -> dict:
    if request.tournament_size > request.population_size:
        raise HTTPException(
            status_code=422,
            detail="tournament_size não pode exceder population_size.",
        )
    config = GAConfig(
        population_size=request.population_size,
        generations=request.generations,
        mutation_rate=request.mutation_rate,
        tournament_size=request.tournament_size,
        seed=request.seed,
    )
    try:
        return run_route_optimization(config, source=request.source)
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/resultado")
def get_latest_result(source: ScenarioSource = "demo") -> dict:
    try:
        return load_route_plan(source)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/historico")
def get_route_history(source: ScenarioSource = "demo") -> dict:
    try:
        plan = load_route_plan(source)
    except FileNotFoundError:
        plan = None
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    try:
        return route_history_summary(current_plan=plan, source=source)
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/assistente")
def route_assistant(request: RouteContentRequest) -> dict:
    try:
        plan = load_route_plan(request.source)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    try:
        history = route_history_summary(current_plan=plan, source=request.source)
        return generate_route_content(
            plan,
            request.action,
            request.question,
            history=history,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except LLMConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except LLMProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
