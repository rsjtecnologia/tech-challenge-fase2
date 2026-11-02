from typing import Literal
import unicodedata

from fastapi import APIRouter, HTTPException, Query
import numpy as np
import pandas as pd

from src.web.services.data_loader import (
    DataUnavailableError,
    clear_cache,
    load_charts,
    load_metrics,
    load_predictions,
    load_samples,
    records,
)


router = APIRouter(prefix="/api")


def _model_sort_key(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    return "".join(character for character in normalized if not unicodedata.combining(character)).casefold()


def _data_error(error: DataUnavailableError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(error))


@router.get("/health")
def health() -> dict:
    try:
        metrics = load_metrics()
        return {
            "status": "ok",
            "contrato": metrics.get("versao_contrato"),
            "amostras": metrics.get("quantidade_amostras"),
        }
    except DataUnavailableError as error:
        raise _data_error(error) from error


@router.get("/metricas")
def metrics() -> dict:
    try:
        return load_metrics()
    except DataUnavailableError as error:
        raise _data_error(error) from error


@router.get("/modelos")
def models() -> dict:
    try:
        metrics_data = load_metrics()
        return {
            "melhor_modelo": metrics_data["melhor_modelo"],
            "modelos": sorted(metrics_data["por_modelo"], key=_model_sort_key),
            "atributos": metrics_data["atributos"],
        }
    except DataUnavailableError as error:
        raise _data_error(error) from error


@router.get("/graficos")
def charts() -> dict:
    try:
        return load_charts()
    except DataUnavailableError as error:
        raise _data_error(error) from error


@router.get("/amostras")
def samples(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=15, ge=0, le=5000),
    diagnostico: Literal["Maligno", "Benigno"] | None = None,
    modelo: str | None = None,
    acerto: bool | None = None,
    consenso: bool | None = None,
    prob_min: float | None = Query(default=None, ge=0, le=1),
    prob_max: float | None = Query(default=None, ge=0, le=1),
    sort: str = "id",
    order: Literal["asc", "desc"] = "asc",
) -> dict:
    try:
        frame = load_samples().copy()
        metrics_data = load_metrics()
    except DataUnavailableError as error:
        raise _data_error(error) from error

    model_names = sorted(metrics_data["por_modelo"], key=_model_sort_key)
    if modelo and modelo not in model_names:
        raise HTTPException(status_code=400, detail=f"Modelo desconhecido: {modelo}")
    if prob_min is not None and prob_max is not None and prob_min > prob_max:
        raise HTTPException(status_code=400, detail="prob_min não pode superar prob_max")

    if diagnostico:
        frame = frame[frame["diagnostico_real"] == diagnostico]
    if consenso is not None:
        frame = frame[frame["consenso"] == consenso]
    if modelo:
        if acerto is not None:
            frame = frame[frame[f"acerto_{modelo}"] == acerto]
        if prob_min is not None:
            frame = frame[frame[f"prob_{modelo}"] >= prob_min]
        if prob_max is not None:
            frame = frame[frame[f"prob_{modelo}"] <= prob_max]
    elif acerto is not None or prob_min is not None or prob_max is not None:
        raise HTTPException(
            status_code=400,
            detail="Selecione um modelo para filtrar por acerto ou probabilidade.",
        )

    for name in model_names:
        frame[f"confianca_{name}"] = np.where(
            frame[f"pred_{name}"] == 1,
            frame[f"prob_{name}"],
            1 - frame[f"prob_{name}"],
        )

    allowed_sort = {
        "id",
        "diagnostico_real",
        "n_modelos_acertaram",
        "amplitude_probabilidade",
        *(f"prob_{name}" for name in model_names),
        *(f"confianca_{name}" for name in model_names),
    }
    if sort not in allowed_sort:
        raise HTTPException(status_code=400, detail=f"Ordenação não permitida: {sort}")

    frame = frame.sort_values(sort, ascending=order == "asc")
    total = len(frame)
    page_size = total if size == 0 else size
    current_page = 1 if size == 0 else page
    start = (current_page - 1) * page_size
    page_frame = frame.iloc[start : start + page_size]

    feature_names = metrics_data["atributos"]
    compact_columns = [
        "id",
        "rotulo_real",
        "diagnostico_real",
        *(column for name in model_names for column in (
            f"pred_{name}",
            f"prob_{name}",
            f"confianca_{name}",
            f"acerto_{name}",
        )),
        "n_modelos_acertaram",
        "consenso",
        "amplitude_probabilidade",
    ]

    return {
        "items": records(page_frame[compact_columns]),
        "pagination": {
            "page": current_page,
            "size": page_size,
            "total": total,
            "pages": max(1, (total + page_size - 1) // page_size) if page_size else 1,
            "all": size == 0,
        },
        "modelos": model_names,
        "atributos": feature_names,
    }


@router.get("/amostras/{sample_id}")
def sample_detail(sample_id: int) -> dict:
    try:
        frame = load_samples()
        metrics_data = load_metrics()
    except DataUnavailableError as error:
        raise _data_error(error) from error

    selected = frame[frame["id"] == sample_id]
    if selected.empty:
        raise HTTPException(status_code=404, detail="Amostra não encontrada")

    row = records(selected)[0]
    features = {
        name: row[name]
        for name in metrics_data["atributos"]
    }
    predictions = [
        {
            "modelo": name,
            "rotulo_previsto": row[f"pred_{name}"],
            "prob_maligno": row[f"prob_{name}"],
            "confianca": (
                row[f"prob_{name}"]
                if row[f"pred_{name}"] == 1
                else 1 - row[f"prob_{name}"]
            ),
            "acerto": row[f"acerto_{name}"],
        }
        for name in sorted(metrics_data["por_modelo"], key=_model_sort_key)
    ]
    return {
        "id": row["id"],
        "rotulo_real": row["rotulo_real"],
        "diagnostico_real": row["diagnostico_real"],
        "n_modelos_acertaram": row["n_modelos_acertaram"],
        "consenso": row["consenso"],
        "amplitude_probabilidade": row["amplitude_probabilidade"],
        "predicoes": predictions,
        "atributos": features,
    }


@router.get("/predicoes/resumo")
def prediction_summary(modelo: str | None = None) -> dict:
    try:
        frame = load_predictions().copy()
        metrics_data = load_metrics()
    except DataUnavailableError as error:
        raise _data_error(error) from error

    model_names = sorted(metrics_data["por_modelo"], key=_model_sort_key)
    if modelo:
        if modelo not in model_names:
            raise HTTPException(status_code=400, detail=f"Modelo desconhecido: {modelo}")
        frame = frame[frame["modelo"] == modelo]

    bins = np.linspace(0, 1, 11)
    frame["faixa_probabilidade"] = pd.cut(
        frame["prob_maligno"],
        bins=bins,
        include_lowest=True,
        labels=[f"{bins[i]:.1f}–{bins[i + 1]:.1f}" for i in range(len(bins) - 1)],
    )

    distributions = (
        frame.groupby(["modelo", "faixa_probabilidade"], observed=False)
        .size()
        .rename("quantidade")
        .reset_index()
    )
    accuracy = (
        frame.groupby("modelo", observed=True)["acerto"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "acertos", "count": "total"})
    )
    accuracy["erros"] = accuracy["total"] - accuracy["acertos"]

    return {
        "distribuicao_probabilidade": records(distributions),
        "acertos_erros": records(accuracy),
    }


@router.post("/cache/recarregar")
def reload_cache() -> dict:
    clear_cache()
    return {"status": "cache recarregado"}
