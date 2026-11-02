from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SRC_DIR / "db" / "outputs" / "breast-cancer-wisconsin-data" / "08-export-web"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
METRICS_DIR = DATA_DIR / "metrics"

SAMPLES_PATH = ARTIFACTS_DIR / "amostras_comparativo.parquet"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predicoes_longas.parquet"
METRICS_PATH = METRICS_DIR / "metricas_comparativo.json"
CHARTS_PATH = METRICS_DIR / "graficos.json"


class DataUnavailableError(RuntimeError):
    """Raised when the export notebook has not produced the web contract."""


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise DataUnavailableError(
            f"Arquivo não encontrado: {path}. Execute notebooks/breast-cancer-wisconsin-data/08_export_web.ipynb."
        )


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    _ensure_exists(METRICS_PATH)
    with METRICS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def load_charts() -> dict[str, Any]:
    _ensure_exists(CHARTS_PATH)
    with CHARTS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def load_samples() -> pd.DataFrame:
    _ensure_exists(SAMPLES_PATH)
    return pd.read_parquet(SAMPLES_PATH)


@lru_cache(maxsize=1)
def load_predictions() -> pd.DataFrame:
    _ensure_exists(PREDICTIONS_PATH)
    return pd.read_parquet(PREDICTIONS_PATH)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame into JSON-safe records, including NumPy values."""
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def clear_cache() -> None:
    """Useful after regenerating the contract while the server is running."""
    load_metrics.cache_clear()
    load_charts.cache_clear()
    load_samples.cache_clear()
    load_predictions.cache_clear()
