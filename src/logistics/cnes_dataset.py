"""Build a simulated medical-delivery scenario from a local CNES CSV export."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import logging
import math
from pathlib import Path
import re
import unicodedata

import pandas as pd

from src.logistics.genetic import haversine_km
from src.logistics.scenario import Delivery, Depot, Scenario, Vehicle


LOGGER = logging.getLogger(__name__)
CNES_DATASET_DIR = (
    Path(__file__).resolve().parents[1] / "db" / "models" / "dataset_cnes"
)
CNES_FILENAME = "cnes_estabelecimentos_sp.csv"
KAGGLE_INVENTORY_PATH = (
    CNES_DATASET_DIR / "hospital_supply_chain" / "inventory_data.csv"
)
MAX_DESTINATIONS = 18
SP_BOUNDS = (-25.4, -19.7, -53.2, -44.0)
DEPOT = Depot(
    "central-sp-simulado",
    "Depósito simulado — São Paulo",
    -23.5505,
    -46.6333,
)
VEHICLES = tuple(
    Vehicle(
        f"cnes-veiculo-{index + 1}",
        f"Veículo estadual {index + 1}" + (" refrigerado" if index < 2 else ""),
        50.0,
        1800.0,
        65.0,
        index < 2,
    )
    for index in range(5)
)

_COLUMN_ALIASES = {
    "id": ("cnes_id", "co_cnes", "co_unidade", "nu_cnes", "codigo_cnes"),
    "name": ("name", "nome", "no_fantasia", "no_razao_social"),
    "type": (
        "facility_type", "tipo_estabelecimento", "tipo_unidade",
        "ds_tipo_unidade", "tp_unidade", "co_tipo_unidade",
    ),
    "municipality": (
        "municipality", "municipio", "no_municipio", "municipio_gestor",
    ),
    "uf": ("uf", "sg_uf", "co_uf"),
    "latitude": ("latitude", "nu_latitude"),
    "longitude": ("longitude", "nu_longitude"),
    "priority_score": ("priority_score", "prioridade_insumo", "prioridade"),
    "medicine_kg": ("medicine_kg", "demanda_medicamento_kg"),
    "supply_kg": ("supply_kg", "demanda_insumo_kg"),
}


def _normalize(value: object) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or "").casefold())
    return "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )


def _column_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _normalize(value)).strip("_")


def _read_cnes_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="latin-1")
    except (pd.errors.ParserError, OSError) as error:
        raise ValueError(f"Não foi possível ler o CSV CNES {path.name}: {error}") from error


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    actual = {_column_key(column): column for column in frame.columns}
    selected: dict[str, str] = {}
    missing = []
    for field, aliases in _COLUMN_ALIASES.items():
        match = next((actual[alias] for alias in aliases if alias in actual), None)
        if match is None:
            if field in {"id", "name", "type", "uf", "latitude", "longitude"}:
                missing.append(field)
        else:
            selected[field] = match
    if missing:
        raise ValueError(
            "CSV CNES sem as colunas obrigatórias (ou equivalentes): "
            + ", ".join(missing)
        )
    demand_fields = {"medicine_kg", "supply_kg"} & selected.keys()
    if demand_fields and demand_fields != {"medicine_kg", "supply_kg"}:
        raise ValueError(
            "Informe medicine_kg e supply_kg juntos, ou omita ambas as colunas."
        )
    canonical = pd.DataFrame(index=frame.index)
    for field, column in selected.items():
        canonical[field] = frame[column]
    if "municipality" not in canonical:
        canonical["municipality"] = ""
    for field in ("priority_score", "medicine_kg", "supply_kg"):
        if field not in canonical:
            canonical[field] = pd.NA
    return canonical


def _is_sp(value: object) -> bool:
    normalized = _normalize(value).strip()
    return normalized in {"sp", "35", "035"}


def _priority_default(facility_type: str) -> int:
    kind = _normalize(facility_type)
    if any(term in kind for term in ("upa", "hospital", "urgencia", "emergencia", "pronto atendimento")):
        return 9
    if any(term in kind for term in ("ubs", "farmacia", "posto", "centro de saude")):
        return 6
    return 4


def _priority_name(score: int) -> str:
    if score >= 9:
        return "critica"
    if score >= 7:
        return "alta"
    return "regular"


def _select_destinations(frame: pd.DataFrame) -> pd.DataFrame:
    remaining = frame.sort_values(
        ["priority_score", "id", "name"], ascending=[False, True, True]
    ).copy()
    selected = [remaining.index[0]]
    remaining = remaining.drop(index=selected[0])
    while len(selected) < MAX_DESTINATIONS and not remaining.empty:
        best_index = None
        best_distance = -1.0
        for index, candidate in remaining.iterrows():
            nearest_distance = min(
                haversine_km(
                    float(candidate.latitude), float(candidate.longitude),
                    float(frame.loc[chosen, "latitude"]),
                    float(frame.loc[chosen, "longitude"]),
                )
                for chosen in selected
            )
            if nearest_distance > best_distance:
                best_index = index
                best_distance = nearest_distance
        selected.append(best_index)
        remaining = remaining.drop(index=best_index)
    return frame.loc[selected].reset_index(drop=True)


def load_kaggle_inventory_profile(path: Path) -> dict | None:
    if not path.is_file():
        return None
    required = {
        "Item_Type",
        "Item_Name",
        "Avg_Usage_Per_Day",
        "Current_Stock",
        "Min_Required",
        "Restock_Lead_Time",
    }
    try:
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, dtype=str, encoding="latin-1")
    except (pd.errors.ParserError, OSError) as error:
        raise ValueError(
            f"Não foi possível ler inventory_data.csv do Kaggle: {error}"
        ) from error
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "inventory_data.csv do Kaggle sem colunas obrigatórias: "
            + ", ".join(sorted(missing))
        )
    for field in (
        "Avg_Usage_Per_Day",
        "Current_Stock",
        "Min_Required",
        "Restock_Lead_Time",
    ):
        raw = frame[field]
        numeric = pd.to_numeric(
            raw.astype("string").str.replace(",", ".", regex=False),
            errors="coerce",
        )
        malformed = raw.notna() & raw.astype("string").str.strip().ne("") & numeric.isna()
        if malformed.any() or numeric.isna().any():
            raise ValueError(
                f"inventory_data.csv do Kaggle contém valores inválidos em {field}."
            )
        if not numeric.map(math.isfinite).all() or (numeric < 0).any():
            raise ValueError(
                f"inventory_data.csv do Kaggle requer valores não negativos e finitos em {field}."
            )
        frame[field] = numeric
    frame["Item_Type"] = frame.Item_Type.fillna("").astype(str).str.strip()
    frame["Item_Name"] = frame.Item_Name.fillna("").astype(str).str.strip()
    if frame.Item_Type.eq("").any() or frame.Item_Name.eq("").any():
        raise ValueError("inventory_data.csv do Kaggle tem tipo ou nome de item vazio.")
    grouped = frame.groupby(["Item_Type", "Item_Name"], sort=True).agg(
        records=("Avg_Usage_Per_Day", "size"),
        average_usage_per_day=("Avg_Usage_Per_Day", "mean"),
        current_stock=("Current_Stock", "mean"),
        minimum_required=("Min_Required", "mean"),
        restock_lead_time_days=("Restock_Lead_Time", "mean"),
    ).reset_index()
    with path.open("rb") as source_file:
        digest = hashlib.file_digest(source_file, "sha256").hexdigest()
    return {
        "dataset": "Hospital Supply Chain — Kaggle",
        "url": "https://www.kaggle.com/datasets/vanpatangan/hospital-supply-chain",
        "license": "MIT (metadados da página Kaggle consultados em 2026-10-24)",
        "file": path.name,
        "file_sha256": digest,
        "records": len(frame),
        "item_types": sorted(frame.Item_Type.unique().tolist()),
        "items": [
            {
                "item_type": str(row.Item_Type),
                "item_name": str(row.Item_Name),
                "records": int(row.records),
                "average_usage_per_day": round(float(row.average_usage_per_day), 2),
                "average_current_stock": round(float(row.current_stock), 2),
                "average_minimum_required": round(float(row.minimum_required), 2),
                "average_restock_lead_time_days": round(
                    float(row.restock_lead_time_days), 2
                ),
            }
            for row in grouped.itertuples(index=False)
        ],
        "scope": (
            "Perfil global de referência do Kaggle; sem CNES, localização ou "
            "demanda específica por estabelecimento. Não usado como carga ou kg."
        ),
    }


def load_cnes_scenario(directory: Path = CNES_DATASET_DIR) -> Scenario:
    try:
        directory = directory.resolve()
        path = directory / CNES_FILENAME
        if not path.is_file():
            raise ValueError(
                f"Dataset CNES ausente: adicione {CNES_FILENAME} em {directory}."
            )
        signature = (path.stat().st_size, path.stat().st_mtime_ns)
        profile_path = directory / "hospital_supply_chain" / "inventory_data.csv"
        profile_signature = (
            (profile_path.stat().st_size, profile_path.stat().st_mtime_ns)
            if profile_path.is_file()
            else None
        )
        return _load_cached(str(path), (signature, profile_signature))
    except (pd.errors.ParserError, UnicodeError, OSError) as error:
        raise ValueError(f"Não foi possível carregar o dataset CNES: {error}") from error


@lru_cache(maxsize=2)
def _load_cached(
    path_value: str,
    signature: tuple[tuple[int, int], tuple[int, int] | None],
) -> Scenario:
    del signature
    path = Path(path_value)
    source = _read_cnes_csv(path)
    inventory_profile = load_kaggle_inventory_profile(
        path.parent / "hospital_supply_chain" / "inventory_data.csv"
    )
    frame = _canonicalize_columns(source)
    frame = frame.loc[frame.uf.map(_is_sp)].copy()
    for field in ("latitude", "longitude", "priority_score", "medicine_kg", "supply_kg"):
        raw_values = frame[field]
        numeric_values = pd.to_numeric(
            raw_values.astype("string").str.replace(",", ".", regex=False),
            errors="coerce",
        )
        malformed = (
            raw_values.notna()
            & raw_values.astype("string").str.strip().ne("")
            & numeric_values.isna()
        )
        if field in {"priority_score", "medicine_kg", "supply_kg"} and malformed.any():
            raise ValueError(f"A coluna {field} contém valores não numéricos.")
        frame[field] = numeric_values
    min_lat, max_lat, min_lon, max_lon = SP_BOUNDS
    valid_coordinates = (
        frame.latitude.between(min_lat, max_lat)
        & frame.longitude.between(min_lon, max_lon)
    )
    frame = frame.loc[valid_coordinates].copy()
    if frame.empty:
        raise ValueError("O CSV CNES não contém estabelecimentos SP com coordenadas válidas.")
    if len(frame) < 2:
        raise ValueError("São necessários ao menos dois estabelecimentos CNES válidos em SP.")
    establishments_with_coordinates = len(frame)
    frame["id"] = frame.id.fillna("").str.strip()
    frame["name"] = frame.name.fillna("").str.strip()
    frame["type"] = frame.type.fillna("").str.strip()
    frame["municipality"] = frame.municipality.fillna("").str.strip()
    frame["priority_is_simulated"] = frame.priority_score.isna()
    if frame.id.eq("").any() or frame.id.duplicated().any():
        raise ValueError("O CSV CNES contém identificador ausente ou duplicado.")
    frame.loc[frame.name.eq(""), "name"] = frame.loc[frame.name.eq(""), "id"]
    frame.loc[frame.type.eq(""), "type"] = "Estabelecimento de saúde"
    missing_priority = frame.priority_score.isna()
    frame["demand_is_simulated"] = frame.medicine_kg.isna() & frame.supply_kg.isna()
    partial_demand = frame.medicine_kg.isna() != frame.supply_kg.isna()
    if partial_demand.any():
        raise ValueError("Preencha medicine_kg e supply_kg juntos em cada estabelecimento.")
    provided_priority = frame.priority_score.dropna()
    if (
        not provided_priority.between(1, 10).all()
        or not provided_priority.mod(1).eq(0).all()
    ):
        raise ValueError("A prioridade informada deve ser um número inteiro entre 1 e 10.")
    frame.loc[missing_priority, "priority_score"] = frame.loc[
        missing_priority, "type"
    ].map(_priority_default)
    frame["priority_score"] = frame.priority_score.astype(int)
    for field in ("medicine_kg", "supply_kg"):
        values = frame[field].dropna()
        if (values < 0).any() or not values.map(math.isfinite).all():
            raise ValueError(f"A coluna {field} deve conter demandas não negativas e finitas.")
    simulated_rows = frame.demand_is_simulated
    frame.loc[simulated_rows, "medicine_kg"] = (
        frame.loc[simulated_rows, "priority_score"] * 0.25
    ).round(2)
    frame.loc[simulated_rows, "supply_kg"] = (
        frame.loc[simulated_rows, "priority_score"] * 0.15
    ).round(2)
    for field in ("medicine_kg", "supply_kg"):
        frame[field] = frame[field].fillna(0)
    frame = frame.loc[frame.medicine_kg.add(frame.supply_kg).gt(0)].copy()
    if len(frame) < 2:
        raise ValueError("São necessários ao menos dois estabelecimentos CNES com demanda em SP.")
    selected = _select_destinations(frame)

    deliveries = []
    aggregates = []
    for index, row in enumerate(selected.itertuples(index=False)):
        delivery_id = f"cnes-entrega-{index + 1:02d}"
        score = int(row.priority_score)
        facility_type = str(row.type)
        municipality = str(row.municipality or "SP")
        medicine_kg = float(row.medicine_kg)
        supply_kg = float(row.supply_kg)
        demand_is_simulated = bool(row.demand_is_simulated)
        medicine_label = (
            "medicamento simulado"
            if demand_is_simulated
            else "medicamento (demanda do CSV)"
        )
        supply_label = (
            "insumo médico simulado"
            if demand_is_simulated
            else "insumo médico (demanda do CSV)"
        )
        items = tuple(
            label for condition, label in (
                (medicine_kg > 0, medicine_label),
                (supply_kg > 0, supply_label),
            ) if condition
        )
        if not items:
            items = ("insumo médico simulado",)
        demand_kg = medicine_kg + supply_kg
        if demand_kg <= 0:
            demand_kg = round(4 + 8 * score / 10, 2)
            supply_kg = demand_kg
        requires_refrigeration = medicine_kg > 0 and score >= 8
        distance = haversine_km(DEPOT.latitude, DEPOT.longitude, row.latitude, row.longitude)
        deadline = 8 * 60 + math.ceil(distance / 65 * 60) + (
            120 if score >= 9 else 240 if score >= 7 else 480
        )
        deliveries.append(Delivery(
            delivery_id,
            f"{str(row.name)} — {municipality}",
            float(row.latitude),
            float(row.longitude),
            round(demand_kg, 2),
            _priority_name(score),
            deadline,
            requires_refrigeration,
            15,
            items,
        ))
        aggregates.append({
            "delivery_id": delivery_id,
            "cnes_id": str(row.id),
            "facility_name": str(row.name),
            "facility_type": facility_type,
            "municipality": municipality,
            "priority_score": score,
            "priority_is_simulated": bool(row.priority_is_simulated),
            "medicine_demand_kg": round(medicine_kg, 2),
            "supply_demand_kg": round(supply_kg, 2),
            "demand_is_simulated": demand_is_simulated,
        })

    with path.open("rb") as source_file:
        digest = hashlib.file_digest(source_file, "sha256").hexdigest()
    profile_digest = inventory_profile["file_sha256"] if inventory_profile else None
    scenario_digest = hashlib.sha256(
        (digest + (profile_digest or "")).encode()
    ).hexdigest()
    file_hashes = {path.name: digest}
    if inventory_profile:
        file_hashes[inventory_profile["file"]] = profile_digest
    data_source = {
        "kind": "cnes_sp",
        "dataset": "CNES — estabelecimentos de saúde do estado de São Paulo",
        "state": "SP",
        "files_sha256": file_hashes,
        "establishments_sp_with_coordinates": establishments_with_coordinates,
        "selected_establishments": len(deliveries),
        "demand_source": (
            "Demanda do CSV local quando informada; valores ausentes simulados por tipo/prioridade"
            if not frame.demand_is_simulated.all()
            else "Demanda e prioridade simuladas por tipo de estabelecimento"
        ),
        "priority_source": (
            "Coluna local quando informada; valores ausentes simulados pelo tipo de estabelecimento"
        ),
        "coordinates": (
            "Latitude/longitude do CNES; envelope estadual aplicado para descartar outliers."
        ),
        "depot": "Ponto central simulado em São Paulo; não representa uma distribuidora real.",
        "demand_is_simulated": bool(frame.demand_is_simulated.any()),
        "facility_aggregates": aggregates,
        "kaggle_inventory_profile": inventory_profile,
    }
    LOGGER.info(
        "cnes_sp_loaded establishments=%d selected=%d simulated_demand=%s",
        establishments_with_coordinates, len(deliveries),
        bool(frame.demand_is_simulated.any()),
    )
    return Scenario(
        id=f"cnes-sp-medical-simulation-{scenario_digest[:16]}",
        name="São Paulo — estabelecimentos CNES e demanda médica simulada",
        description=(
            f"{len(deliveries)} estabelecimentos de saúde selecionados de um CSV CNES local. "
            "Coordenadas e identificação vêm do CNES; demanda, itens médicos, frota, "
            "prazos e depósito são simulados, salvo demanda explicitamente fornecida no CSV."
        ),
        synthetic=True,
        depot=DEPOT,
        vehicles=VEHICLES,
        deliveries=tuple(deliveries),
        data_source=data_source,
    )
