"""Reproducible fictional allocations over Olist/CNES geography and Kaggle stock."""

from dataclasses import replace
import hashlib
import math
from pathlib import Path

import pandas as pd

from src.logistics.cnes_dataset import (
    CNES_DATASET_DIR, CNES_FILENAME, KAGGLE_INVENTORY_PATH,
    load_cnes_scenario, load_kaggle_inventory_profile,
)
from src.logistics.datasets import load_olist_scenario
from src.logistics.genetic import haversine_km
from src.logistics.scenario import Scenario


MODEL_VERSION = "distribution-v1"
UNITS_SCALE = 20
MAX_UNITS_PER_LINE = 30


def build_distribution_scenario(base: Scenario, inventory_path: Path) -> Scenario:
    profile = load_kaggle_inventory_profile(inventory_path)
    if profile is None:
        raise ValueError(
            "Inventário Hospital Supply Chain ausente. Execute "
            r".\scripts\download_kaggle_inventory.ps1 antes de usar a distribuição híbrida."
        )
    inventory = pd.read_csv(inventory_path, encoding="utf-8-sig")
    inventory = inventory.loc[
        inventory.Item_Type.str.casefold().eq("consumable")
        & inventory.Avg_Usage_Per_Day.gt(0)
    ]
    if inventory.empty:
        raise ValueError("Inventário Kaggle sem consumíveis com uso diário positivo.")
    deliveries = []
    manifests = []
    for index, delivery in enumerate(base.deliveries):
        # Hash assignment is fictional: the inventory has no facility join key.
        offset = int(hashlib.sha256(delivery.id.encode()).hexdigest()[:8], 16)
        categories = (
            ("medicamento", "insumo") if index % 3 == 0
            else ("medicamento",) if index % 3 == 1
            else ("insumo",)
        )
        lines = []
        for line_index, category in enumerate(categories):
            source_index = (offset + line_index) % len(inventory)
            row = inventory.iloc[source_index]
            usage = max(1, math.ceil(float(row.Avg_Usage_Per_Day) / UNITS_SCALE))
            stock = math.floor(float(row.Current_Stock) / UNITS_SCALE)
            minimum = math.ceil(float(row.Min_Required) / UNITS_SCALE)
            lead_days = float(row.Restock_Lead_Time)
            coverage = stock / usage
            priority = (
                "critica" if stock < minimum or coverage < 1
                else "alta" if coverage < lead_days else "regular"
            )
            quantity = min(MAX_UNITS_PER_LINE, max(usage, minimum - stock, 1))
            unit_weight = 0.05 if category == "medicamento" else 0.10
            name = (
                "Kit de medicamentos fictício"
                if category == "medicamento"
                else f"Kit de insumos ({row.Item_Name})"
            )
            lines.append({
                "category": category,
                "name": name,
                "quantity": quantity,
                "unit": "kit simulado",
                "unit_weight_kg": unit_weight,
                "load_kg": round(quantity * unit_weight, 2),
                "stock_units": stock,
                "minimum_stock_units": minimum,
                "daily_usage_units": usage,
                "stock_coverage_days": round(coverage, 2),
                "restock_lead_time_days": lead_days,
                "priority": priority,
                "kaggle_row": int(row.name) + 1,
                "reference_item": str(row.Item_Name),
                "simulated": True,
            })
        priority = min(
            (line["priority"] for line in lines),
            key={"critica": 0, "alta": 1, "regular": 2}.__getitem__,
        )
        distance = haversine_km(
            base.depot.latitude, base.depot.longitude,
            delivery.latitude, delivery.longitude,
        )
        deadline = 480 + math.ceil(distance / 65 * 60) + {
            "critica": 120, "alta": 240, "regular": 480,
        }[priority]
        load = round(sum(line["load_kg"] for line in lines), 2)
        items = tuple(
            f"{line['category']} simulado: {line['name']} — {line['quantity']} kits"
            for line in lines
        )
        deliveries.append(replace(
            delivery, demand_kg=load, priority=priority, deadline_minute=deadline,
            requires_refrigeration=(
                "medicamento" in categories and index % 5 == 0
            ),
            items=items,
        ))
        manifests.append({
            "delivery_id": delivery.id,
            "priority": priority,
            "load_kg": load,
            "lines": lines,
            "simulated": True,
        })
    digest = hashlib.sha256(
        f"{MODEL_VERSION}:{base.id}:{profile['file_sha256']}".encode()
    ).hexdigest()[:16]
    source = {
        **base.data_source,
        "kind": "hybrid_sp",
        "geography_source": base.data_source.get("kind", "demo"),
        "kaggle_inventory_profile": profile,
        "distribution_manifest": manifests,
        "demand_is_simulated": True,
        "demand_source": (
            "Associação fictícia determinística de estoque Kaggle aos destinos. "
            "Medicamentos e pesos por kit são fictícios; não são produtos Kaggle."
        ),
        "distribution_method": {
            "version": MODEL_VERSION,
            "units_scale": UNITS_SCALE,
            "max_units_per_line": MAX_UNITS_PER_LINE,
            "medicine_kit_kg": 0.05,
            "supply_kit_kg": 0.10,
            "priority_rule": (
                "Crítica: estoque abaixo do mínimo ou cobertura < 1 dia; "
                "alta: cobertura menor que prazo de reposição; demais regulares."
            ),
        },
    }
    source["files_sha256"] = {
        **base.data_source.get("files_sha256", {}),
        profile["file"]: profile["file_sha256"],
    }
    if "facility_aggregates" in source:
        by_delivery = {manifest["delivery_id"]: manifest for manifest in manifests}
        source["facility_aggregates"] = [
            {
                **facility,
                "priority_score": {"critica": 9, "alta": 7, "regular": 4}[
                    by_delivery[facility["delivery_id"]]["priority"]
                ],
                "priority_is_simulated": True,
                "demand_is_simulated": True,
                **{
                    field: round(sum(
                        line["load_kg"]
                        for line in by_delivery[facility["delivery_id"]]["lines"]
                        if line["category"] == category
                    ), 2)
                    for field, category in (
                        ("medicine_demand_kg", "medicamento"),
                        ("supply_demand_kg", "insumo"),
                    )
                },
            }
            for facility in source["facility_aggregates"]
        ]
    return replace(
        base,
        id=f"{MODEL_VERSION}-{digest}",
        name="Distribuição híbrida — medicamentos e insumos (simulação)",
        description=(
            f"Distribuição fictícia em {len(deliveries)} destinos. Base geográfica: "
            f"{source['geography_source']}. Estoques Kaggle associados artificialmente; "
            "kits de medicamentos, pesos, demandas e cadeia fria são simulados. "
            "Não há vínculo real de consumo entre Kaggle, Olist e CNES. "
            "Distâncias Haversine em linha reta, sem ruas ou trânsito."
        ),
        deliveries=tuple(deliveries), synthetic=True, data_source=source,
    )


def load_distribution_scenario() -> Scenario:
    if (CNES_DATASET_DIR / CNES_FILENAME).is_file():
        base = load_cnes_scenario()
    else:
        base = load_olist_scenario()
    return build_distribution_scenario(base, KAGGLE_INVENTORY_PATH)
