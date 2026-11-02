"""Aggregate local Olist data into a simulated medical VRP in Sao Paulo state."""

from functools import lru_cache
import hashlib
import logging
import math
from pathlib import Path
import unicodedata

import pandas as pd

from src.logistics.genetic import haversine_km
from src.logistics.cnes_dataset import (
    KAGGLE_INVENTORY_PATH,
    load_kaggle_inventory_profile,
)
from src.logistics.item_types import classify_medical_item_type
from src.logistics.scenario import Delivery, Depot, Scenario, Vehicle


LOGGER = logging.getLogger(__name__)
DATASET_DIR = Path(__file__).resolve().parents[1] / "db" / "models" / "dataset_rotasBR"
FILES = {
    "customers": ("olist_customers_dataset.csv", [
        "customer_id", "customer_zip_code_prefix", "customer_city", "customer_state",
    ]),
    "geolocation": ("olist_geolocation_dataset.csv", [
        "geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
        "geolocation_state",
    ]),
    "orders": ("olist_orders_dataset.csv", [
        "order_id", "customer_id", "order_status", "order_purchase_timestamp",
    ]),
    "items": ("olist_order_items_dataset.csv", ["order_id", "product_id"]),
    "products": ("olist_products_dataset.csv", [
        "product_id", "product_weight_g", "product_category_name",
    ]),
}
MAX_DESTINATIONS = 18


def _normalize_city(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", value.strip().casefold())
        if not unicodedata.combining(char)
    )


def load_olist_scenario(directory: Path = DATASET_DIR) -> Scenario:
    try:
        directory = directory.resolve()
        missing = [name for name, _ in FILES.values() if not (directory / name).is_file()]
        if missing:
            raise ValueError(
                "Dataset Olist indisponível em dataset_rotasBR. Arquivos ausentes: "
                + ", ".join(missing)
            )
        signature = tuple(
            ((directory / name).stat().st_size, (directory / name).stat().st_mtime_ns)
            for name, _ in FILES.values()
        )
        profile_path = (
            KAGGLE_INVENTORY_PATH
            if directory == DATASET_DIR.resolve()
            else directory / "hospital_supply_chain" / "inventory_data.csv"
        )
        profile_signature = (
            (profile_path.stat().st_size, profile_path.stat().st_mtime_ns)
            if profile_path.is_file()
            else (0, 0)
        )
        signature += (profile_signature,)
        return _load_cached(str(directory), signature)
    except (pd.errors.ParserError, UnicodeError, OSError) as error:
        raise ValueError(f"Não foi possível ler dataset_rotasBR: {error}") from error


@lru_cache(maxsize=2)
def _load_cached(directory: str, signature: tuple[tuple[int, int], ...]) -> Scenario:
    root = Path(directory)
    frames = {}
    hashes = {}
    for label, (filename, columns) in FILES.items():
        path = root / filename
        with path.open("rb") as source:
            hashes[filename] = hashlib.file_digest(source, "sha256").hexdigest()
        if label != "geolocation":
            try:
                frames[label] = pd.read_csv(path, usecols=columns, dtype=str)
            except ValueError as error:
                raise ValueError(f"Colunas inválidas em {filename}: {error}") from error
    root = Path(directory)
    profile_path = (
        KAGGLE_INVENTORY_PATH
        if root == DATASET_DIR
        else root / "hospital_supply_chain" / "inventory_data.csv"
    )
    inventory_profile = load_kaggle_inventory_profile(profile_path)
    if inventory_profile:
        hashes[inventory_profile["file"]] = inventory_profile["file_sha256"]

    customers = frames["customers"]
    customers = customers.loc[customers.customer_state.eq("SP")].copy()
    customers["city"] = customers.customer_city.fillna("").map(_normalize_city)
    customers["zip"] = pd.to_numeric(customers.customer_zip_code_prefix, errors="coerce")
    if customers.customer_id.isna().any() or customers.customer_id.duplicated().any():
        raise ValueError("Olist SP contém customer_id ausente ou duplicado.")

    geo_parts = []
    sp_geo_rows = invalid_geo_rows = 0
    filename, columns = FILES["geolocation"]
    try:
        chunks = pd.read_csv(root / filename, usecols=columns, dtype=str, chunksize=100_000)
        for chunk in chunks:
            part = chunk.loc[chunk.geolocation_state.eq("SP")].copy()
            sp_geo_rows += len(part)
            part["latitude"] = pd.to_numeric(part.geolocation_lat, errors="coerce")
            part["longitude"] = pd.to_numeric(part.geolocation_lng, errors="coerce")
            part["zip"] = pd.to_numeric(part.geolocation_zip_code_prefix, errors="coerce")
            valid = (
                part.latitude.between(-25.4, -19.7)
                & part.longitude.between(-53.2, -44.0)
                & part.zip.between(0, 99999)
                & part.zip.mod(1).eq(0)
            )
            invalid_geo_rows += int((~valid).sum())
            geo_parts.append(part.loc[valid, ["zip", "latitude", "longitude"]])
    except ValueError as error:
        raise ValueError(f"Geolocalização Olist inválida: {error}") from error
    if not geo_parts:
        raise ValueError("Não há geolocalização SP no dataset Olist.")
    geo = pd.concat(geo_parts).groupby("zip")[["latitude", "longitude"]].median()
    city_zip = customers.loc[customers.city.ne(""), ["city", "zip"]].drop_duplicates()
    coordinates = city_zip.merge(geo, on="zip", how="inner").groupby("city")[
        ["latitude", "longitude"]
    ].median()

    orders = frames["orders"]
    if orders.order_id.isna().any() or orders.order_id.duplicated().any():
        raise ValueError("Olist contém order_id ausente ou duplicado.")
    orders = orders.loc[orders.order_status.eq("delivered")].merge(
        customers[["customer_id", "city"]], on="customer_id", validate="many_to_one"
    )
    purchase_dates = pd.to_datetime(
        orders.order_purchase_timestamp, format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    if purchase_dates.empty or purchase_dates.isna().any():
        raise ValueError("Pedidos entregues SP sem data de compra válida no Olist.")
    products = frames["products"]
    if products.product_id.isna().any() or products.product_id.duplicated().any():
        raise ValueError("Olist contém product_id ausente ou duplicado.")
    products["weight_kg"] = pd.to_numeric(products.product_weight_g, errors="coerce") / 1000
    products.loc[
        ~products.weight_kg.gt(0) | ~products.weight_kg.lt(float("inf")), "weight_kg"
    ] = float("nan")
    products["medical_item_type"] = products.product_category_name.map(
        classify_medical_item_type
    )
    items = frames["items"].merge(
        products[["product_id", "weight_kg", "medical_item_type"]],
        on="product_id", how="left",
        validate="many_to_one",
    )
    items["medical_item_type"] = items.medical_item_type.fillna("nao_classificado")
    category_fields = {
        "historical_medicine_items": "medicamento",
        "historical_supply_items": "insumo",
        "historical_health_beauty_items": "saude_beleza_nao_confirmado",
        "historical_unclassified_items": "nao_classificado",
    }
    for field, category in category_fields.items():
        items[field] = items.medical_item_type.eq(category)
    order_items = items.groupby("order_id").agg(
        item_count=("product_id", "size"),
        known_weight_items=("weight_kg", "count"),
        observed_weight_kg=("weight_kg", "sum"),
        **{field: (field, "sum") for field in category_fields},
    )
    orders = orders.merge(order_items, on="order_id", how="left", validate="one_to_one")
    demand = orders.groupby("city").agg(
        delivered_orders=("order_id", "size"),
        item_count=("item_count", "sum"),
        known_weight_items=("known_weight_items", "sum"),
        observed_weight_kg=("observed_weight_kg", "sum"),
        historical_medicine_items=("historical_medicine_items", "sum"),
        historical_supply_items=("historical_supply_items", "sum"),
        historical_health_beauty_items=("historical_health_beauty_items", "sum"),
        historical_unclassified_items=("historical_unclassified_items", "sum"),
    )
    eligible = demand.join(coordinates, how="inner")
    if "sao paulo" not in eligible.index:
        raise ValueError("Dataset sem coordenadas e pedidos entregues para São Paulo (depósito).")
    selected = eligible.drop(index="sao paulo").reset_index().sort_values(
        ["delivered_orders", "city"], ascending=[False, True]
    ).head(MAX_DESTINATIONS)
    if len(selected) < 2:
        raise ValueError("São necessários ao menos dois municípios SP além do depósito.")
    central = eligible.loc["sao paulo"]
    depot = Depot(
        "central-sp", "Depósito simulado — São Paulo",
        round(float(central.latitude), 6), round(float(central.longitude), 6),
    )
    vehicles = tuple(
        Vehicle(
            f"sp-veiculo-{index + 1}",
            f"Veículo estadual {index + 1}" + (" refrigerado" if index < 2 else ""),
            50.0, 1800.0, 65.0, index < 2,
        )
        for index in range(5)
    )
    deliveries = []
    aggregates = []
    largest = float(selected.delivered_orders.max())
    for index, row in enumerate(selected.itertuples(index=False)):
        refrigerated = index % 5 == 0
        priority = "critica" if refrigerated else ("alta" if index % 3 == 0 else "regular")
        distance = haversine_km(depot.latitude, depot.longitude, row.latitude, row.longitude)
        deadline = 8 * 60 + math.ceil(distance / 65 * 60) + (
            120 if priority == "critica" else 240 if priority == "alta" else 480
        )
        delivery_id = f"sp-entrega-{index + 1:02d}"
        deliveries.append(Delivery(
            delivery_id, f"Polo simulado — {row.city.title()}/SP",
            round(float(row.latitude), 6), round(float(row.longitude), 6),
            round(4 + 8 * math.sqrt(row.delivered_orders / largest), 2),
            priority, deadline, refrigerated, 15,
            ("medicamento refrigerado simulado",) if refrigerated else ("insumos médicos simulados",),
        ))
        aggregates.append({
            "delivery_id": delivery_id,
            "municipality": row.city,
            "historical_delivered_orders": int(row.delivered_orders),
            "historical_items": int(row.item_count),
            "items_with_known_weight": int(row.known_weight_items),
            "observed_product_weight_kg": round(float(row.observed_weight_kg), 2),
            "historical_medicine_items": int(row.historical_medicine_items),
            "historical_supply_items": int(row.historical_supply_items),
            "historical_health_beauty_items": int(row.historical_health_beauty_items),
            "historical_unclassified_items": int(row.historical_unclassified_items),
        })
    version = hashlib.sha256(
        ("olist-sp-v1" + "".join(hashes.values())).encode()
    ).hexdigest()[:16]
    data_source = {
        "kind": "olist_sp",
        "dataset": "Olist — comércio eletrônico brasileiro",
        "state": "SP",
        "files_sha256": hashes,
        "delivered_orders_sp": len(orders),
        "purchase_period": {
            "start": purchase_dates.min().isoformat(),
            "end": purchase_dates.max().isoformat(),
        },
        "geolocation_sp_rows": sp_geo_rows,
        "discarded_geolocation_rows": invalid_geo_rows,
        "municipalities_with_orders": len(demand),
        "municipalities_with_coordinates": len(eligible),
        "municipalities_without_coordinates": len(demand.index.difference(coordinates.index)),
        "orders_without_items": int(orders.item_count.isna().sum()),
        "selected_municipalities": len(deliveries),
        "selection": "Até 18 municípios por pedidos entregues, desempate por nome; capital é o depósito.",
        "coordinates": "Mediana por prefixo de CEP; depois mediana dos prefixos por município. Não são endereços.",
        "geographic_filter": "UF SP e envelope lat [-25.4,-19.7], lon [-53.2,-44.0]; não é polígono estadual.",
        "simulated_fields": ["carga médica", "prioridade", "prazo", "frota", "cadeia fria", "serviço"],
        "medical_item_classification": (
        "Itens históricos classificados somente quando product_category_name "
        "contém termos explícitos de medicamento ou insumo; categorias genéricas "
        "de saúde/beleza são destacadas separadamente, sem confirmação de que "
        "sejam medicamentos ou insumos; os demais produtos ficam sem classificação."
        ),
        "kaggle_inventory_profile": inventory_profile,
        "municipal_aggregates": aggregates,
    }
    LOGGER.info(
        "olist_sp_loaded orders=%d destinations=%d discarded_geo=%d",
        len(orders), len(deliveries), invalid_geo_rows,
    )
    return Scenario(
        id=f"olist-sp-medical-simulation-v1-{version}",
        name="Estado de São Paulo — Olist com operação médica simulada",
        description=(
            f"Recorte de {len(deliveries)} municípios do estado de São Paulo com "
            "coordenadas agregadas e demanda histórica do Olist (comércio eletrônico). "
            "Não são hospitais, pacientes ou rotas reais: medicamentos, carga médica, "
            "prioridades, frota e prazos são simulados. Distâncias em linha reta, sem vias ou trânsito."
        ),
        synthetic=True, depot=depot, vehicles=vehicles,
        deliveries=tuple(deliveries), data_source=data_source,
    )
