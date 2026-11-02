"""Synthetic delivery scenario for the medical route optimization demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Depot:
    id: str
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Vehicle:
    id: str
    name: str
    capacity_kg: float
    max_distance_km: float
    speed_kmh: float
    refrigerated: bool
    start_minute: int = 8 * 60


@dataclass(frozen=True)
class Delivery:
    id: str
    name: str
    latitude: float
    longitude: float
    demand_kg: float
    priority: str
    deadline_minute: int
    requires_refrigeration: bool
    service_minutes: int
    items: tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    description: str
    synthetic: bool
    depot: Depot
    vehicles: tuple[Vehicle, ...]
    deliveries: tuple[Delivery, ...]
    data_source: dict[str, Any] = field(default_factory=dict)


DEMO_SCENARIO = Scenario(
    id="demo-sp-capital-sintetico-v2",
    name="Distribuição hospitalar — cidade de São Paulo (simulação)",
    description=(
        "Simulação fictícia distribuída por diferentes regiões da cidade de São "
        "Paulo. Coordenadas ilustrativas não correspondem a endereços ou unidades "
        "reais; cargas, prazos, frota e operações são inteiramente demonstrativos."
    ),
    synthetic=True,
    depot=Depot(
        id="central",
        name="Central de distribuição hospitalar",
        latitude=-23.5505,
        longitude=-46.6333,
    ),
    vehicles=(
        Vehicle("veiculo-1", "Veículo refrigerado A", 32.0, 110.0, 32.0, True),
        Vehicle("veiculo-2", "Veículo refrigerado B", 30.0, 105.0, 30.0, True),
        Vehicle("veiculo-3", "Veículo utilitário C", 34.0, 115.0, 31.0, False),
        Vehicle("veiculo-4", "Veículo utilitário D", 34.0, 115.0, 30.0, False),
        Vehicle("veiculo-5", "Veículo utilitário E", 32.0, 105.0, 29.0, False),
    ),
    deliveries=(
        Delivery(
            "entrega-01",
            "Unidade hospitalar Norte (fictícia)",
            -23.4300,
            -46.6300,
            8.0,
            "critica",
            9 * 60 + 30,
            True,
            12,
            ("medicamento refrigerado", "material de emergência"),
        ),
        Delivery(
            "entrega-02",
            "Unidade básica Leste (fictícia)",
            -23.5500,
            -46.4800,
            6.0,
            "regular",
            13 * 60 + 30,
            False,
            10,
            ("insumos de curativo",),
        ),
        Delivery(
            "entrega-03",
            "Unidade hospitalar Sul (fictícia)",
            -23.6800,
            -46.6400,
            10.0,
            "alta",
            11 * 60 + 30,
            False,
            15,
            ("medicamento prioritário",),
        ),
        Delivery(
            "entrega-04",
            "Unidade básica Oeste (fictícia)",
            -23.5600,
            -46.7700,
            5.0,
            "regular",
            13 * 60 + 45,
            False,
            10,
            ("insumos de coleta",),
        ),
        Delivery(
            "entrega-05",
            "Atendimento domiciliar simulado A",
            -23.6500,
            -46.5300,
            3.0,
            "critica",
            9 * 60 + 45,
            True,
            12,
            ("medicamento refrigerado",),
        ),
        Delivery(
            "entrega-06",
            "Unidade de pronto atendimento (fictícia)",
            -23.4600,
            -46.7200,
            7.0,
            "alta",
            11 * 60,
            False,
            12,
            ("material de emergência",),
        ),
        Delivery(
            "entrega-07",
            "Unidade básica Centro (fictícia)",
            -23.5600,
            -46.6400,
            4.0,
            "regular",
            14 * 60,
            False,
            10,
            ("insumos de curativo",),
        ),
        Delivery(
            "entrega-08",
            "Atendimento domiciliar simulado B",
            -23.4800,
            -46.5200,
            2.0,
            "alta",
            11 * 60 + 15,
            False,
            12,
            ("medicamento prioritário",),
        ),
        Delivery(
            "entrega-09",
            "Unidade hospitalar Sudeste (fictícia)",
            -23.6000,
            -46.4500,
            9.0,
            "critica",
            10 * 60,
            True,
            15,
            ("medicamento refrigerado", "material de emergência"),
        ),
        Delivery(
            "entrega-10",
            "Unidade básica Noroeste (fictícia)",
            -23.4200,
            -46.7000,
            5.0,
            "regular",
            14 * 60 + 15,
            False,
            10,
            ("insumos de coleta",),
        ),
        Delivery(
            "entrega-11",
            "Unidade de apoio Sul (fictícia)",
            -23.6900,
            -46.7400,
            6.0,
            "alta",
            11 * 60 + 45,
            False,
            12,
            ("medicamento prioritário",),
        ),
        Delivery(
            "entrega-12",
            "Atendimento domiciliar simulado C",
            -23.6400,
            -46.7800,
            2.0,
            "regular",
            14 * 60 + 30,
            False,
            12,
            ("insumos de curativo",),
        ),
        Delivery(
            "entrega-13",
            "Unidade hospitalar Sudeste 2 (fictícia)",
            -23.7000,
            -46.4900,
            8.0,
            "alta",
            12 * 60,
            False,
            15,
            ("medicamento prioritário", "insumos hospitalares"),
        ),
        Delivery(
            "entrega-14",
            "Unidade de apoio Nordeste (fictícia)",
            -23.4100,
            -46.5400,
            5.0,
            "critica",
            10 * 60 + 15,
            True,
            12,
            ("medicamento refrigerado",),
        ),
        Delivery(
            "entrega-15",
            "Unidade básica Noroeste 2 (fictícia)",
            -23.4700,
            -46.8000,
            7.0,
            "regular",
            14 * 60 + 45,
            False,
            10,
            ("insumos de coleta", "insumos de curativo"),
        ),
        Delivery(
            "entrega-16",
            "Unidade hospitalar Leste (fictícia)",
            -23.6600,
            -46.4400,
            9.0,
            "alta",
            12 * 60 + 15,
            False,
            15,
            ("medicamento prioritário", "material de emergência"),
        ),
        Delivery(
            "entrega-17",
            "Atendimento domiciliar simulado D",
            -23.6100,
            -46.6200,
            3.0,
            "critica",
            10 * 60 + 30,
            True,
            12,
            ("medicamento refrigerado",),
        ),
        Delivery(
            "entrega-18",
            "Unidade básica Oeste 2 (fictícia)",
            -23.5400,
            -46.7300,
            4.0,
            "regular",
            15 * 60,
            False,
            10,
            ("insumos de curativo",),
        ),
    ),
)

PRIORITY_WEIGHTS = {"critica": 8.0, "alta": 3.0, "regular": 1.0}
