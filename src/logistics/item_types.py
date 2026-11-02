"""Classification for explicit medical product or supply labels."""

from __future__ import annotations

import re
import unicodedata


def classify_medical_item_type(value: object) -> str:
    """Classify explicit medicine/supply labels and flag broad health categories separately."""
    normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
    normalized = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    words = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    if re.search(
        r"\b(medicamento|medicamentos|remedio|remedios|farmacia|farmaceutico|"
        r"farmaceutica|medicine|medicines|medication|medications|drug|drugs|"
        r"pharmaceutical|pharmaceuticals)\b",
        words,
    ):
        return "medicamento"
    if re.search(
        r"\b(insumo|insumos|suprimento|suprimentos|curativo|curativos|"
        r"material de emergencia|materiais de emergencia|"
        r"material medico|materiais medicos|material hospitalar|"
        r"materiais hospitalares|medical supply|medical supplies|"
        r"hospital supply|hospital supplies|surgical supply|surgical supplies)\b",
        words,
    ):
        return "insumo"
    if words in {"beleza saude", "saude beleza", "health beauty", "beauty health"}:
        return "saude_beleza_nao_confirmado"
    return "nao_classificado"
