"""Charge-lot normalization layered on the causal Charge state V1 contract."""

from __future__ import annotations

import re

from Mooney_Prediction_Pipeline.feature_engineering.causal_charge_state import (
    build_causal_charge_features,
    material_family,
)


def normalize_charge_lot(value: object) -> str:
    text = str(value).strip().upper().replace(" ", "")
    match = re.fullmatch(r"S\d{2}(.+)", text)
    return match.group(1) if match else text


__all__ = ["build_causal_charge_features", "material_family", "normalize_charge_lot"]
