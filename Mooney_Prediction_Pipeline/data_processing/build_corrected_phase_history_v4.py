"""V4 facade: varchar OrderID binding plus active/archive post-cache dedup."""

from __future__ import annotations

import pandas as pd

from Mooney_Prediction_Pipeline.data_processing import build_corrected_phase_history_v1 as v1
from Mooney_Prediction_Pipeline.data_processing import build_corrected_phase_history_v2 as v2
from Mooney_Prediction_Pipeline.data_processing.canonical_material_contract import (
    deduplicate_cross_database_rows,
)


def _varchar_order_scope(order_ids: list[str]) -> tuple[list[str], list[str]]:
    numeric, rejected = [], []
    for value in order_ids:
        text = str(value).strip()
        (numeric if text.isdigit() else rejected).append(text)
    return sorted(set(numeric)), sorted(set(rejected))


_raw_cached_query = v2._cached_query


def _deduplicated_cached_query(*args, **kwargs):
    source = args[0] if args else kwargs["source"]
    frame = _raw_cached_query(*args, **kwargs)
    if frame.empty:
        return frame
    if source == "batch_headers":
        return deduplicate_cross_database_rows(frame, ["OrderID", "BatchNumber"])
    if source == "batch_charges":
        frame = frame.loc[
            frame["ChargeID"].astype(str).str.strip().str.upper().ne("UNKNOWN LOT")
        ]
        return deduplicate_cross_database_rows(
            frame, ["OrderID", "BatchNumber", "MaterialCode", "ChargeID"],
        )
    return frame


_original_aggregate = v1.aggregate_order_material_percentages


def _safe_aggregate(materials: pd.DataFrame, headers: pd.DataFrame) -> pd.DataFrame:
    result = _original_aggregate(materials, headers)
    expected = {
        "weight_pct_solid_elastomer", "weight_pct_natural_rubber",
        "weight_pct_silica", "weight_pct_oil", "weight_pct_silian",
        "weight_pct_carbon_black",
    }
    for column in expected - set(result.columns):
        result[column] = 0.0
    return result


v2._numeric_orders = _varchar_order_scope
v2._cached_query = _deduplicated_cached_query
v1.aggregate_order_material_percentages = _safe_aggregate


if __name__ == "__main__":
    v2.main()
