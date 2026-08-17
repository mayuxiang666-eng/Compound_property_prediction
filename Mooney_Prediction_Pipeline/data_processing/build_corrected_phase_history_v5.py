"""Final facade for cached corrected-history extraction."""

from __future__ import annotations

import pandas as pd

from Mooney_Prediction_Pipeline.data_processing import build_corrected_phase_history_v1 as v1
from Mooney_Prediction_Pipeline.data_processing import build_corrected_phase_history_v2 as v2
from Mooney_Prediction_Pipeline.data_processing.canonical_material_contract import (
    deduplicate_cross_database_rows,
    material_family,
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


def _safe_aggregate(materials: pd.DataFrame, headers: pd.DataFrame) -> pd.DataFrame:
    work = deduplicate_cross_database_rows(
        materials,
        [c for c in ("OrderID", "MaterialCode", "MaterialName", "MaterialWeight") if c in materials],
    )
    work["family"] = work["MaterialCode"].map(material_family)
    work["MaterialWeight"] = pd.to_numeric(work["MaterialWeight"], errors="coerce")
    totals = work.loc[work["family"].ne("other")].pivot_table(
        index="OrderID", columns="family", values="MaterialWeight", aggfunc="sum",
    ).reset_index()
    result = headers.merge(totals, on="OrderID", how="left", validate="many_to_one")
    result["BatchWeight"] = pd.to_numeric(result["BatchWeight"], errors="coerce")
    outputs = {
        "solid_elastomer": "weight_pct_solid_elastomer",
        "natural_rubber": "weight_pct_natural_rubber",
        "silica": "weight_pct_silica", "oil": "weight_pct_oil",
        "silane": "weight_pct_silian", "carbon_black": "weight_pct_carbon_black",
    }
    for family, column in outputs.items():
        source = result[family] if family in result else pd.Series(0.0, index=result.index)
        result[column] = pd.to_numeric(source, errors="coerce").fillna(0.0) / result["BatchWeight"] * 100.0
    return result[["OrderID", "BatchNumber", *outputs.values()]]


v2._numeric_orders = _varchar_order_scope
v2._cached_query = _deduplicated_cached_query
v1.aggregate_order_material_percentages = _safe_aggregate


if __name__ == "__main__":
    v2.main()
