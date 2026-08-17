"""Order-completion incremental Phase builder for the corrected V3.7 path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from Mooney_Prediction_Pipeline.data_processing.canonical_mms_source import (
    query_batch_curves,
    query_batch_headers,
)
from Mooney_Prediction_Pipeline.data_processing.canonical_mms_source_v2 import build_corrected_static_features
from Mooney_Prediction_Pipeline.data_processing.function_definitions_M1 import get_hefei_weather
from Mooney_Prediction_Pipeline.feature_engineering.corrected_phase_contract import (
    CONTRACT_VERSION,
    apply_corrected_phase_contract,
)
from Mooney_Prediction_Pipeline.model_analysis.predict_latest_sfe_order import extract_features_for_row
from Mooney_Prediction_Pipeline.model_analysis.predict_order_v3 import query_order_rows


WEATHER_COLUMNS = [
    "env_temp_mean", "env_temp_min", "env_temp_max",
    "env_humidity_mean", "env_humidity_min", "env_humidity_max",
]
LAB_COLUMNS = ["MNY", "Mooney_Viscosity", "actual_mny", "current_batch_mny"]


def _model_columns(metadata_path: str | Path | None) -> list[str]:
    if metadata_path is None:
        return []
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return list(dict.fromkeys([*metadata["s1_features"], *metadata["s2_features"]]))


def build_incremental_phase_order(
    order_id: str,
    *,
    metadata_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = query_order_rows(str(order_id)).copy()
    if source.empty:
        raise ValueError(f"No completed MMS curve rows for OrderID={order_id}")
    source["OrderID"] = source["OrderID"].astype(str).str.strip()
    source["BatchNumber"] = pd.to_numeric(source["BatchNumber"], errors="coerce").astype(int)

    curves = query_batch_curves(str(order_id)).copy()
    curves["OrderID"] = curves["OrderID"].astype(str).str.strip()
    curves["BatchNumber"] = pd.to_numeric(curves["BatchNumber"], errors="coerce").astype(int)
    curves = curves.rename(columns={
        "Curve1": "temp", "Curve2": "power", "Curve5": "Torque",
        "Curve6": "RotorSpeed", "Curve7": "WayofRam",
    })
    curve_columns = ["temp", "power", "Torque", "RotorSpeed", "WayofRam"]
    source = source.drop(columns=curve_columns, errors="ignore").merge(
        curves[["OrderID", "BatchNumber", *curve_columns]],
        on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one",
    )

    static = build_corrected_static_features(str(order_id))
    static["OrderID"] = static["OrderID"].astype(str).str.strip()
    static["BatchNumber"] = pd.to_numeric(static["BatchNumber"], errors="coerce").astype(int)
    static_columns = [column for column in static if column not in {"OrderID", "BatchNumber"}]
    source = source.drop(columns=[column for column in static_columns if column in source], errors="ignore")
    source = source.merge(static, on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one")

    headers = query_batch_headers(str(order_id))
    headers["OrderID"] = headers["OrderID"].astype(str).str.strip()
    headers["BatchNumber"] = pd.to_numeric(headers["BatchNumber"], errors="coerce").astype(int)
    source = source.drop(columns=["PalletID"], errors="ignore").merge(
        headers[["OrderID", "BatchNumber", "PalletID"]],
        on=["OrderID", "BatchNumber"], how="left", validate="one_to_one",
    )
    if source["PalletID"].isna().any():
        raise ValueError("PalletID is incomplete for incremental Phase order")

    source["_weather_date"] = pd.to_datetime(source["OrderStartTime"], errors="coerce").dt.date
    weather = get_hefei_weather(source["OrderStartTime"])
    source = source.merge(
        weather, left_on="_weather_date", right_on="date", how="left", validate="many_to_one",
    ).drop(columns=["_weather_date", "date"], errors="ignore")
    if source[WEATHER_COLUMNS].isna().any(axis=None):
        raise ValueError("Weather is incomplete for OrderStartTime production dates")

    features = []
    failures = []
    for _, row in source.iterrows():
        try:
            feature = extract_features_for_row(row)
            feature["PalletID"] = str(row["PalletID"]).strip()
            for column in WEATHER_COLUMNS:
                feature[column] = row[column]
            for column in static_columns:
                feature[column] = row[column]
            features.append(feature)
        except Exception as exc:
            failures.append({
                "OrderID": str(order_id), "BatchNumber": int(row["BatchNumber"]),
                "error_type": type(exc).__name__, "error": str(exc)[:500],
            })
    if failures:
        raise ValueError(f"Incremental Phase extraction failed: {failures[:5]}")

    result = apply_corrected_phase_contract(pd.DataFrame(features))
    result = result.drop(columns=LAB_COLUMNS, errors="ignore")
    result["LabelGroupID"] = result["OrderID"].astype(str) + "|" + result["PalletID"].astype(str)
    model_columns = _model_columns(metadata_path)
    missing_columns = sorted(set(model_columns) - set(result.columns))
    missing_cells = int(result[model_columns].isna().sum().sum()) if model_columns and not missing_columns else None
    if missing_columns or (missing_cells is not None and missing_cells > 0):
        raise ValueError({"missing_model_columns": missing_columns, "missing_model_cells": missing_cells})
    audit = {
        "order_id": str(order_id),
        "batch_rows": int(len(result)),
        "pallet_groups": int(result["LabelGroupID"].nunique()),
        "weather_missing_cells": int(result[WEATHER_COLUMNS].isna().sum().sum()),
        "model_feature_count": int(len(model_columns)),
        "missing_model_columns": missing_columns,
        "missing_model_cells": missing_cells,
        "material_contract_version": "MATERIAL-TAXONOMY-HF-V2",
        "phase_contract_version": CONTRACT_VERSION,
        "lab_fields_used": False,
        "automatic_control": "DISABLED",
    }
    return result, audit
