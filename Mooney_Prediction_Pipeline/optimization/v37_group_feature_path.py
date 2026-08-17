"""Operational no-Lab V3.7 path using the original Phase weather/group contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.feature_engineering.cb_dispersion_feature_builder import build_cb_dispersion_features
from Mooney_Prediction_Pipeline.feature_engineering.clustering import cluster_silica_carbon_black
from Mooney_Prediction_Pipeline.feature_engineering.silica_pid_feature_builder import build_silica_pid_features
from Mooney_Prediction_Pipeline.model_analysis.predict_latest_sfe_order import extract_features_for_row
from Mooney_Prediction_Pipeline.model_analysis.predict_order_v3 import query_order_rows
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor
from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection


ROOT = Path(__file__).resolve().parents[2]
PHASE_HISTORY = ROOT / "data" / "stage_statistics_enriched_all_features_weather_v4.csv"
WEATHER_COLUMNS = [
    "env_temp_mean", "env_temp_min", "env_temp_max",
    "env_humidity_mean", "env_humidity_min", "env_humidity_max",
]


@dataclass(frozen=True)
class V37GroupFeaturePathResult:
    batch_features: pd.DataFrame
    batch_predictions: pd.DataFrame
    group_predictions: pd.DataFrame
    audit: dict[str, Any]


def _cached_weather_by_date(order_times: pd.Series) -> pd.DataFrame:
    requested = pd.to_datetime(order_times, errors="coerce").dt.date
    if requested.isna().any():
        raise ValueError("OrderStartTime is required for weather alignment.")
    history = pd.read_csv(
        PHASE_HISTORY,
        usecols=["OrderStartTime", *WEATHER_COLUMNS],
        low_memory=False,
    )
    history["date"] = pd.to_datetime(history["OrderStartTime"], errors="coerce").dt.date
    weather = history.groupby("date", as_index=False)[WEATHER_COLUMNS].median()
    requested_frame = pd.DataFrame({"date": sorted(set(requested))})
    result = requested_frame.merge(weather, on="date", how="left", validate="one_to_one")
    if result[WEATHER_COLUMNS].isna().any(axis=None):
        missing_dates = result.loc[result[WEATHER_COLUMNS].isna().any(axis=1), "date"].astype(str).tolist()
        raise ValueError(
            "Weather is not available in the original Phase cache for production dates: "
            + ", ".join(missing_dates)
        )
    return result


def _query_pallet_map(order_id: str) -> pd.DataFrame:
    query = """
        SELECT CAST(OrderID AS VARCHAR(40)) AS OrderID, BatchNumber,
               CAST(PalletID AS VARCHAR(40)) AS PalletID
        FROM dbo.BatchHeader WITH (NOLOCK) WHERE OrderID = ?
        UNION ALL
        SELECT CAST(OrderID AS VARCHAR(40)) AS OrderID, BatchNumber,
               CAST(PalletID AS VARCHAR(40)) AS PalletID
        FROM SFEPLANT_ARCHIVE.dbo.BatchHeader WITH (NOLOCK) WHERE OrderID = ?
    """
    connection = get_sfe_connection()
    try:
        frame = pd.read_sql_query(query, connection, params=[str(order_id), str(order_id)])
    finally:
        connection.close()
    frame["OrderID"] = frame["OrderID"].astype(str).str.strip()
    frame["PalletID"] = frame["PalletID"].astype(str).str.strip()
    frame["BatchNumber"] = pd.to_numeric(frame["BatchNumber"], errors="coerce")
    return frame.dropna(subset=["BatchNumber", "PalletID"]).drop_duplicates(["OrderID", "BatchNumber"])


def build_v37_group_feature_path(
    order_id: str,
    predictor: V37RawPredictor,
) -> V37GroupFeaturePathResult:
    source = query_order_rows(str(order_id)).copy()
    if source.empty:
        raise ValueError(f"No complete curve source rows for OrderID={order_id}.")
    source["OrderID"] = source["OrderID"].astype(str).str.strip()
    source["BatchNumber"] = pd.to_numeric(source["BatchNumber"], errors="coerce")
    source["date"] = pd.to_datetime(source["OrderStartTime"], errors="coerce").dt.date
    weather = _cached_weather_by_date(source["OrderStartTime"])
    source = source.merge(weather, on="date", how="left", validate="many_to_one")
    source = source.merge(_query_pallet_map(str(order_id)), on=["OrderID", "BatchNumber"], how="left", validate="one_to_one")
    if source["PalletID"].isna().any():
        raise ValueError(f"PalletID mapping is incomplete for OrderID={order_id}.")
    if source[WEATHER_COLUMNS].isna().any(axis=None):
        raise ValueError(f"Weather alignment is incomplete for OrderID={order_id}.")

    rows = []
    failures = []
    for _, row in source.iterrows():
        try:
            features = extract_features_for_row(row)
            features["PalletID"] = row["PalletID"]
            rows.append(features)
        except Exception as exc:
            failures.append({
                "OrderID": str(order_id),
                "BatchNumber": int(row["BatchNumber"]),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            })
    if failures:
        raise ValueError(f"Feature extraction failed for {len(failures)} batch rows: {failures[:3]}")
    features = pd.DataFrame(rows)
    pid = build_silica_pid_features(features)
    cb = build_cb_dispersion_features(features)
    for column in pid.columns:
        features[column] = pid[column]
    for column in cb.columns:
        features[column] = cb[column]
    features = cluster_silica_carbon_black(features)

    missing_columns = sorted(set(predictor.feature_columns) - set(features.columns))
    if missing_columns:
        raise ValueError(f"V3.7 artifact columns missing after original Phase extraction: {missing_columns}")
    weather_missing = int(features[WEATHER_COLUMNS].isna().sum().sum())
    if weather_missing:
        raise ValueError(f"Weather contains {weather_missing} missing feature cells after extraction.")
    fill_counts = {}
    for column in predictor.feature_columns:
        values = pd.to_numeric(features[column], errors="coerce")
        fill_counts[column] = int(values.isna().sum())
        features[column] = values.fillna(0.0)

    prediction = predictor.predict_raw(features).final_prediction
    batch = features[["OrderID", "PalletID", "BatchNumber", "CompoundName", "OrderStartTime"]].copy()
    batch["predicted_mny_raw"] = prediction
    group = batch.groupby(["OrderID", "PalletID"], as_index=False).agg(
        batch_count=("BatchNumber", "size"),
        predicted_mny_group_mean=("predicted_mny_raw", "mean"),
        predicted_mny_batch_std=("predicted_mny_raw", "std"),
        first_batch=("BatchNumber", "min"),
        last_batch=("BatchNumber", "max"),
    )
    audit = {
        "order_id": str(order_id),
        "batch_rows": int(len(batch)),
        "pallet_groups": int(len(group)),
        "weather_source": str(PHASE_HISTORY),
        "weather_missing_cells": weather_missing,
        "missing_v37_columns": missing_columns,
        "training_compatible_zero_fill_cells": int(sum(fill_counts.values())),
        "training_compatible_zero_fill_by_column": {k: v for k, v in fill_counts.items() if v},
        "finite_prediction_rate": float(np.isfinite(prediction).mean()),
        "model_path_status": "MODEL_PATH_OK" if np.isfinite(prediction).all() else "BLOCKED_NONFINITE_PREDICTION",
        "lab_fields_used": False,
        "automatic_control": "DISABLED",
    }
    return V37GroupFeaturePathResult(features, batch, group, audit)
