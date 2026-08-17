"""Compare corrected MMS static inputs with the historical Phase rows.

This is a diagnostic only. It does not write predictions to SQL and it never
uses laboratory values as model input.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Mooney_Prediction_Pipeline"))
sys.path.insert(0, str(ROOT / "Mooney_Prediction_Pipeline" / "model_analysis"))

from Mooney_Prediction_Pipeline.feature_engineering.cb_dispersion_feature_builder import build_cb_dispersion_features
from Mooney_Prediction_Pipeline.feature_engineering.clustering import cluster_silica_carbon_black
from Mooney_Prediction_Pipeline.feature_engineering.silica_pid_feature_builder import build_silica_pid_features
from Mooney_Prediction_Pipeline.model_analysis.predict_latest_sfe_order import extract_features_for_row
from Mooney_Prediction_Pipeline.model_analysis.predict_order_v3 import query_order_rows
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor
from Mooney_Prediction_Pipeline.data_processing.canonical_mms_source_v2 import build_corrected_static_features


PHASE = ROOT / "data" / "stage_statistics_enriched_all_features_weather_v4.csv"
PACKAGE = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v37_production_model_package"
OUT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow" / "mms_source_contract_v2"
WEATHER = ["env_temp_mean", "env_temp_min", "env_temp_max", "env_humidity_mean", "env_humidity_min", "env_humidity_max"]


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for derived in (build_silica_pid_features(work), build_cb_dispersion_features(work)):
        for column in derived.columns:
            work[column] = derived[column]
    return cluster_silica_carbon_black(work)


def main() -> None:
    phase = pd.read_csv(PHASE, low_memory=False)
    phase = phase.loc[phase["OrderID"].astype(str).eq("2016836")].copy()
    static = build_corrected_static_features("2016836")
    source = query_order_rows("2016836").copy()
    source["OrderID"] = source["OrderID"].astype(str)
    source["BatchNumber"] = pd.to_numeric(source["BatchNumber"], errors="coerce").astype(int)
    source = source.drop(columns=[column for column in static.columns if column in source.columns and column not in ["OrderID", "BatchNumber"]], errors="ignore")
    source = source.merge(static, on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one")

    operational_rows = []
    for _, row in source.iterrows():
        feature = extract_features_for_row(row)
        feature.update(static.loc[static["BatchNumber"] == int(row["BatchNumber"])].iloc[0].to_dict())
        operational_rows.append(feature)
    operational = pd.DataFrame(operational_rows)
    operational = operational.merge(
        phase[["OrderID", "BatchNumber", "PalletID", *WEATHER]],
        on=["OrderID", "BatchNumber"], how="left", validate="one_to_one",
    )
    operational = add_derived(operational)

    reference = phase.copy()
    reference = reference.merge(static, on=["OrderID", "BatchNumber"], how="inner", suffixes=("", "__corrected"), validate="one_to_one")
    for column in static.columns:
        if column not in {"OrderID", "BatchNumber", "MaterialContractVersion"}:
            reference[column] = reference[f"{column}__corrected"]
            reference.drop(columns=[f"{column}__corrected"], inplace=True)
    reference = add_derived(reference)

    predictor = V37RawPredictor(PACKAGE).load()
    for frame in (reference, operational):
        for column in predictor.feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    phase_prediction = predictor.predict_raw(reference).final_prediction
    operational_prediction = predictor.predict_raw(operational).final_prediction
    keys = ["OrderID", "BatchNumber"]
    feature_rows = []
    for feature in predictor.feature_columns:
        delta = pd.to_numeric(operational[feature], errors="coerce") - pd.to_numeric(reference[feature], errors="coerce")
        feature_rows.append({
            "feature": feature, "reference_mean": float(reference[feature].mean()),
            "operational_mean": float(operational[feature].mean()),
            "mean_delta": float(delta.mean()), "mean_abs_delta": float(delta.abs().mean()),
            "max_abs_delta": float(delta.abs().max()),
        })
    drift = pd.DataFrame(feature_rows).sort_values("mean_abs_delta", ascending=False)
    prediction = operational[keys].copy()
    prediction["reference_prediction"] = phase_prediction
    prediction["operational_prediction"] = operational_prediction
    prediction["prediction_delta"] = operational_prediction - phase_prediction
    OUT.mkdir(parents=True, exist_ok=True)
    drift.to_csv(OUT / "order_2016836_corrected_operational_feature_drift.csv", index=False)
    prediction.to_csv(OUT / "order_2016836_corrected_operational_prediction_drift.csv", index=False)
    summary = {
        "order_id": "2016836", "common_batches": int(len(prediction)),
        "phase_feature_missing_cells": int(reference[predictor.feature_columns].isna().sum().sum()),
        "operational_feature_missing_cells": int(operational[predictor.feature_columns].isna().sum().sum()),
        "max_prediction_abs_delta": float(np.abs(prediction["prediction_delta"]).max()),
        "mean_prediction_abs_delta": float(np.abs(prediction["prediction_delta"]).mean()),
        "prediction_gate_status": "PASS" if float(np.abs(prediction["prediction_delta"]).max()) <= 0.1 else "BLOCKED_PREDICTION_DRIFT",
        "stage6_ram_candidate": {
            "reference_mean": float(reference["Stage6_BottomMixing_WayofRam_Mean"].mean()),
            "operational_mean": float(operational["Stage6_BottomMixing_WayofRam_Mean"].mean()),
        },
        "automatic_control": "DISABLED",
    }
    with (OUT / "order_2016836_corrected_operational_parity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
