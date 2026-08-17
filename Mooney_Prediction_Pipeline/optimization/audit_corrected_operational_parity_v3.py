"""Strict same-batch corrected Phase/MMS parity audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.model_analysis.predict_latest_sfe_order import extract_features_for_row
from Mooney_Prediction_Pipeline.model_analysis.predict_order_v3 import query_order_rows
from Mooney_Prediction_Pipeline.optimization.audit_corrected_operational_parity_v2 import (
    OUT,
    PACKAGE,
    PHASE,
    WEATHER,
    add_derived,
)
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor
from Mooney_Prediction_Pipeline.data_processing.canonical_mms_source_v2 import build_corrected_static_features


def main() -> None:
    phase = pd.read_csv(PHASE, low_memory=False)
    phase = phase.loc[phase["OrderID"].astype(str).eq("2016836")].copy()
    phase["OrderID"] = phase["OrderID"].astype(str)
    phase["BatchNumber"] = pd.to_numeric(phase["BatchNumber"], errors="coerce").astype(int)
    keys = phase[["OrderID", "BatchNumber"]].drop_duplicates()

    static = build_corrected_static_features("2016836")
    static["OrderID"] = static["OrderID"].astype(str)
    static["BatchNumber"] = pd.to_numeric(static["BatchNumber"], errors="coerce").astype(int)
    source = query_order_rows("2016836").copy()
    source["OrderID"] = source["OrderID"].astype(str)
    source["BatchNumber"] = pd.to_numeric(source["BatchNumber"], errors="coerce").astype(int)
    source = source.merge(keys, on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one")
    corrected_columns = [column for column in static.columns if column not in {"OrderID", "BatchNumber"}]
    source = source.drop(columns=[column for column in corrected_columns if column in source], errors="ignore")
    source = source.merge(static, on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one")

    rows = []
    for _, row in source.iterrows():
        feature = extract_features_for_row(row)
        feature.update(static.loc[static["BatchNumber"].eq(int(row["BatchNumber"]))].iloc[0].to_dict())
        rows.append(feature)
    operational = pd.DataFrame(rows)
    operational["OrderID"] = operational["OrderID"].astype(str)
    operational["BatchNumber"] = pd.to_numeric(operational["BatchNumber"], errors="coerce").astype(int)
    operational = operational.merge(
        phase[["OrderID", "BatchNumber", "PalletID", *WEATHER]],
        on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one",
    ).sort_values(["OrderID", "BatchNumber"]).reset_index(drop=True)
    operational = add_derived(operational)

    reference = phase.merge(
        static, on=["OrderID", "BatchNumber"], how="inner", suffixes=("", "__corrected"),
        validate="one_to_one",
    ).sort_values(["OrderID", "BatchNumber"]).reset_index(drop=True)
    for column in corrected_columns:
        corrected = f"{column}__corrected"
        if corrected in reference:
            reference[column] = reference[corrected]
            reference.drop(columns=[corrected], inplace=True)
    reference = add_derived(reference)

    predictor = V37RawPredictor(PACKAGE).load()
    raw_missing = {
        "reference": int(reference[predictor.feature_columns].isna().sum().sum()),
        "operational": int(operational[predictor.feature_columns].isna().sum().sum()),
    }
    for frame in (reference, operational):
        for column in predictor.feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    ref_prediction = predictor.predict_raw(reference).final_prediction
    op_prediction = predictor.predict_raw(operational).final_prediction

    drift_rows = []
    for feature in predictor.feature_columns:
        delta = operational[feature] - reference[feature]
        drift_rows.append({
            "feature": feature,
            "reference_mean": float(reference[feature].mean()),
            "operational_mean": float(operational[feature].mean()),
            "mean_delta": float(delta.mean()),
            "mean_abs_delta": float(delta.abs().mean()),
            "max_abs_delta": float(delta.abs().max()),
        })
    drift = pd.DataFrame(drift_rows).sort_values("mean_abs_delta", ascending=False)
    prediction = operational[["OrderID", "BatchNumber", "PalletID"]].copy()
    prediction["reference_prediction"] = ref_prediction
    prediction["operational_prediction"] = op_prediction
    prediction["prediction_delta"] = op_prediction - ref_prediction
    OUT.mkdir(parents=True, exist_ok=True)
    drift.to_csv(OUT / "order_2016836_corrected_operational_feature_drift_v3.csv", index=False)
    prediction.to_csv(OUT / "order_2016836_corrected_operational_prediction_drift_v3.csv", index=False)
    max_delta = float(prediction["prediction_delta"].abs().max())
    summary = {
        "order_id": "2016836",
        "same_batch_count": int(len(prediction)),
        "same_batch_numbers": prediction["BatchNumber"].astype(int).tolist(),
        "raw_missing_feature_cells": raw_missing,
        "mean_prediction_abs_delta": float(prediction["prediction_delta"].abs().mean()),
        "max_prediction_abs_delta": max_delta,
        "prediction_tolerance_mny": 0.1,
        "prediction_gate_status": "PARITY_PASS" if max_delta <= 0.1 else "BLOCKED_PREDICTION_DRIFT",
        "stage6_ram_reference_mean": float(reference["Stage6_BottomMixing_WayofRam_Mean"].mean()),
        "stage6_ram_operational_mean": float(operational["Stage6_BottomMixing_WayofRam_Mean"].mean()),
        "automatic_control": "DISABLED",
    }
    with (OUT / "order_2016836_corrected_operational_parity_summary_v3.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
