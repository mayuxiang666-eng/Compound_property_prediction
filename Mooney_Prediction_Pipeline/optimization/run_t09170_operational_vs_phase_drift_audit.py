"""Compare live no-Lab extraction against original Phase rows for the same batches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.feature_engineering.cb_dispersion_feature_builder import build_cb_dispersion_features
from Mooney_Prediction_Pipeline.feature_engineering.clustering import cluster_silica_carbon_black
from Mooney_Prediction_Pipeline.feature_engineering.silica_pid_feature_builder import build_silica_pid_features
from Mooney_Prediction_Pipeline.optimization.v37_group_feature_path import build_v37_group_feature_path
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "stage_statistics_enriched_all_features_weather_v4.csv"
PACKAGE = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v37_production_model_package"
OUT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"
ORDER_ID = "2016836"


def _prepare_phase(predictor):
    phase = pd.read_csv(DATA, low_memory=False)
    phase = phase[phase["OrderID"].astype(str).eq(ORDER_ID)].copy()
    pid = build_silica_pid_features(phase)
    cb = build_cb_dispersion_features(phase)
    for column in pid.columns:
        phase[column] = pid[column]
    for column in cb.columns:
        phase[column] = cb[column]
    phase = cluster_silica_carbon_black(phase)
    for column in predictor.feature_columns:
        phase[column] = pd.to_numeric(phase[column], errors="coerce").fillna(0.0)
    return phase


def main() -> None:
    predictor = V37RawPredictor(PACKAGE).load()
    operational_result = build_v37_group_feature_path(ORDER_ID, predictor)
    operational = operational_result.batch_features.copy()
    phase = _prepare_phase(predictor)
    operational["BatchNumber"] = pd.to_numeric(operational["BatchNumber"], errors="coerce")
    phase["BatchNumber"] = pd.to_numeric(phase["BatchNumber"], errors="coerce")
    common = sorted(set(operational["BatchNumber"]) & set(phase["BatchNumber"]))
    op = operational[operational["BatchNumber"].isin(common)].set_index("BatchNumber").sort_index()
    ph = phase[phase["BatchNumber"].isin(common)].drop_duplicates("BatchNumber").set_index("BatchNumber").sort_index()
    op = op.loc[ph.index]

    rows = []
    for feature in predictor.feature_columns:
        left = pd.to_numeric(ph[feature], errors="coerce").fillna(0.0)
        right = pd.to_numeric(op[feature], errors="coerce").fillna(0.0)
        delta = right - left
        rows.append({
            "feature": feature,
            "phase_mean": float(left.mean()),
            "operational_mean": float(right.mean()),
            "mean_delta": float(delta.mean()),
            "mean_abs_delta": float(delta.abs().mean()),
            "max_abs_delta": float(delta.abs().max()),
            "same_rate": float(np.isclose(left, right, rtol=1e-9, atol=1e-9).mean()),
        })
    drift = pd.DataFrame(rows).sort_values(["mean_abs_delta", "max_abs_delta"], ascending=False)

    phase_pred = predictor.predict_raw(ph.reset_index()).final_prediction
    op_pred = predictor.predict_raw(op.reset_index()).final_prediction
    matched = pd.DataFrame({
        "OrderID": ORDER_ID,
        "BatchNumber": ph.index,
        "PalletID": phase.drop_duplicates("BatchNumber").set_index("BatchNumber").loc[ph.index, "PalletID"].astype(str),
        "actual_mny": phase.drop_duplicates("BatchNumber").set_index("BatchNumber").loc[ph.index, "MNY"].astype(float),
        "phase_feature_prediction": phase_pred,
        "operational_feature_prediction": op_pred,
        "prediction_delta": op_pred - phase_pred,
    })
    summary = {
        "order_id": ORDER_ID,
        "matched_batches": len(common),
        "phase_prediction_mean": float(np.mean(phase_pred)),
        "operational_prediction_mean": float(np.mean(op_pred)),
        "prediction_mean_delta": float(np.mean(op_pred - phase_pred)),
        "prediction_mae_between_paths": float(np.mean(np.abs(op_pred - phase_pred))),
        "features_with_any_drift": int((drift["same_rate"] < 1.0).sum()),
        "top_20_drift_features": drift.head(20).to_dict(orient="records"),
        "status": "PASS_FEATURE_PARITY" if np.allclose(phase_pred, op_pred, rtol=1e-9, atol=1e-9) else "BLOCKED_FEATURE_PARITY_DRIFT",
        "automatic_control": "DISABLED",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    drift_path = OUT / "t09170ws_02004_operational_vs_phase_feature_drift.csv"
    matched_path = OUT / "t09170ws_02004_operational_vs_phase_prediction_drift.csv"
    summary_path = OUT / "t09170ws_02004_operational_vs_phase_drift_summary.json"
    drift.to_csv(drift_path, index=False, encoding="utf-8-sig")
    matched.to_csv(matched_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"drift={drift_path}")
    print(f"predictions={matched_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
