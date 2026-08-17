"""Reproduce the original Phase/V3.7 Pallet-group prediction contract.

One Lab MNY label belongs to an OrderID+PalletID group containing one or more
process batches.  The canonical V3.7 audit predicts every batch with the
original Phase feature pipeline and averages predictions within the label
group.  A mean-feature comparison is also reported because the older group
base model aggregated Phase features before prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Mooney_Prediction_Pipeline.feature_engineering.cb_dispersion_feature_builder import build_cb_dispersion_features
from Mooney_Prediction_Pipeline.feature_engineering.clustering import cluster_silica_carbon_black
from Mooney_Prediction_Pipeline.feature_engineering.silica_pid_feature_builder import build_silica_pid_features
from Mooney_Prediction_Pipeline.model_training.label_group_handler import (
    add_label_group_information,
    audit_label_group_conflicts,
)
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "stage_statistics_enriched_all_features_weather_v4.csv"
PACKAGE = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v37_production_model_package"
OUT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"
COMPOUND = "M1-T09170WS-- 02 004"
WEATHER_COLUMNS = [
    "env_temp_mean", "env_temp_min", "env_temp_max",
    "env_humidity_mean", "env_humidity_min", "env_humidity_max",
]


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    y = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)),
        "within_2_mny_rate": float(np.mean(np.abs(y - p) <= 2.0)),
        "within_3_mny_rate": float(np.mean(np.abs(y - p) <= 3.0)),
        "bias": float(np.mean(p - y)),
    }


def _prepare_original_phase_features(candidate: pd.DataFrame, predictor: V37RawPredictor):
    work = candidate.copy()
    pid = build_silica_pid_features(work)
    cb = build_cb_dispersion_features(work)
    for column in pid.columns:
        work[column] = pid[column]
    for column in cb.columns:
        work[column] = cb[column]
    work = cluster_silica_carbon_black(work)
    work = add_label_group_information(work)

    missing_columns = sorted(set(predictor.feature_columns) - set(work.columns))
    raw_missing_cells = {
        column: int(pd.to_numeric(work[column], errors="coerce").isna().sum())
        for column in predictor.feature_columns
        if column in work.columns
    }
    for column in predictor.feature_columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
    return work, missing_columns, raw_missing_cells


def _predict_group_mean_features(work: pd.DataFrame, predictor: V37RawPredictor) -> pd.DataFrame:
    numeric = work.groupby("_label_group_id", as_index=False)[predictor.feature_columns].mean()
    context = work.groupby("_label_group_id", as_index=False).agg(
        CompoundName=("CompoundName", "first"),
        material_system=("material_system", "first"),
        actual_mny=("MNY", "first"),
    )
    group_features = context.merge(numeric, on="_label_group_id", how="inner", validate="one_to_one")
    prediction = predictor.predict_raw(group_features)
    group_features["prediction_from_mean_features"] = prediction.final_prediction
    return group_features[["_label_group_id", "prediction_from_mean_features"]]


def _group_predictions(work: pd.DataFrame, batch_predictions: np.ndarray) -> pd.DataFrame:
    result = work[["_label_group_id", "OrderID", "PalletID", "BatchNumber", "MNY", "OrderStartTime"]].copy()
    result["baseline_batch_prediction"] = batch_predictions
    result["OrderStartTime"] = pd.to_datetime(result["OrderStartTime"], errors="coerce")
    return result.groupby("_label_group_id", as_index=False).agg(
        OrderID=("OrderID", "first"),
        PalletID=("PalletID", "first"),
        batch_count=("BatchNumber", "size"),
        distinct_batch_count=("BatchNumber", "nunique"),
        actual_mny=("MNY", "first"),
        actual_mny_nunique=("MNY", "nunique"),
        order_start_time=("OrderStartTime", "first"),
        baseline_group_prediction=("baseline_batch_prediction", "mean"),
        batch_prediction_std=("baseline_batch_prediction", "std"),
        batch_prediction_min=("baseline_batch_prediction", "min"),
        batch_prediction_max=("baseline_batch_prediction", "max"),
    )


def _run_replay(work: pd.DataFrame, predictor: V37RawPredictor, baseline: np.ndarray) -> pd.DataFrame:
    rows = []
    base = work[["_label_group_id"]].copy()
    base["baseline"] = baseline
    base_group = base.groupby("_label_group_id")["baseline"].mean()
    for parameter, feature, deltas in (
        ("Stage6_duration", "Stage6_BottomMixing_Duration", (-10.0, 10.0)),
        ("discharge_temperature", "phys_discharge_temp", (-1.0, 1.0)),
    ):
        for delta in deltas:
            candidate = work.copy()
            candidate[feature] = pd.to_numeric(candidate[feature], errors="coerce") + delta
            prediction = predictor.predict_raw(candidate).final_prediction
            arm = candidate[["_label_group_id"]].copy()
            arm["candidate"] = prediction
            group_candidate = arm.groupby("_label_group_id")["candidate"].mean()
            aligned = pd.concat([base_group, group_candidate], axis=1).dropna()
            response = aligned["candidate"] - aligned["baseline"]
            abs_median = float(response.abs().median())
            path_ok = bool(response.abs().max() > 1e-12)
            rows.append({
                "parameter": parameter,
                "phase_feature": feature,
                "delta": delta,
                "direction": "DOWN" if delta < 0 else "UP",
                "label_group_count": int(len(aligned)),
                "baseline_mny_median": float(aligned["baseline"].median()),
                "candidate_mny_median": float(aligned["candidate"].median()),
                "response_median": float(response.median()),
                "response_abs_median": abs_median,
                "response_p10": float(response.quantile(0.10)),
                "response_p90": float(response.quantile(0.90)),
                "nonzero_response_rate": float((response.abs() > 1e-12).mean()),
                "model_path_status": "MODEL_PATH_OK" if path_ok else "NO_MODEL_PATH",
                "minimum_response_status": "PASS" if abs_median >= 0.25 else "REJECTED_MINIMUM_MOONEY_RESPONSE",
                "family_reference_spec_prediction_pass_rate": float(aligned["candidate"].between(45.0, 77.0).mean()),
                "specification_status": "FAMILY_SPEC_INHERITANCE_UNCONFIRMED",
                "process_transfer_status": "DIAGNOSTIC_DIRECT_PHASE_PERTURBATION",
                "thermal_budget_status": "NOT_EVALUATED_DIRECT_PHASE_REPLAY",
                "temperature_cap_status": "NOT_EVALUATED_DIRECT_PHASE_REPLAY",
                "final_status": "DIAGNOSTIC_ONLY",
            })
    return pd.DataFrame(rows)


def main() -> None:
    source = pd.read_csv(DATA, low_memory=False)
    candidate = source[source["CompoundName"].astype(str).eq(COMPOUND)].copy()
    candidate["MNY"] = pd.to_numeric(candidate["MNY"], errors="coerce")
    candidate = candidate.dropna(subset=["MNY", "OrderID", "PalletID", "BatchNumber"]).copy()
    predictor = V37RawPredictor(PACKAGE).load()
    work, missing_columns, raw_missing_cells = _prepare_original_phase_features(candidate, predictor)
    if missing_columns:
        raise ValueError(f"Original Phase data is missing V3.7 artifact columns: {missing_columns}")

    first = predictor.predict_raw(work).final_prediction
    second = predictor.predict_raw(work).final_prediction
    deterministic_max_abs_delta = float(np.max(np.abs(first - second)))
    group = _group_predictions(work, first)
    mean_features = _predict_group_mean_features(work, predictor)
    group = group.merge(mean_features, on="_label_group_id", how="left", validate="one_to_one")
    replay = _run_replay(work, predictor, first)

    _, conflicts, _ = audit_label_group_conflicts(work, mny_col="MNY")
    batch_output = work[["_label_group_id", "OrderID", "PalletID", "BatchNumber", "MNY", "OrderStartTime"]].copy()
    batch_output["baseline_batch_prediction"] = first

    ordered_groups = group.sort_values("order_start_time", kind="stable").reset_index(drop=True)
    tail_start = max(0, int(len(ordered_groups) * 0.85))
    tail = ordered_groups.iloc[tail_start:]
    weather_missing = {column: int(candidate[column].isna().sum()) for column in WEATHER_COLUMNS}
    summary = {
        "compound_name": COMPOUND,
        "source": str(DATA),
        "batch_rows": int(len(work)),
        "label_groups": int(group["_label_group_id"].nunique()),
        "average_batches_per_label_group": float(len(work) / group["_label_group_id"].nunique()),
        "label_group_conflicts": int(len(conflicts)),
        "weather_missing_by_column": weather_missing,
        "weather_path_status": "PASS" if sum(weather_missing.values()) == 0 else "BLOCKED_MISSING_WEATHER",
        "v37_feature_count": len(predictor.feature_columns),
        "missing_v37_columns": missing_columns,
        "raw_missing_feature_cells_total": int(sum(raw_missing_cells.values())),
        "raw_missing_feature_cells_by_column": {k: v for k, v in raw_missing_cells.items() if v},
        "finite_prediction_rate": float(np.isfinite(first).mean()),
        "deterministic_max_abs_delta": deterministic_max_abs_delta,
        "prediction_determinism_status": "PASS" if deterministic_max_abs_delta <= 1e-12 else "FAIL",
        "canonical_contract": "V3.7 predicts each Phase batch, then averages predictions within OrderID+PalletID label group",
        "canonical_group_metrics": _metrics(group["actual_mny"], group["baseline_group_prediction"]),
        "mean_feature_comparison_metrics": _metrics(group["actual_mny"], group["prediction_from_mean_features"]),
        "canonical_vs_mean_feature_prediction_mae": float(mean_absolute_error(group["baseline_group_prediction"], group["prediction_from_mean_features"])),
        "latest_15pct_monitoring_metrics_not_independent_oot": _metrics(tail["actual_mny"], tail["baseline_group_prediction"]),
        "model_path_status": "MODEL_PATH_OK" if not missing_columns and np.isfinite(first).all() else "BLOCKED_V37_INPUT_PATH",
        "automatic_control": "DISABLED",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    batch_path = OUT / "t09170ws_02004_phase_v37_batch_baseline_predictions.csv"
    group_path = OUT / "t09170ws_02004_phase_v37_group_baseline_predictions.csv"
    replay_path = OUT / "t09170ws_02004_phase_v37_group_bidirectional_replay.csv"
    summary_path = OUT / "t09170ws_02004_phase_v37_group_reproduction_summary.json"
    batch_output.to_csv(batch_path, index=False, encoding="utf-8-sig")
    group.to_csv(group_path, index=False, encoding="utf-8-sig")
    replay.to_csv(replay_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(replay.to_string(index=False))
    print(f"batch={batch_path}")
    print(f"group={group_path}")
    print(f"replay={replay_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
