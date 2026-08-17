"""Reproduce the original V3.7 leak-free split and report T09170 test metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Mooney_Prediction_Pipeline.feature_engineering.cb_dispersion_feature_builder import build_cb_dispersion_features
from Mooney_Prediction_Pipeline.feature_engineering.clustering import cluster_silica_carbon_black
from Mooney_Prediction_Pipeline.feature_engineering.silica_pid_feature_builder import build_silica_pid_features
from Mooney_Prediction_Pipeline.feature_engineering.stage1_recipe_features import extract_stage1_recipe_features
from Mooney_Prediction_Pipeline.feature_engineering.stage2_process_features import extract_stage2_process_features
from Mooney_Prediction_Pipeline.model_training.effective_weighting import compute_effective_sample_weights
from Mooney_Prediction_Pipeline.model_training.hybrid_unified_model import HybridUnifiedMooneyModel
from Mooney_Prediction_Pipeline.model_training.label_group_handler import add_label_group_information
from Mooney_Prediction_Pipeline.model_training.split_builder import generate_stratified_recipe_splits
from Mooney_Prediction_Pipeline.model_training.stage3_online_calibration import Stage3DelayedFeedbackCalibrator


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "stage_statistics_enriched_all_features_weather_v4.csv"
OUT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"
COMPOUND = "M1-T09170WS-- 02 004"


def _metrics(actual, predicted):
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)) if len(y) > 1 else None,
        "within_2_mny_rate": float(np.mean(np.abs(y - p) <= 2.0)),
        "within_3_mny_rate": float(np.mean(np.abs(y - p) <= 3.0)),
        "bias": float(np.mean(p - y)),
    }


def _group_metrics(frame, prediction_column):
    grouped = frame.groupby("_label_group_id", as_index=False).agg(
        actual_mny=("MNY", "first"),
        predicted_mny=(prediction_column, "mean"),
        batch_count=("BatchNumber", "size"),
        actual_mny_nunique=("MNY", "nunique"),
    )
    return grouped, _metrics(grouped["actual_mny"], grouped["predicted_mny"])


def main() -> None:
    df = pd.read_csv(DATA, low_memory=False)
    df["MNY"] = pd.to_numeric(df["MNY"], errors="coerce")
    df = df.dropna(subset=["MNY"]).copy()
    df = df.sort_values(["OrderID"], kind="stable").reset_index(drop=True)

    pid = build_silica_pid_features(df)
    cb = build_cb_dispersion_features(df)
    for column in pid.columns:
        df[column] = pid[column]
    for column in cb.columns:
        df[column] = cb[column]
    df = cluster_silica_carbon_black(df)
    df = add_label_group_information(df)
    df = compute_effective_sample_weights(df)
    df = generate_stratified_recipe_splits(df, test_size=0.15, val_size=0.15, random_state=42)

    s1 = extract_stage1_recipe_features(df)
    s2 = list(set(extract_stage2_process_features(df) + list(pid.columns) + list(cb.columns)))
    for column in set(s1 + s2):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    train = df[df["_split"].eq("train")].copy()
    test = df[df["_split"].eq("test")].copy().reset_index(drop=True)
    model = HybridUnifiedMooneyModel(
        use_material_route_matrix=True,
        use_silica_subsystem=True,
        use_cb_subsystem=True,
    )
    model.fit(train, s1, s2, target_col="MNY", cluster_col="material_system")
    raw, _, _, _ = model.predict(test, cluster_col="material_system")
    calibrated, _ = Stage3DelayedFeedbackCalibrator(
        lag_k=3, alpha_base=0.3, adaptive=True,
    ).calibrate_time_series(test, raw, target_col="MNY", group_col="CompoundName")
    test["raw_prediction"] = raw
    test["calibrated_prediction"] = calibrated
    candidate = test[test["CompoundName"].astype(str).eq(COMPOUND)].copy()

    raw_groups, raw_metrics = _group_metrics(candidate, "raw_prediction")
    calibrated_groups, calibrated_metrics = _group_metrics(candidate, "calibrated_prediction")
    output = raw_groups.rename(columns={"predicted_mny": "raw_group_prediction"}).merge(
        calibrated_groups[["_label_group_id", "predicted_mny"]].rename(columns={"predicted_mny": "calibrated_group_prediction"}),
        on="_label_group_id", how="inner", validate="one_to_one",
    )
    summary = {
        "compound_name": COMPOUND,
        "split_contract": "Original V3.7 recipe/label-group leak-free split, random_state=42",
        "train_batches": int(len(train)),
        "test_batches_all": int(len(test)),
        "candidate_test_batches": int(len(candidate)),
        "candidate_test_label_groups": int(candidate["_label_group_id"].nunique()),
        "candidate_test_recipe_codes": int(candidate["_recipe_code"].nunique()),
        "label_group_overlap_train_test": int(len(set(train["_label_group_id"]) & set(candidate["_label_group_id"]))),
        "recipe_code_overlap_train_test": int(len(set(train["_recipe_code"]) & set(candidate["_recipe_code"]))),
        "raw_group_metrics": raw_metrics,
        "delayed_calibrated_group_metrics": calibrated_metrics,
        "status": "PASS_LEAKFREE_TEST_PATH" if len(candidate) and not (set(train["_label_group_id"]) & set(candidate["_label_group_id"])) else "BLOCKED_SPLIT_OR_COVERAGE",
        "automatic_control": "DISABLED",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "t09170ws_02004_v37_leakfree_test_group_predictions.csv"
    json_path = OUT / "t09170ws_02004_v37_leakfree_test_summary.json"
    output.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"predictions={csv_path}")
    print(f"summary={json_path}")


if __name__ == "__main__":
    main()
