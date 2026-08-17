"""R13 residual expert with an allowlisted process contract and leakage gate."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training import run_v371_r11_step_gated_delta as core
from Mooney_Prediction_Pipeline.model_training.feature_leakage_guard import assert_no_target_leakage
from Mooney_Prediction_Pipeline.model_training.order_atomic_trend_metrics import candidate_score, metric_record
from Mooney_Prediction_Pipeline.model_training.run_v371_r11e_checkpointed_order_model import estimator


BASE = core.BASE
SOURCE = BASE / "v371_r11e_checkpointed_order_model"
R12_OOF = BASE / "v371_r12_causal_residual_expert" / "r12_group_predictions.csv"
LEVEL_MODEL = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r11e_checkpointed_order_model" / "r11e_level_checkpoint.joblib"
CHARGE_SOURCE = BASE / "t15760_causal_charge_state_v2" / "t15760_causal_charge_features_v2.csv"
OUTPUT = BASE / "v371_r13_leakage_blocked_residual"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r13_leakage_blocked_residual"


PROCESS_EXACT = {
    "ProcessVersion", "regime_upset", "regime_incomplete", "physical_batch_count",
    "abnormal_batch_count", "abnormal_batch_rate", "max_tail_step11_duration",
    "missing_bottom_batch_count", "supplier_rubber_viscosity_avg",
    "supplier_silica_moisture_avg", "supplier_silica_surface_area_avg",
}


def is_process_feature(column: str) -> bool:
    return column.startswith("step_phase__") or column.startswith("step_exec__") or column.startswith("Stage") or column in PROCESS_EXACT


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    groups = pd.read_pickle(SOURCE / "r11e_level_checkpoint.pkl").copy()
    level_bundle = joblib.load(LEVEL_MODEL)
    assert_no_target_leakage(groups, list(level_bundle["bundle"]["features"]))
    oof = pd.read_csv(R12_OOF, usecols=["LabelGroupID", "OOFLevelPrediction"], low_memory=False)
    groups = groups.merge(oof, on="LabelGroupID", how="left", validate="one_to_one")
    groups["ResidualTarget"] = groups["MNY"] - groups["OOFLevelPrediction"]
    charge = pd.read_csv(CHARGE_SOURCE, low_memory=False)
    charge_features = [column for column in charge if column.startswith("charge_state__")]
    groups = groups.merge(charge, on="LabelGroupID", how="left", validate="one_to_one")

    process_pool = []
    for column in groups.columns:
        if not is_process_feature(column):
            continue
        values = pd.to_numeric(groups[column], errors="coerce")
        if values.notna().sum() >= 35 and values.nunique(dropna=True) >= 2:
            groups[column] = values
            process_pool.append(column)
    for column in charge_features:
        groups[column] = pd.to_numeric(groups[column], errors="coerce")
    assert_no_target_leakage(groups, process_pool + charge_features)

    train = groups.loc[groups["split"].eq("train") & groups["ResidualTarget"].notna()].copy()
    process_selected = core.select_features(train, process_pool, "ResidualTarget", 30, [])
    charge_usable = [column for column in charge_features if train[column].notna().sum() >= 35 and train[column].nunique(dropna=True) >= 2]
    combined_pool = list(dict.fromkeys(process_pool + charge_usable))
    combined_selected = core.select_features(train, combined_pool, "ResidualTarget", 35, charge_usable)
    assert_no_target_leakage(groups, process_selected + combined_selected)
    print(f"LEAKAGE_GATE_OK residual_train={len(train)} process_pool={len(process_pool)}", flush=True)

    candidates, rows, models = {}, [], {}
    baseline = groups["LevelPrediction"].to_numpy(copy=True)
    for family, features in (("PROCESS_ONLY", process_selected), ("PROCESS_CHARGE", combined_selected)):
        fitted = estimator("regression_l1", 3, 12, 140)
        fitted.fit(train[features], train["ResidualTarget"])
        residual = fitted.predict(groups[features])
        models[family] = {"model": fitted, "features": features}
        for gamma in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5):
            prediction = baseline + gamma * residual
            name = f"R13_{family}_G{int(gamma*100):03d}"
            score, level, trend = candidate_score(groups, prediction)
            candidates[name] = prediction
            rows.append({"candidate": name, "family": family, "gamma": gamma, "score": score, **level, **trend})
        pd.DataFrame(rows).to_csv(OUTPUT / "r13_validation_search.csv", index=False)
        print(f"FIT {family} features={len(features)} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    search = pd.DataFrame(rows).sort_values(["score", "mae", "candidate"], kind="stable")
    selected_name = str(search.iloc[0]["candidate"])
    selected_family = str(search.iloc[0]["family"])
    selected_gamma = float(search.iloc[0]["gamma"])
    groups["FinalPrediction"] = candidates[selected_name]
    records = []
    for split_name in ("val", "oot"):
        records.append(metric_record(groups, "LevelPrediction", split_name, "R11E_LEVEL_CLEAN"))
        records.append(metric_record(groups, "FinalPrediction", split_name, "R13_FINAL"))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "r13_metrics.csv", index=False)
    pd.DataFrame({"feature": process_selected}).to_csv(OUTPUT / "r13_process_features.csv", index=False)
    pd.DataFrame({"feature": combined_selected, "is_charge": [f in charge_features for f in combined_selected]}).to_csv(OUTPUT / "r13_process_charge_features.csv", index=False)
    groups[["LabelGroupID", "OrderID", "PalletID", "RecipeID", "split", "regime", "MNY", "LevelPrediction", "OOFLevelPrediction", "ResidualTarget", "FinalPrediction", "production_start", "lab_time"]].to_csv(OUTPUT / "r13_group_predictions.csv", index=False)
    bundle = {
        "contract": "V3.7.1-R13-LEAKAGE-BLOCKED-RESIDUAL-V1", "level": level_bundle,
        "selected_family": selected_family, "residual": models[selected_family], "gamma": selected_gamma,
        "feature_allowlist": "Step/Phase execution, explicit regime/QM, optional causal Charge only",
        "target_leakage_gate": "PASSED", "trend_evaluation": "WITHIN_ORDERID_ONLY",
        "stage3_calibration_enabled": False, "automatic_control": "DISABLED",
    }
    joblib.dump(bundle, MODEL_OUTPUT / "r13_leakage_blocked_residual_shadow.joblib")
    val = metrics.loc[(metrics["candidate"] == "R13_FINAL") & (metrics["split"] == "val")].iloc[0]
    oot = metrics.loc[(metrics["candidate"] == "R13_FINAL") & (metrics["split"] == "oot")].iloc[0]
    summary = {
        "contract": bundle["contract"], "groups": len(groups), "oof_residual_groups": len(train),
        "selected_candidate": selected_name, "selected_family": selected_family, "selected_gamma": selected_gamma,
        "validation": val.to_dict(), "oot": oot.to_dict(), "target_leakage_gate": "PASSED",
        "invalidated_predecessor": "R12 contained actual_mny in residual feature set",
        "target_mae_mny": 2.5,
        "target_trend": {"delta_pearson_min": 0.5, "direction_accuracy_min": 0.65, "delta_std_ratio_range": [0.7, 1.3]},
        "release_status": "NOT_RELEASED_GATES_FAILED_OR_BLIND_WINDOW_PENDING",
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED", "elapsed_seconds": time.perf_counter() - started,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "r13_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
