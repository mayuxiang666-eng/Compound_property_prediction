"""R12 direct residual expert with causal Charge-state ablation.

The base level model remains the corrected R11E model.  Residual targets are
formed only from expanding-time cross-fitted base predictions, never from the
in-sample base fit.  Candidate selection uses strict within-OrderID trends.
"""

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
from Mooney_Prediction_Pipeline.model_training.order_atomic_trend_metrics import candidate_score, metric_record
from Mooney_Prediction_Pipeline.model_training.run_v371_r11e_checkpointed_order_model import estimator


BASE = core.BASE
SOURCE = BASE / "v371_r11e_checkpointed_order_model"
LEVEL_MODEL = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r11e_checkpointed_order_model" / "r11e_level_checkpoint.joblib"
CHARGE_SOURCE = BASE / "t15760_causal_charge_state_v2" / "t15760_causal_charge_features_v2.csv"
OUTPUT = BASE / "v371_r12_causal_residual_expert"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r12_causal_residual_expert"


def expanding_oof_level(train: pd.DataFrame, features: list[str], amplitude: float) -> pd.Series:
    ordered = train.sort_values(["production_start", "OrderID", "PalletID", "LabelGroupID"], kind="stable")
    n = len(ordered)
    boundaries = [0, int(n * 0.40), int(n * 0.60), int(n * 0.80), n]
    output = pd.Series(np.nan, index=train.index, dtype=float)
    for fold in range(1, 4):
        fit_index = ordered.index[: boundaries[fold]]
        predict_index = ordered.index[boundaries[fold] : boundaries[fold + 1]]
        if len(predict_index) == 0:
            continue
        fitted = estimator("regression_l1", 3, 18, 140)
        fitted.fit(train.loc[fit_index, features], train.loc[fit_index, "MNY"])
        fold_center = float(train.loc[fit_index, "MNY"].mean())
        raw = fitted.predict(train.loc[predict_index, features])
        output.loc[predict_index] = fold_center + amplitude * (raw - fold_center)
    return output


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    groups = pd.read_pickle(SOURCE / "r11e_level_checkpoint.pkl").copy()
    level_bundle = joblib.load(LEVEL_MODEL)
    base = level_bundle["bundle"]
    level_features = list(base["features"])
    amplitude = float(base["amplitude"])
    charge = pd.read_csv(CHARGE_SOURCE, low_memory=False)
    charge_features = [column for column in charge.columns if column.startswith("charge_state__")]
    groups = groups.merge(charge, on="LabelGroupID", how="left", validate="one_to_one")
    for column in charge_features:
        groups[column] = pd.to_numeric(groups[column], errors="coerce")

    train_mask = groups["split"].eq("train")
    oof = expanding_oof_level(groups.loc[train_mask], level_features, amplitude)
    groups["OOFLevelPrediction"] = np.nan
    groups.loc[oof.index, "OOFLevelPrediction"] = oof
    groups["ResidualTarget"] = groups["MNY"] - groups["OOFLevelPrediction"]
    residual_train = groups.loc[train_mask & groups["ResidualTarget"].notna()].copy()
    print(f"OOF_RESIDUAL groups={len(residual_train)} charge_features={len(charge_features)} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    identifiers = {"LabelGroupID", "OrderID", "PalletID", "RecipeID", "CompoundName", "split", "regime", "MNY", "LevelPrediction", "OOFLevelPrediction", "ResidualTarget", "production_start", "lab_time"}
    process_pool = []
    for column in groups.columns:
        if column in identifiers or column in charge_features:
            continue
        values = pd.to_numeric(groups[column], errors="coerce")
        if values.notna().sum() >= 35 and values.nunique(dropna=True) >= 2:
            groups[column] = values
            process_pool.append(column)
    process_selected = core.select_features(residual_train, process_pool, "ResidualTarget", 30, [])
    charge_usable = [column for column in charge_features if residual_train[column].notna().sum() >= 35 and residual_train[column].nunique(dropna=True) >= 2]
    combined_pool = list(dict.fromkeys(process_pool + charge_usable))
    combined_selected = core.select_features(residual_train, combined_pool, "ResidualTarget", 35, charge_usable)

    candidates: dict[str, np.ndarray] = {"R12_BASE_LEVEL": groups["LevelPrediction"].to_numpy(copy=True)}
    rows: list[dict] = []
    models: dict[str, dict] = {}
    for family, features in (("PROCESS_ONLY", process_selected), ("PROCESS_CHARGE", combined_selected)):
        fitted = estimator("regression_l1", 3, 12, 140)
        fitted.fit(residual_train[features], residual_train["ResidualTarget"])
        residual_prediction = fitted.predict(groups[features])
        models[family] = {"model": fitted, "features": features}
        for gamma in (0.0, 0.5, 0.75, 1.0, 1.25):
            prediction = groups["LevelPrediction"].to_numpy(copy=True) + gamma * residual_prediction
            name = f"R12_{family}_G{int(gamma*100):03d}"
            score, level, trend = candidate_score(groups, prediction)
            candidates[name] = prediction
            rows.append({"candidate": name, "family": family, "gamma": gamma, "score": score, **level, **trend})
        pd.DataFrame(rows).to_csv(OUTPUT / "r12_validation_search.csv", index=False)
        print(f"RESIDUAL_FIT family={family} features={len(features)} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    search = pd.DataFrame(rows).sort_values(["score", "mae", "candidate"], kind="stable")
    selected_name = str(search.iloc[0]["candidate"])
    groups["FinalPrediction"] = candidates[selected_name]
    selected_family = str(search.iloc[0]["family"])
    selected_gamma = float(search.iloc[0]["gamma"])

    records = []
    for split_name in ("val", "oot"):
        records.append(metric_record(groups, "LevelPrediction", split_name, "R11E_LEVEL_CLEAN"))
        records.append(metric_record(groups, "FinalPrediction", split_name, "R12_FINAL"))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "r12_metrics.csv", index=False)
    pd.DataFrame({"feature": process_selected}).to_csv(OUTPUT / "r12_process_residual_features.csv", index=False)
    pd.DataFrame({"feature": combined_selected, "is_charge": [feature in charge_features for feature in combined_selected]}).to_csv(OUTPUT / "r12_process_charge_residual_features.csv", index=False)
    groups[["LabelGroupID", "OrderID", "PalletID", "RecipeID", "split", "regime", "MNY", "LevelPrediction", "OOFLevelPrediction", "ResidualTarget", "FinalPrediction", "production_start", "lab_time"]].to_csv(OUTPUT / "r12_group_predictions.csv", index=False)
    bundle = {
        "contract": "V3.7.1-R12-CAUSAL-RESIDUAL-EXPERT-V1",
        "level": level_bundle,
        "selected_family": selected_family,
        "residual": models[selected_family],
        "gamma": selected_gamma,
        "charge_contract": "only Lab events with lab_time < production_start",
        "trend_evaluation": "WITHIN_ORDERID_ONLY",
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
    }
    joblib.dump(bundle, MODEL_OUTPUT / "r12_causal_residual_shadow.joblib")
    val = metrics.loc[(metrics["candidate"] == "R12_FINAL") & (metrics["split"] == "val")].iloc[0]
    oot = metrics.loc[(metrics["candidate"] == "R12_FINAL") & (metrics["split"] == "oot")].iloc[0]
    summary = {
        "contract": bundle["contract"], "groups": len(groups), "oof_residual_groups": len(residual_train),
        "selected_candidate": selected_name, "selected_family": selected_family, "selected_gamma": selected_gamma,
        "validation": val.to_dict(), "oot": oot.to_dict(), "target_mae_mny": 2.5,
        "target_trend": {"delta_pearson_min": 0.5, "direction_accuracy_min": 0.65, "delta_std_ratio_range": [0.7, 1.3]},
        "release_status": "NOT_RELEASED_GATES_FAILED_OR_BLIND_WINDOW_PENDING",
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED", "elapsed_seconds": time.perf_counter() - started,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "r12_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
