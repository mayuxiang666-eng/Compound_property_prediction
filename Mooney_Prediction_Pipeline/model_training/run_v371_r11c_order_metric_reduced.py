"""Reduced R11 search selected strictly by within-OrderID Lab/Pallet trends."""

from __future__ import annotations

import json
import sys
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
from Mooney_Prediction_Pipeline.model_training.run_v371_r8_batch_group_mean import GROUP_SOURCE, METADATA_SOURCE, R6_PREDICTIONS, T15760_NAME


OUTPUT = core.BASE / "v371_r11c_order_metric_reduced"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r11c_order_metric_reduced"


def prepare():
    metadata = json.loads(METADATA_SOURCE.read_text(encoding="utf-8"))
    source = pd.read_csv(GROUP_SOURCE, low_memory=False)
    split = pd.read_csv(R6_PREDICTIONS, usecols=["LabelGroupID", "split"], low_memory=False)
    groups = source.loc[source["CompoundName"].astype(str).eq(T15760_NAME)].drop(columns=["split"], errors="ignore").merge(split, on="LabelGroupID", how="inner", validate="one_to_one")
    corrected = core.aggregate_corrected(pd.read_csv(core.CORRECTED_SOURCE, low_memory=False))
    step = pd.read_csv(core.STEP_GROUP_SOURCE, low_memory=False)
    step_columns = [column for column in step.columns if column.startswith("step_exec__") or column in {"LabelGroupID", "physical_batch_count", "abnormal_batch_count", "abnormal_batch_rate", "max_tail_step11_duration", "missing_bottom_batch_count"}]
    step = step[step_columns]
    groups = groups.merge(corrected, on="LabelGroupID", how="inner", validate="one_to_one").merge(step, on="LabelGroupID", how="left", validate="one_to_one")
    groups["production_start"] = pd.to_datetime(groups["production_start"], errors="coerce")
    groups["lab_time"] = pd.to_datetime(groups["lab_time"], errors="coerce")
    groups["ProcessVersion"] = pd.to_numeric(groups["ProcessVersion"], errors="coerce")
    groups["regime_upset"] = groups["regime"].eq("PROCESS_UPSET").astype(float)
    groups["regime_incomplete"] = groups["regime"].eq("DATA_INCOMPLETE").astype(float)
    static = [feature for feature in metadata["s1_features"] if feature in groups]
    corrected_features = [column for column in corrected if column not in {"LabelGroupID", "regime", "production_start"}]
    step_features = [column for column in step if column != "LabelGroupID"]
    pool = list(dict.fromkeys(["ProcessVersion", "regime_upset", "regime_incomplete"] + static + corrected_features + step_features))
    usable = []
    for feature in pool:
        groups[feature] = pd.to_numeric(groups[feature], errors="coerce")
        if groups[feature].notna().sum() >= 35 and groups[feature].nunique(dropna=True) >= 2:
            usable.append(feature)
    mandatory = ["ProcessVersion", "regime_upset", "regime_incomplete", "physical_batch_count", "abnormal_batch_rate", "missing_bottom_batch_count", "max_tail_step11_duration", "supplier_rubber_viscosity_avg", "supplier_silica_moisture_avg", "supplier_silica_surface_area_avg"]
    return groups, usable, mandatory


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    groups, usable, mandatory = prepare()
    train = groups.loc[groups["split"].eq("train")]
    center = float(train["MNY"].mean())
    candidates, bundles, rows = {}, {}, []
    for top_k in (30, 60, 100):
        features = core.select_features(train, usable, "MNY", top_k, mandatory)
        for objective in ("regression_l1", "huber"):
            for leaves, min_child in ((3, 18), (5, 25)):
                fitted = core.estimator(objective, leaves, min_child)
                fitted.fit(train[features], train["MNY"])
                raw = fitted.predict(groups[features])
                for amplitude in (1.0, 1.25, 1.5, 1.75):
                    prediction = center + amplitude * (raw - center)
                    name = f"R11C_K{top_k}_{objective.upper()}_L{leaves}_A{int(amplitude*100):03d}"
                    score, level, trend = candidate_score(groups, prediction)
                    candidates[name] = prediction
                    bundles[name] = {"model": fitted, "features": features, "amplitude": amplitude, "center": center}
                    rows.append({"candidate": name, "score": score, **level, **trend})
    search = pd.DataFrame(rows)
    level_name = search.sort_values(["score", "mae", "candidate"], kind="stable").iloc[0]["candidate"]
    groups["LevelPrediction"] = candidates[level_name]
    features = bundles[level_name]["features"]

    pairs = core.build_pairs(groups, features)
    train_pairs = pairs.loc[pairs["split"].eq("train")]
    delta_pool = [column for column in pairs if column.startswith("delta__")]
    delta_candidates, delta_bundles, delta_rows = {}, {}, []
    for top_k in (10, 15, 25):
        selected = core.select_features(train_pairs, delta_pool, "ActualDelta", top_k, [])
        for leaves, min_child in ((3, 5), (5, 8)):
            fitted = core.estimator("regression_l1", leaves, min_child)
            fitted.set_params(n_estimators=250, learning_rate=0.02)
            fitted.fit(train_pairs[selected], train_pairs["ActualDelta"])
            pair_prediction = pairs[["from_group", "to_group", "OrderID", "split", "ActualDelta"]].copy()
            pair_prediction["DeltaPrediction"] = fitted.predict(pairs[selected])
            for beta in (0.0, 0.25, 0.50, 0.75, 1.0):
                prediction = core.recursive_prediction(groups, groups["LevelPrediction"].to_numpy(), pair_prediction, beta)
                name = f"R11C_DELTA_K{top_k}_L{leaves}_B{int(beta*100):03d}"
                score, level, trend = candidate_score(groups, prediction)
                delta_candidates[name] = prediction
                delta_bundles[name] = {"model": fitted, "features": selected, "beta": beta, "pair_predictions": pair_prediction}
                delta_rows.append({"candidate": name, "score": score, **level, **trend})
    delta_search = pd.DataFrame(delta_rows)
    delta_name = delta_search.sort_values(["score", "mae", "candidate"], kind="stable").iloc[0]["candidate"]
    groups["FinalPrediction"] = delta_candidates[delta_name]

    records = []
    for column, name in (("LevelPrediction", "R11C_LEVEL"), ("FinalPrediction", "R11C_FINAL")):
        for split_name in ("val", "oot"):
            records.append(metric_record(groups, column, split_name, name))
            for regime in ("NORMAL_COMPLETE_ROUTE", "PROCESS_UPSET"):
                record = metric_record(groups.loc[groups["regime"].eq(regime)], column, split_name, f"{name}_{regime}")
                record["regime"] = regime
                records.append(record)
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "r11c_metrics.csv", index=False)
    search.to_csv(OUTPUT / "r11c_level_validation_search.csv", index=False)
    delta_search.to_csv(OUTPUT / "r11c_delta_validation_search.csv", index=False)
    groups[["LabelGroupID", "OrderID", "PalletID", "RecipeID", "split", "regime", "MNY", "LevelPrediction", "FinalPrediction", "production_start", "lab_time"]].to_csv(OUTPUT / "r11c_group_predictions.csv", index=False)
    delta_bundles[delta_name]["pair_predictions"].to_csv(OUTPUT / "r11c_pair_predictions.csv", index=False)
    bundle = {"contract": "V3.7.1-R11C-STEP-BOUNDARY-ORDER-DELTA-V1", "level_candidate": level_name, "level": bundles[level_name], "delta_candidate": delta_name, "delta": {k:v for k,v in delta_bundles[delta_name].items() if k != "pair_predictions"}, "trend_evaluation": "WITHIN_ORDERID_ONLY", "stage3_calibration_enabled": False, "automatic_control": "DISABLED"}
    joblib.dump(bundle, MODEL_OUTPUT / "r11c_step_order_delta_shadow.joblib")
    val = metrics.loc[metrics["candidate"].eq("R11C_FINAL") & metrics["split"].eq("val")].iloc[0]
    oot = metrics.loc[metrics["candidate"].eq("R11C_FINAL") & metrics["split"].eq("oot")].iloc[0]
    summary = {"contract": bundle["contract"], "groups": len(groups), "regime_counts": groups["regime"].value_counts().to_dict(), "selected_level": level_name, "selected_delta": delta_name, "validation": val.to_dict(), "oot": oot.to_dict(), "target_mae_mny": 2.5, "target_trend": {"delta_pearson_min": 0.5, "direction_accuracy_min": 0.65, "delta_std_ratio_range": [0.7,1.3]}, "release_status": "NOT_RELEASED_GATES_FAILED_OR_BLIND_WINDOW_PENDING", "database_access_during_run": False, "local_mms_access": False, "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED", "generated_utc": datetime.now(timezone.utc).isoformat()}
    (OUTPUT / "r11c_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
