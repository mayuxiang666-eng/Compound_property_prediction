"""R10b: phase-aligned current-batch curve-state challenger for T15760."""

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

from Mooney_Prediction_Pipeline.model_training.r10b_phase_curve_features import (
    extract_phase_curve_row,
)
from Mooney_Prediction_Pipeline.model_training.run_v371_r10_curve_state import (
    load_t15760_curves,
    normalize_id,
)
from Mooney_Prediction_Pipeline.model_training.run_v371_r8_batch_group_mean import (
    BASE,
    GROUP_SOURCE,
    METADATA_SOURCE,
    PHASE_SOURCE,
    R6_PREDICTIONS,
    T15760_NAME,
    aggregate_prediction,
    candidate_score,
    metric_record,
    model,
    select_features,
)


OUTPUT = BASE / "v371_r10b_phase_curve_state"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r10b_phase_curve_state"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(METADATA_SOURCE.read_text(encoding="utf-8"))
    group_source = pd.read_csv(GROUP_SOURCE, low_memory=False)
    r6 = pd.read_csv(
        R6_PREDICTIONS,
        usecols=["LabelGroupID", "split", "BasePrediction", "LevelPrediction"],
        low_memory=False,
    ).rename(columns={"LevelPrediction": "R6LevelPrediction"})
    groups = group_source.loc[
        group_source["CompoundName"].astype(str).eq(T15760_NAME),
        ["LabelGroupID", "OrderID", "PalletID", "RecipeID", "ProcessVersion", "CompoundName", "lab_time", "MNY"],
    ].merge(r6, on="LabelGroupID", how="inner", validate="one_to_one")

    phase = pd.read_csv(PHASE_SOURCE, low_memory=False)
    batch = phase.loc[phase["CompoundName"].astype(str).eq(T15760_NAME)].copy()
    batch["OrderID"] = batch["OrderID"].map(normalize_id)
    batch["BatchNumber"] = batch["BatchNumber"].map(normalize_id)
    required_keys = set(zip(batch["OrderID"], batch["BatchNumber"]))
    curves = load_t15760_curves(required_keys)
    source = batch.merge(curves, on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one")
    extracted, blocked = [], []
    for _, row in source.iterrows():
        try:
            extracted.append(extract_phase_curve_row(row, points=8))
        except ValueError as exc:
            blocked.append({"OrderID": row["OrderID"], "BatchNumber": row["BatchNumber"], "reason": str(exc)})
    curve_features = pd.DataFrame(extracted)
    if blocked:
        pd.DataFrame(blocked).to_csv(OUTPUT / "r10b_blocked_batches.csv", index=False)
    if len(curve_features) != len(required_keys):
        raise AssertionError(f"R10b coverage incomplete: expected {len(required_keys)}, extracted {len(curve_features)}")

    feature_columns = [
        column
        for column in curve_features.columns
        if column.startswith("phase_curve__") and column != "phase_curve__boundary_status"
    ]
    batch = batch.merge(curve_features, on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one")
    batch = batch.merge(
        groups[["LabelGroupID", "split", "MNY", "BasePrediction"]],
        on="LabelGroupID",
        how="inner",
        validate="many_to_one",
    )
    batch["BatchGroupSize"] = batch.groupby("LabelGroupID")["LabelGroupID"].transform("size")
    batch["GroupEqualWeight"] = 1.0 / batch["BatchGroupSize"]
    weight_audit = batch.groupby("LabelGroupID")["GroupEqualWeight"].sum()
    if not np.allclose(weight_audit.to_numpy(dtype=float), 1.0):
        raise AssertionError("R10b Lab/Pallet group total weights are not one")

    base_pool = list(dict.fromkeys(["BasePrediction", "BatchWeight"] + metadata["s1_features"] + metadata["s2_features"]))
    feature_pool = list(dict.fromkeys([feature for feature in base_pool if feature in batch.columns] + feature_columns))
    for feature in feature_pool:
        batch[feature] = pd.to_numeric(batch[feature], errors="coerce")
    mandatory = [
        "BasePrediction",
        "BatchWeight",
        "supplier_rubber_viscosity_avg",
        "supplier_silica_moisture_avg",
        "supplier_silica_surface_area_avg",
        "phase_curve__Stage2_DryMixing__temp__last",
        "phase_curve__Stage2_DryMixing__power__integral",
        "phase_curve__Stage4_WetMixing__temp__last",
        "phase_curve__Stage4_WetMixing__power__integral",
        "phase_curve__Stage5_PID__temp__last",
        "phase_curve__Stage5_PID__power__integral",
        "phase_curve__Stage6_BottomMixing__temp__last",
        "phase_curve__Stage6_BottomMixing__power__integral",
    ]
    batch_train = batch.loc[batch["split"].eq("train")].copy()
    group_train = groups.loc[groups["split"].eq("train")].copy()
    candidates, models, feature_sets, search_rows = {}, {}, {}, []
    for mode in ("residual", "direct"):
        batch_train["TrainingTarget"] = batch_train["MNY"] - batch_train["BasePrediction"] if mode == "residual" else batch_train["MNY"]
        group_train["TrainingTarget"] = group_train["MNY"] - group_train["BasePrediction"] if mode == "residual" else group_train["MNY"]
        for top_k in (30, 60, 100, 160, 220):
            features = select_features(batch_train, group_train, feature_pool, "TrainingTarget", top_k, mandatory)
            for objective in ("regression_l1", "huber"):
                for leaves, min_child in ((3, 25), (5, 35), (8, 50)):
                    name = f"R10B_{mode.upper()}_K{top_k}_{objective.upper()}_L{leaves}"
                    estimator = model(objective, leaves, min_child)
                    estimator.fit(batch_train[features], batch_train["TrainingTarget"], sample_weight=batch_train["GroupEqualWeight"])
                    raw = estimator.predict(batch[features])
                    raw_name = "RawCorrection" if mode == "residual" else "RawLevel"
                    raw_group = aggregate_prediction(batch, raw).rename(columns={"Prediction": raw_name})
                    models[name], feature_sets[name] = estimator, features
                    weights = (0.50, 0.75, 1.00, 1.25) if mode == "residual" else (0.75, 1.00)
                    for weight in weights:
                        temp = groups[["LabelGroupID", "BasePrediction"]].merge(raw_group, on="LabelGroupID", how="left", validate="one_to_one")
                        prediction = (
                            temp["BasePrediction"].to_numpy() + weight * temp[raw_name].to_numpy()
                            if mode == "residual"
                            else weight * temp[raw_name].to_numpy() + (1.0 - weight) * temp["BasePrediction"].to_numpy()
                        )
                        candidate = f"{name}_W{int(weight * 100):03d}"
                        total, level, trend = candidate_score(groups, prediction)
                        candidates[candidate] = prediction
                        search_rows.append({"candidate": candidate, "score": total, **level, **trend})

    search = pd.DataFrame(search_rows)
    selected = search.sort_values(["score", "mae", "candidate"], kind="stable").iloc[0]["candidate"]
    model_name, weight_text = selected.rsplit("_W", 1)
    groups["R10bPrediction"] = candidates[selected]
    records = []
    for column, candidate in (("BasePrediction", "R6_BASE"), ("R6LevelPrediction", "R6_LEVEL"), ("R10bPrediction", "R10B_SELECTED")):
        for split in ("val", "oot"):
            records.append(metric_record(groups, column, split, candidate))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "r10b_metrics.csv", index=False)
    search.to_csv(OUTPUT / "r10b_validation_search.csv", index=False)
    groups.to_csv(OUTPUT / "r10b_t15760_group_predictions.csv", index=False)
    curve_features.to_csv(OUTPUT / "r10b_t15760_batch_phase_curve_features.csv", index=False)

    estimator, features = models[model_name], feature_sets[model_name]
    pd.DataFrame({"feature": features, "importance": estimator.feature_importances_}).sort_values(
        ["importance", "feature"], ascending=[False, True]
    ).to_csv(OUTPUT / "r10b_feature_importance.csv", index=False)
    bundle = {
        "contract": "V3.7.1-R10B-PHASE-ALIGNED-CURRENT-BATCH-CURVE-V1",
        "candidate": selected,
        "model": estimator,
        "features": features,
        "weight": int(weight_text) / 100.0,
        "points_per_phase_signal": 8,
        "evaluation": "PHYSICAL_BATCH_PREDICTION_MEAN_PER_ORDERID_PALLETID",
        "boundary_fill": "CURRENT_BATCH_INTERPOLATION_ONLY",
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
    }
    joblib.dump(bundle, MODEL_OUTPUT / "r10b_phase_curve_state_shadow.joblib")
    val = metrics.loc[metrics["candidate"].eq("R10B_SELECTED") & metrics["split"].eq("val")].iloc[0]
    oot = metrics.loc[metrics["candidate"].eq("R10B_SELECTED") & metrics["split"].eq("oot")].iloc[0]
    summary = {
        "contract": bundle["contract"],
        "selected": selected,
        "curve_batches_expected": len(required_keys),
        "curve_batches_found": int(len(curve_features)),
        "phase_curve_feature_count": len(feature_columns),
        "selected_feature_count": len(features),
        "group_weight_min": float(weight_audit.min()),
        "group_weight_max": float(weight_audit.max()),
        "validation": val.to_dict(),
        "oot": oot.to_dict(),
        "target_mae_mny": 2.5,
        "target_trend": {"delta_pearson_min": 0.50, "direction_accuracy_min": 0.65, "delta_std_ratio_range": [0.70, 1.30]},
        "release_status": "NOT_RELEASED_GATES_FAILED_OR_BLIND_WINDOW_PENDING",
        "stage3_calibration_enabled": False,
        "local_mms_access": False,
        "database_access_during_run": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "automatic_control": "DISABLED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "r10b_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
