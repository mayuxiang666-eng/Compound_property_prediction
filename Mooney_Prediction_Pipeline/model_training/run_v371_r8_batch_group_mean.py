"""R8 T15760 experiment: predict physical batches, score Lab-group means.

Each Lab/Pallet group contributes total training weight 1.0 regardless of its
number of physical batches.  Candidate selection uses strict validation group
MAE/trend; OOT is evaluated only after selection.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.r5_order_state_features import (
    build_group_target_features,
)
from Mooney_Prediction_Pipeline.model_training.run_v371_r6_recipe_delta import (
    BASE,
    GROUP_SOURCE,
    PHASE_SOURCE,
    T15760_NAME,
    level_metrics,
    trend_metrics,
)


METADATA_SOURCE = (
    ROOT
    / "Mooney_Prediction_Pipeline"
    / "models"
    / "v371_challenger_r3_expert_ablation"
    / "feature_metadata.json"
)
R6_PREDICTIONS = BASE / "v371_r6_recipe_delta" / "r6_group_predictions.csv"
OUTPUT = BASE / "v371_r8_batch_group_mean"
MODEL_OUTPUT = (
    ROOT
    / "Mooney_Prediction_Pipeline"
    / "models"
    / "v371_challenger_r8_batch_group_mean"
)


def model(objective, leaves, min_child):
    return LGBMRegressor(
        objective=objective,
        n_estimators=350,
        learning_rate=0.02,
        num_leaves=leaves,
        min_child_samples=min_child,
        colsample_bytree=0.75,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=4.0,
        reg_lambda=15.0,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


def aggregate_prediction(batch, prediction):
    work = batch[["LabelGroupID"]].copy()
    work["Prediction"] = prediction
    return work.groupby("LabelGroupID", as_index=False)["Prediction"].mean()


def candidate_score(group_frame, prediction):
    frame = group_frame.copy()
    frame["Prediction"] = prediction
    val = frame.loc[frame["split"].eq("val")]
    level = level_metrics(val, "Prediction")
    trend = trend_metrics(val, "Prediction")
    correlation = trend["delta_pearson"] if pd.notna(trend["delta_pearson"]) else -1.0
    direction = trend["direction_accuracy"] if pd.notna(trend["direction_accuracy"]) else 0.0
    ratio = trend["delta_std_ratio"] if pd.notna(trend["delta_std_ratio"]) else 0.0
    score = float(level["mae"])
    score += 1.5 * max(0.0, 0.50 - float(correlation))
    score += 1.5 * max(0.0, 0.70 - float(direction))
    score += 0.75 * max(0.0, 0.70 - float(ratio))
    score += 0.75 * max(0.0, float(ratio) - 1.30)
    return score, level, trend


def select_features(batch_train, group_train, pool, target, top_k, mandatory):
    means = batch_train.groupby("LabelGroupID")[pool].mean(numeric_only=True)
    audit = group_train.set_index("LabelGroupID")[[target]].join(means, how="inner")
    scored = []
    for feature in pool:
        values = pd.to_numeric(audit[feature], errors="coerce")
        valid = values.notna() & audit[target].notna()
        if int(valid.sum()) < 40 or values.loc[valid].nunique() < 3:
            continue
        correlation = values.loc[valid].corr(audit.loc[valid, target], method="spearman")
        scored.append((abs(float(correlation)) if pd.notna(correlation) else 0.0, feature))
    ranked = [name for _, name in sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]]
    return list(dict.fromkeys([name for name in mandatory if name in pool] + ranked))


def metric_record(frame, prediction_col, split, candidate):
    part = frame.loc[frame["split"].eq(split)]
    return {
        "candidate": candidate,
        "split": split,
        **level_metrics(part, prediction_col),
        **trend_metrics(part, prediction_col),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(METADATA_SOURCE.read_text(encoding="utf-8"))
    group_source = pd.read_csv(GROUP_SOURCE, low_memory=False)
    r6 = pd.read_csv(
        R6_PREDICTIONS,
        usecols=[
            "LabelGroupID",
            "split",
            "BasePrediction",
            "LevelPrediction",
        ],
        low_memory=False,
    ).rename(columns={"LevelPrediction": "R6LevelPrediction"})
    groups = group_source.loc[
        group_source["CompoundName"].astype(str).eq(T15760_NAME),
        [
            "LabelGroupID",
            "OrderID",
            "PalletID",
            "RecipeID",
            "ProcessVersion",
            "CompoundName",
            "lab_time",
            "MNY",
        ],
    ].merge(r6, on="LabelGroupID", how="inner", validate="one_to_one")

    phase = pd.read_csv(PHASE_SOURCE, low_memory=False)
    target_contract = build_group_target_features(
        phase[
            [
                "LabelGroupID",
                "TargetVectorRaw",
                "TargetVectorHash",
                "StepPatternHash",
                "TemplateKey",
                "BatchStartTime",
                "BatchEndTime",
            ]
        ]
    )
    target_features = sorted(target_contract.feature_map)
    target_frame = target_contract.frame[["LabelGroupID"] + target_features]
    batch = phase.loc[phase["CompoundName"].astype(str).eq(T15760_NAME)].copy()
    batch = batch.merge(target_frame, on="LabelGroupID", how="inner", validate="many_to_one")
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
        raise AssertionError("R8 Lab/Pallet groups do not have equal total weight")

    feature_pool = list(
        dict.fromkeys(
            ["BasePrediction", "BatchWeight"]
            + metadata["s1_features"]
            + metadata["s2_features"]
            + target_features
        )
    )
    feature_pool = [feature for feature in feature_pool if feature in batch.columns]
    for feature in feature_pool:
        batch[feature] = pd.to_numeric(batch[feature], errors="coerce")
    mandatory = [
        "BasePrediction",
        "BatchWeight",
        "supplier_rubber_viscosity_avg",
        "supplier_silica_moisture_avg",
        "supplier_silica_surface_area_avg",
        "ProcessVersion",
    ]
    batch_train = batch.loc[batch["split"].eq("train")].copy()
    group_train = groups.loc[groups["split"].eq("train")].copy()
    candidates = {}
    models = {}
    feature_sets = {}
    search_rows = []
    for mode in ("residual", "direct"):
        batch_train["TrainingTarget"] = (
            batch_train["MNY"] - batch_train["BasePrediction"]
            if mode == "residual"
            else batch_train["MNY"]
        )
        group_train["TrainingTarget"] = (
            group_train["MNY"] - group_train["BasePrediction"]
            if mode == "residual"
            else group_train["MNY"]
        )
        for top_k in (15, 30, 60, 100):
            features = select_features(
                batch_train,
                group_train,
                feature_pool,
                "TrainingTarget",
                top_k,
                mandatory,
            )
            for objective in ("regression_l1", "huber"):
                for leaves, min_child in ((3, 25), (5, 35), (8, 50)):
                    name = f"R8_{mode.upper()}_K{top_k}_{objective.upper()}_L{leaves}"
                    estimator = model(objective, leaves, min_child)
                    estimator.fit(
                        batch_train[features],
                        batch_train["TrainingTarget"],
                        sample_weight=batch_train["GroupEqualWeight"],
                    )
                    raw_batch = estimator.predict(batch[features])
                    if mode == "residual":
                        raw_group = aggregate_prediction(batch, raw_batch).rename(
                            columns={"Prediction": "RawCorrection"}
                        )
                    else:
                        raw_group = aggregate_prediction(batch, raw_batch).rename(
                            columns={"Prediction": "RawLevel"}
                        )
                    models[name] = estimator
                    feature_sets[name] = features
                    for weight in ((0.50, 0.75, 1.00, 1.25) if mode == "residual" else (0.75, 1.00)):
                        temp = groups[["LabelGroupID", "BasePrediction"]].merge(
                            raw_group, on="LabelGroupID", how="left", validate="one_to_one"
                        )
                        if mode == "residual":
                            prediction = temp["BasePrediction"].to_numpy() + weight * temp["RawCorrection"].to_numpy()
                        else:
                            prediction = weight * temp["RawLevel"].to_numpy() + (1.0 - weight) * temp["BasePrediction"].to_numpy()
                        candidate = f"{name}_W{int(weight * 100):03d}"
                        score, level, trend = candidate_score(groups, prediction)
                        candidates[candidate] = prediction
                        search_rows.append(
                            {"candidate": candidate, "score": score, **level, **trend}
                        )
    search = pd.DataFrame(search_rows)
    selected = search.sort_values(["score", "mae", "candidate"], kind="stable").iloc[0]["candidate"]
    model_name, weight_text = selected.rsplit("_W", 1)
    groups["R8Prediction"] = candidates[selected]
    records = []
    for column, candidate in (
        ("BasePrediction", "R6_BASE"),
        ("R6LevelPrediction", "R6_LEVEL"),
        ("R8Prediction", "R8_SELECTED"),
    ):
        for split in ("val", "oot"):
            records.append(metric_record(groups, column, split, candidate))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "r8_metrics.csv", index=False)
    search.to_csv(OUTPUT / "r8_validation_search.csv", index=False)
    groups.to_csv(OUTPUT / "r8_t15760_group_predictions.csv", index=False)

    estimator = models[model_name]
    features = feature_sets[model_name]
    pd.DataFrame(
        {"feature": features, "importance": estimator.feature_importances_}
    ).sort_values(["importance", "feature"], ascending=[False, True]).to_csv(
        OUTPUT / "r8_feature_importance.csv", index=False
    )
    bundle = {
        "contract": "V3.7.1-R8-BATCH-PREDICT-GROUP-MEAN-V1",
        "candidate": selected,
        "model": estimator,
        "features": features,
        "weight": int(weight_text) / 100.0,
        "lab_group_training_total_weight": 1.0,
        "evaluation": "MEAN_BATCH_PREDICTION_PER_ORDERID_PALLETID",
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
    }
    joblib.dump(bundle, MODEL_OUTPUT / "r8_batch_group_mean_shadow.joblib")
    val = metrics.loc[
        metrics["candidate"].eq("R8_SELECTED") & metrics["split"].eq("val")
    ].iloc[0]
    oot = metrics.loc[
        metrics["candidate"].eq("R8_SELECTED") & metrics["split"].eq("oot")
    ].iloc[0]
    summary = {
        "contract": bundle["contract"],
        "selected": selected,
        "batch_rows": int(len(batch)),
        "lab_groups": int(groups["LabelGroupID"].nunique()),
        "feature_count": len(features),
        "group_weight_min": float(weight_audit.min()),
        "group_weight_max": float(weight_audit.max()),
        "validation": val.to_dict(),
        "oot": oot.to_dict(),
        "target_mae_mny": 2.5,
        "release_status": "NOT_RELEASED_GATES_FAILED_OR_BLIND_WINDOW_PENDING",
        "stage3_calibration_enabled": False,
        "local_mms_access": False,
        "database_access_during_run": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "automatic_control": "DISABLED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "r8_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
