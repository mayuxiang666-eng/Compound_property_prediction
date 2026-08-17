"""R7 T15760 expert: raw-material QM + current Phase + prior group state.

Selection is validation-only.  OOT is evaluated once after the candidate is
frozen.  No database access, Stage3 calibration, recommendation, or control.
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

from Mooney_Prediction_Pipeline.model_training.run_v371_r6_recipe_delta import (
    BASE,
    MODEL_OUTPUT as R6_MODEL_OUTPUT,
    T15760_NAME,
    level_metrics,
    prepare_frame,
    trend_metrics,
)


R6_PREDICTIONS = BASE / "v371_r6_recipe_delta" / "r6_group_predictions.csv"
OUTPUT = BASE / "v371_r7_qm_order_state"
MODEL_OUTPUT = (
    ROOT
    / "Mooney_Prediction_Pipeline"
    / "models"
    / "v371_challenger_r7_qm_order_state"
)


def select_features(train, pool, target, top_k, mandatory):
    scored = []
    for feature in pool:
        values = pd.to_numeric(train[feature], errors="coerce")
        valid = values.notna() & target.notna()
        if int(valid.sum()) < 40 or values.loc[valid].nunique() < 3:
            continue
        correlation = values.loc[valid].corr(target.loc[valid], method="spearman")
        scored.append((abs(float(correlation)) if pd.notna(correlation) else 0.0, feature))
    selected = [name for _, name in sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]]
    return list(dict.fromkeys([name for name in mandatory if name in pool] + selected))


def model(objective, leaves, min_child):
    return LGBMRegressor(
        objective=objective,
        n_estimators=300,
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


def validation_score(frame, prediction):
    part = frame.loc[frame["split"].eq("val")].copy()
    part["Prediction"] = prediction[part.index]
    level = level_metrics(part, "Prediction")
    trend = trend_metrics(part, "Prediction")
    correlation = trend["delta_pearson"] if pd.notna(trend["delta_pearson"]) else -1.0
    direction = trend["direction_accuracy"] if pd.notna(trend["direction_accuracy"]) else 0.0
    ratio = trend["delta_std_ratio"] if pd.notna(trend["delta_std_ratio"]) else 0.0
    score = float(level["mae"])
    score += 1.5 * max(0.0, 0.50 - float(correlation))
    score += 1.5 * max(0.0, 0.70 - float(direction))
    score += 0.75 * max(0.0, 0.70 - float(ratio))
    score += 0.75 * max(0.0, float(ratio) - 1.30)
    return score, level, trend


def metric_record(frame, prediction_col, split, candidate):
    part = frame.loc[frame["split"].eq(split)].copy()
    return {
        "candidate": candidate,
        "split": split,
        **level_metrics(part, prediction_col),
        **trend_metrics(part, prediction_col),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    groups, metadata, target_feature_map, lag_features, split_audit = prepare_frame()
    old = pd.read_csv(R6_PREDICTIONS, usecols=["LabelGroupID", "BasePrediction", "LevelPrediction"])
    old = old.rename(columns={"LevelPrediction": "R6LevelPrediction"})
    groups = groups.merge(old, on="LabelGroupID", how="inner", validate="one_to_one")
    tmask = groups["CompoundName"].astype(str).eq(T15760_NAME)
    t = groups.loc[tmask].copy().reset_index(drop=True)

    base_features = list(dict.fromkeys(metadata["s1_features"] + metadata["s2_features"]))
    dispersion = [
        f"{feature}{suffix}"
        for feature in base_features
        for suffix in ("__group_std", "__group_min", "__group_max")
        if f"{feature}{suffix}" in t.columns
    ]
    target_features = sorted(target_feature_map)
    pool = list(
        dict.fromkeys(
            [
                "BasePrediction",
                "ProcessVersion",
                "PhaseBatchCount",
                "batch_count",
                "prior_order_state_available",
                "prior_group_gap_hours",
                "order_group_index",
            ]
            + base_features
            + dispersion
            + target_features
            + lag_features
        )
    )
    pool = [feature for feature in pool if feature in t.columns]
    mandatory_qm = [
        feature
        for feature in metadata["s1_features"]
        if feature.startswith("supplier_") and not feature.endswith("__missing")
    ] + ["BasePrediction", "ProcessVersion", "prior_order_state_available", "order_group_index"]

    train = t.loc[t["split"].eq("train")].copy()
    candidates = {}
    validation_rows = []
    fitted = {}
    feature_sets = {}
    for target_mode in ("residual", "direct"):
        target = train["MNY"] - train["BasePrediction"] if target_mode == "residual" else train["MNY"]
        for top_k in (15, 30, 60, 100):
            features = select_features(train, pool, target, top_k, mandatory_qm)
            for objective in ("regression_l1", "huber"):
                for leaves, min_child in ((3, 20), (5, 25), (8, 35)):
                    model_name = f"R7_{target_mode.upper()}_K{top_k}_{objective.upper()}_L{leaves}"
                    estimator = model(objective, leaves, min_child)
                    estimator.fit(train[features], target)
                    raw = estimator.predict(t[features])
                    fitted[model_name] = estimator
                    feature_sets[model_name] = features
                    weights = (0.50, 0.75, 1.00, 1.25) if target_mode == "residual" else (0.75, 1.00)
                    for weight in weights:
                        if target_mode == "residual":
                            prediction = t["BasePrediction"].to_numpy() + weight * raw
                        else:
                            prediction = weight * raw + (1.0 - weight) * t["BasePrediction"].to_numpy()
                        candidate = f"{model_name}_W{int(weight * 100):03d}"
                        score, level, trend = validation_score(t, prediction)
                        candidates[candidate] = prediction
                        validation_rows.append(
                            {"candidate": candidate, "score": score, **level, **trend}
                        )
    validation = pd.DataFrame(validation_rows)
    selected = validation.sort_values(["score", "mae", "candidate"], kind="stable").iloc[0]["candidate"]
    model_name, weight_text = selected.rsplit("_W", 1)
    weight = int(weight_text) / 100.0
    t["R7Prediction"] = candidates[selected]

    records = []
    for column, candidate in (
        ("BasePrediction", "R6_BASE"),
        ("R6LevelPrediction", "R6_LEVEL"),
        ("R7Prediction", "R7_SELECTED"),
    ):
        for split in ("val", "oot"):
            records.append(metric_record(t, column, split, candidate))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "r7_metrics.csv", index=False)
    validation.to_csv(OUTPUT / "r7_validation_search.csv", index=False)
    t[
        [
            "LabelGroupID",
            "OrderID",
            "PalletID",
            "RecipeID",
            "ProcessVersion",
            "lab_time",
            "GroupProductionStart",
            "split",
            "MNY",
            "BasePrediction",
            "R6LevelPrediction",
            "R7Prediction",
        ]
    ].to_csv(OUTPUT / "r7_t15760_predictions.csv", index=False)

    estimator = fitted[model_name]
    features = feature_sets[model_name]
    importances = pd.DataFrame(
        {"feature": features, "importance": estimator.feature_importances_}
    ).sort_values(["importance", "feature"], ascending=[False, True])
    importances.to_csv(OUTPUT / "r7_feature_importance.csv", index=False)
    bundle = {
        "contract": "V3.7.1-R7-T15760-QM-ORDER-STATE-V1",
        "model": estimator,
        "features": features,
        "candidate": selected,
        "weight": weight,
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
    }
    joblib.dump(bundle, MODEL_OUTPUT / "r7_t15760_qm_order_state_shadow.joblib")

    val = metrics.loc[
        metrics["candidate"].eq("R7_SELECTED") & metrics["split"].eq("val")
    ].iloc[0]
    oot = metrics.loc[
        metrics["candidate"].eq("R7_SELECTED") & metrics["split"].eq("oot")
    ].iloc[0]
    summary = {
        "contract": bundle["contract"],
        "selected": selected,
        "feature_count": len(features),
        "qm_features_forced_into_candidate": [feature for feature in mandatory_qm if feature in features],
        "strict_split_audit": split_audit,
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
    (OUTPUT / "r7_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
