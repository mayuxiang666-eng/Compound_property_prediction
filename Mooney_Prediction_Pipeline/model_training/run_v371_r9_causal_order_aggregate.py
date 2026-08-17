"""R9 T15760 causal cumulative-order process-state expert."""

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
    T15760_NAME,
    level_metrics,
    prepare_frame,
    trend_metrics,
)


R6_PREDICTIONS = BASE / "v371_r6_recipe_delta" / "r6_group_predictions.csv"
OUTPUT = BASE / "v371_r9_causal_order_aggregate"
MODEL_OUTPUT = (
    ROOT
    / "Mooney_Prediction_Pipeline"
    / "models"
    / "v371_challenger_r9_causal_order_aggregate"
)


def add_causal_order_aggregates(frame, features):
    work = frame.copy()
    work["_position"] = np.arange(len(work))
    work = work.sort_values(
        ["OrderID", "GroupProductionStart", "LabelGroupID"], kind="stable"
    ).reset_index(drop=True)
    grouped = work.groupby("OrderID", sort=False)
    work["causal_order_group_count"] = grouped.cumcount() + 1
    added = ["causal_order_group_count"]
    blocks = []
    for feature in features:
        value = pd.to_numeric(work[feature], errors="coerce")
        temporary = pd.DataFrame({"OrderID": work["OrderID"], "value": value})
        order_value = temporary.groupby("OrderID", sort=False)["value"]
        block = pd.DataFrame(
            {
                f"causal_mean__{feature}": order_value.expanding().mean().reset_index(level=0, drop=True),
                f"causal_min__{feature}": order_value.expanding().min().reset_index(level=0, drop=True),
                f"causal_max__{feature}": order_value.expanding().max().reset_index(level=0, drop=True),
            },
            index=work.index,
        )
        blocks.append(block)
        added.extend(block.columns)
    work = pd.concat([work] + blocks, axis=1)
    work = work.sort_values("_position", kind="stable").drop(columns="_position").reset_index(drop=True)
    return work, added


def select_features(train, pool, target, top_k, mandatory):
    scores = []
    for feature in pool:
        value = pd.to_numeric(train[feature], errors="coerce")
        valid = value.notna() & target.notna()
        if int(valid.sum()) < 40 or value.loc[valid].nunique() < 3:
            continue
        correlation = value.loc[valid].corr(target.loc[valid], method="spearman")
        scores.append((abs(float(correlation)) if pd.notna(correlation) else 0.0, feature))
    ranked = [name for _, name in sorted(scores, key=lambda item: (-item[0], item[1]))[:top_k]]
    return list(dict.fromkeys([name for name in mandatory if name in pool] + ranked))


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


def score(frame, prediction):
    part = frame.loc[frame["split"].eq("val")].copy()
    part["Prediction"] = prediction[part.index]
    level = level_metrics(part, "Prediction")
    trend = trend_metrics(part, "Prediction")
    correlation = trend["delta_pearson"] if pd.notna(trend["delta_pearson"]) else -1.0
    direction = trend["direction_accuracy"] if pd.notna(trend["direction_accuracy"]) else 0.0
    ratio = trend["delta_std_ratio"] if pd.notna(trend["delta_std_ratio"]) else 0.0
    total = float(level["mae"])
    total += 1.5 * max(0.0, 0.50 - float(correlation))
    total += 1.5 * max(0.0, 0.70 - float(direction))
    total += 0.75 * max(0.0, 0.70 - float(ratio))
    total += 0.75 * max(0.0, float(ratio) - 1.30)
    return total, level, trend


def metric_record(frame, column, split, candidate):
    part = frame.loc[frame["split"].eq(split)]
    return {
        "candidate": candidate,
        "split": split,
        **level_metrics(part, column),
        **trend_metrics(part, column),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    groups, metadata, target_map, lag_features, split_audit = prepare_frame()
    old = pd.read_csv(
        R6_PREDICTIONS,
        usecols=["LabelGroupID", "BasePrediction", "LevelPrediction"],
    ).rename(columns={"LevelPrediction": "R6LevelPrediction"})
    groups = groups.merge(old, on="LabelGroupID", how="inner", validate="one_to_one")
    t = groups.loc[groups["CompoundName"].astype(str).eq(T15760_NAME)].copy().reset_index(drop=True)

    physical = list(dict.fromkeys(metadata["s1_features"] + metadata["s2_features"]))
    physical = [feature for feature in physical if feature in t.columns]
    t, causal_features = add_causal_order_aggregates(t, physical)
    pool = list(
        dict.fromkeys(
            ["BasePrediction", "ProcessVersion", "prior_order_state_available", "order_group_index"]
            + physical
            + sorted(target_map)
            + lag_features
            + causal_features
        )
    )
    pool = [feature for feature in pool if feature in t.columns]
    mandatory = [
        "BasePrediction",
        "ProcessVersion",
        "causal_order_group_count",
        "causal_mean__supplier_rubber_viscosity_avg",
        "causal_mean__supplier_silica_moisture_avg",
        "causal_mean__supplier_silica_surface_area_avg",
        "causal_mean__Stage2_DryMixing_Specific_Energy",
        "causal_mean__Stage4_WetMixing_Specific_Energy",
        "causal_mean__Stage6_BottomMixing_Duration",
        "causal_mean__phys_discharge_temp",
    ]
    train = t.loc[t["split"].eq("train")].copy()
    candidates, models, feature_sets, search_rows = {}, {}, {}, []
    residual = train["MNY"] - train["BasePrediction"]
    for top_k in (15, 30, 60, 100):
        features = select_features(train, pool, residual, top_k, mandatory)
        for objective in ("regression_l1", "huber"):
            for leaves, min_child in ((3, 20), (5, 25), (8, 35)):
                name = f"R9_K{top_k}_{objective.upper()}_L{leaves}"
                estimator = model(objective, leaves, min_child)
                estimator.fit(train[features], residual)
                correction = estimator.predict(t[features])
                models[name] = estimator
                feature_sets[name] = features
                for weight in (0.50, 0.75, 1.00, 1.25):
                    candidate = f"{name}_W{int(weight * 100):03d}"
                    prediction = t["BasePrediction"].to_numpy() + weight * correction
                    total, level, trend = score(t, prediction)
                    candidates[candidate] = prediction
                    search_rows.append({"candidate": candidate, "score": total, **level, **trend})
    search = pd.DataFrame(search_rows)
    selected = search.sort_values(["score", "mae", "candidate"], kind="stable").iloc[0]["candidate"]
    model_name, weight_text = selected.rsplit("_W", 1)
    t["R9Prediction"] = candidates[selected]
    records = []
    for column, candidate in (
        ("BasePrediction", "R6_BASE"),
        ("R6LevelPrediction", "R6_LEVEL"),
        ("R9Prediction", "R9_SELECTED"),
    ):
        for split in ("val", "oot"):
            records.append(metric_record(t, column, split, candidate))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "r9_metrics.csv", index=False)
    search.to_csv(OUTPUT / "r9_validation_search.csv", index=False)
    t[
        [
            "LabelGroupID",
            "OrderID",
            "PalletID",
            "RecipeID",
            "ProcessVersion",
            "GroupProductionStart",
            "split",
            "MNY",
            "BasePrediction",
            "R6LevelPrediction",
            "R9Prediction",
            "causal_order_group_count",
        ]
    ].to_csv(OUTPUT / "r9_t15760_predictions.csv", index=False)
    estimator = models[model_name]
    features = feature_sets[model_name]
    pd.DataFrame(
        {"feature": features, "importance": estimator.feature_importances_}
    ).sort_values(["importance", "feature"], ascending=[False, True]).to_csv(
        OUTPUT / "r9_feature_importance.csv", index=False
    )
    bundle = {
        "contract": "V3.7.1-R9-CAUSAL-ORDER-AGGREGATE-V1",
        "candidate": selected,
        "model": estimator,
        "features": features,
        "weight": int(weight_text) / 100.0,
        "state_contract": "CURRENT_AND_ALL_PRIOR_COMPLETED_GROUP_PROCESS_ONLY_NO_LAB",
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
    }
    joblib.dump(bundle, MODEL_OUTPUT / "r9_causal_order_aggregate_shadow.joblib")
    val = metrics.loc[
        metrics["candidate"].eq("R9_SELECTED") & metrics["split"].eq("val")
    ].iloc[0]
    oot = metrics.loc[
        metrics["candidate"].eq("R9_SELECTED") & metrics["split"].eq("oot")
    ].iloc[0]
    summary = {
        "contract": bundle["contract"],
        "selected": selected,
        "feature_count": len(features),
        "causal_feature_count": len(causal_features),
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
    (OUTPUT / "r9_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
