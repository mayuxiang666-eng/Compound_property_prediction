"""Diagnose the deployable and oracle signal ceiling for T15760.

All deployable history features use a Lab result only when its result timestamp
is no later than the current Pallet group's production start.  Oracle baselines
are reported separately and are never model candidates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.r5_order_state_features import (
    build_group_target_features,
)


BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
GROUP_SOURCE = BASE / "pallet_group_training_frame_c2_prepared.csv"
PHASE_SOURCE = BASE / "phase_batch_features_active_v1.csv"
R6_PREDICTIONS = BASE / "v371_r6_recipe_delta" / "r6_group_predictions.csv"
OUTPUT = BASE / "v371_r6_recipe_delta" / "signal_ceiling"
T15760_NAME = "M1-T15760---- 06 002"


def metric(actual, prediction):
    valid = np.isfinite(actual) & np.isfinite(prediction)
    actual = np.asarray(actual, dtype=float)[valid]
    prediction = np.asarray(prediction, dtype=float)[valid]
    if not len(actual):
        return {"groups": 0, "mae": None, "rmse": None, "bias": None, "r2": None}
    return {
        "groups": int(len(actual)),
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
        "bias": float(np.mean(prediction - actual)),
        "r2": float(r2_score(actual, prediction)) if len(actual) > 1 else None,
    }


def causal_lab_features(frame):
    ordered = frame.sort_values(
        ["GroupProductionStart", "lab_time", "OrderID", "LabelGroupID"],
        kind="stable",
    ).copy()
    lab_events = frame.sort_values("lab_time", kind="stable")
    event_times = pd.to_datetime(lab_events["lab_time"], utc=True).to_numpy()
    event_values = lab_events["MNY"].to_numpy(dtype=float)
    latest = []
    rolling3 = []
    ewma50 = []
    available_count = []
    for start in pd.to_datetime(ordered["GroupProductionStart"], utc=True):
        position = int(np.searchsorted(event_times, np.datetime64(start), side="right"))
        values = event_values[:position]
        available_count.append(position)
        latest.append(values[-1] if len(values) else np.nan)
        rolling3.append(float(np.mean(values[-3:])) if len(values) else np.nan)
        if len(values):
            state = float(values[0])
            for value in values[1:]:
                state = 0.5 * float(value) + 0.5 * state
            ewma50.append(state)
        else:
            ewma50.append(np.nan)
    ordered["causal_lab_count"] = available_count
    ordered["causal_latest_lab"] = latest
    ordered["causal_rolling3_lab"] = rolling3
    ordered["causal_ewma50_lab"] = ewma50
    return ordered.sort_index()


def trend_metric(frame, prediction_col):
    part = frame.sort_values(["RecipeID", "lab_time", "LabelGroupID"], kind="stable")
    actual_delta = part.groupby("RecipeID")["MNY"].diff()
    predicted_delta = part.groupby("RecipeID")[prediction_col].diff()
    valid = actual_delta.notna() & predicted_delta.notna()
    actual = actual_delta.loc[valid].to_numpy(dtype=float)
    predicted = predicted_delta.loc[valid].to_numpy(dtype=float)
    significant = np.abs(actual) >= 2.0
    return {
        "pairs": int(len(actual)),
        "significant_pairs": int(significant.sum()),
        "delta_pearson": (
            float(np.corrcoef(actual, predicted)[0, 1])
            if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0
            else None
        ),
        "direction_accuracy": (
            float(np.mean(np.sign(actual[significant]) == np.sign(predicted[significant])))
            if significant.any()
            else None
        ),
        "delta_std_ratio": (
            float(np.std(predicted, ddof=1) / np.std(actual, ddof=1))
            if len(actual) > 1 and np.std(actual, ddof=1) > 0
            else None
        ),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    groups = pd.read_csv(GROUP_SOURCE, low_memory=False)
    predictions = pd.read_csv(R6_PREDICTIONS, low_memory=False)
    phase = pd.read_csv(
        PHASE_SOURCE,
        usecols=[
            "LabelGroupID",
            "TargetVectorRaw",
            "TargetVectorHash",
            "StepPatternHash",
            "TemplateKey",
            "BatchStartTime",
            "BatchEndTime",
        ],
        low_memory=False,
    )
    target = build_group_target_features(phase).frame[
        [
            "LabelGroupID",
            "TargetVectorHash",
            "StepPatternHash",
            "TemplateKey",
            "GroupProductionStart",
            "GroupProductionComplete",
        ]
    ]
    t = groups.loc[groups["CompoundName"].astype(str).eq(T15760_NAME)].copy()
    t = t.drop(columns=[column for column in target.columns if column != "LabelGroupID" and column in t])
    t = t.merge(target, on="LabelGroupID", how="inner", validate="one_to_one")
    t = t.merge(
        predictions[
            [
                "LabelGroupID",
                "split",
                "BasePrediction",
                "LevelPrediction",
                "FinalPrediction",
            ]
        ],
        on="LabelGroupID",
        how="inner",
        validate="one_to_one",
    )
    t["lab_time"] = pd.to_datetime(t["lab_time"], errors="coerce", utc=True)
    t["GroupProductionStart"] = pd.to_datetime(
        t["GroupProductionStart"], errors="coerce", utc=True
    )
    t["MNY"] = pd.to_numeric(t["MNY"], errors="coerce")
    t = causal_lab_features(t)

    train = t.loc[t["split"].eq("train")]
    train_mean = float(train["MNY"].mean())
    t["train_mean"] = train_mean
    for key, name in (
        ("ProcessVersion", "train_process_version_mean"),
        ("TargetVectorHash", "train_target_vector_mean"),
        ("TemplateKey", "train_template_mean"),
    ):
        mapping = train.groupby(key)["MNY"].mean()
        t[name] = t[key].map(mapping).fillna(train_mean)

    # Explicit non-deployable ceilings, kept separate from candidates.
    t["oracle_full_order_mean"] = t.groupby("OrderID")["MNY"].transform("mean")
    t["oracle_full_target_mean"] = t.groupby("TargetVectorHash")["MNY"].transform("mean")

    rows = []
    columns = [
        ("R6_BASE", "BasePrediction", True),
        ("R6_LEVEL", "LevelPrediction", True),
        ("R6_FINAL", "FinalPrediction", True),
        ("TRAIN_MEAN", "train_mean", True),
        ("TRAIN_PROCESS_VERSION_MEAN", "train_process_version_mean", True),
        ("TRAIN_TARGET_VECTOR_MEAN", "train_target_vector_mean", True),
        ("TRAIN_TEMPLATE_MEAN", "train_template_mean", True),
        ("CAUSAL_LATEST_LAB", "causal_latest_lab", True),
        ("CAUSAL_ROLLING3_LAB", "causal_rolling3_lab", True),
        ("CAUSAL_EWMA50_LAB", "causal_ewma50_lab", True),
        ("ORACLE_FULL_ORDER_MEAN", "oracle_full_order_mean", False),
        ("ORACLE_FULL_TARGET_MEAN", "oracle_full_target_mean", False),
    ]
    for split in ("val", "oot"):
        part = t.loc[t["split"].eq(split)]
        for name, column, deployable in columns:
            rows.append(
                {
                    "split": split,
                    "baseline": name,
                    "deployable": deployable,
                    **metric(
                        part["MNY"].to_numpy(dtype=float),
                        pd.to_numeric(part[column], errors="coerce").to_numpy(dtype=float),
                    ),
                    **trend_metric(part, column),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "t15760_signal_ceiling_metrics.csv", index=False)

    order_stats = t.groupby("OrderID")["MNY"].agg(["count", "mean", "std"])
    repeated_orders = order_stats.loc[order_stats["count"].ge(2)]
    target_stats = t.groupby("TargetVectorHash")["MNY"].agg(["count", "std"])
    repeated_targets = target_stats.loc[target_stats["count"].ge(5)]
    qm_columns = [column for column in t.columns if column.startswith("supplier_")]
    qm_coverage = {
        column: float(pd.to_numeric(t[column], errors="coerce").notna().mean())
        for column in qm_columns
    }
    summary = {
        "compound": T15760_NAME,
        "groups": int(len(t)),
        "orders": int(t["OrderID"].nunique()),
        "split_groups": t["split"].value_counts().to_dict(),
        "process_versions": int(t["ProcessVersion"].nunique()),
        "target_vectors": int(t["TargetVectorHash"].nunique()),
        "templates": int(t["TemplateKey"].nunique()),
        "actual_std": float(t["MNY"].std(ddof=1)),
        "repeated_orders": int(len(repeated_orders)),
        "median_within_order_std": (
            float(repeated_orders["std"].median()) if len(repeated_orders) else None
        ),
        "oracle_full_order_mae_all": float(
            mean_absolute_error(t["MNY"], t["oracle_full_order_mean"])
        ),
        "repeated_target_vectors": int(len(repeated_targets)),
        "median_within_target_std": (
            float(repeated_targets["std"].median()) if len(repeated_targets) else None
        ),
        "causal_lab_available_before_production_rate": float(
            t["causal_latest_lab"].notna().mean()
        ),
        "supplier_qm_nonmissing_rate": qm_coverage,
        "local_mms_access": False,
        "database_access_during_run": False,
        "oracle_warning": "FULL_ORDER/FULL_TARGET use future labels and are diagnostic only",
    }
    (OUTPUT / "t15760_signal_ceiling_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    keep = [
        "LabelGroupID",
        "OrderID",
        "PalletID",
        "ProcessVersion",
        "TargetVectorHash",
        "lab_time",
        "GroupProductionStart",
        "split",
        "MNY",
        "BasePrediction",
        "LevelPrediction",
        "FinalPrediction",
        "causal_lab_count",
        "causal_latest_lab",
        "causal_rolling3_lab",
        "causal_ewma50_lab",
        "oracle_full_order_mean",
    ]
    t.loc[:, keep].to_csv(OUTPUT / "t15760_signal_ceiling_rows.csv", index=False)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
