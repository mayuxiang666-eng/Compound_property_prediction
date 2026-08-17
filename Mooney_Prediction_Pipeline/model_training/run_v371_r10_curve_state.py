"""R10 T15760 raw-curve shape challenger on frozen RDS Active snapshots."""

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

from Mooney_Prediction_Pipeline.model_training.r10_curve_shape_features import (
    SIGNALS,
    extract_curve_row,
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
    level_metrics,
    metric_record,
    model,
    select_features,
    trend_metrics,
)


CURVE_DIR = BASE / "curves_by_order_chunks10"
OUTPUT = BASE / "v371_r10_curve_state"
MODEL_OUTPUT = (
    ROOT
    / "Mooney_Prediction_Pipeline"
    / "models"
    / "v371_challenger_r10_curve_state"
)


def normalize_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def load_t15760_curves(required_keys: set[tuple[str, str]]) -> pd.DataFrame:
    pieces = []
    required_orders = {order for order, _ in required_keys}
    columns = ["OrderID", "BatchNumber", *SIGNALS]
    for path in sorted(CURVE_DIR.glob("curves_orders_*.csv")):
        piece = pd.read_csv(path, usecols=columns, dtype=str, low_memory=False)
        piece["OrderID"] = piece["OrderID"].map(normalize_id)
        selected = piece.loc[piece["OrderID"].isin(required_orders)]
        if len(selected):
            pieces.append(selected)
    if not pieces:
        raise RuntimeError("No frozen curve rows matched T15760 batches")
    curves = pd.concat(pieces, ignore_index=True)
    curves["BatchNumber"] = curves["BatchNumber"].map(normalize_id)
    curves = curves.loc[
        [key in required_keys for key in zip(curves["OrderID"], curves["BatchNumber"])]
    ].copy()
    duplicate = curves.duplicated(["OrderID", "BatchNumber"], keep=False)
    if duplicate.any():
        raise ValueError("Frozen curve snapshot has duplicate physical batch keys")
    return curves


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
    batch = phase.loc[phase["CompoundName"].astype(str).eq(T15760_NAME)].copy()
    batch["OrderID"] = batch["OrderID"].map(normalize_id)
    batch["BatchNumber"] = batch["BatchNumber"].map(normalize_id)
    required_keys = set(zip(batch["OrderID"], batch["BatchNumber"]))
    curves = load_t15760_curves(required_keys)
    curve_rows = [extract_curve_row(row) for _, row in curves.iterrows()]
    curve_features = pd.DataFrame(curve_rows)
    curve_feature_columns = [
        column for column in curve_features.columns if column.startswith("curve__")
    ]
    batch = batch.merge(
        curve_features,
        on=["OrderID", "BatchNumber"],
        how="inner",
        validate="one_to_one",
    )
    if len(batch) != len(required_keys):
        raise AssertionError(
            f"R10 curve coverage incomplete: expected {len(required_keys)}, found {len(batch)}"
        )
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
        raise AssertionError("R10 Lab/Pallet group weights are not equal")

    base_pool = list(
        dict.fromkeys(
            ["BasePrediction", "BatchWeight"]
            + metadata["s1_features"]
            + metadata["s2_features"]
        )
    )
    feature_pool = [feature for feature in base_pool if feature in batch.columns]
    feature_pool += curve_feature_columns
    feature_pool = list(dict.fromkeys(feature_pool))
    for feature in feature_pool:
        batch[feature] = pd.to_numeric(batch[feature], errors="coerce")
    mandatory = [
        "BasePrediction",
        "BatchWeight",
        "supplier_rubber_viscosity_avg",
        "supplier_silica_moisture_avg",
        "supplier_silica_surface_area_avg",
        "curve__temp__max",
        "curve__temp__last",
        "curve__temp__tail30_slope",
        "curve__temp__time_to_100_frac",
        "curve__temp__time_to_120_frac",
        "curve__power__integral_norm",
        "curve__torque__integral_norm",
        "curve__rotor__integral_norm",
        "curve__ram__mean",
        "curve__length_mismatch",
    ]
    batch_train = batch.loc[batch["split"].eq("train")].copy()
    group_train = groups.loc[groups["split"].eq("train")].copy()
    candidates, models, feature_sets, search_rows = {}, {}, {}, []
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
        for top_k in (30, 60, 100, 160):
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
                    name = f"R10_{mode.upper()}_K{top_k}_{objective.upper()}_L{leaves}"
                    estimator = model(objective, leaves, min_child)
                    estimator.fit(
                        batch_train[features],
                        batch_train["TrainingTarget"],
                        sample_weight=batch_train["GroupEqualWeight"],
                    )
                    raw_batch = estimator.predict(batch[features])
                    raw_name = "RawCorrection" if mode == "residual" else "RawLevel"
                    raw_group = aggregate_prediction(batch, raw_batch).rename(
                        columns={"Prediction": raw_name}
                    )
                    models[name] = estimator
                    feature_sets[name] = features
                    weights = (0.50, 0.75, 1.00, 1.25) if mode == "residual" else (0.75, 1.00)
                    for weight in weights:
                        temp = groups[["LabelGroupID", "BasePrediction"]].merge(
                            raw_group, on="LabelGroupID", how="left", validate="one_to_one"
                        )
                        if mode == "residual":
                            prediction = temp["BasePrediction"].to_numpy() + weight * temp[raw_name].to_numpy()
                        else:
                            prediction = weight * temp[raw_name].to_numpy() + (1.0 - weight) * temp["BasePrediction"].to_numpy()
                        candidate = f"{name}_W{int(weight * 100):03d}"
                        total, level, trend = candidate_score(groups, prediction)
                        candidates[candidate] = prediction
                        search_rows.append(
                            {"candidate": candidate, "score": total, **level, **trend}
                        )
    search = pd.DataFrame(search_rows)
    selected = search.sort_values(["score", "mae", "candidate"], kind="stable").iloc[0]["candidate"]
    model_name, weight_text = selected.rsplit("_W", 1)
    groups["R10Prediction"] = candidates[selected]
    records = []
    for column, candidate in (
        ("BasePrediction", "R6_BASE"),
        ("R6LevelPrediction", "R6_LEVEL"),
        ("R10Prediction", "R10_SELECTED"),
    ):
        for split in ("val", "oot"):
            records.append(metric_record(groups, column, split, candidate))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "r10_metrics.csv", index=False)
    search.to_csv(OUTPUT / "r10_validation_search.csv", index=False)
    groups.to_csv(OUTPUT / "r10_t15760_group_predictions.csv", index=False)
    curve_features.to_csv(OUTPUT / "r10_t15760_batch_curve_features.csv", index=False)

    estimator = models[model_name]
    features = feature_sets[model_name]
    pd.DataFrame(
        {"feature": features, "importance": estimator.feature_importances_}
    ).sort_values(["importance", "feature"], ascending=[False, True]).to_csv(
        OUTPUT / "r10_feature_importance.csv", index=False
    )
    bundle = {
        "contract": "V3.7.1-R10-RAW-CURVE-SHAPE-V1",
        "candidate": selected,
        "model": estimator,
        "features": features,
        "weight": int(weight_text) / 100.0,
        "curve_signals": SIGNALS,
        "curve_points_per_signal": 32,
        "evaluation": "BATCH_PREDICTION_MEAN_PER_ORDERID_PALLETID",
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
    }
    joblib.dump(bundle, MODEL_OUTPUT / "r10_curve_state_shadow.joblib")
    val = metrics.loc[
        metrics["candidate"].eq("R10_SELECTED") & metrics["split"].eq("val")
    ].iloc[0]
    oot = metrics.loc[
        metrics["candidate"].eq("R10_SELECTED") & metrics["split"].eq("oot")
    ].iloc[0]
    summary = {
        "contract": bundle["contract"],
        "selected": selected,
        "curve_batches_expected": len(required_keys),
        "curve_batches_found": int(len(curve_features)),
        "curve_feature_count": len(curve_feature_columns),
        "selected_feature_count": len(features),
        "all_signals_present_rate": float(curve_features["curve__all_signals_present"].mean()),
        "length_mismatch_rate": float(curve_features["curve__length_mismatch"].mean()),
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
    (OUTPUT / "r10_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
