"""Chronological order-atomic endpoint response test for B00458 MSB Step 3 duration."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from Mooney_Prediction_Pipeline.optimization.expert_process_history import build_process_response_history
from Mooney_Prediction_Pipeline.optimization.expert_process_response import PROCESS_OUTPUTS, TEMPERATURE_OUTPUTS


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history" / "active_exact_pilot_step_history_v1"
OUTPUT = BASE / "b00458_main_template_process_response_v1"
RECIPE = "9922A00458BT_PV77_004"
TEMPLATE = "0A497A75CABE"
ACTION_FEATURE = "AVR_MSB_S3_duration"


def order_splits(history: pd.DataFrame) -> pd.DataFrame:
    orders = history.groupby("OrderID", as_index=False).agg(
        OrderTime=("OrderStartTime", "max"), BatchCount=("BatchNumber", "size")
    ).sort_values(["OrderTime", "OrderID"], kind="stable").reset_index(drop=True)
    cumulative = orders["BatchCount"].cumsum()
    total = float(orders["BatchCount"].sum())
    orders["Split"] = np.select(
        [cumulative <= 0.70 * total, cumulative <= 0.85 * total],
        ["train", "validation"], default="oot_test"
    )
    return orders


def main() -> None:
    mapped = pd.read_csv(BASE / "active_exact_pilot_step_vectors.csv", low_memory=False)
    assignments = pd.read_csv(BASE / "active_exact_pilot_template_assignments_v2.csv", low_memory=False)
    for frame in (mapped, assignments):
        frame["OrderID"] = frame["OrderID"].astype(str).str.replace(r"\.0$", "", regex=True)
        frame["BatchNumber"] = pd.to_numeric(frame["BatchNumber"], errors="raise").astype(int)
    selected = assignments.loc[
        assignments["RecipeID"].astype(str).eq(RECIPE) & assignments["TemplateID"].astype(str).eq(TEMPLATE),
        ["OrderID", "BatchNumber", "TemplateID"],
    ]
    steps = mapped.merge(selected, on=["OrderID", "BatchNumber"], how="inner", validate="many_to_one")
    history, target_features = build_process_response_history(steps, time_column="OrderStartTime")
    history["OrderID"] = history["OrderID"].astype(str).str.replace(r"\.0$", "", regex=True)
    history["OrderStartTime"] = pd.to_datetime(history["OrderStartTime"], errors="raise")
    splits = order_splits(history)
    history = history.merge(splits[["OrderID", "Split"]], on="OrderID", how="left", validate="many_to_one")
    usable_features = tuple(
        column for column in target_features
        if pd.to_numeric(history[column], errors="coerce").notna().all()
    )
    if ACTION_FEATURE not in usable_features:
        raise ValueError(f"Missing identifiable action feature: {ACTION_FEATURE}")

    train = history.loc[history["Split"].eq("train")]
    validation = history.loc[history["Split"].eq("validation")]
    oot = history.loc[history["Split"].eq("oot_test")]
    models, radii, metrics = {}, {}, {}
    predictions = oot[["OrderID", "BatchNumber", "OrderStartTime", "Split"]].copy()
    x_train = train.loc[:, usable_features].apply(pd.to_numeric, errors="raise")
    x_validation = validation.loc[:, usable_features].apply(pd.to_numeric, errors="raise")
    x_oot = oot.loc[:, usable_features].apply(pd.to_numeric, errors="raise")

    for output in PROCESS_OUTPUTS:
        train_ok = train[output].notna()
        validation_ok = validation[output].notna()
        oot_ok = oot[output].notna()
        model = GradientBoostingRegressor(loss="squared_error", random_state=17)
        model.fit(x_train.loc[train_ok], pd.to_numeric(train.loc[train_ok, output], errors="raise"))
        validation_actual = pd.to_numeric(validation.loc[validation_ok, output], errors="raise").to_numpy(float)
        validation_prediction = model.predict(x_validation.loc[validation_ok])
        radius = float(np.quantile(np.abs(validation_actual - validation_prediction), 0.90))
        oot_actual = pd.to_numeric(oot.loc[oot_ok, output], errors="raise").to_numpy(float)
        oot_prediction = model.predict(x_oot.loc[oot_ok])
        mae = float(mean_absolute_error(oot_actual, oot_prediction))
        coverage = float((np.abs(oot_actual - oot_prediction) <= radius).mean())
        threshold = 2.0 if output in TEMPERATURE_OUTPUTS else 5.0
        passed = bool(len(oot_actual) >= 20 and mae <= threshold and coverage >= 0.80)
        metrics[output] = {
            "train_samples": int(train_ok.sum()), "validation_samples": int(validation_ok.sum()),
            "oot_samples": int(oot_ok.sum()), "oot_mae": mae,
            "conformal_90_radius": radius, "oot_interval_coverage": coverage,
            "mae_threshold": threshold, "passed": passed,
        }
        models[output], radii[output] = model, radius
        predictions.loc[oot_ok, f"{output}_actual"] = oot_actual
        predictions.loc[oot_ok, f"{output}_prediction"] = oot_prediction
        predictions.loc[oot_ok, f"{output}_low"] = oot_prediction - radius
        predictions.loc[oot_ok, f"{output}_high"] = oot_prediction + radius

    action_levels = {
        split: sorted(pd.to_numeric(history.loc[history["Split"].eq(split), ACTION_FEATURE], errors="coerce").dropna().unique().tolist())
        for split in ("train", "validation", "oot_test")
    }
    endpoint_approved = bool(all(item["passed"] for item in metrics.values()))
    bundle = {
        "models": models, "conformal_radius": radii, "feature_columns": usable_features,
        "action_feature": ACTION_FEATURE, "recipe_id": RECIPE, "template_id": TEMPLATE,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUTPUT / "endpoint_response_shadow.joblib")
    history.to_csv(OUTPUT / "process_response_history.csv", index=False, encoding="utf-8-sig")
    splits.to_csv(OUTPUT / "order_atomic_splits.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT / "oot_endpoint_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "contract": "B00458-MAIN-TEMPLATE-PROCESS-RESPONSE-V1",
        "recipe_id": RECIPE, "template_id": TEMPLATE,
        "action": "AVR_MSB Step 3 duration",
        "history_batches": int(len(history)), "orders": int(history["OrderID"].nunique()),
        "partition_batches": {split: int(history["Split"].eq(split).sum()) for split in ("train", "validation", "oot_test")},
        "partition_orders": {split: int(history.loc[history["Split"].eq(split), "OrderID"].nunique()) for split in ("train", "validation", "oot_test")},
        "action_levels_by_split": action_levels,
        "endpoint_metrics": metrics,
        "endpoint_response_approved": endpoint_approved,
        "formal_v37_control_transfer_status": "BLOCKED_MISSING_ACTUAL_PHASE_FEATURE_RESPONSE",
        "reason": "Endpoint temperatures cannot be substituted for V3.7 Phase means; all affected actual Phase features still require response validation.",
        "database_writes": False, "automatic_control": "DISABLED",
        "recommendation_status": "NOT_GENERATED_ENDPOINT_MODEL_ONLY",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
