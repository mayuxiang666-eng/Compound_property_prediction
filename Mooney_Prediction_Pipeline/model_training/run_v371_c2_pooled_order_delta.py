"""C2 pooled within-order process-delta expert anchored by the clean C1 model."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training import run_v371_r1_temporal_group as r1
from Mooney_Prediction_Pipeline.model_training import run_v371_r11_step_gated_delta as pair_core
from Mooney_Prediction_Pipeline.model_training.feature_leakage_guard import assert_no_target_leakage
from Mooney_Prediction_Pipeline.model_training.order_atomic_trend_metrics import candidate_score, metric_record
from Mooney_Prediction_Pipeline.model_training.order_delta_recursion_v2 import recursive_order_prediction


BASE = r1.DEFAULT_BASE
C1_MODEL = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_c1_strict_order_selected" / "hybrid_model_shadow.joblib"
OUTPUT = BASE / "v371_c2_pooled_order_delta"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_c2_pooled_order_delta"


def ridge(alpha: float) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", RobustScaler()), ("ridge", Ridge(alpha=alpha))])


def add_production_start(frame: pd.DataFrame) -> pd.DataFrame:
    phase = pd.read_csv(BASE / "phase_batch_features_active_v1.csv", usecols=["LabelGroupID", "BatchStartTime"], low_memory=False)
    phase["BatchStartTime"] = pd.to_datetime(phase["BatchStartTime"], errors="coerce")
    start = phase.groupby("LabelGroupID", as_index=False)["BatchStartTime"].min().rename(columns={"BatchStartTime": "production_start"})
    result = frame.merge(start, on="LabelGroupID", how="left", validate="one_to_one")
    if result["production_start"].isna().any():
        raise ValueError("BLOCKED_MISSING_GROUP_PRODUCTION_START")
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, metadata = r1.load_contract(r1.DEFAULT_FRAME, r1.DEFAULT_C3_METADATA)
    frame = add_production_start(frame)
    model = joblib.load(C1_MODEL)
    assert_no_target_leakage(frame, metadata["s2_features"])
    prediction, *_ = model.predict(frame)
    frame["LevelPrediction"] = prediction
    for route in ("Silica", "CarbonBlack"):
        frame[f"route__material__{route}"] = frame["material_system"].astype(str).eq(route).astype(float)
    for route in ("oil_wet", "no_oil_dry"):
        frame[f"route__phase__{route}"] = frame["phase_route"].astype(str).eq(route).astype(float)
    route_features = [column for column in frame if column.startswith("route__")]
    feature_pool = [feature for feature in metadata["s2_features"] if feature in frame] + route_features
    assert_no_target_leakage(frame, feature_pool)
    for feature in feature_pool:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    pairs = pair_core.build_pairs(frame, feature_pool)
    delta_pool = [column for column in pairs if column.startswith("delta__")]
    train_pairs = pairs.loc[pairs["split"].eq("train")].copy()
    if len(train_pairs) < 50:
        raise ValueError("BLOCKED_INSUFFICIENT_POOLED_ORDER_PAIRS")

    candidates = {"C1_LEVEL": frame["LevelPrediction"].to_numpy(copy=True)}
    rows, bundles, pair_outputs = [], {}, {}
    for top_k in (15, 30, 50):
        features = pair_core.select_features(train_pairs, delta_pool, "ActualDelta", top_k, [])
        for alpha in (10.0, 50.0, 200.0):
            fitted = ridge(alpha)
            fitted.fit(train_pairs[features], train_pairs["ActualDelta"])
            pair_prediction = pairs[["from_group", "to_group", "OrderID", "split", "ActualDelta"]].copy()
            pair_prediction["DeltaPrediction"] = fitted.predict(pairs[features])
            key = f"K{top_k}_A{int(alpha)}"
            bundles[key] = {"model": fitted, "features": features, "alpha": alpha, "top_k": top_k}
            pair_outputs[key] = pair_prediction
            for beta in (0.50, 0.75, 0.90, 1.0):
                output = recursive_order_prediction(frame, frame["LevelPrediction"].to_numpy(copy=True), pair_prediction, beta)
                name = f"C2_POOL_DELTA_{key}_B{int(beta*100):03d}"
                score, level, trend = candidate_score(frame, output)
                candidates[name] = output
                rows.append({"candidate": name, "bundle_key": key, "beta": beta, "score": score, **level, **trend})
    search = pd.DataFrame(rows).sort_values(["score", "mae", "candidate"], kind="stable")
    selected_name = str(search.iloc[0]["candidate"])
    selected_key = str(search.iloc[0]["bundle_key"])
    selected_beta = float(search.iloc[0]["beta"])
    frame["FinalPrediction"] = candidates[selected_name]
    records = []
    for split_name in ("val", "oot"):
        for scope, mask in (
            ("global", pd.Series(True, index=frame.index)),
            ("T15760", frame["CompoundName"].astype(str).str.contains("T15760", regex=False)),
            ("B00458", frame["CompoundName"].astype(str).str.contains("B00458", regex=False)),
        ):
            part = frame.loc[mask]
            if part.loc[part["split"].eq(split_name)].empty:
                continue
            records.append({"scope": scope, **metric_record(part, "FinalPrediction", split_name, "C2_FINAL")})
    metrics = pd.DataFrame(records)
    search.to_csv(OUTPUT / "c2_validation_search.csv", index=False)
    metrics.to_csv(OUTPUT / "c2_metrics.csv", index=False)
    pairs.to_csv(OUTPUT / "c2_order_pairs.csv", index=False)
    pair_outputs[selected_key].to_csv(OUTPUT / "c2_selected_pair_predictions.csv", index=False)
    frame[["LabelGroupID", "OrderID", "PalletID", "RecipeID", "CompoundName", "ProcessVersion", "split", "MNY", "LevelPrediction", "FinalPrediction", "production_start", "lab_time"]].to_csv(OUTPUT / "c2_group_predictions.csv", index=False)
    bundle = {
        "contract": "V3.7.1-C2-POOLED-ORDER-DELTA-V1", "level_model": model,
        "delta": bundles[selected_key], "beta": selected_beta,
        "trend_contract": "within OrderID adjacent LabelGroups by production_start",
        "target_leakage_gate": "PASSED", "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
    }
    joblib.dump(bundle, MODEL_OUTPUT / "c2_pooled_order_delta_shadow.joblib")
    summary = {
        "contract": bundle["contract"], "groups": len(frame),
        "pair_counts_by_split": pairs["split"].value_counts().to_dict(),
        "selected_candidate": selected_name, "selected_beta": selected_beta,
        "metrics": metrics.to_dict("records"), "target_leakage_gate": "PASSED",
        "release_status": "NOT_RELEASED_GATES_FAILED_OR_OOT_ALREADY_VIEWED",
        "stage3_calibration_enabled": False, "automatic_control": "DISABLED",
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
