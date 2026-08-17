"""C2.1 T15760 expert-mapped low-dimensional Step delta challenger."""

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
STEP_SOURCE = BASE / "t15760_within_order_contrast" / "step_execution" / "t15760_group_step_execution_features.csv"
OUTPUT = BASE / "v371_c21_t15760_expert_step_delta"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_c21_t15760_expert_step_delta"


def value(frame: pd.DataFrame, group: str, step: int, field: str) -> pd.Series:
    column = f"step_exec__{group}__S{step}__{field}"
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index)


def build_expert_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = pd.DataFrame(index=frame.index)
    # Explicit expert mapping: MST 1-2 loading, 3 Dry, 4 Oil, 5-9 Wet,
    # 10 controlled PID, 11 post-PID tail; MSB steps remain separate.
    for label, step in (("dry", 3), ("oil", 4), ("pid", 10), ("tail", 11)):
        for field in ("time_increment", "temp", "rotor", "energy_increment"):
            result[f"expert__{label}__{field}"] = value(frame, "AVR_MST", step, field)
    for step in range(5, 10):
        for field in ("time_increment", "temp", "rotor", "energy_increment"):
            result[f"expert__wet_s{step}__{field}"] = value(frame, "AVR_MST", step, field)
    for step in (1, 2, 3):
        for field in ("time_increment", "temp", "rotor", "energy_increment"):
            result[f"expert__msb_s{step}__{field}"] = value(frame, "AVR_MSB", step, field)
    wet_duration = sum((result[f"expert__wet_s{s}__time_increment"] for s in range(5, 10)), start=pd.Series(0.0, index=result.index))
    wet_energy = sum((result[f"expert__wet_s{s}__energy_increment"] for s in range(5, 10)), start=pd.Series(0.0, index=result.index))
    result["expert__wet_total_duration"] = wet_duration
    result["expert__wet_total_energy"] = wet_energy
    result["expert__wet_endpoint_temp"] = result["expert__wet_s9__temp"]
    result["expert__oil_to_wet_temp_rise"] = result["expert__wet_endpoint_temp"] - result["expert__oil__temp"]
    result["expert__dry_to_oil_temp_rise"] = result["expert__oil__temp"] - result["expert__dry__temp"]
    result["expert__wet_rotor_time"] = sum(
        (result[f"expert__wet_s{s}__rotor"] * result[f"expert__wet_s{s}__time_increment"] for s in range(5, 10)),
        start=pd.Series(0.0, index=result.index),
    )
    for source in ("abnormal_batch_rate", "max_tail_step11_duration", "missing_bottom_batch_count", "physical_batch_count"):
        if source in frame:
            result[f"expert__{source}"] = pd.to_numeric(frame[source], errors="coerce")
    usable = [column for column in result if result[column].notna().sum() >= 35 and result[column].nunique(dropna=True) >= 2]
    return result[usable], usable


def ridge(alpha: float) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", RobustScaler()), ("ridge", Ridge(alpha=alpha))])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, _ = r1.load_contract(r1.DEFAULT_FRAME, r1.DEFAULT_C3_METADATA)
    frame = frame.loc[frame["CompoundName"].astype(str).str.contains("T15760", regex=False)].copy()
    step = pd.read_csv(STEP_SOURCE, low_memory=False).drop(columns=["OrderID", "PalletID", "CompoundName", "MNY", "ProcessVersion", "lab_time"], errors="ignore")
    frame = frame.merge(step, on="LabelGroupID", how="inner", validate="one_to_one")
    phase = pd.read_csv(BASE / "phase_batch_features_active_v1.csv", usecols=["LabelGroupID", "BatchStartTime"], low_memory=False)
    phase["BatchStartTime"] = pd.to_datetime(phase["BatchStartTime"], errors="coerce")
    start = phase.groupby("LabelGroupID", as_index=False)["BatchStartTime"].min().rename(columns={"BatchStartTime": "production_start"})
    frame = frame.merge(start, on="LabelGroupID", how="left", validate="one_to_one")
    level_model = joblib.load(C1_MODEL)
    frame["LevelPrediction"] = level_model.predict(frame)[0]
    expert, features = build_expert_features(frame)
    frame = pd.concat([frame.reset_index(drop=True), expert.reset_index(drop=True)], axis=1)
    assert_no_target_leakage(frame, features)
    pairs = pair_core.build_pairs(frame, features)
    delta_features = [column for column in pairs if column.startswith("delta__")]
    train_pairs = pairs.loc[pairs["split"].eq("train")]
    candidates, rows, bundles, outputs = {}, [], {}, {}
    for alpha in (5.0, 20.0, 100.0, 500.0):
        fitted = ridge(alpha)
        fitted.fit(train_pairs[delta_features], train_pairs["ActualDelta"])
        pair_prediction = pairs[["from_group", "to_group", "OrderID", "split", "ActualDelta"]].copy()
        pair_prediction["DeltaPrediction"] = fitted.predict(pairs[delta_features])
        bundles[alpha] = fitted
        outputs[alpha] = pair_prediction
        for beta in (0.50, 0.75, 0.90, 1.0):
            prediction = recursive_order_prediction(frame, frame["LevelPrediction"].to_numpy(copy=True), pair_prediction, beta)
            name = f"C21_T15760_EXPERT_A{int(alpha):03d}_B{int(beta*100):03d}"
            score, level, trend = candidate_score(frame, prediction)
            candidates[name] = prediction
            rows.append({"candidate": name, "alpha": alpha, "beta": beta, "score": score, **level, **trend})
    search = pd.DataFrame(rows).sort_values(["score", "mae", "candidate"], kind="stable")
    selected_name = str(search.iloc[0]["candidate"])
    selected_alpha = float(search.iloc[0]["alpha"])
    selected_beta = float(search.iloc[0]["beta"])
    frame["FinalPrediction"] = candidates[selected_name]
    metrics = pd.DataFrame([metric_record(frame, "FinalPrediction", split, "C21_FINAL") for split in ("val", "oot")])
    search.to_csv(OUTPUT / "c21_validation_search.csv", index=False)
    metrics.to_csv(OUTPUT / "c21_metrics.csv", index=False)
    pd.DataFrame({"feature": features, "expert_phase_contract": "MST3 Dry; 4 Oil; 5-9 Wet; 10 PID; 11 Tail; MSB1-3 separate"}).to_csv(OUTPUT / "c21_feature_contract.csv", index=False)
    outputs[selected_alpha].to_csv(OUTPUT / "c21_pair_predictions.csv", index=False)
    frame[["LabelGroupID", "OrderID", "PalletID", "RecipeID", "ProcessVersion", "TemplateKey", "split", "MNY", "LevelPrediction", "FinalPrediction", "production_start", "lab_time"]].to_csv(OUTPUT / "c21_group_predictions.csv", index=False)
    bundle = {
        "contract": "V3.7.1-C2.1-T15760-EXPERT-STEP-DELTA-V1", "level_model": level_model,
        "delta_model": bundles[selected_alpha], "features": features, "alpha": selected_alpha, "beta": selected_beta,
        "expert_mapping": "MST1-2 Loading; 3 Dry; 4 Oil; 5-9 Wet; 10 PID; 11 Tail; MSB1-3 separate",
        "stage3_calibration_enabled": False, "automatic_control": "DISABLED",
    }
    joblib.dump(bundle, MODEL_OUTPUT / "c21_t15760_expert_step_delta_shadow.joblib")
    summary = {
        "contract": bundle["contract"], "groups": len(frame), "feature_count": len(features),
        "pair_counts_by_split": pairs["split"].value_counts().to_dict(),
        "selected_candidate": selected_name, "selected_alpha": selected_alpha, "selected_beta": selected_beta,
        "metrics": metrics.to_dict("records"), "target_leakage_gate": "PASSED",
        "timing_mapping_conflict": "Old PHASE-CORRECTED timing candidate maps MST7-11 to PID; expert contract overrides it",
        "release_status": "NOT_RELEASED_GATES_FAILED_OR_OOT_ALREADY_VIEWED",
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED", "automatic_control": "DISABLED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
