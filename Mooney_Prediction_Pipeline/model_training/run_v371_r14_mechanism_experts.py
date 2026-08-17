"""R14 low-dimensional mixing-mechanism residual and within-order experts."""

from __future__ import annotations

import json
import sys
import time
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

from Mooney_Prediction_Pipeline.model_training import run_v371_r11_step_gated_delta as core
from Mooney_Prediction_Pipeline.model_training.feature_leakage_guard import assert_no_target_leakage
from Mooney_Prediction_Pipeline.model_training.order_atomic_trend_metrics import candidate_score, metric_record
from Mooney_Prediction_Pipeline.model_training.order_delta_recursion_v2 import recursive_order_prediction


BASE = core.BASE
SOURCE = BASE / "v371_r11e_checkpointed_order_model" / "r11e_level_checkpoint.pkl"
R12_OOF = BASE / "v371_r12_causal_residual_expert" / "r12_group_predictions.csv"
OUTPUT = BASE / "v371_r14_mechanism_experts"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r14_mechanism_experts"


PHASES = [
    "Stage2_DryMixing", "Stage3_OilLoading", "Stage4_WetMixing", "Stage5_PID_Controlled",
    "Stage6_MSB_PID_Step1", "Stage6_MSB_PID_Step2", "Stage6_MSB_PID_Step3", "PostPID_Tail",
]
PHASE_MEASURES = [
    ("duration", None), ("temp", "last"), ("temp", "mean"), ("power", "integral"),
    ("torque", "mean"), ("rotor", "mean"),
]


def source_name(phase: str, signal: str, statistic: str | None) -> str:
    if signal == "duration":
        return f"step_phase__{phase}__duration__group_mean"
    return f"step_phase__{phase}__{signal}__{statistic}__group_mean"


def mechanism_frame(groups: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = pd.DataFrame(index=groups.index)
    for phase in PHASES:
        for signal, statistic in PHASE_MEASURES:
            source = source_name(phase, signal, statistic)
            if source in groups:
                name = f"mech__{phase}__{signal}_{statistic or 'actual'}"
                output[name] = pd.to_numeric(groups[source], errors="coerce")
    for step in (3, 4, 5, 6, 7, 8, 9, 10, 11):
        for measure in ("time_increment", "temp", "rotor", "energy_increment"):
            source = f"step_exec__AVR_MST__S{step}__{measure}"
            if source in groups:
                output[f"mech__MST_S{step}__{measure}"] = pd.to_numeric(groups[source], errors="coerce")
    for column in ("abnormal_batch_rate", "max_tail_step11_duration", "missing_bottom_batch_count", "physical_batch_count", "ProcessVersion"):
        if column in groups:
            output[f"mech__{column}"] = pd.to_numeric(groups[column], errors="coerce")

    def col(phase: str, measure: str) -> pd.Series:
        name = f"mech__{phase}__{measure}"
        return output[name] if name in output else pd.Series(np.nan, index=output.index)

    dry_temp = col("Stage2_DryMixing", "temp_last")
    oil_temp = col("Stage3_OilLoading", "temp_last")
    wet_temp = col("Stage4_WetMixing", "temp_last")
    wet_duration = col("Stage4_WetMixing", "duration_actual")
    wet_rotor = col("Stage4_WetMixing", "rotor_mean")
    dry_duration = col("Stage2_DryMixing", "duration_actual")
    pid_duration = col("Stage5_PID_Controlled", "duration_actual")
    msb_duration = sum((col(f"Stage6_MSB_PID_Step{i}", "duration_actual") for i in (1, 2, 3)), start=pd.Series(0.0, index=output.index))
    output["mech__oil_to_wet_temp_rise"] = wet_temp - oil_temp
    output["mech__dry_to_oil_temp_rise"] = oil_temp - dry_temp
    output["mech__wet_thermal_exposure"] = wet_duration * (oil_temp + wet_temp) / 2.0
    output["mech__dry_thermal_exposure"] = dry_duration * dry_temp
    output["mech__pid_msb_total_duration"] = pid_duration + msb_duration
    output["mech__wet_rotor_time"] = wet_duration * wet_rotor
    oil_center = oil_temp.median(skipna=True)
    wet_duration_center = wet_duration.median(skipna=True)
    output["mech__oil_hot_wet_short_interaction"] = (oil_temp - oil_center).clip(lower=0) * (wet_duration_center - wet_duration).clip(lower=0)
    usable = [column for column in output if output[column].notna().sum() >= 35 and output[column].nunique(dropna=True) >= 2]
    return output[usable], usable


def ridge(alpha: float) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", RobustScaler()), ("ridge", Ridge(alpha=alpha))])


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    groups = pd.read_pickle(SOURCE).copy()
    oof = pd.read_csv(R12_OOF, usecols=["LabelGroupID", "OOFLevelPrediction"], low_memory=False)
    groups = groups.merge(oof, on="LabelGroupID", how="left", validate="one_to_one")
    groups["ResidualTarget"] = groups["MNY"] - groups["OOFLevelPrediction"]
    mechanism, features = mechanism_frame(groups)
    groups = pd.concat([groups.reset_index(drop=True), mechanism.reset_index(drop=True)], axis=1)
    assert_no_target_leakage(groups, features)
    mechanism.to_csv(OUTPUT / "r14_mechanism_features.csv", index=False)
    train = groups.loc[groups["split"].eq("train") & groups["ResidualTarget"].notna()].copy()
    baseline = groups["LevelPrediction"].to_numpy(copy=True)

    candidates, rows, residual_models = {}, [], {}
    for alpha in (1.0, 10.0, 50.0, 200.0):
        model = ridge(alpha)
        model.fit(train[features], train["ResidualTarget"])
        residual = model.predict(groups[features])
        residual_models[alpha] = model
        for gamma in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5):
            prediction = baseline + gamma * residual
            name = f"R14_RESID_A{int(alpha):03d}_G{int(gamma*100):03d}"
            score, level, trend = candidate_score(groups, prediction)
            candidates[name] = prediction
            rows.append({"candidate": name, "stage": "residual", "alpha": alpha, "gamma": gamma, "beta": np.nan, "score": score, **level, **trend})
    search = pd.DataFrame(rows).sort_values(["score", "mae", "candidate"], kind="stable")
    residual_name = str(search.iloc[0]["candidate"])
    residual_base = candidates[residual_name]
    selected_alpha = float(search.iloc[0]["alpha"])
    selected_gamma = float(search.iloc[0]["gamma"])
    print(f"RESIDUAL selected={residual_name} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    pairs = core.build_pairs(groups, features)
    delta_features = [f"delta__{feature}" for feature in features]
    train_pairs = pairs.loc[pairs["split"].eq("train")].copy()
    pair_models = {}
    pair_outputs = {}
    for alpha in (10.0, 50.0, 200.0):
        model = ridge(alpha)
        model.fit(train_pairs[delta_features], train_pairs["ActualDelta"])
        pair_prediction = pairs[["from_group", "to_group", "OrderID", "split", "ActualDelta"]].copy()
        pair_prediction["DeltaPrediction"] = model.predict(pairs[delta_features])
        pair_models[alpha] = model
        pair_outputs[alpha] = pair_prediction
        for beta in (0.50, 0.75, 0.90, 1.0):
            prediction = recursive_order_prediction(groups, residual_base, pair_prediction, beta)
            name = f"R14_PAIR_A{int(alpha):03d}_B{int(beta*100):03d}"
            score, level, trend = candidate_score(groups, prediction)
            candidates[name] = prediction
            rows.append({"candidate": name, "stage": "pair", "alpha": alpha, "gamma": selected_gamma, "beta": beta, "score": score, **level, **trend})
    search = pd.DataFrame(rows).sort_values(["score", "mae", "candidate"], kind="stable")
    selected_name = str(search.iloc[0]["candidate"])
    selected = search.iloc[0]
    groups["FinalPrediction"] = candidates[selected_name]
    records = []
    for split_name in ("val", "oot"):
        records.append(metric_record(groups, "LevelPrediction", split_name, "R11E_LEVEL_CLEAN"))
        records.append(metric_record(groups, "FinalPrediction", split_name, "R14_FINAL"))
    metrics = pd.DataFrame(records)
    search.to_csv(OUTPUT / "r14_validation_search.csv", index=False)
    metrics.to_csv(OUTPUT / "r14_metrics.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(OUTPUT / "r14_feature_contract.csv", index=False)
    groups[["LabelGroupID", "OrderID", "PalletID", "RecipeID", "split", "regime", "MNY", "LevelPrediction", "OOFLevelPrediction", "ResidualTarget", "FinalPrediction", "production_start", "lab_time"]].to_csv(OUTPUT / "r14_group_predictions.csv", index=False)

    selected_stage = str(selected["stage"])
    bundle = {
        "contract": "V3.7.1-R14-MECHANISM-EXPERTS-V1", "features": features,
        "residual": {"model": residual_models[selected_alpha], "alpha": selected_alpha, "gamma": selected_gamma},
        "selected_stage": selected_stage, "selected_candidate": selected_name,
        "pair": None,
        "target_leakage_gate": "PASSED", "trend_evaluation": "WITHIN_ORDERID_ONLY",
        "stage3_calibration_enabled": False, "automatic_control": "DISABLED",
    }
    if selected_stage == "pair":
        pair_alpha = float(selected["alpha"])
        bundle["pair"] = {"model": pair_models[pair_alpha], "alpha": pair_alpha, "beta": float(selected["beta"])}
        pair_outputs[pair_alpha].to_csv(OUTPUT / "r14_selected_pair_predictions.csv", index=False)
    joblib.dump(bundle, MODEL_OUTPUT / "r14_mechanism_experts_shadow.joblib")
    val = metrics.loc[(metrics["candidate"] == "R14_FINAL") & (metrics["split"] == "val")].iloc[0]
    oot = metrics.loc[(metrics["candidate"] == "R14_FINAL") & (metrics["split"] == "oot")].iloc[0]
    summary = {
        "contract": bundle["contract"], "groups": len(groups), "oof_residual_groups": len(train),
        "mechanism_feature_count": len(features), "pair_counts_by_split": pairs["split"].value_counts().to_dict(),
        "selected_candidate": selected_name, "selected_stage": selected_stage,
        "validation": val.to_dict(), "oot": oot.to_dict(), "target_leakage_gate": "PASSED",
        "target_mae_mny": 2.5,
        "target_trend": {"delta_pearson_min": 0.5, "direction_accuracy_min": 0.65, "delta_std_ratio_range": [0.7, 1.3]},
        "release_status": "NOT_RELEASED_GATES_FAILED_OR_BLIND_WINDOW_PENDING",
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED", "elapsed_seconds": time.perf_counter() - started,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "r14_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
