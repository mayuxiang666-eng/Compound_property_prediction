"""Replay the frozen V3.7 four-stage expert artifact on corrected pilot Phase rows.

This is an inference/identity audit, not a training entry point.  It fails closed
unless the production artifact is the expected Stage 1 + 1b + Material/Route
Stage 2 expert model.  Stage 3 is simulated once per Lab label group, never once
per replicated batch label.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor
from services.mooney_predict_service.phase_group_inference_v3 import prepare_phase_batches_v3


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports" / "corrected_phase_history" / "phase_exact_pilots_corrected_static_v1.csv"
TRAINING_SOURCE = ROOT / "data" / "stage_statistics_enriched_all_features_weather_v4.csv"
PACKAGE = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v37_production_model_package"
OUTPUT = ROOT / "reports" / "corrected_phase_history" / "v37_frozen_four_stage_exact_pilot_replay_v2"
EXPECTED_MODEL_SHA256 = "9AA220AC0369AEA3768663DC87C6728C6C3D30B1F2C04D82ACFA92BEE552D794"
EXPECTED_ROUTE_EXPERTS = {
    ("CarbonBlack", "no_oil_dry"): "CarbonBlackSubsystemPredictor",
    ("CarbonBlack", "oil_wet"): "CarbonBlackSubsystemPredictor",
    ("Silica", "no_oil_dry"): "SilicaSubsystemPredictor",
    ("Silica", "oil_wet"): "SilicaSubsystemPredictor",
}
PILOTS = {
    "B00458": "M1-B00458---- 12 004",
    "T09170WS": "M1-T09170WS-- 02 004",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_four_stage_artifact(model: Any, calibrator: Any, model_hash: str) -> dict[str, Any]:
    """Reject any artifact that is not the documented V3.7 expert architecture."""
    errors: list[str] = []
    if model_hash != EXPECTED_MODEL_SHA256:
        errors.append(f"model_sha256={model_hash}")
    expected_flags = {
        "use_material_route_matrix": True,
        "use_silica_subsystem": True,
        "use_cb_subsystem": True,
    }
    for name, expected in expected_flags.items():
        if getattr(model, name, None) is not expected:
            errors.append(f"{name}={getattr(model, name, None)!r}")
    if float(getattr(model, "shrinkage_k", np.nan)) != 5.0:
        errors.append(f"shrinkage_k={getattr(model, 'shrinkage_k', None)!r}")

    actual_keys = set(getattr(model, "stage2_experts_", {}))
    if actual_keys != set(EXPECTED_ROUTE_EXPERTS):
        errors.append(f"stage2_route_keys={sorted(map(str, actual_keys))}")
    expert_contract: dict[str, Any] = {}
    for route, expected_class in EXPECTED_ROUTE_EXPERTS.items():
        expert = getattr(model, "stage2_experts_", {}).get(route)
        actual_class = type(expert).__name__ if expert is not None else "MISSING"
        if actual_class != expected_class:
            errors.append(f"route={route!r}:class={actual_class}")
        combiner = getattr(expert, "combiner_", None)
        if combiner is None or not bool(getattr(combiner, "positive", False)):
            errors.append(f"route={route!r}:non_negative_combiner_missing")
        expert_contract[str(route)] = {
            "class": actual_class,
            "combiner": type(combiner).__name__ if combiner is not None else None,
            "combiner_positive": bool(getattr(combiner, "positive", False)),
            "combiner_coefficients": (
                np.asarray(getattr(combiner, "coef_", []), dtype=float).tolist()
                if combiner is not None else []
            ),
        }

    calibrator_contract = {
        "class": type(calibrator).__name__,
        "lag_k": int(getattr(calibrator, "lag_k", -1)),
        "alpha_base": float(getattr(calibrator, "alpha_base", np.nan)),
        "max_offset_abs": float(getattr(calibrator, "max_offset_abs", np.nan)),
        "adaptive": bool(getattr(calibrator, "adaptive", False)),
    }
    if calibrator_contract != {
        "class": "Stage3DelayedFeedbackCalibrator",
        "lag_k": 3,
        "alpha_base": 0.3,
        "max_offset_abs": 4.0,
        "adaptive": True,
    }:
        errors.append(f"stage3={calibrator_contract}")
    if errors:
        raise RuntimeError("Frozen V3.7 four-stage identity gate failed: " + "; ".join(errors))
    return {
        "model_sha256": model_hash,
        "stage1": "Shared Global Recipe Surface GBDT",
        "stage1b": "Regularized Compound Bias Shrinkage k=5.0",
        "stage2": "MaterialSystem x PhaseRoute residual expert matrix",
        "stage2_routes": expert_contract,
        "stage3": calibrator_contract,
        "identity_status": "V37_FOUR_STAGE_EXPERT_ARTIFACT_CONFIRMED",
    }


def normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["OrderID"] = out["OrderID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out["PalletID"] = out["PalletID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out["BatchNumber"] = pd.to_numeric(out["BatchNumber"], errors="raise").astype(int)
    out["LabelGroupID"] = out["OrderID"] + "|" + out["PalletID"]
    out["BatchKey"] = out["OrderID"] + "|" + out["BatchNumber"].astype(str)
    return out


def add_training_membership(source: pd.DataFrame, training_source: Path) -> pd.DataFrame:
    training = pd.read_csv(
        training_source,
        low_memory=False,
        usecols=lambda column: column in {"OrderID", "BatchNumber", "MNY"},
    ).dropna(subset=["MNY"])
    training = normalise_keys(training.assign(PalletID="TRAINING_MEMBERSHIP_ONLY"))
    seen = set(training["BatchKey"])
    out = source.copy()
    out["V37TrainingMembership"] = np.where(
        out["BatchKey"].isin(seen), "SEEN_IN_FROZEN_V37_TRAINING", "UNSEEN_BY_BATCH_KEY"
    )
    return out


def add_subexpert_audit(model: Any, features: pd.DataFrame, batch: pd.DataFrame) -> float:
    """Expose every Stage 2 subexpert and verify weighted closure to Stage 2."""
    x_delta, routes = model._transform_process_deltas(features, "material_system")
    audit_columns = [
        "SilicaPID", "SilicaWet", "SilicaBottom", "SilicaMaterial",
        "CBPrep", "CBBottom", "CBMaterial",
    ]
    for column in audit_columns:
        batch[f"{column}ExpertRaw"] = np.nan
        batch[f"{column}ExpertWeighted"] = np.nan
    weighted_sum = np.zeros(len(features), dtype=float)
    route_names = np.empty(len(features), dtype=object)

    name_map = {
        "pid": "SilicaPID", "wet": "SilicaWet", "bottom": "SilicaBottom",
        "material": "SilicaMaterial", "prep": "CBPrep",
    }
    for route in routes.unique():
        mask = routes.eq(route)
        positions = np.flatnonzero(mask.to_numpy())
        expert = model.stage2_experts_[route]
        predictions = expert.predict_experts(x_delta.loc[mask])
        coefficients = np.asarray(expert.combiner_.coef_, dtype=float)
        for index, (expert_name, values) in enumerate(predictions.items()):
            display = name_map[expert_name]
            if route[0] == "CarbonBlack" and expert_name in {"bottom", "material"}:
                display = "CB" + expert_name.capitalize()
            raw = np.asarray(values, dtype=float)
            weighted = raw * coefficients[index]
            batch.iloc[positions, batch.columns.get_loc(f"{display}ExpertRaw")] = raw
            batch.iloc[positions, batch.columns.get_loc(f"{display}ExpertWeighted")] = weighted
            weighted_sum[positions] += weighted
        route_names[positions] = f"{route[0]}|{route[1]}"
    batch["Stage2ExpertRoute"] = route_names
    batch["Stage2WeightedExpertSum"] = weighted_sum
    return float(np.max(np.abs(batch["Stage2Delta"].to_numpy(dtype=float) - weighted_sum)))


def metric_block(groups: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    if len(groups) < 2:
        return {"groups": int(len(groups)), "mae": None, "rmse": None, "r2": None, "bias": None}
    y = groups["ActualMNY"].to_numpy(dtype=float)
    p = groups[prediction_column].to_numpy(dtype=float)
    return {
        "groups": int(len(groups)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)),
        "bias": float(np.mean(p - y)),
        "actual_std": float(np.std(y)),
        "prediction_std": float(np.std(p)),
        "pearson_r": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else None,
    }


def run() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = normalise_keys(pd.read_csv(SOURCE, low_memory=False))
    source["MNY"] = pd.to_numeric(source["MNY"], errors="coerce")
    source["OrderStartTime"] = pd.to_datetime(source["OrderStartTime"], errors="raise")
    source = add_training_membership(source, TRAINING_SOURCE)

    predictor = V37RawPredictor(PACKAGE).load()
    calibrator = joblib.load(PACKAGE / "stage3_calibrator.joblib")
    model_hash = sha256_file(PACKAGE / "hybrid_model.joblib")
    architecture = validate_four_stage_artifact(predictor.model, calibrator, model_hash)
    prepared = prepare_phase_batches_v3(source, predictor.metadata)
    raw = predictor.predict_raw(prepared.features)
    repeated = predictor.predict_raw(prepared.features)
    deterministic_delta = float(np.max(np.abs(raw.final_prediction - repeated.final_prediction)))

    identity = [
        "OrderID", "BatchNumber", "PalletID", "LabelGroupID", "CompoundName",
        "OrderStartTime", "material_system",
    ]
    batch = prepared.features[identity].copy().reset_index(drop=True)
    batch["Stage1Prediction"] = raw.stage1_prediction
    batch["Stage1bBias"] = raw.stage1b_bias
    batch["Stage2Delta"] = raw.stage2_residual
    batch["RawPrediction"] = raw.final_prediction
    membership = source[["OrderID", "BatchNumber", "V37TrainingMembership"]].drop_duplicates()
    batch = batch.merge(membership, on=["OrderID", "BatchNumber"], how="left", validate="one_to_one")
    expert_closure_delta = add_subexpert_audit(predictor.model, prepared.features.reset_index(drop=True), batch)

    quality = prepared.quality.copy()
    expected = quality.groupby("LabelGroupID", as_index=False).agg(
        ExpectedBatchCount=("BatchNumber", "size"),
        AcceptedBatchCount=("FeatureQualityStatus", lambda values: int((values == "MODEL_INPUT_OK").sum())),
    )
    labels = source.groupby("LabelGroupID", as_index=False).agg(
        ActualMNY=("MNY", "median"), ActualMNYMin=("MNY", "min"), ActualMNYMax=("MNY", "max"),
        LabResultTime=("test_result_start_time", "max"),
    )
    labels["MNYConflict"] = (labels["ActualMNYMax"] - labels["ActualMNYMin"]).abs().gt(1e-9)
    groups = batch.groupby(
        ["OrderID", "PalletID", "LabelGroupID", "CompoundName"], as_index=False
    ).agg(
        BatchCount=("BatchNumber", "size"), ProductTime=("OrderStartTime", "max"),
        GroupRawPrediction=("RawPrediction", "mean"), BatchPredictionStd=("RawPrediction", "std"),
        Stage1GroupMean=("Stage1Prediction", "mean"), Stage1bGroupMean=("Stage1bBias", "mean"),
        Stage2GroupMean=("Stage2Delta", "mean"),
        SeenBatchCount=("V37TrainingMembership", lambda values: int((values == "SEEN_IN_FROZEN_V37_TRAINING").sum())),
    )
    groups["BatchPredictionStd"] = groups["BatchPredictionStd"].fillna(0.0)
    groups = groups.merge(expected, on="LabelGroupID", how="right", validate="one_to_one")
    groups = groups.merge(labels, on="LabelGroupID", how="left", validate="one_to_one")
    groups["GroupComplete"] = groups["ExpectedBatchCount"].eq(groups["AcceptedBatchCount"])
    groups["TrainingMembership"] = np.select(
        [groups["SeenBatchCount"].eq(groups["BatchCount"]), groups["SeenBatchCount"].eq(0)],
        ["ALL_SEEN_IN_FROZEN_V37_TRAINING", "ALL_UNSEEN_BY_BATCH_KEY"],
        default="MIXED_TRAINING_MEMBERSHIP",
    )
    blocked = groups.loc[~groups["GroupComplete"] | groups["MNYConflict"]].copy()
    complete = groups.loc[groups["GroupComplete"] & ~groups["MNYConflict"]].copy()
    complete = complete.sort_values(["ProductTime", "OrderID", "PalletID"], kind="stable").reset_index(drop=True)

    stage3_input = complete[["CompoundName", "ActualMNY"]].rename(columns={"ActualMNY": "MNY"})
    calibrated, offsets = calibrator.calibrate_time_series(
        stage3_input, complete["GroupRawPrediction"].to_numpy(dtype=float),
        target_col="MNY", group_col="CompoundName",
    )
    complete["Stage3Offset"] = offsets
    complete["GroupCalibratedPrediction"] = calibrated
    complete["Stage3SimulationContract"] = "ONE_UPDATE_PER_LABEL_GROUP_LAG3_CHRONOLOGICAL"
    complete["EvaluationScope"] = np.where(
        complete["TrainingMembership"].eq("ALL_UNSEEN_BY_BATCH_KEY"),
        "STRICT_UNSEEN_BY_KEY", "TRAINING_SEEN_IDENTITY_REPLAY_NOT_OOT",
    )

    metrics: dict[str, Any] = {}
    for name, compound in PILOTS.items():
        selected = complete.loc[complete["CompoundName"].eq(compound)]
        unseen = selected.loc[selected["EvaluationScope"].eq("STRICT_UNSEEN_BY_KEY")]
        metrics[name] = {
            "all_complete_groups_raw_identity_replay": metric_block(selected, "GroupRawPrediction"),
            "all_complete_groups_stage3_simulation": metric_block(selected, "GroupCalibratedPrediction"),
            "strict_unseen_group_count": int(len(unseen)),
            "strict_oot_metrics": (
                metric_block(unseen, "GroupRawPrediction") if len(unseen) else None
            ),
        }

    batch.to_csv(OUTPUT / "batch_four_stage_predictions.csv", index=False, encoding="utf-8-sig")
    complete.to_csv(OUTPUT / "complete_group_predictions.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(OUTPUT / "blocked_group_audit.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUTPUT / "feature_quality_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "experiment": "FROZEN-V3.7-FOUR-STAGE-EXPERT-EXACT-PILOT-REPLAY-V2",
        "architecture": architecture,
        "source": str(SOURCE),
        "training_membership_source": str(TRAINING_SOURCE),
        "source_batch_rows": int(len(source)),
        "accepted_batch_rows": int(len(batch)),
        "complete_group_count": int(len(complete)),
        "blocked_group_count": int(len(blocked)),
        "training_membership_counts": {
            str(key): int(value) for key, value in complete["TrainingMembership"].value_counts().items()
        },
        "deterministic_max_abs_delta": deterministic_delta,
        "stage2_subexpert_closure_max_abs_delta": expert_closure_delta,
        "metrics": metrics,
        "oot_interpretation": (
            "Frozen production V3.7 was fitted on 100% of its 18,783-row source. "
            "Seen groups are identity/stability evidence only and are never reported as OOT."
        ),
        "model_path_status": "MODEL_PATH_OK" if len(complete) else "BLOCKED_NO_COMPLETE_GROUP",
        "automatic_control": "DISABLED",
        "manual_review_only_issued": False,
        "database_write": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
