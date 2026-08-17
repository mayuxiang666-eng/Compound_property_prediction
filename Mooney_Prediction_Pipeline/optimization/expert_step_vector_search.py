"""Fail-closed expert-rule Step-vector search for shadow-only recommendations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.optimization.action_feature_lineage_contract import (
    build_action_feature_lineage,
    candidate_has_model_path,
)
from Mooney_Prediction_Pipeline.optimization.counterfactual_phase_features import merge_counterfactual_features
from Mooney_Prediction_Pipeline.optimization.expert_process_response import (
    ProcessResponseModel,
    flatten_step_targets,
    process_predictions_to_v37_features,
)
from Mooney_Prediction_Pipeline.optimization.expert_shadow_policy import (
    DRY_TEMPERATURE_CAP_C,
    EndpointBand,
    identify_discharge_temperature_link,
    policy_action_allowed,
    policy_reasons,
    prepare_recommendation_vector,
    refresh_recommended_time,
)
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor


@dataclass(frozen=True)
class ExpertVectorSearchResult:
    baseline_prediction: float | None
    recommended_steps: pd.DataFrame | None
    audit_table: pd.DataFrame
    status: str
    reason_codes: tuple[str, ...]


def _number(row: pd.Series, column: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else None


def _bounded(row: pd.Series, value: float, minimum_col: str, maximum_col: str) -> bool:
    minimum, maximum = _number(row, minimum_col), _number(row, maximum_col)
    return minimum is not None and maximum is not None and minimum <= value <= maximum


def _action_options(row: pd.Series, material_system: str, wet_discharge_step: int | None, process_cap: float | None) -> list[dict[str, float]]:
    """Expert-approved atomic actions for exactly one Step; no-op is added by caller."""
    phase = str(row.get("mapped_phase", ""))
    step_no = int(row["StepNo"])
    options: list[dict[str, float]] = []
    if phase == "Stage2_DryMixing":
        if policy_action_allowed(row, material_system, "duration"):
            for delta in (5.0, 10.0):
                if _bounded(row, float(row["dur_SP"]) + delta, "step_time_min", "step_time_max"):
                    options.append({"duration": delta})
        if policy_action_allowed(row, material_system, "rotor"):
            for delta in (5.0,):
                value = float(row["rotor_SP"]) + delta
                if 5.0 <= value <= 50.0 and _bounded(row, value, "step_rotor_min", "step_rotor_max"):
                    options.append({"rotor": delta})
        if policy_action_allowed(row, material_system, "temperature"):
            for delta in (1.0, 2.0):
                value = float(row["temp_SP"]) + delta
                if value <= DRY_TEMPERATURE_CAP_C and _bounded(row, value, "step_temp_min", "step_temp_max"):
                    options.append({"temperature": delta})
    elif phase == "Stage4_WetMixing_Ramp":
        # Wet duration is intentionally available only as a lower-RPM coupled action.
        if policy_action_allowed(row, material_system, "rotor") and policy_action_allowed(row, material_system, "duration"):
            rotor_value = float(row["rotor_SP"]) - 5.0
            if 5.0 <= rotor_value <= 50.0 and _bounded(row, rotor_value, "step_rotor_min", "step_rotor_max"):
                for duration_delta in (5.0, 10.0):
                    if _bounded(row, float(row["dur_SP"]) + duration_delta, "step_time_min", "step_time_max"):
                        options.append({"rotor": -5.0, "duration": duration_delta})
        if policy_action_allowed(row, material_system, "temperature") and process_cap is not None and process_cap > 0:
            for delta in (-2.0, -1.0, 1.0, 2.0):
                value = float(row["temp_SP"]) + delta
                if value > 0.0 and value <= process_cap and _bounded(row, value, "step_temp_min", "step_temp_max"):
                    options.append({"temperature": delta})
    elif phase == "Stage5_MST_PID_Silanization" and material_system == "CarbonBlack":
        if policy_action_allowed(row, material_system, "duration"):
            for delta in (-10.0, -5.0, 5.0, 10.0):
                if _bounded(row, float(row["dur_SP"]) + delta, "step_time_min", "step_time_max"):
                    options.append({"duration": delta})
        if step_no == wet_discharge_step and policy_action_allowed(row, material_system, "temperature"):
            for delta in (-1.0, 1.0):
                if _bounded(row, float(row["temp_SP"]) + delta, "step_temp_min", "step_temp_max"):
                    options.append({"temperature": delta})
    elif phase.startswith("Stage6_MSB_PID") and material_system in {"CarbonBlack", "Silica"}:
        duration_deltas = (-10.0, 10.0) if material_system == "Silica" else (-10.0, -5.0, 5.0, 10.0)
        if policy_action_allowed(row, material_system, "duration"):
            for delta in duration_deltas:
                if _bounded(row, float(row["dur_SP"]) + delta, "step_time_min", "step_time_max"):
                    options.append({"duration": delta})
        if step_no == wet_discharge_step and policy_action_allowed(row, material_system, "temperature"):
            for delta in (-1.0, 1.0):
                if _bounded(row, float(row["temp_SP"]) + delta, "step_temp_min", "step_temp_max"):
                    options.append({"temperature": delta})
    return options


def _apply_action(vector: pd.DataFrame, row_index: int, action: dict[str, float]) -> pd.DataFrame:
    out = vector.copy()
    if "duration" in action:
        out.loc[row_index, "recommended_dur_SP"] = float(out.loc[row_index, "recommended_dur_SP"]) + action["duration"]
    if "rotor" in action:
        out.loc[row_index, "recommended_rotor_SP"] = float(out.loc[row_index, "recommended_rotor_SP"]) + action["rotor"]
    if "temperature" in action:
        out.loc[row_index, "recommended_temp_SP"] = float(out.loc[row_index, "recommended_temp_SP"]) + action["temperature"]
    return refresh_recommended_time(out)


def _action_cost(before: pd.DataFrame, candidate: pd.DataFrame) -> float:
    columns = (("dur_SP", "recommended_dur_SP"), ("rotor_SP", "recommended_rotor_SP"), ("temp_SP", "recommended_temp_SP"))
    return float(sum((pd.to_numeric(candidate[after], errors="coerce") - pd.to_numeric(before[before_col], errors="coerce")).abs().sum() for before_col, after in columns))


def _endpoint_reasons(prediction: pd.Series, bands: dict[str, EndpointBand], process_cap: float | None) -> list[str]:
    reasons: list[str] = []
    dry_p90 = float(prediction["dry_end_temp_actual_p90"])
    wet_p90 = float(prediction["wet_end_temp_actual_p90"])
    if dry_p90 > DRY_TEMPERATURE_CAP_C:
        reasons.append("REJECTED_DRY_TEMPERATURE_CAP")
    if process_cap is None or process_cap <= 0:
        reasons.append("BLOCKED_MISSING_PROCESS_TARGET_TEMPERATURE")
    elif wet_p90 > process_cap:
        reasons.append("REJECTED_WET_TEMPERATURE_CAP")
    for output, band in bands.items():
        p50_column = f"{output}_p50"
        if p50_column not in prediction.index:
            continue
        value = float(prediction[p50_column])
        if value < band.recommended_low or value > band.recommended_high:
            reasons.append(f"REJECTED_ENDPOINT_TARGET_{output.upper()}")
        if value < band.safe_low or value > band.safe_high:
            reasons.append(f"REJECTED_ENDPOINT_SAFETY_{output.upper()}")
    post_oil, wet_duration = bands.get("post_oil_temp_actual"), bands.get("wet_duration_actual")
    if post_oil and wet_duration and float(prediction["post_oil_temp_actual_p50"]) > post_oil.safe_high and float(prediction["wet_duration_actual_p50"]) < wet_duration.recommended_low:
        reasons.append("REJECTED_THERMAL_BUDGET")
    return reasons


def _v37_process_features(vector: pd.DataFrame, baseline_feature_row: pd.Series, process_prediction: pd.DataFrame) -> pd.DataFrame:
    # The Target-only aggregate is an intermediate shape adapter.  Every affected
    # temperature/duration feature is then overwritten by predicted actual process values.
    target_shape = merge_counterfactual_features(vector, baseline_feature_row)
    return process_predictions_to_v37_features(target_shape, process_prediction)


def search_expert_step_vectors(
    steps: pd.DataFrame,
    baseline_feature_row: pd.Series,
    raw_predictor: V37RawPredictor,
    response_model: ProcessResponseModel,
    *,
    material_system: str,
    process_target_temperature_cap: float | None,
    endpoint_bands: dict[str, EndpointBand],
    spec_center: float,
    spec_low: float,
    spec_high: float,
    beam_width: int = 100,
    max_candidates: int = 5000,
) -> ExpertVectorSearchResult:
    """Search complete Dry/Wet/PID Target vectors with hard expert gates."""
    if material_system not in {"CarbonBlack", "Silica"}:
        raise ValueError("material_system must be CarbonBlack or Silica")
    if not response_model.approved:
        return ExpertVectorSearchResult(None, None, pd.DataFrame(), "DIAGNOSTIC_ONLY", ("CONTROL_TRANSFER_UNAVAILABLE",))
    before = prepare_recommendation_vector(steps)
    process_baseline = response_model.predict(flatten_step_targets(before))
    baseline_features = _v37_process_features(before, baseline_feature_row, process_baseline)
    baseline_prediction = float(raw_predictor.predict_raw(baseline_features).final_prediction[0])
    discharge = identify_discharge_temperature_link(before)
    discharge_step = None if discharge is None else discharge.control_step_no
    beam: list[tuple[pd.DataFrame, list[dict[str, Any]]]] = [(before, [])]
    eligible = before.sort_values(["GroupName", "StepNo"], kind="stable")
    for row_index, row in eligible.iterrows():
        choices = _action_options(row, material_system, discharge_step, process_target_temperature_cap)
        if not choices:
            continue
        expanded: list[tuple[float, pd.DataFrame, list[dict[str, Any]]]] = []
        for vector, actions in beam:
            expanded.append((_action_cost(before, vector), vector, actions))
            for action in choices:
                candidate = _apply_action(vector, row_index, action)
                reasons = policy_reasons(before, candidate, material_system=material_system, process_target_temperature_cap=process_target_temperature_cap)
                if reasons:
                    continue
                prediction = response_model.predict(flatten_step_targets(candidate)).iloc[0]
                endpoint_penalty = sum(abs(float(prediction[f"{output}_p50"]) - band.center) for output, band in endpoint_bands.items() if f"{output}_p50" in prediction.index)
                expanded.append((_action_cost(before, candidate) + endpoint_penalty, candidate, actions + [{"GroupName": str(row["GroupName"]), "StepNo": int(row["StepNo"]), **action}]))
        expanded.sort(key=lambda item: item[0])
        beam = [(vector, actions) for _, vector, actions in expanded[:beam_width]]
    audits: list[dict[str, Any]] = []
    accepted: list[tuple[tuple[float, float], pd.DataFrame]] = []
    for candidate, actions in beam[:max_candidates]:
        if not actions:
            continue
        process_prediction = response_model.predict(flatten_step_targets(candidate))
        endpoint = process_prediction.iloc[0]
        reasons = list(policy_reasons(before, candidate, material_system=material_system, process_target_temperature_cap=process_target_temperature_cap))
        reasons.extend(_endpoint_reasons(endpoint, endpoint_bands, process_target_temperature_cap))
        candidate_features = _v37_process_features(candidate, baseline_feature_row, process_prediction)
        lineage = build_action_feature_lineage(before, candidate, baseline_features, candidate_features, raw_predictor.feature_columns)
        model_path = candidate_has_model_path(lineage)
        if not model_path:
            reasons.append("NO_MODEL_PATH")
            recommended = None
            response_delta = None
        else:
            recommended = float(raw_predictor.predict_raw(candidate_features).final_prediction[0])
            repeated = float(raw_predictor.predict_raw(candidate_features).final_prediction[0])
            response_delta = recommended - baseline_prediction
            if abs(recommended - repeated) > 1e-6:
                reasons.append("REJECTED_RAW_REPREDICTION_MISMATCH")
            if abs(response_delta) < 0.25:
                reasons.append("REJECTED_MINIMUM_MOONEY_RESPONSE")
            if recommended < spec_low or recommended > spec_high:
                reasons.append("REJECTED_MOONEY_OUT_OF_SPEC")
            if abs(spec_center - recommended) >= abs(spec_center - baseline_prediction):
                reasons.append("REJECTED_OVERSHOOT")
        status = "MANUAL_REVIEW_ONLY" if not reasons else ("NO_MODEL_PATH" if "NO_MODEL_PATH" in reasons else "REJECTED")
        audit = {
            "policy_version": "EXPERT_STEP_VECTOR_V1_20260811",
            "material_system": material_system,
            "actions": json.dumps(actions, sort_keys=True),
            "action_template": "STEP_VECTOR",
            "process_target_temperature_cap": process_target_temperature_cap,
            "dry_temperature_cap_status": "PASSED" if "REJECTED_DRY_TEMPERATURE_CAP" not in reasons else "REJECTED",
            "wet_temperature_cap_status": "PASSED" if "REJECTED_WET_TEMPERATURE_CAP" not in reasons else "REJECTED",
            "thermal_budget_status": "PASSED" if "REJECTED_THERMAL_BUDGET" not in reasons else "REJECTED",
            "control_transfer_status": "PASSED",
            "endpoint_actual_prediction": json.dumps({key: float(value) for key, value in endpoint.items()}, sort_keys=True),
            "baseline_prediction": baseline_prediction,
            "recommended_prediction": recommended,
            "model_response_delta": response_delta,
            "lineage_json": json.dumps(lineage, sort_keys=True),
            "model_path_status": "MODEL_PATH_OK" if model_path else "NO_MODEL_PATH",
            "energy_status": "NOT_RANKED",
            "status": status,
            "reason_codes": json.dumps(list(dict.fromkeys(reasons))),
        }
        audits.append(audit)
        if status == "MANUAL_REVIEW_ONLY" and recommended is not None:
            accepted.append(((abs(spec_center - recommended), _action_cost(before, candidate)), candidate))
    audit_frame = pd.DataFrame(audits)
    if not accepted:
        reason_codes = tuple(sorted({reason for encoded in audit_frame.get("reason_codes", pd.Series(dtype=str)) for reason in json.loads(encoded)}))
        return ExpertVectorSearchResult(baseline_prediction, None, audit_frame, "NO_VALID_CANDIDATE", reason_codes or ("NO_CANDIDATES",))
    accepted.sort(key=lambda item: item[0])
    return ExpertVectorSearchResult(baseline_prediction, accepted[0][1], audit_frame, "MANUAL_REVIEW_ONLY", ())
