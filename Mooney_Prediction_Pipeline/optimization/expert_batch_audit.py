"""Read-only per-batch audit for the expert Step-vector policy."""

from __future__ import annotations

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.expert_shadow_policy import (
    DRY_TEMPERATURE_CAP_C,
    endpoint_actuals,
    identify_discharge_temperature_link,
    policy_action_allowed,
    prepare_recommendation_vector,
)


def audit_expert_batch(steps: pd.DataFrame, *, material_system: str, process_target_temperature_cap: float | None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Expose what is eligible, locked, or blocked before any search is attempted."""
    vector = prepare_recommendation_vector(steps)
    records = []
    for _, row in vector.iterrows():
        for parameter in ("duration", "rotor", "temperature", "ram"):
            allowed = policy_action_allowed(row, material_system, parameter)
            reason = "ELIGIBLE_BY_POLICY" if allowed else "LOCKED_OR_NOT_SETTABLE"
            if parameter == "temperature" and str(row.get("mapped_phase")) == "Stage2_DryMixing" and float(row.get("temp_SP", 0.0) or 0.0) > DRY_TEMPERATURE_CAP_C:
                allowed, reason = False, "BLOCKED_DRY_BASELINE_TEMPERATURE_CAP"
            if parameter == "temperature" and str(row.get("mapped_phase")) == "Stage4_WetMixing_Ramp":
                current = pd.to_numeric(pd.Series([row.get("temp_SP")]), errors="coerce").iloc[0]
                if process_target_temperature_cap is None or process_target_temperature_cap <= 0:
                    allowed, reason = False, "BLOCKED_MISSING_PROCESS_TARGET_TEMPERATURE"
                elif pd.notna(current) and current > process_target_temperature_cap:
                    allowed, reason = False, "BLOCKED_BASELINE_WET_TARGET_EXCEEDS_PROCESS_CAP"
            records.append({
                "GroupName": row["GroupName"], "StepNo": int(row["StepNo"]), "mapped_phase": row.get("mapped_phase"),
                "parameter": parameter, "current_target": row.get({"duration": "dur_SP", "rotor": "rotor_SP", "temperature": "temp_SP", "ram": "ram_SP"}[parameter]),
                "policy_eligible": bool(allowed), "policy_status": reason,
            })
    discharge = identify_discharge_temperature_link(vector)
    summary: dict[str, object] = {
        "policy_version": "EXPERT_STEP_VECTOR_V1_20260811",
        "material_system": material_system,
        "process_target_temperature_cap": process_target_temperature_cap,
        "endpoints_actual": endpoint_actuals(vector),
        "discharge_control_step": None if discharge is None else discharge.control_step_no,
        "discharge_final_step": None if discharge is None else discharge.final_step_no,
        "discharge_target": None if discharge is None else discharge.target,
        "discharge_actual": None if discharge is None else discharge.actual,
    }
    return pd.DataFrame(records), summary
