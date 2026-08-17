"""Corrected runtime policy entry point for expert shadow recommendations."""

from __future__ import annotations

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.expert_shadow_policy import *  # noqa: F401,F403
from Mooney_Prediction_Pipeline.optimization import expert_shadow_policy as _base


def policy_reasons(steps_before: pd.DataFrame, steps_after: pd.DataFrame, *, material_system: str, process_target_temperature_cap: float | None) -> tuple[str, ...]:
    """V2 fix: GroupName/StepNo are index levels during vector comparison."""
    before = steps_before.set_index(["GroupName", "StepNo"])
    after = steps_after.set_index(["GroupName", "StepNo"])
    if set(before.index) != set(after.index):
        return ("REJECTED_STEP_VECTOR_MISMATCH",)
    reasons: list[str] = []
    for key, old in before.iterrows():
        new = after.loc[key]
        group_name, step_no = str(key[0]), int(key[1])
        phase = str(old.get("mapped_phase", ""))
        def delta(before_col: str, after_col: str) -> float:
            old_value = pd.to_numeric(pd.Series([old.get(before_col)]), errors="coerce").iloc[0]
            new_value = pd.to_numeric(pd.Series([new.get(after_col)]), errors="coerce").iloc[0]
            return 0.0 if pd.isna(old_value) and pd.isna(new_value) else float(new_value - old_value)
        changes = {
            "duration": delta("dur_SP", "recommended_dur_SP"),
            "rotor": delta("rotor_SP", "recommended_rotor_SP"),
            "temperature": delta("temp_SP", "recommended_temp_SP"),
            "ram": delta("ram_SP", "recommended_ram_SP"),
        }
        if step_no in {1, 2} and any(abs(value) > 1e-9 for value in changes.values()):
            reasons.append("REJECTED_EXPERT_LOCKED_LOADING_STEP")
        if abs(changes["ram"]) > 1e-9:
            reasons.append("REJECTED_EXPERT_RAM_LOCKED")
        policy_row = old.copy()
        policy_row["GroupName"] = group_name
        policy_row["StepNo"] = step_no
        for parameter, value in changes.items():
            if abs(value) > 1e-9 and not _base.policy_action_allowed(policy_row, material_system, parameter):
                reasons.append("REJECTED_EXPERT_ACTION_SCOPE")
        if phase == "Stage2_DryMixing" and float(new.get("recommended_temp_SP", old["temp_SP"])) > _base.DRY_TEMPERATURE_CAP_C + 1e-9:
            reasons.append("REJECTED_DRY_TEMPERATURE_CAP")
        if phase == "Stage4_WetMixing_Ramp":
            target = float(new.get("recommended_temp_SP", old["temp_SP"]))
            if abs(changes["temperature"]) > 1e-9 and (process_target_temperature_cap is None or process_target_temperature_cap <= 0):
                reasons.append("BLOCKED_MISSING_PROCESS_TARGET_TEMPERATURE")
            elif abs(changes["temperature"]) > 1e-9 and target > float(process_target_temperature_cap) + 1e-9:
                reasons.append("REJECTED_WET_TEMPERATURE_CAP")
            if changes["rotor"] < -1e-9 and changes["duration"] < -1e-9:
                reasons.append("REJECTED_WET_ROTOR_DURATION_COUPLING")
            if abs(changes["rotor"]) > 1e-9 and abs(changes["rotor"]) < 5.0 - 1e-9:
                reasons.append("REJECTED_ROTOR_MINIMUM_INCREMENT")
        if group_name == "AVR_MST" and abs(changes["rotor"]) > 1e-9:
            rotor = float(new["recommended_rotor_SP"])
            if rotor < 5.0 - 1e-9 or rotor > 50.0 + 1e-9:
                reasons.append("REJECTED_ROTOR_ENGINEERING_RANGE")
    return tuple(dict.fromkeys(reasons))
