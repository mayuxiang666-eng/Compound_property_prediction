"""Versioned expert-rule contracts for shadow-only step-vector recommendations.

No function in this module writes PLC, MMS, or model state.  It turns the
agreed process-engineering rules into deterministic, fail-closed gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


POLICY_VERSION = "EXPERT_STEP_VECTOR_V1_20260811"
DRY_TEMPERATURE_CAP_C = 120.0
TRACKING_GATES = {
    "duration": {"mae": 2.0, "p95": 5.0, "excursion": 0.05},
    "rotor": {"mae": 1.0, "p95": 2.0, "excursion": 0.05},
    "temperature": {"mae": 2.0, "p95": 5.0, "excursion": 0.05},
}


@dataclass(frozen=True)
class EndpointBand:
    """Historical qualified endpoint targets and absolute operating domain."""

    center: float
    recommended_low: float
    recommended_high: float
    safe_low: float
    safe_high: float
    sample_count: int


@dataclass(frozen=True)
class DischargeTemperatureLink:
    """Cross-step target/actual linkage for lower-mixer discharge temperature."""

    control_index: int
    control_step_no: int
    final_index: int
    final_step_no: int
    target: float
    actual: float | None


def _as_number(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else None


def prepare_recommendation_vector(steps: pd.DataFrame) -> pd.DataFrame:
    """Create one complete Target vector with cumulative recommended time.

    The function deliberately does not infer new controls: a missing or zero
    Target remains non-settable and cannot become a recommendation target.
    """
    required = {"OrderID", "BatchNumber", "GroupName", "StepNo", "time_SP", "rotor_SP", "ram_SP", "temp_SP"}
    missing = required - set(steps.columns)
    if missing:
        raise ValueError(f"Missing Step-vector columns: {sorted(missing)}")
    out = steps.copy().sort_values(["OrderID", "BatchNumber", "GroupName", "StepNo"], kind="stable")
    keys = ["OrderID", "BatchNumber", "GroupName"]
    out["dur_SP"] = pd.to_numeric(out["time_SP"], errors="coerce") - out.groupby(keys)["time_SP"].shift(1, fill_value=0.0)
    if out["dur_SP"].isna().any() or (out["dur_SP"] < 0).any():
        raise ValueError("time_SP must be numeric and non-decreasing within each mixer group.")
    out["recommended_dur_SP"] = out["dur_SP"]
    out["recommended_rotor_SP"] = pd.to_numeric(out["rotor_SP"], errors="coerce")
    out["recommended_ram_SP"] = pd.to_numeric(out["ram_SP"], errors="coerce")
    out["recommended_temp_SP"] = pd.to_numeric(out["temp_SP"], errors="coerce")
    out["recommended_time_SP"] = out.groupby(keys)["recommended_dur_SP"].cumsum()
    return out


def refresh_recommended_time(steps: pd.DataFrame) -> pd.DataFrame:
    """Rebuild cumulative end times after a per-Step duration modification."""
    out = steps.copy().sort_values(["OrderID", "BatchNumber", "GroupName", "StepNo"], kind="stable")
    out["recommended_time_SP"] = out.groupby(["OrderID", "BatchNumber", "GroupName"])["recommended_dur_SP"].cumsum()
    return out


def is_existing_control(row: pd.Series, parameter: str) -> bool:
    column = {"duration": "time_SP", "rotor": "rotor_SP", "temperature": "temp_SP", "ram": "ram_SP"}[parameter]
    value = _as_number(row.get(column))
    return value is not None and value > 0.0 and bool(row.get(f"is_{'time' if parameter == 'duration' else parameter}_settable", value > 0.0))


def identify_discharge_temperature_link(steps: pd.DataFrame) -> DischargeTemperatureLink | None:
    """Find last MSB temperature Target and final MSB actual temperature Step.

    A trailing no-Target anti-stick/discharge step is intentionally excluded
    from the control action while retained as the actual discharge observation.
    """
    msb = steps[steps["GroupName"].eq("AVR_MSB")].sort_values("StepNo", kind="stable")
    if msb.empty:
        return None
    target_rows = msb[pd.to_numeric(msb["temp_SP"], errors="coerce").fillna(0.0).gt(0.0)]
    if target_rows.empty:
        return None
    control = target_rows.iloc[-1]
    final = msb.iloc[-1]
    actual = _as_number(final.get("temp_AV"))
    return DischargeTemperatureLink(
        control_index=int(control.name), control_step_no=int(control["StepNo"]),
        final_index=int(final.name), final_step_no=int(final["StepNo"]),
        target=float(control["temp_SP"]), actual=actual,
    )


def endpoint_actuals(steps: pd.DataFrame) -> dict[str, float | None]:
    """Extract actual terminal temperatures and the actual Wet duration."""
    values: dict[str, float | None] = {}
    phase_names = {
        "dry_end_temp_actual": "Stage2_DryMixing",
        "post_oil_temp_actual": "Stage3_OilLoading",
        "wet_end_temp_actual": "Stage4_WetMixing_Ramp",
    }
    for output, phase in phase_names.items():
        rows = steps[steps.get("mapped_phase", pd.Series("", index=steps.index)).eq(phase)].sort_values("StepNo", kind="stable")
        temp = pd.to_numeric(rows.get("temp_AV", pd.Series(dtype=float)), errors="coerce").dropna()
        values[output] = float(temp.iloc[-1]) if not temp.empty else None
    wet = steps[steps.get("mapped_phase", pd.Series("", index=steps.index)).eq("Stage4_WetMixing_Ramp")]
    duration = pd.to_numeric(wet.get("dur_SP", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    values["wet_duration_actual"] = float(duration) if pd.notna(duration) else None
    discharge = identify_discharge_temperature_link(steps)
    values["discharge_temp_actual"] = None if discharge is None else discharge.actual
    return values


def endpoint_band(values: Iterable[object], *, cap: float | None = None, minimum_samples: int = 150) -> EndpointBand | None:
    """Build qualified-history endpoint band; cap clips both recommendation and safety limits."""
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if len(numeric) < minimum_samples:
        return None
    low, high = np.percentile(numeric, [25, 75])
    safe_low, safe_high = np.percentile(numeric, [5, 95])
    if cap is not None:
        high, safe_high = min(float(high), cap), min(float(safe_high), cap)
    if high < low or safe_high < safe_low:
        return None
    return EndpointBand(
        center=float(np.median(numeric)), recommended_low=float(low), recommended_high=float(high),
        safe_low=float(safe_low), safe_high=float(safe_high), sample_count=int(len(numeric)),
    )


def tracking_summary(errors: Iterable[object], *, limit: float) -> dict[str, float | bool]:
    """Return the common MAE/P95/excursion execution-capability audit."""
    absolute = pd.to_numeric(pd.Series(list(errors)), errors="coerce").abs().dropna()
    if absolute.empty:
        return {"mae": np.nan, "p95": np.nan, "excursion_rate": np.nan, "passed": False}
    mae, p95 = float(absolute.mean()), float(np.percentile(absolute, 95))
    excursion = float((absolute > limit).mean())
    return {"mae": mae, "p95": p95, "excursion_rate": excursion, "passed": mae <= 2.0 and p95 <= limit and excursion <= 0.05}


def policy_action_allowed(row: pd.Series, material_system: str, parameter: str) -> bool:
    """Hard expert-rule action scope; direction/bounds are checked separately."""
    step_no, phase = int(row["StepNo"]), str(row.get("mapped_phase", ""))
    if step_no in {1, 2} or parameter == "ram" or not is_existing_control(row, parameter):
        return False
    if phase == "Stage2_DryMixing":
        return str(row["GroupName"]) == "AVR_MST" and parameter in {"duration", "rotor", "temperature"}
    if phase == "Stage4_WetMixing_Ramp":
        return str(row["GroupName"]) == "AVR_MST" and parameter in {"duration", "rotor", "temperature"}
    if material_system == "CarbonBlack" and (phase == "Stage5_MST_PID_Silanization" or phase.startswith("Stage6_MSB_PID")):
        return parameter in {"duration", "temperature"}
    if material_system == "Silica" and phase.startswith("Stage6_MSB_PID"):
        return parameter in {"duration", "temperature"}
    return False


def policy_reasons(steps_before: pd.DataFrame, steps_after: pd.DataFrame, *, material_system: str, process_target_temperature_cap: float | None) -> tuple[str, ...]:
    """Validate hard expert locks, temperature caps, and Wet rotor-duration coupling."""
    before = steps_before.set_index(["GroupName", "StepNo"])
    after = steps_after.set_index(["GroupName", "StepNo"])
    reasons: list[str] = []
    if set(before.index) != set(after.index):
        return ("REJECTED_STEP_VECTOR_MISMATCH",)
    for key, old in before.iterrows():
        new = after.loc[key]
        step_no, phase = int(old.name[1]), str(old.get("mapped_phase", ""))
        changes = {
            "duration": float(new.get("recommended_dur_SP", old["dur_SP"])) - float(old["dur_SP"]),
            "rotor": float(new.get("recommended_rotor_SP", old["rotor_SP"])) - float(old["rotor_SP"]),
            "temperature": float(new.get("recommended_temp_SP", old["temp_SP"])) - float(old["temp_SP"]),
            "ram": float(new.get("recommended_ram_SP", old["ram_SP"])) - float(old["ram_SP"]),
        }
        if step_no in {1, 2} and any(abs(value) > 1e-9 for value in changes.values()):
            reasons.append("REJECTED_EXPERT_LOCKED_LOADING_STEP")
        if abs(changes["ram"]) > 1e-9:
            reasons.append("REJECTED_EXPERT_RAM_LOCKED")
        for parameter, delta in changes.items():
            if abs(delta) > 1e-9 and not policy_action_allowed(old, material_system, parameter):
                reasons.append("REJECTED_EXPERT_ACTION_SCOPE")
        if phase == "Stage2_DryMixing" and float(new.get("recommended_temp_SP", old["temp_SP"])) > DRY_TEMPERATURE_CAP_C + 1e-9:
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
        if str(old["GroupName"]) == "AVR_MST" and abs(changes["rotor"]) > 1e-9:
            rotor = float(new["recommended_rotor_SP"])
            if rotor < 5.0 - 1e-9 or rotor > 50.0 + 1e-9:
                reasons.append("REJECTED_ROTOR_ENGINEERING_RANGE")
    return tuple(dict.fromkeys(reasons))
