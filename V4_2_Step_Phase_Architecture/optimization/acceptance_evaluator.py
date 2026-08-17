"""Fail-closed acceptance gates for V3.7 Phase-to-Step recommendation candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


LAB_ONLY_COLUMNS = {"actual_mny", "current_batch_mny"}
VALID_TARGET_SOURCES = {"SPEC_CENTER", "HISTORICAL_QUALIFIED_MEAN"}
PARAMETER_SPECS = {
    "time": ("dur_SP", "recommended_dur_SP", "is_time_settable", "step_time_min", "step_time_max"),
    "rotor": ("rotor_SP", "recommended_rotor_SP", "is_rotor_settable", "step_rotor_min", "step_rotor_max"),
    "ram": ("ram_SP", "recommended_ram_SP", "is_ram_settable", "step_ram_min", "step_ram_max"),
    "temp": ("temp_SP", "recommended_temp_SP", "is_temp_settable", "step_temp_min", "step_temp_max"),
}


@dataclass(frozen=True)
class AcceptanceResult:
    """Decision and machine-readable reasons for exactly one complete step vector."""

    accepted: bool
    status: str
    reason_codes: tuple[str, ...]
    pull_effective: bool
    overshoot_flag: bool


class AcceptanceEvaluator:
    """Evaluates Level 0/1 requirements and strict predicted Mooney improvement.

    The evaluator is deliberately fail-closed: missing bounds, missing settable flags,
    inconsistent identities, or unavailable prediction evidence reject the candidate.
    """

    def __init__(self, single_shift_fraction: float = 0.5, tolerance: float = 1e-6):
        self.single_shift_fraction = single_shift_fraction
        self.tolerance = tolerance

    @staticmethod
    def _has_lab_columns(frame: pd.DataFrame) -> bool:
        return bool({str(column).strip().lower() for column in frame.columns} & LAB_ONLY_COLUMNS)

    @staticmethod
    def _identity_index(frame: pd.DataFrame) -> pd.MultiIndex:
        required = {"GroupName", "StepNo"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing composite step identity columns: {sorted(missing)}")
        return pd.MultiIndex.from_frame(frame[["GroupName", "StepNo"]])

    def _validate_step_vectors(
        self,
        before_steps: pd.DataFrame,
        after_steps: pd.DataFrame,
        expected_phase: str | None,
        expected_time_direction: int | None,
    ) -> list[str]:
        reasons: list[str] = []
        if before_steps.empty or after_steps.empty:
            return ["REJECTED_MISSING_STEP_VECTOR"]
        if self._has_lab_columns(before_steps) or self._has_lab_columns(after_steps):
            return ["REJECTED_LAB_LEAKAGE"]

        try:
            before_index = self._identity_index(before_steps)
            after_index = self._identity_index(after_steps)
        except ValueError:
            return ["REJECTED_STEP_IDENTITY_MISSING"]
        if before_index.has_duplicates or after_index.has_duplicates:
            reasons.append("REJECTED_DUPLICATE_COMPOSITE_STEP_ID")
        if set(before_index) != set(after_index):
            reasons.append("REJECTED_STEP_VECTOR_MISMATCH")
            return reasons

        before = before_steps.set_index(["GroupName", "StepNo"]).sort_index()
        after = after_steps.set_index(["GroupName", "StepNo"]).sort_index()

        for parameter, (before_col, after_col, settable_col, minimum_col, maximum_col) in PARAMETER_SPECS.items():
            if after_col not in after.columns:
                continue
            if before_col not in before.columns:
                reasons.append(f"REJECTED_{parameter.upper()}_CONTRACT_INCOMPLETE")
                continue
            current = pd.to_numeric(before[before_col], errors="coerce")
            recommended = pd.to_numeric(after[after_col], errors="coerce")
            changed = (recommended - current).abs() > self.tolerance
            if not changed.any():
                continue
            required = {before_col, after_col, settable_col, minimum_col, maximum_col}
            available_columns = set(before.columns) | set(after.columns)
            if not required.issubset(available_columns):
                reasons.append(f"REJECTED_{parameter.upper()}_CONTRACT_INCOMPLETE")
                continue

            settable = after[settable_col].fillna(False).astype(bool)
            minimum = pd.to_numeric(after[minimum_col], errors="coerce")
            maximum = pd.to_numeric(after[maximum_col], errors="coerce")

            if current.isna().any() or recommended.isna().any() or minimum.isna().any() or maximum.isna().any():
                reasons.append(f"REJECTED_{parameter.upper()}_VALUE_OR_BOUND_MISSING")
                continue
            if (minimum > maximum).any() or (~recommended.between(minimum, maximum)).any():
                reasons.append("REJECTED_IMPLAUSIBLE")
            if (changed & ~settable).any():
                reasons.append(f"REJECTED_{parameter.upper()}_NOT_SETTABLE")
            if parameter == "temp" and (changed & (current <= 0)).any():
                reasons.append("REJECTED_TEMP_NON_CONTROL_STEP")

            allowed_shift = (maximum - minimum) * self.single_shift_fraction
            if ((recommended - current).abs() - allowed_shift > self.tolerance).any():
                reasons.append("REJECTED_SINGLE_STEP_SHIFT_LIMIT")

        if {"recommended_rotor_SP", "recommended_dur_SP"}.issubset(after.columns):
            rotor = pd.to_numeric(after["recommended_rotor_SP"], errors="coerce")
            duration = pd.to_numeric(after["recommended_dur_SP"], errors="coerce")
            if ((rotor < 10.0) & (duration > 60.0)).any():
                reasons.append("REJECTED_LOW_ROTOR_LONG_TIME")

        if expected_phase is not None and expected_time_direction in {-1, 1}:
            phase_mask = after.get("mapped_phase", pd.Series("", index=after.index)).eq(expected_phase)
            if not phase_mask.any():
                reasons.append("REJECTED_CORRECTED_PHASE_UNMAPPED")
            elif "recommended_dur_SP" not in after.columns or "dur_SP" not in before.columns:
                reasons.append("REJECTED_TIME_CONTRACT_INCOMPLETE")
            else:
                net_delta = (
                    pd.to_numeric(after.loc[phase_mask, "recommended_dur_SP"], errors="coerce")
                    - pd.to_numeric(before.loc[phase_mask, "dur_SP"], errors="coerce")
                ).sum()
                if net_delta * expected_time_direction <= self.tolerance:
                    reasons.append("REJECTED_NET_ACTION_DIRECTION")
        return reasons

    def evaluate(
        self,
        before_steps: pd.DataFrame,
        after_steps: pd.DataFrame,
        *,
        target_center: float,
        target_source: str,
        baseline_prediction: float,
        recommended_prediction: float,
        re_predicted_recommendation: float,
        reproducible: bool,
        spec_lower: float,
        spec_upper: float,
        expected_phase: str | None = None,
        expected_time_direction: int | None = None,
    ) -> AcceptanceResult:
        """Apply all hard gates that can be known before real lab feedback arrives."""
        reasons = self._validate_step_vectors(
            before_steps,
            after_steps,
            expected_phase,
            expected_time_direction,
        )
        if target_source not in VALID_TARGET_SOURCES:
            reasons.append("REJECTED_ILLEGAL_TARGET_SOURCE")
        if not np.isfinite(target_center):
            reasons.append("REJECTED_TARGET_CENTER_MISSING")
        if not reproducible:
            reasons.append("REJECTED_PREDICTION_NOT_REPRODUCIBLE")

        prediction_values = (baseline_prediction, recommended_prediction, re_predicted_recommendation)
        if not all(np.isfinite(value) for value in prediction_values):
            reasons.append("REJECTED_RAW_PREDICTION_UNAVAILABLE")
        elif abs(recommended_prediction - re_predicted_recommendation) > self.tolerance:
            reasons.append("REJECTED_RAW_REPREDICTION_MISMATCH")

        gap_before = abs(target_center - baseline_prediction)
        gap_after = abs(target_center - recommended_prediction)
        overshoot_flag = gap_after > gap_before + self.tolerance
        if overshoot_flag:
            reasons.append("REJECTED_OVERSHOOT")
        if recommended_prediction < spec_lower - self.tolerance or recommended_prediction > spec_upper + self.tolerance:
            reasons.append("REJECTED_MOONEY_OUT_OF_SPEC")

        deduplicated_reasons = tuple(dict.fromkeys(reasons))
        accepted = not deduplicated_reasons
        pull_effective = accepted and gap_after < gap_before - self.tolerance and not overshoot_flag
        return AcceptanceResult(
            accepted=accepted,
            status="PASSED" if accepted else "REJECTED",
            reason_codes=deduplicated_reasons,
            pull_effective=pull_effective,
            overshoot_flag=overshoot_flag,
        )