"""Build physical curve boundaries from current-batch cumulative Step execution.

The builder never infers an MST/MSB split from another batch and never fills a
missing route with zero-valued process features.  It uses the cumulative actual
`SCP-1-Step-Time-rel-s` values already recorded for the current physical batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


TIME_VARIABLE = "SCP-1-Step-Time-rel-s"
TOP_GROUPS = {"MST", "AVR_MST"}
BOTTOM_GROUPS = {"MSB", "AVR_MSB"}


@dataclass(frozen=True)
class BoundaryResult:
    steps: pd.DataFrame
    phases: pd.DataFrame
    audit: dict[str, object]


def _canonical_group(value: object) -> str:
    text = str(value).strip().upper()
    if text in TOP_GROUPS:
        return "AVR_MST"
    if text in BOTTOM_GROUPS:
        return "AVR_MSB"
    return text


def _time_rows(batchdata: pd.DataFrame) -> pd.DataFrame:
    required = {"GroupName", "StepNo", "VariablePath", "Value"}
    missing = required.difference(batchdata.columns)
    if missing:
        raise ValueError(f"Missing BatchData columns: {sorted(missing)}")
    rows = batchdata.loc[batchdata["VariablePath"].astype(str).eq(TIME_VARIABLE)].copy()
    rows["GroupName"] = rows["GroupName"].map(_canonical_group)
    rows["StepNo"] = pd.to_numeric(rows["StepNo"], errors="coerce")
    rows["ActualCumulativeTime"] = pd.to_numeric(rows["Value"], errors="coerce")
    if "Target" in rows:
        rows["TargetCumulativeTime"] = pd.to_numeric(rows["Target"], errors="coerce")
    else:
        rows["TargetCumulativeTime"] = np.nan
    rows = rows.dropna(subset=["StepNo", "ActualCumulativeTime"])
    rows["StepNo"] = rows["StepNo"].astype(int)
    duplicate = rows.duplicated(["GroupName", "StepNo"], keep=False)
    if duplicate.any():
        conflicting = rows.loc[duplicate].groupby(["GroupName", "StepNo"])["ActualCumulativeTime"].nunique().gt(1)
        if conflicting.any():
            raise ValueError("BLOCKED_CONFLICTING_STEP_EXECUTION_ROWS")
        rows = rows.drop_duplicates(["GroupName", "StepNo"], keep="last")
    return rows.sort_values(["GroupName", "StepNo"], kind="stable")


def _add_incremental_times(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    previous = out.groupby("GroupName")["ActualCumulativeTime"].shift(1).fillna(0.0)
    out["ActualStepDuration"] = out["ActualCumulativeTime"] - previous
    target_previous = out.groupby("GroupName")["TargetCumulativeTime"].shift(1).fillna(0.0)
    out["TargetStepDuration"] = out["TargetCumulativeTime"] - target_previous
    return out


def build_batch_step_boundaries(
    batchdata: pd.DataFrame,
    curve_length: int,
    phase_mapping: Mapping[tuple[str, int], str],
    *,
    tail_step: tuple[str, int] | None = ("AVR_MST", 11),
    long_tail_threshold_s: float = 75.0,
    timing_tolerance_s: float = 3.0,
    temperature_curve: np.ndarray | None = None,
    discharge_drop_min_c: float = 10.0,
) -> BoundaryResult:
    """Return Step and Phase curve spans for one physical batch.

    When the current-batch temperature curve is supplied, MSB starts at the
    first >= ``discharge_drop_min_c`` instantaneous temperature drop at/after
    the recorded MST endpoint. This is the governed B1 start. The old
    end-anchor remains only as an explicit diagnostic fallback when no
    temperature curve is supplied.
    """

    if int(curve_length) <= 0:
        raise ValueError("BLOCKED_INVALID_CURVE_LENGTH")
    rows = _add_incremental_times(_time_rows(batchdata))
    top = rows.loc[rows["GroupName"].eq("AVR_MST")].copy()
    bottom = rows.loc[rows["GroupName"].eq("AVR_MSB")].copy()
    if top.empty:
        raise ValueError("BLOCKED_MISSING_MST_STEP_EXECUTION")
    if (top["ActualStepDuration"] < -timing_tolerance_s).any() or (bottom["ActualStepDuration"] < -timing_tolerance_s).any():
        raise ValueError("BLOCKED_NONMONOTONIC_CUMULATIVE_STEP_TIME")

    top_total = float(top["ActualCumulativeTime"].max())
    bottom_total = float(bottom["ActualCumulativeTime"].max()) if len(bottom) else np.nan
    bottom_executed = bool(pd.notna(bottom_total) and bottom_total > timing_tolerance_s)
    bottom_end_anchor_start = float(curve_length) - bottom_total if bottom_executed else float(curve_length)
    bottom_start = bottom_end_anchor_start
    discharge_drop_index = None
    discharge_drop_magnitude = None
    msb_start_basis = "CURVE_END_MINUS_MSB_CUMULATIVE_TIME_FALLBACK"
    if temperature_curve is not None and bottom_executed:
        temp = np.asarray(temperature_curve, dtype=float)
        if len(temp) != int(curve_length):
            raise ValueError("BLOCKED_TEMPERATURE_CURVE_LENGTH_MISMATCH")
        differences = temp[:-1] - temp[1:]
        indexes = np.arange(1, len(temp))
        search_start = max(1, int(np.floor(top_total - timing_tolerance_s)))
        candidates = indexes[
            (indexes >= search_start)
            & np.isfinite(temp[:-1])
            & np.isfinite(temp[1:])
            & (differences >= float(discharge_drop_min_c))
        ]
        if len(candidates):
            discharge_drop_index = int(candidates[0])
            discharge_drop_magnitude = float(differences[discharge_drop_index - 1])
            bottom_start = float(discharge_drop_index)
            msb_start_basis = "FIRST_TEMP_DROP_GE_10C_AFTER_MST_ENDPOINT"
    overlap = top_total - bottom_start
    bottom_curve_end = bottom_start + bottom_total if bottom_executed else bottom_start
    if temperature_curve is not None and bottom_executed and discharge_drop_index is None:
        status = "BLOCKED_MSB_DISCHARGE_DROP_NOT_FOUND"
    elif overlap > timing_tolerance_s:
        status = "BLOCKED_STEP_TIME_EXCEEDS_CURVE_LENGTH"
    elif bottom_curve_end > float(curve_length) + timing_tolerance_s:
        status = "BLOCKED_MSB_STEP_TIME_EXCEEDS_CURVE_AFTER_DISCHARGE_DROP"
    elif not bottom_executed:
        status = "PROCESS_UPSET_MSB_NOT_EXECUTED"
    else:
        status = "NORMAL_COMPLETE_ROUTE"

    rows["CurveStart"] = np.nan
    rows["CurveEnd"] = np.nan
    for index, row in rows.iterrows():
        if row["GroupName"] == "AVR_MST":
            start = row["ActualCumulativeTime"] - row["ActualStepDuration"]
            end = row["ActualCumulativeTime"]
        elif row["GroupName"] == "AVR_MSB" and bottom_executed:
            start = bottom_start + row["ActualCumulativeTime"] - row["ActualStepDuration"]
            end = bottom_start + row["ActualCumulativeTime"]
        else:
            continue
        rows.at[index, "CurveStart"] = max(0.0, min(float(curve_length), float(start)))
        rows.at[index, "CurveEnd"] = max(0.0, min(float(curve_length), float(end)))

    mapping = {(_canonical_group(group), int(step)): phase for (group, step), phase in phase_mapping.items()}
    rows["MappedPhase"] = [mapping.get((group, int(step)), "UNMAPPED") for group, step in zip(rows["GroupName"], rows["StepNo"])]
    if tail_step is not None:
        tail_key = (_canonical_group(tail_step[0]), int(tail_step[1]))
        tail_mask = rows["GroupName"].eq(tail_key[0]) & rows["StepNo"].eq(tail_key[1])
        rows.loc[tail_mask, "MappedPhase"] = "PostPID_Tail"
        tail_duration = float(rows.loc[tail_mask, "ActualStepDuration"].iloc[0]) if tail_mask.any() else np.nan
    else:
        tail_duration = np.nan
    long_tail = bool(pd.notna(tail_duration) and tail_duration > long_tail_threshold_s)
    if status == "NORMAL_COMPLETE_ROUTE" and long_tail:
        status = "PROCESS_UPSET_LONG_POST_PID_TAIL"

    phase_rows = []
    valid = rows.dropna(subset=["CurveStart", "CurveEnd"])
    for phase, part in valid.groupby("MappedPhase", sort=False):
        if phase == "UNMAPPED":
            continue
        phase_rows.append(
            {
                "Phase": phase,
                "CurveStart": float(part["CurveStart"].min()),
                "CurveEnd": float(part["CurveEnd"].max()),
                "ActualDuration": float(part["ActualStepDuration"].clip(lower=0.0).sum()),
                "StepCount": int(len(part)),
            }
        )
    transfer_start = min(float(curve_length), max(0.0, top_total))
    transfer_end = min(float(curve_length), max(transfer_start, bottom_start))
    phase_rows.append(
        {
            "Phase": "Transfer",
            "CurveStart": transfer_start,
            "CurveEnd": transfer_end,
            "ActualDuration": max(0.0, transfer_end - transfer_start),
            "StepCount": 0,
        }
    )
    phases = pd.DataFrame(phase_rows)
    audit = {
        "BoundaryContract": "CURRENT_BATCH_CUMULATIVE_STEP_TIME_V1",
        "BoundaryStatus": status,
        "CurveLength": int(curve_length),
        "MSTActualDuration": top_total,
        "MSBActualDuration": bottom_total,
        "MSBExecuted": bottom_executed,
        "MSBCurveStart": bottom_start,
        "MSBCurveEnd": bottom_curve_end,
        "MSBEndAnchorStart": bottom_end_anchor_start,
        "MSBStartBasis": msb_start_basis,
        "DischargeDropIndex": discharge_drop_index,
        "DischargeDropMagnitude_C": discharge_drop_magnitude,
        "DischargeDropMinimum_C": float(discharge_drop_min_c),
        "TransferDuration": max(0.0, transfer_end - transfer_start),
        "PostPIDTailDuration": tail_duration,
        "LongPostPIDTail": long_tail,
        "TimingOverlap": overlap,
        "UnmappedStepCount": int(rows["MappedPhase"].eq("UNMAPPED").sum()),
        "CrossBatchFill": False,
    }
    return BoundaryResult(steps=rows.reset_index(drop=True), phases=phases, audit=audit)


__all__ = ["BoundaryResult", "TIME_VARIABLE", "build_batch_step_boundaries"]
