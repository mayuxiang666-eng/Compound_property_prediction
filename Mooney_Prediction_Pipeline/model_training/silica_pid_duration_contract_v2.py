"""Correct Silica PID duration semantics.

SCP-1-Step-Time is a cumulative in-mixer time point.  A PID interval starts
at the first MST step whose actual temperature reaches PID target minus 3 C,
then continues through the final MSB discharge step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_step_duration(frame: pd.DataFrame, value_col: str, output_col: str) -> pd.DataFrame:
    result = frame.copy()
    result = result.sort_values(["LabelGroupID", "GroupName", "StepNo"], kind="stable")
    previous = result.groupby(["LabelGroupID", "GroupName"])[value_col].shift(1)
    result[output_col] = pd.to_numeric(result[value_col], errors="coerce") - pd.to_numeric(previous, errors="coerce")
    first = previous.isna()
    result.loc[first, output_col] = pd.to_numeric(result.loc[first, value_col], errors="coerce")
    # A reset or malformed timestamp cannot create negative physical time.
    result.loc[result[output_col] < 0, output_col] = np.nan
    return result


def find_pid_start(frame: pd.DataFrame) -> pd.DataFrame:
    temps = frame[(frame["GroupName"] == "AVR_MST") & (frame["kind"] == "temperature")].copy()
    temps["actual_value_mean"] = pd.to_numeric(temps["actual_value_mean"], errors="coerce")
    temps["Target_Temperature"] = pd.to_numeric(temps["Target_Temperature"], errors="coerce")
    temps["threshold_temperature"] = temps["Target_Temperature"] - 3.0
    temps = temps[temps["actual_value_mean"] >= temps["threshold_temperature"]]
    return (
        temps.sort_values(["LabelGroupID", "StepNo"], kind="stable")
        .groupby("LabelGroupID", as_index=False)
        .first()[["LabelGroupID", "StepNo", "actual_value_mean", "threshold_temperature"]]
        .rename(columns={"StepNo": "pid_start_step", "actual_value_mean": "pid_start_actual_temperature"})
    )


def build_pid_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """Return target and actual PID durations split into MST top/MSB bottom."""
    result = frame.copy()
    result["StepNo"] = pd.to_numeric(result["StepNo"], errors="coerce")
    duration_rows = result[result["kind"].eq("duration")].copy()
    duration_rows = add_step_duration(duration_rows, "target_mean", "target_step_duration_s")
    duration_rows = add_step_duration(duration_rows, "actual_value_mean", "actual_step_duration_s")
    starts = find_pid_start(result)
    result = duration_rows.merge(starts, on="LabelGroupID", how="inner", validate="many_to_one")
    result = result[
        ((result["GroupName"] == "AVR_MST") & (result["StepNo"] >= result["pid_start_step"]))
        | (result["GroupName"] == "AVR_MSB")
    ].copy()
    result["pid_section"] = np.where(result["GroupName"].eq("AVR_MST"), "TOP_MST", "BOTTOM_MSB")
    durations = (
        result.groupby(["LabelGroupID", "Target_Temperature", "pid_start_step", "pid_start_actual_temperature", "threshold_temperature", "pid_section"], as_index=False)
        .agg(target_pid_duration_s=("target_step_duration_s", "sum"), actual_pid_duration_s=("actual_step_duration_s", "sum"))
    )
    wide = durations.pivot_table(
        index=["LabelGroupID", "Target_Temperature", "pid_start_step", "pid_start_actual_temperature", "threshold_temperature"],
        columns="pid_section", values=["target_pid_duration_s", "actual_pid_duration_s"], aggfunc="first",
    ).reset_index()
    wide.columns = ["_".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else str(c) for c in wide.columns]
    for col in ("target_pid_duration_s_TOP_MST", "target_pid_duration_s_BOTTOM_MSB", "actual_pid_duration_s_TOP_MST", "actual_pid_duration_s_BOTTOM_MSB"):
        if col not in wide:
            wide[col] = np.nan
    wide["target_pid_total_duration_s"] = wide["target_pid_duration_s_TOP_MST"] + wide["target_pid_duration_s_BOTTOM_MSB"]
    wide["actual_pid_total_duration_s"] = wide["actual_pid_duration_s_TOP_MST"] + wide["actual_pid_duration_s_BOTTOM_MSB"]
    return wide


__all__ = ["add_step_duration", "find_pid_start", "build_pid_intervals"]
