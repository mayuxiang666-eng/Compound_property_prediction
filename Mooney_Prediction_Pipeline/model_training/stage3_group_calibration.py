"""Leak-safe, group-event Stage3 replay used only as an offline diagnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .group_metric_contract import assert_one_row_per_lab_group


def replay_group_stage3(
    groups: pd.DataFrame,
    *,
    group_col: str = "LabelGroupID",
    scope_col: str = "CalibrationScope",
    actual_col: str = "ActualMNY",
    raw_prediction_col: str = "GroupRawPrediction",
    prediction_time_col: str = "PredictionTime",
    feedback_time_col: str = "FeedbackTime",
    history_groups: int = 3,
    max_abs_offset: float = 5.0,
) -> pd.DataFrame:
    """Replay Stage3 using only prior independent Lab events available in time."""

    if history_groups < 1:
        raise ValueError("history_groups must be positive")
    assert_one_row_per_lab_group(groups, group_col=group_col)
    required = {
        group_col, scope_col, actual_col, raw_prediction_col,
        prediction_time_col, feedback_time_col,
    }
    missing = sorted(required.difference(groups.columns))
    if missing:
        raise ValueError(f"Missing Stage3 group-event columns: {missing}")

    work = groups.copy()
    work[prediction_time_col] = pd.to_datetime(work[prediction_time_col], errors="coerce", utc=True)
    work[feedback_time_col] = pd.to_datetime(work[feedback_time_col], errors="coerce", utc=True)
    work[actual_col] = pd.to_numeric(work[actual_col], errors="coerce")
    work[raw_prediction_col] = pd.to_numeric(work[raw_prediction_col], errors="coerce")
    if work[list(required)].isna().any().any():
        raise ValueError("Stage3 group-event inputs contain missing values")
    work = work.sort_values([prediction_time_col, group_col], kind="stable").reset_index(drop=True)

    offsets: list[float] = []
    support_counts: list[int] = []
    support_ids: list[str] = []
    for _, current in work.iterrows():
        eligible = work.loc[
            work[scope_col].eq(current[scope_col])
            & work[feedback_time_col].le(current[prediction_time_col])
            & work[group_col].ne(current[group_col])
            & work[prediction_time_col].lt(current[prediction_time_col])
        ].sort_values([feedback_time_col, group_col], kind="stable").tail(int(history_groups))
        errors = eligible[actual_col].to_numpy(dtype=float) - eligible[raw_prediction_col].to_numpy(dtype=float)
        offset = float(np.median(errors)) if len(errors) else 0.0
        offsets.append(float(np.clip(offset, -abs(max_abs_offset), abs(max_abs_offset))))
        support_counts.append(int(len(eligible)))
        support_ids.append("|".join(eligible[group_col].astype(str)))

    work["Stage3CandidateOffset"] = offsets
    work["Stage3SupportGroupCount"] = support_counts
    work["Stage3SupportGroupIDs"] = support_ids
    work["CalibratedPrediction"] = work[raw_prediction_col] + work["Stage3CandidateOffset"]
    work["Stage3Status"] = np.where(
        work["Stage3SupportGroupCount"].ge(history_groups),
        "DIAGNOSTIC_CANDIDATE", "INSUFFICIENT_PRIOR_LAB_GROUPS",
    )
    work["Stage3Enabled"] = False
    return work


__all__ = ["replay_group_stage3"]
