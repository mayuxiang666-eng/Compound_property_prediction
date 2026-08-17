"""Build actual-process response labels from mapped historical Step vectors."""

from __future__ import annotations

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.expert_process_response import flatten_step_targets
from Mooney_Prediction_Pipeline.optimization.expert_shadow_policy import identify_discharge_temperature_link, prepare_recommendation_vector


def actual_process_labels(steps: pd.DataFrame) -> dict[str, float | None]:
    """Extract endpoint labels; Wet duration is derived only from actual cumulative time."""
    vector = prepare_recommendation_vector(steps)
    vector = vector.sort_values(["OrderID", "BatchNumber", "GroupName", "StepNo"], kind="stable")
    keys = ["OrderID", "BatchNumber", "GroupName"]
    actual_time = pd.to_numeric(vector.get("time_AV"), errors="coerce")
    vector["actual_dur"] = actual_time - actual_time.groupby([vector[key] for key in keys]).shift(1, fill_value=0.0)
    vector.loc[vector["actual_dur"] < 0, "actual_dur"] = pd.NA
    values: dict[str, float | None] = {}
    for output, phase in (
        ("dry_end_temp_actual", "Stage2_DryMixing"),
        ("post_oil_temp_actual", "Stage3_OilLoading"),
        ("wet_end_temp_actual", "Stage4_WetMixing_Ramp"),
    ):
        subset = vector[vector["mapped_phase"].eq(phase)].sort_values("StepNo", kind="stable")
        observed = pd.to_numeric(subset.get("temp_AV", pd.Series(dtype=float)), errors="coerce").dropna()
        values[output] = float(observed.iloc[-1]) if not observed.empty else None
    wet = vector[vector["mapped_phase"].eq("Stage4_WetMixing_Ramp")]
    duration = pd.to_numeric(wet["actual_dur"], errors="coerce").sum(min_count=1)
    values["wet_duration_actual"] = float(duration) if pd.notna(duration) else None
    discharge = identify_discharge_temperature_link(vector)
    values["discharge_temp_actual"] = None if discharge is None else discharge.actual
    return values


def build_process_response_history(mapped_steps: pd.DataFrame, *, time_column: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Flatten historical Target vectors and attach actual endpoint labels per batch."""
    required = {"OrderID", "BatchNumber", "GroupName", "StepNo", "mapped_phase", time_column}
    missing = required - set(mapped_steps.columns)
    if missing:
        raise ValueError(f"Missing process-history columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    target_columns: set[str] = set()
    for (order_id, batch_number), batch in mapped_steps.groupby(["OrderID", "BatchNumber"], sort=False):
        targets = flatten_step_targets(prepare_recommendation_vector(batch)).iloc[0].to_dict()
        labels = actual_process_labels(batch)
        row = {"OrderID": order_id, "BatchNumber": batch_number, time_column: batch[time_column].iloc[0], **targets, **labels}
        target_columns.update(targets)
        rows.append(row)
    history = pd.DataFrame(rows)
    for column in target_columns:
        if column not in history:
            history[column] = pd.NA
    # One response model requires an identical Target-vector schema; missing
    # target fields are explicit non-controls and represented as zero.
    history.loc[:, sorted(target_columns)] = history.loc[:, sorted(target_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return history, tuple(sorted(target_columns))
