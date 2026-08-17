"""Build a causal M1-only prior-order state contract for R1B.

Only process measurements from an earlier completed Lab group in the same
OrderID are used.  Prior numeric state is masked when TemplateKey changes;
the version/template transition itself remains an explicit flag.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
SOURCE_BATCH = BASE / "v371_r1b_m1_only_snapshot_v1" / "m1_only_batch_pid_quality_features.csv"
SOURCE_GROUP = BASE / "v371_r1b_m1_only_snapshot_v1" / "m1_only_group_label_snapshot_with_pid_quality.csv"
SOURCE_METADATA = BASE / "v371_r1b_m1_only_snapshot_v1" / "m1_only_metadata_v1.json"
OUT = BASE / "v371_r1b_m1_order_state_snapshot_v1"


PROCESS_STATE_FEATURES = [
    "Stage2_DryMixing_Duration", "Stage2_DryMixing_temp_Mean",
    "Stage2_DryMixing_temp_Std", "Stage2_DryMixing_power_Integral",
    "Stage2_DryMixing_Torque_Integral", "Stage2_DryMixing_RotorSpeed_Mean",
    "Stage3_OilLoading_Duration", "Stage3_OilLoading_temp_Mean",
    "Stage3_OilLoading_Torque_Integral", "Stage3_OilLoading_Specific_Energy",
    "Stage4_WetMixing_Duration", "Stage4_WetMixing_temp_Mean",
    "Stage4_WetMixing_temp_Std", "Stage4_WetMixing_Torque_Integral",
    "Stage4_WetMixing_Specific_Energy", "Stage4_WetMixing_RotorSpeed_Mean",
    "Stage5_PID_Duration", "Stage5_PID_temp_Mean",
    "Stage5_PID_Torque_Integral", "Stage5_PID_Specific_Energy",
    "Stage5_PID_RotorSpeed_Mean", "Stage6_BottomMixing_Duration",
    "Stage6_BottomMixing_temp_Mean", "Stage6_BottomMixing_Torque_Integral",
    "Stage6_BottomMixing_Specific_Energy", "Stage6_BottomMixing_RotorSpeed_Mean",
    "phys_discharge_temp", "phys_temp_integral", "phys_shear_history_total",
    "pid_actual_total_duration_s", "pid_target_total_duration_s",
    "pid_actual_top_duration_s", "pid_actual_bottom_duration_s",
]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    batch = pd.read_csv(SOURCE_BATCH, low_memory=False)
    group = pd.read_csv(SOURCE_GROUP, low_memory=False)
    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))

    batch = batch.loc[batch["CompoundName"].astype(str).str.startswith("M1-")].copy()
    group = group.loc[group["CompoundName"].astype(str).str.startswith("M1-")].copy()
    for column in ("BatchStartTime", "BatchEndTime"):
        batch[column] = pd.to_datetime(batch[column], errors="coerce", utc=True)
    if batch[["BatchStartTime", "BatchEndTime"]].isna().any().any():
        raise ValueError("M1_ORDER_STATE_INVALID_BATCH_TIME")

    available = [column for column in PROCESS_STATE_FEATURES if column in batch.columns]
    if len(available) < 20:
        raise ValueError(f"M1_ORDER_STATE_TOO_FEW_PROCESS_FEATURES: {len(available)}")

    state_agg = {
        "GroupProductionStart": ("BatchStartTime", "min"),
        "GroupProductionComplete": ("BatchEndTime", "max"),
        **{column: (column, "mean") for column in available},
    }
    if "StepPatternHash" not in group.columns and "StepPatternHash" in batch.columns:
        state_agg["StepPatternHash"] = ("StepPatternHash", "first")
    group_state = batch.groupby("LabelGroupID", as_index=False).agg(
        **state_agg,
    )
    group_state["LabelGroupID"] = group_state["LabelGroupID"].astype(str)
    group["LabelGroupID"] = group["LabelGroupID"].astype(str)
    replace_columns = [column for column in state_agg if column in group.columns]
    if replace_columns:
        group = group.drop(columns=replace_columns)
    group = group.merge(group_state, on="LabelGroupID", how="inner", validate="one_to_one")
    if len(group) != group["LabelGroupID"].nunique():
        raise ValueError("M1_ORDER_STATE_GROUP_DUPLICATE")

    group["OrderID"] = group["OrderID"].astype(str)
    group = group.sort_values(
        ["OrderID", "GroupProductionStart", "LabelGroupID"], kind="stable"
    ).reset_index(drop=True)
    order = group.groupby("OrderID", sort=False)
    previous_complete = order["GroupProductionComplete"].shift(1)
    previous_template = order["TemplateKey"].shift(1)
    previous_version = order["ProcessVersion"].shift(1)
    previous_step = order["StepPatternHash"].shift(1) if "StepPatternHash" in group else pd.Series(index=group.index, dtype=object)
    current_start = group["GroupProductionStart"]
    valid_prior = previous_complete.notna() & previous_complete.le(current_start)
    same_template = valid_prior & group["TemplateKey"].eq(previous_template)
    same_version = valid_prior & group["ProcessVersion"].eq(previous_version)
    same_step = valid_prior & group["StepPatternHash"].eq(previous_step) if "StepPatternHash" in group else valid_prior

    group["state_prior_available"] = valid_prior.astype(int)
    group["state_prior_same_template"] = same_template.astype(int)
    group["state_prior_same_process_version"] = same_version.astype(int)
    group["state_prior_same_step_pattern"] = same_step.astype(int)
    group["state_template_changed_from_prior"] = (valid_prior & ~same_template).astype(int)
    group["state_process_version_changed_from_prior"] = (valid_prior & ~same_version).astype(int)
    group["state_step_pattern_changed_from_prior"] = (valid_prior & ~same_step).astype(int)
    group["state_prior_gap_hours"] = (
        (current_start - previous_complete).dt.total_seconds() / 3600.0
    ).where(valid_prior)
    group["state_order_group_index"] = order.cumcount()

    added = [
        "state_prior_available", "state_prior_same_template",
        "state_prior_same_process_version", "state_prior_same_step_pattern",
        "state_template_changed_from_prior", "state_process_version_changed_from_prior",
        "state_step_pattern_changed_from_prior", "state_prior_gap_hours",
        "state_order_group_index",
    ]
    for feature in available:
        current = _numeric(group, feature)
        prior = pd.to_numeric(order[feature].shift(1), errors="coerce")
        prior = prior.where(same_template)
        group[f"state_prior__{feature}"] = prior
        group[f"state_delta_prior__{feature}"] = current - prior
        added.extend([f"state_prior__{feature}", f"state_delta_prior__{feature}"])

    # Process-execution residuals are current-batch observations, not targets.
    group["state_discharge_minus_target_temp"] = (
        _numeric(group, "phys_discharge_temp") - _numeric(group, "Target_Temperature")
    )
    group["state_pid_actual_minus_target_duration"] = (
        _numeric(group, "pid_actual_total_duration_s") - _numeric(group, "pid_target_total_duration_s")
    )
    added.extend([
        "state_discharge_minus_target_temp",
        "state_pid_actual_minus_target_duration",
    ])

    # Restore original order only after constructing causal states.
    group = group.sort_values("LabelGroupID", kind="stable").reset_index(drop=True)
    state_path = OUT / "m1_only_group_order_state_features.csv"
    group_path = OUT / "m1_only_group_label_snapshot_with_order_state.csv"
    batch_path = OUT / "m1_only_batch_pid_quality_with_order_state.csv"
    group[ ["LabelGroupID", *added] ].to_csv(state_path, index=False, encoding="utf-8-sig")
    group.to_csv(group_path, index=False, encoding="utf-8-sig")
    batch = batch.merge(group[["LabelGroupID", *added]], on="LabelGroupID", how="inner", validate="many_to_one")
    batch.to_csv(batch_path, index=False, encoding="utf-8-sig")

    metadata["contract_version"] = "V3.7.1-R1B-M1-ONLY-ORDER-STATE-V1"
    metadata["scope"] = "M1_ONLY"
    metadata["order_state_contract"] = {
        "causal_rule": "previous group must be completed before current group starts",
        "numeric_prior_rule": "prior numeric process state is retained only when TemplateKey is unchanged",
        "template_change_rule": "template/version/step changes are explicit flags; no cross-template numeric prior",
        "uses_lab_value": False,
        "uses_future_batch": False,
        "state_features": added,
    }
    metadata["s2_features"] = list(dict.fromkeys(metadata["s2_features"] + added))
    metadata["generated_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_path = OUT / "m1_only_order_state_metadata_v1.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "contract_version": metadata["contract_version"],
        "scope": "M1_ONLY",
        "source_batch": str(SOURCE_BATCH),
        "source_group": str(SOURCE_GROUP),
        "batch_rows": int(len(batch)),
        "lab_groups": int(group["LabelGroupID"].nunique()),
        "process_state_features": available,
        "added_state_features": added,
        "prior_available_rate": float(group["state_prior_available"].mean()),
        "prior_same_template_rate_among_available": float(
            group.loc[group["state_prior_available"].eq(1), "state_prior_same_template"].mean()
        ) if group["state_prior_available"].eq(1).any() else 0.0,
        "template_change_count": int(group["state_template_changed_from_prior"].sum()),
        "process_version_change_count": int(group["state_process_version_changed_from_prior"].sum()),
        "step_pattern_change_count": int(group["state_step_pattern_changed_from_prior"].sum()),
        "uses_lab_value": False,
        "uses_future_batch": False,
        "database_access_during_build": False,
        "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "m1_only_order_state_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
