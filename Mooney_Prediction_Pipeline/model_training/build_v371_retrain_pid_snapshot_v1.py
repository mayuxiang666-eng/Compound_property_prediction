"""Build a full Active-only retrain snapshot with corrected PID semantics."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.silica_pid_hard_gate_v1 import validate_silica_pid_total_target_time

BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUT = BASE / "v371_retrain_pid_snapshot_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    static_path = BASE / "batch_static_template_skeleton.csv"
    batchdata_path = BASE / "batchdata_targets_values_dedup.csv"
    group_path = BASE / "pallet_group_training_frame_c2_prepared.csv"

    static_cols = ["OrderID", "BatchNumber", "LabelGroupID", "CompoundName", "material_system", "RecipeID", "ProcessVersion", "Target_Temperature"]
    static = pd.read_csv(static_path, usecols=static_cols, low_memory=False).drop_duplicates(["OrderID", "BatchNumber"])
    static["BatchNumber"] = pd.to_numeric(static["BatchNumber"], errors="coerce")
    static["Target_Temperature"] = pd.to_numeric(static["Target_Temperature"], errors="coerce")

    cols = ["OrderID", "BatchNumber", "GroupName", "StepNo", "VariablePath", "Target", "Value", "Timestamp"]
    raw = pd.read_csv(batchdata_path, usecols=cols, low_memory=False)
    raw["BatchNumber"] = pd.to_numeric(raw["BatchNumber"], errors="coerce")
    raw = raw.merge(static, on=["OrderID", "BatchNumber"], how="inner", validate="many_to_one")
    raw["StepNo"] = pd.to_numeric(raw["StepNo"], errors="coerce")
    raw["Target"] = pd.to_numeric(raw["Target"], errors="coerce")
    raw["Value"] = pd.to_numeric(raw["Value"], errors="coerce")
    variable = raw["VariablePath"].astype(str)
    raw["kind"] = np.select([variable.str.contains("SCP-1-Step-Time", case=False, na=False), variable.str.contains("SCP-3-Temperature", case=False, na=False)], ["duration", "temperature"], default=None)
    raw = raw[raw["kind"].notna() & raw["GroupName"].isin(["AVR_MST", "AVR_MSB"])].copy()
    raw = raw.sort_values(["OrderID", "BatchNumber", "GroupName", "StepNo", "kind"], kind="stable")

    batch_keys = ["OrderID", "BatchNumber", "GroupName"]
    duration_rows = raw[raw["kind"].eq("duration")].copy()
    for source, output in (("Target", "target_step_duration_s"), ("Value", "actual_step_duration_s")):
        previous = duration_rows.groupby(batch_keys)[source].shift(1)
        duration_rows[output] = duration_rows[source] - previous
        first = previous.isna()
        duration_rows.loc[first, output] = duration_rows.loc[first, source]
        duration_rows.loc[duration_rows[output] < 0, output] = np.nan
    raw = raw.drop(columns=["target_step_duration_s", "actual_step_duration_s"], errors="ignore")
    raw = raw.merge(duration_rows[["OrderID", "BatchNumber", "GroupName", "StepNo", "kind", "target_step_duration_s", "actual_step_duration_s"]], on=["OrderID", "BatchNumber", "GroupName", "StepNo", "kind"], how="left", validate="one_to_one")
    raw["duration_target_value"] = np.where(raw["kind"].eq("duration"), raw["target_step_duration_s"], np.nan)
    raw["duration_actual_value"] = np.where(raw["kind"].eq("duration"), raw["actual_step_duration_s"], np.nan)

    # Detect the PID start independently for every physical batch.
    temps = raw[(raw["GroupName"] == "AVR_MST") & (raw["kind"] == "temperature")].copy()
    temps["threshold_temperature"] = temps["Target_Temperature"] - 3.0
    starts = temps[temps["Value"] >= temps["threshold_temperature"]].sort_values(["OrderID", "BatchNumber", "StepNo"], kind="stable")
    starts = starts.groupby(["OrderID", "BatchNumber"], as_index=False).first()[["OrderID", "BatchNumber", "StepNo", "Value", "threshold_temperature"]]
    starts = starts.rename(columns={"StepNo": "pid_start_step", "Value": "pid_start_actual_temperature"})
    raw = raw.merge(starts, on=["OrderID", "BatchNumber"], how="left", validate="many_to_one")
    raw["pid_start_found"] = raw["pid_start_step"].notna()
    in_pid = ((raw["GroupName"] == "AVR_MST") & raw["pid_start_step"].notna() & (raw["StepNo"] >= raw["pid_start_step"])) | (raw["GroupName"] == "AVR_MSB")
    pid_dur = raw[in_pid & raw["kind"].eq("duration")].copy()
    pid_batch = pid_dur.groupby(["OrderID", "BatchNumber", "LabelGroupID", "CompoundName", "material_system", "RecipeID", "ProcessVersion", "Target_Temperature", "GroupName"], as_index=False).agg(target_duration_s=("duration_target_value", "sum"), actual_duration_s=("duration_actual_value", "sum"))
    pid_wide = pid_batch.pivot_table(index=["OrderID", "BatchNumber", "LabelGroupID", "CompoundName", "material_system", "RecipeID", "ProcessVersion", "Target_Temperature"], columns="GroupName", values=["target_duration_s", "actual_duration_s"], aggfunc="first").reset_index()
    pid_wide.columns = ["_".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else str(c) for c in pid_wide.columns]
    for col in ("target_duration_s_AVR_MST", "target_duration_s_AVR_MSB", "actual_duration_s_AVR_MST", "actual_duration_s_AVR_MSB"):
        if col not in pid_wide:
            pid_wide[col] = np.nan
    pid_wide["pid_target_total_duration_s"] = pid_wide["target_duration_s_AVR_MST"] + pid_wide["target_duration_s_AVR_MSB"]
    pid_wide["pid_actual_total_duration_s"] = pid_wide["actual_duration_s_AVR_MST"] + pid_wide["actual_duration_s_AVR_MSB"]
    pid_wide = pid_wide.merge(starts, on=["OrderID", "BatchNumber"], how="left", validate="one_to_one")

    group_agg = pid_wide.groupby(["LabelGroupID", "CompoundName", "material_system", "RecipeID", "ProcessVersion"], as_index=False).agg(
        pid_target_top_duration_s=("target_duration_s_AVR_MST", "mean"), pid_target_bottom_duration_s=("target_duration_s_AVR_MSB", "mean"),
        pid_actual_top_duration_s=("actual_duration_s_AVR_MST", "mean"), pid_actual_bottom_duration_s=("actual_duration_s_AVR_MSB", "mean"),
        pid_target_total_duration_s=("pid_target_total_duration_s", "mean"), pid_actual_total_duration_s=("pid_actual_total_duration_s", "mean"),
        pid_start_step_mean=("pid_start_step", "mean"), pid_start_found_rate=("pid_start_step", lambda x: float(x.notna().mean())),
        pid_target_min=("Target_Temperature", "min"), pid_target_max=("Target_Temperature", "max"),
    )
    group_agg["pid_target_temperature"] = group_agg["pid_target_min"]
    group_agg["pid_target_time_gate_status"] = "NOT_APPLICABLE_NON_SILICA"
    silica = group_agg["material_system"].astype(str).eq("Silica")
    group_agg.loc[silica, "pid_target_time_gate_status"] = [
        validate_silica_pid_total_target_time(temp, top, bottom)["status"] if pd.notna(top) and pd.notna(bottom) else "BLOCKED_MISSING_SILICA_PID_TOP_BOTTOM_TARGET_TIME"
        for temp, top, bottom in zip(group_agg.loc[silica, "pid_target_temperature"], group_agg.loc[silica, "pid_target_top_duration_s"], group_agg.loc[silica, "pid_target_bottom_duration_s"])
    ]
    group_agg.to_csv(OUT / "pid_corrected_group_features.csv", index=False, encoding="utf-8-sig")

    group = pd.read_csv(group_path, low_memory=False)
    merged = group.merge(group_agg, on=["LabelGroupID", "CompoundName", "material_system", "RecipeID", "ProcessVersion"], how="left", validate="one_to_one")
    merged.to_csv(OUT / "v371_retrain_group_feature_snapshot.csv", index=False, encoding="utf-8-sig")
    summary = {
        "contract": "V3.7.1-RETRAIN-PID-SNAPSHOT-V1", "source_policy": "RDS Active snapshot only; local MMS/Archive not accessed",
        "input_rows": int(len(raw)), "physical_batches": int(raw[["OrderID", "BatchNumber"]].drop_duplicates().shape[0]),
        "group_rows": int(len(merged)), "group_rows_with_pid": int(merged["pid_target_total_duration_s"].notna().sum()),
        "group_rows_pid_start_complete": int((merged["pid_start_found_rate"] >= 0.95).sum()),
        "silica_gate_status": merged.loc[merged["material_system"].eq("Silica"), "pid_target_time_gate_status"].value_counts(dropna=False).to_dict(),
        "feature_snapshot_sha256": sha256(OUT / "v371_retrain_group_feature_snapshot.csv"),
        "source_sha256": {str(p.name): sha256(p) for p in (static_path, batchdata_path, group_path)},
        "generated_utc": datetime.now(timezone.utc).isoformat(), "stage3_calibration_enabled": False, "automatic_control": "DISABLED",
    }
    (OUT / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
