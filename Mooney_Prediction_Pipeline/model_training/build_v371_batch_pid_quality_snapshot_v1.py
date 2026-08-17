"""Build batch-level corrected PID and curve-quality snapshot for R1B."""

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
OUT = BASE / "v371_r1b_batch_pid_quality_snapshot_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phase_path = BASE / "phase_batch_features_active_v1.csv"
    group_path = BASE / "pallet_group_training_frame_c2_prepared.csv"
    batchdata_path = BASE / "batchdata_targets_values_dedup.csv"
    quality_path = BASE / "v371_curve_quality_v1" / "batch_curve_quality_audit.csv"

    phase = pd.read_csv(phase_path, low_memory=False)
    group = pd.read_csv(group_path, usecols=["LabelGroupID", "MNY", "lab_time", "split"], low_memory=False)
    phase["OrderID"] = phase["OrderID"].astype(str)
    phase["BatchNumber"] = pd.to_numeric(phase["BatchNumber"], errors="coerce")
    phase["LabelGroupID"] = phase["LabelGroupID"].astype(str)
    phase = phase.merge(group.assign(LabelGroupID=group["LabelGroupID"].astype(str)), on="LabelGroupID", how="inner", validate="many_to_one")

    static = pd.read_csv(BASE / "batch_static_template_skeleton.csv", usecols=["OrderID", "BatchNumber", "Target_Temperature"], low_memory=False)
    static["OrderID"] = static["OrderID"].astype(str)
    static["BatchNumber"] = pd.to_numeric(static["BatchNumber"], errors="coerce")
    static["Target_Temperature"] = pd.to_numeric(static["Target_Temperature"], errors="coerce")
    raw = pd.read_csv(batchdata_path, usecols=["OrderID", "BatchNumber", "GroupName", "StepNo", "VariablePath", "Target", "Value"], low_memory=False)
    raw["OrderID"] = raw["OrderID"].astype(str)
    raw["BatchNumber"] = pd.to_numeric(raw["BatchNumber"], errors="coerce")
    raw = raw.merge(static, on=["OrderID", "BatchNumber"], how="inner", validate="many_to_one")
    raw["StepNo"] = pd.to_numeric(raw["StepNo"], errors="coerce")
    raw["Target"] = pd.to_numeric(raw["Target"], errors="coerce")
    raw["Value"] = pd.to_numeric(raw["Value"], errors="coerce")
    variable = raw["VariablePath"].astype(str)
    raw["kind"] = np.select([
        variable.str.contains("SCP-1-Step-Time", case=False, na=False),
        variable.str.contains("SCP-3-Temperature", case=False, na=False),
    ], ["duration", "temperature"], default=None)
    raw = raw[raw["kind"].notna() & raw["GroupName"].isin(["AVR_MST", "AVR_MSB"])].copy()
    raw = raw.sort_values(["OrderID", "BatchNumber", "GroupName", "StepNo", "kind"], kind="stable")

    duration = raw.loc[raw["kind"].eq("duration")].copy()
    keys = ["OrderID", "BatchNumber", "GroupName"]
    for source, output in (("Target", "pid_target_step_duration_s"), ("Value", "pid_actual_step_duration_s")):
        previous = duration.groupby(keys)[source].shift(1)
        duration[output] = duration[source] - previous
        duration.loc[previous.isna(), output] = duration.loc[previous.isna(), source]
        duration.loc[duration[output] < 0, output] = np.nan

    temps = raw.loc[(raw["GroupName"] == "AVR_MST") & raw["kind"].eq("temperature")].copy()
    temps["threshold_temperature"] = temps["Target_Temperature"] - 3.0
    starts = temps.loc[temps["Value"] >= temps["threshold_temperature"]].sort_values(["OrderID", "BatchNumber", "StepNo"], kind="stable")
    starts = starts.groupby(["OrderID", "BatchNumber"], as_index=False).first()[["OrderID", "BatchNumber", "StepNo"]]
    starts = starts.rename(columns={"StepNo": "pid_start_step"})

    duration = duration.merge(starts, on=["OrderID", "BatchNumber"], how="left", validate="many_to_one")
    in_pid = ((duration["GroupName"] == "AVR_MST") & duration["pid_start_step"].notna() & duration["StepNo"].ge(duration["pid_start_step"])) | (duration["GroupName"] == "AVR_MSB")
    duration = duration.loc[in_pid].copy()
    batch_pid = duration.groupby(["OrderID", "BatchNumber", "GroupName"], as_index=False).agg(
        target_duration_s=("pid_target_step_duration_s", "sum"),
        actual_duration_s=("pid_actual_step_duration_s", "sum"),
    )
    wide = batch_pid.pivot_table(index=["OrderID", "BatchNumber"], columns="GroupName", values=["target_duration_s", "actual_duration_s"], aggfunc="first").reset_index()
    wide.columns = ["_".join(str(v) for v in c if str(v) != "") if isinstance(c, tuple) else str(c) for c in wide.columns]
    for column in ("target_duration_s_AVR_MST", "target_duration_s_AVR_MSB", "actual_duration_s_AVR_MST", "actual_duration_s_AVR_MSB"):
        if column not in wide:
            wide[column] = np.nan
    wide = wide.merge(starts, on=["OrderID", "BatchNumber"], how="left", validate="one_to_one")
    wide["pid_target_top_duration_s"] = wide["target_duration_s_AVR_MST"]
    wide["pid_target_bottom_duration_s"] = wide["target_duration_s_AVR_MSB"]
    wide["pid_actual_top_duration_s"] = wide["actual_duration_s_AVR_MST"]
    wide["pid_actual_bottom_duration_s"] = wide["actual_duration_s_AVR_MSB"]
    wide["pid_target_total_duration_s"] = wide["pid_target_top_duration_s"] + wide["pid_target_bottom_duration_s"]
    wide["pid_actual_total_duration_s"] = wide["pid_actual_top_duration_s"] + wide["pid_actual_bottom_duration_s"]
    wide["pid_start_found_flag"] = wide["pid_start_step"].notna().astype(float)
    wide["pid_target_temperature"] = static.set_index(["OrderID", "BatchNumber"]).reindex(pd.MultiIndex.from_frame(wide[["OrderID", "BatchNumber"]]))["Target_Temperature"].to_numpy()

    result = phase.merge(wide, on=["OrderID", "BatchNumber"], how="left", validate="one_to_one")
    quality = pd.read_csv(quality_path, low_memory=False)
    quality["OrderID"] = quality["OrderID"].astype(str)
    quality["BatchNumber"] = pd.to_numeric(quality["BatchNumber"], errors="coerce")
    qcols = ["OrderID", "BatchNumber", "curve_physical_valid", "curve_statistical_outlier", "curve_if_score", "curve_quality_status", "curve_model_include", "curve_risk_flag"]
    result = result.merge(quality[qcols], on=["OrderID", "BatchNumber"], how="left", validate="one_to_one")
    if result["curve_model_include"].isna().any():
        raise ValueError("Missing curve quality row for a physical batch")
    result = result.loc[result["curve_model_include"].astype(bool)].copy()
    # Keep Lab target/time/split outside the batch process snapshot. The R1B
    # contract joins those fields once from the independent Lab-group frame.
    result = result.drop(columns=["MNY", "lab_time", "split"], errors="ignore")
    result["silica_pid_target_time_gate_status"] = "NOT_APPLICABLE_NON_SILICA"
    silica = result["material_system"].astype(str).eq("Silica")
    result.loc[silica, "silica_pid_target_time_gate_status"] = [
        validate_silica_pid_total_target_time(temp, top, bottom)["status"]
        if pd.notna(temp) and pd.notna(top) and pd.notna(bottom)
        else "BLOCKED_MISSING_SILICA_PID_TOP_BOTTOM_TARGET_TIME"
        for temp, top, bottom in zip(result.loc[silica, "pid_target_temperature"], result.loc[silica, "pid_target_top_duration_s"], result.loc[silica, "pid_target_bottom_duration_s"])
    ]
    output = OUT / "v371_r1b_batch_pid_quality_features.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    manifest = {
        "contract_version": "V3.7.1-R1B-BATCH-PID-QUALITY-SNAPSHOT-V1",
        "input_phase_sha256": sha256(phase_path), "input_batchdata_sha256": sha256(batchdata_path),
        "input_quality_sha256": sha256(quality_path), "output_sha256": sha256(output),
        "input_batches": int(len(phase)), "output_batches": int(len(result)),
        "physical_invalid_excluded": int(len(phase) - len(result)),
        "statistical_outlier_retained": int(result["curve_statistical_outlier"].sum()),
        "pid_start_found": int(result["pid_start_found_flag"].sum()),
        "source_policy": "Cloud RDS Active snapshot only; local MMS/Archive not accessed",
        "stage3_calibration_enabled": False, "automatic_control": "DISABLED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
