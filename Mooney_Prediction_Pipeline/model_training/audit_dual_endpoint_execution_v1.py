"""Template-aware endpoint and execution-quality audit for the two pilots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.silica_pid_duration_contract_v2 import build_pid_intervals

BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUT = BASE / "v371_dual_process_response_v1"
TARGETS = ["M1-T15760---- 06 002", "M3B-B00460---- 01 002"]

# These are explicit pilot-template contracts, not a global step-number rule.
ENDPOINT_CONTRACT = {
    "M1-T15760---- 06 002": {
        "dry_mst_steps": [3], "oil_mst_steps": [4], "wet_mst_steps": [5, 6, 7, 8, 9],
        "route": "oil_wet", "silica_pid_gate": True,
    },
    "M3B-B00460---- 01 002": {
        "dry_mst_steps": [3, 4], "oil_mst_steps": [], "wet_mst_steps": [],
        "route": "no_oil_dry", "silica_pid_gate": False,
    },
}


def endpoint_temp(frame: pd.DataFrame, steps: list[int]) -> float:
    x = frame[(frame["GroupName"] == "AVR_MST") & (frame["kind"] == "temperature") & frame["StepNo"].isin(steps)]
    if x.empty:
        return np.nan
    return float(x.sort_values("StepNo")["actual_value_mean"].iloc[-1])


def duration_sum(frame: pd.DataFrame, group: str, steps: list[int]) -> tuple[float, float]:
    x = frame[(frame["GroupName"] == group) & (frame["kind"] == "duration") & frame["StepNo"].isin(steps)]
    if x.empty:
        return np.nan, np.nan
    return float(x["actual_step_duration_s"].sum()), float(x["target_step_duration_s"].sum())


def main():
    step = pd.read_csv(BASE / "dual_target_readiness" / "dual_pilot_group_step_features.csv", low_memory=False)
    step = step[step["CompoundName"].isin(TARGETS)].copy()
    static = pd.read_csv(BASE / "batch_static_template_skeleton.csv", usecols=["LabelGroupID", "Target_Temperature", "RecipeID", "ProcessVersion"], low_memory=False).drop_duplicates("LabelGroupID")
    step = step.merge(static, on="LabelGroupID", how="left", validate="many_to_one", suffixes=("", "_static"))
    step["StepNo"] = pd.to_numeric(step["StepNo"], errors="coerce")
    step["target_mean"] = pd.to_numeric(step["target_mean"], errors="coerce")
    step["actual_value_mean"] = pd.to_numeric(step["actual_value_mean"], errors="coerce")

    rows = []
    for compound, g in step.groupby("CompoundName", sort=False):
        intervals = build_pid_intervals(g)
        interval_cols = ["LabelGroupID", "pid_start_step", "pid_start_actual_temperature", "threshold_temperature", "target_pid_total_duration_s", "actual_pid_total_duration_s"]
        intervals = intervals[interval_cols].drop_duplicates("LabelGroupID")
        dur = g[g["kind"].eq("duration")].copy()
        dur = dur.sort_values(["LabelGroupID", "GroupName", "StepNo"], kind="stable")
        dur["target_step_duration_s"] = dur.groupby(["LabelGroupID", "GroupName"])["target_mean"].diff()
        dur["actual_step_duration_s"] = dur.groupby(["LabelGroupID", "GroupName"])["actual_value_mean"].diff()
        first = dur.groupby(["LabelGroupID", "GroupName"]).head(1).index
        dur.loc[first, "target_step_duration_s"] = dur.loc[first, "target_mean"]
        dur.loc[first, "actual_step_duration_s"] = dur.loc[first, "actual_value_mean"]
        contract = ENDPOINT_CONTRACT[compound]
        for label_group, x in g.groupby("LabelGroupID", sort=False):
            x_dur = dur[dur["LabelGroupID"].eq(label_group)]
            x_interval = intervals[intervals["LabelGroupID"].eq(label_group)]
            if x_interval.empty:
                pid = {c: np.nan for c in interval_cols[1:]}
                pid_status = "BLOCKED_PID_START_NOT_FOUND"
            else:
                pid = x_interval.iloc[0].to_dict()
                pid_status = "PASS_PID_START_FOUND"
            dry_actual, dry_target = duration_sum(x_dur, "AVR_MST", contract["dry_mst_steps"])
            wet_actual, wet_target = duration_sum(x_dur, "AVR_MST", contract["wet_mst_steps"])
            msb_temp = x[(x["GroupName"] == "AVR_MSB") & (x["kind"] == "temperature")].sort_values("StepNo")
            msb_controls = msb_temp[msb_temp["target_mean"].gt(0)]
            rows.append({
                "CompoundName": compound, "LabelGroupID": label_group, "RecipeID": x["RecipeID"].iloc[0], "ProcessVersion": x["ProcessVersion"].iloc[0],
                "route": contract["route"], "silica_pid_gate_applicable": contract["silica_pid_gate"],
                "dry_endpoint_actual": endpoint_temp(x, contract["dry_mst_steps"]),
                "oil_endpoint_actual": endpoint_temp(x, contract["oil_mst_steps"]),
                "wet_endpoint_actual": endpoint_temp(x, contract["wet_mst_steps"]),
                "dry_duration_actual": dry_actual, "dry_duration_target": dry_target,
                "wet_duration_actual": wet_actual, "wet_duration_target": wet_target,
                "pid_start_step": pid.get("pid_start_step"), "pid_start_actual_temperature": pid.get("pid_start_actual_temperature"),
                "pid_threshold_temperature": pid.get("threshold_temperature"), "pid_target_duration_total": pid.get("target_pid_total_duration_s"),
                "pid_actual_duration_total": pid.get("actual_pid_total_duration_s"), "pid_start_status": pid_status,
                "discharge_control_step": int(msb_controls["StepNo"].iloc[-1]) if not msb_controls.empty else np.nan,
                "discharge_final_step": int(msb_temp["StepNo"].iloc[-1]) if not msb_temp.empty else np.nan,
                "discharge_control_target": float(msb_controls["target_mean"].iloc[-1]) if not msb_controls.empty else np.nan,
                "discharge_actual": float(msb_temp["actual_value_mean"].iloc[-1]) if not msb_temp.empty else np.nan,
            })
    out = pd.DataFrame(rows)
    out["discharge_abs_error"] = (out["discharge_actual"] - out["discharge_control_target"]).abs()
    out["pid_actual_target_gap_s"] = out["pid_actual_duration_total"] - out["pid_target_duration_total"]
    out["pid_execution_outlier"] = out["pid_actual_target_gap_s"].abs().gt(30.0)
    out.to_csv(OUT / "dual_endpoint_execution_audit.csv", index=False, encoding="utf-8-sig")

    summary = {"contract": "V3.7.1-DUAL-ENDPOINT-EXECUTION-AUDIT-V1", "source_policy": "RDS Active snapshot only; local MMS/Archive not accessed", "recommendation_ready": False, "pilots": {}}
    for compound, g in out.groupby("CompoundName"):
        summary["pilots"][compound] = {
            "groups": int(len(g)), "pid_start_found": int(g["pid_start_status"].eq("PASS_PID_START_FOUND").sum()),
            "pid_execution_outlier_gt30s": int(g["pid_execution_outlier"].sum()),
            "discharge_control_rows": int(g["discharge_control_target"].notna().sum()),
            "discharge_mae": float(g["discharge_abs_error"].mean()) if g["discharge_abs_error"].notna().any() else None,
            "discharge_p95": float(g["discharge_abs_error"].dropna().quantile(.95)) if g["discharge_abs_error"].notna().any() else None,
            "route": str(g["route"].iloc[0]), "silica_pid_gate_applicable": bool(g["silica_pid_gate_applicable"].iloc[0]),
        }
    (OUT / "dual_endpoint_execution_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
