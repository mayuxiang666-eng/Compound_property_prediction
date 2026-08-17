"""Correct discharge tracking using the original Phase physical endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUT = BASE / "v371_dual_process_response_v2"
TARGETS = ["M1-T15760---- 06 002", "M3B-B00460---- 01 002"]


def main():
    step = pd.read_csv(BASE / "dual_target_readiness" / "dual_pilot_group_step_features.csv", low_memory=False)
    step = step[step["CompoundName"].isin(TARGETS)].copy()
    msb = step[(step["GroupName"] == "AVR_MSB") & (step["kind"] == "temperature") & pd.to_numeric(step["target_mean"], errors="coerce").gt(0)].copy()
    controls = msb.sort_values(["LabelGroupID", "StepNo"], kind="stable").groupby("LabelGroupID", as_index=False).tail(1)
    controls = controls[["LabelGroupID", "CompoundName", "StepNo", "target_mean"]].rename(columns={"StepNo": "control_step", "target_mean": "control_target"})

    phase = pd.read_csv(BASE / "phase_batch_features_active_v1.csv", usecols=["LabelGroupID", "CompoundName", "phys_discharge_temp"], low_memory=False)
    phase = phase[phase["CompoundName"].isin(TARGETS)].copy()
    phase["phys_discharge_temp"] = pd.to_numeric(phase["phys_discharge_temp"], errors="coerce")
    group_phase = phase.groupby(["LabelGroupID", "CompoundName"], as_index=False).agg(
        phase_discharge_actual_mean=("phys_discharge_temp", "mean"),
        phase_discharge_actual_min=("phys_discharge_temp", "min"),
        phase_discharge_actual_max=("phys_discharge_temp", "max"),
        phase_discharge_batch_count=("phys_discharge_temp", "count"),
    )
    audit = controls.merge(group_phase, on=["LabelGroupID", "CompoundName"], how="inner", validate="one_to_one")
    audit["absolute_tracking_error"] = (audit["phase_discharge_actual_mean"] - audit["control_target"]).abs()
    audit["batch_level_excursion_gt5"] = (audit["phase_discharge_actual_max"] - audit["control_target"]).abs().gt(5) | (audit["phase_discharge_actual_min"] - audit["control_target"]).abs().gt(5)
    audit.to_csv(OUT / "dual_endpoint_discharge_tracking_phase_v2.csv", index=False, encoding="utf-8-sig")

    summary = {"contract": "V3.7.1-DUAL-DISCHARGE-TRACKING-PHASE-V2", "source_policy": "RDS Active snapshot only; local MMS/Archive not accessed", "recommendation_ready": False, "pilots": {}}
    for compound, g in audit.groupby("CompoundName"):
        err = g["absolute_tracking_error"].dropna()
        summary["pilots"][compound] = {
            "groups": int(len(g)), "control_steps": sorted(map(int, g["control_step"].dropna().unique())),
            "mae": float(err.mean()) if len(err) else None,
            "p95_abs_error": float(err.quantile(.95)) if len(err) else None,
            "excursion_gt5_group_rate": float(g["batch_level_excursion_gt5"].mean()) if len(g) else None,
            "passed_formal_gate": bool(len(err) and err.mean() <= 2.0 and err.quantile(.95) <= 5.0 and g["batch_level_excursion_gt5"].mean() <= 0.05),
        }
    (OUT / "dual_endpoint_discharge_tracking_phase_summary_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
