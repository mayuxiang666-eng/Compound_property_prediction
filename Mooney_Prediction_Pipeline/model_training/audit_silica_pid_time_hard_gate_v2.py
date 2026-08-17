"""Audit Silica PID total-time bands with corrected cumulative-time semantics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.silica_pid_duration_contract_v2 import build_pid_intervals
from Mooney_Prediction_Pipeline.model_training.silica_pid_hard_gate_v1 import validate_silica_pid_total_target_time

BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUT = BASE / "v371_dual_process_response_v1"
COMPOUND = "M1-T15760---- 06 002"


def main():
    step = pd.read_csv(BASE / "dual_target_readiness" / "dual_pilot_group_step_features.csv", low_memory=False)
    step = step[step["CompoundName"].eq(COMPOUND)].copy()
    static = pd.read_csv(BASE / "batch_static_template_skeleton.csv", usecols=["LabelGroupID", "Target_Temperature"], low_memory=False).drop_duplicates("LabelGroupID")
    step = step.merge(static, on="LabelGroupID", how="left", validate="many_to_one")
    intervals = build_pid_intervals(step)
    intervals["gate_result"] = [
        validate_silica_pid_total_target_time(t, top, bottom)["status"]
        for t, top, bottom in zip(intervals["Target_Temperature"], intervals["target_pid_duration_s_TOP_MST"], intervals["target_pid_duration_s_BOTTOM_MSB"])
    ]
    intervals.to_csv(OUT / "silica_pid_time_hard_gate_history_audit_v2.csv", index=False, encoding="utf-8-sig")
    summary = {
        "contract": "V3.7.1-SILICA-PID-TOP-BOTTOM-TIME-HARD-GATE-V2",
        "compound": COMPOUND,
        "rows": int(len(intervals)),
        "gate_status_counts": intervals["gate_result"].value_counts(dropna=False).to_dict(),
        "start_rule": "first AVR_MST actual temperature >= Target_Temperature - 3 C",
        "duration_rule": "SCP-1-Step-Time cumulative value converted to within-mixer step duration by first difference",
        "end_rule": "all qualifying AVR_MST duration from start step plus all AVR_MSB duration through final MSB step",
        "target_total_summary": intervals["target_pid_total_duration_s"].describe().to_dict(),
        "actual_total_summary": intervals["actual_pid_total_duration_s"].describe().to_dict(),
        "source_policy": "RDS Active snapshot only; local MMS/Archive not accessed",
        "recommendation_ready": False,
    }
    (OUT / "silica_pid_time_hard_gate_history_summary_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
