"""Audit the new Silica PID time hard gate without releasing it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.silica_pid_hard_gate_v1 import validate_silica_pid_total_target_time

BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUT = BASE / "v371_dual_process_response_v1"
COMPOUND = "M1-T15760---- 06 002"


def main():
    step = pd.read_csv(BASE / "dual_target_readiness" / "dual_pilot_group_step_features.csv", low_memory=False)
    step = step[step["CompoundName"].eq(COMPOUND)].copy()
    static = pd.read_csv(BASE / "batch_static_template_skeleton.csv", usecols=["LabelGroupID", "Target_Temperature"], low_memory=False)
    static = static.drop_duplicates("LabelGroupID")
    durations = step[step["kind"].eq("duration")].copy()
    top = durations[(durations["GroupName"] == "AVR_MST") & (durations["StepNo"] == 10)][["LabelGroupID", "target_mean"]].rename(columns={"target_mean": "top_pid_duration_target_s"})
    bottom = durations[(durations["GroupName"] == "AVR_MSB") & (durations["StepNo"].isin([1, 2, 3]))].groupby("LabelGroupID", as_index=False)["target_mean"].sum().rename(columns={"target_mean": "bottom_pid_duration_target_s"})
    audit = top.merge(bottom, on="LabelGroupID", how="outer").merge(static, on="LabelGroupID", how="left")
    audit["gate_result"] = [
        validate_silica_pid_total_target_time(t, top_s, bottom_s)["status"]
        for t, top_s, bottom_s in zip(audit["Target_Temperature"], audit["top_pid_duration_target_s"], audit["bottom_pid_duration_target_s"])
    ]
    audit["total_pid_duration_target_s"] = audit["top_pid_duration_target_s"] + audit["bottom_pid_duration_target_s"]
    audit.to_csv(OUT / "silica_pid_time_hard_gate_history_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "contract": "V3.7.1-SILICA-PID-TOP-BOTTOM-TIME-HARD-GATE-V1",
        "compound": COMPOUND,
        "rows": int(len(audit)),
        "gate_status_counts": audit["gate_result"].value_counts(dropna=False).to_dict(),
        "current_mapping": "top=MST10 duration Target; bottom=MSB1+MSB2+MSB3 duration Target",
        "mapping_status": "BLOCKED_MAPPING_REQUIRES_PROCESS_CONFIRMATION",
        "reason": "this provisional mapping yields 428 s for the observed T15760 165 C recipe, outside the engineer-confirmed 80-140 s band",
        "recommendation_ready": False,
        "source_policy": "RDS Active snapshot only; local MMS/Archive not accessed",
    }
    (OUT / "silica_pid_time_hard_gate_history_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
