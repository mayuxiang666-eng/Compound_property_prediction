"""Corrected process response runner using within-mixer Step duration."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training import run_v371_dual_process_response_v1 as v1
from Mooney_Prediction_Pipeline.model_training.silica_pid_duration_contract_v2 import add_step_duration

v1.OUT = v1.BASE / "v371_dual_process_response_v2"
v1.MODEL_OUT = v1.ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_dual_process_response_v2"
_original_build_long = v1.build_long


def corrected_build_long():
    d = _original_build_long()
    duration = d[d["kind"].eq("duration")].copy()
    duration = add_step_duration(duration, "target_mean", "target_step_duration_s")
    duration = add_step_duration(duration, "actual_value_mean", "actual_step_duration_s")
    d.loc[duration.index, "target_mean"] = duration["target_step_duration_s"]
    d.loc[duration.index, "actual_value_mean"] = duration["actual_step_duration_s"]
    d["target_actual_gap"] = d["actual_value_mean"] - d["target_mean"]
    d = d[d["actual_value_mean"].notna()].copy()
    return d


v1.build_long = corrected_build_long

if __name__ == "__main__":
    v1.main()
