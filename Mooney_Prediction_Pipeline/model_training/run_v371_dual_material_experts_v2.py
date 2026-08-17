"""Run the dual material diagnostic with the fixed R1_REGRESSION OOF base."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training import run_v371_dual_material_experts_v1 as v1


_read_csv = pd.read_csv


def _read_csv_fixed(path, *args, **kwargs):
    data = _read_csv(path, *args, **kwargs)
    if str(path).replace("\\", "/").endswith("v371_r1_temporal_group/r1_all_group_predictions.csv") and "Candidate" in data.columns:
        data = data.loc[data["Candidate"].eq("R1_REGRESSION")].copy()
    return data


pd.read_csv = _read_csv_fixed

if __name__ == "__main__":
    v1.main()
