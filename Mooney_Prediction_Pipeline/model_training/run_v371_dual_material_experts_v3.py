"""Run dual material diagnostic with an atomic fixed R1 candidate/split."""

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
    normalized = str(path).replace("\\", "/")
    if normalized.endswith("v371_r1_temporal_group/r1_all_group_predictions.csv") and "Candidate" in data.columns:
        data = data.loc[data["Candidate"].eq("R1_REGRESSION")].copy()
    if normalized.endswith("pallet_group_training_frame_c2_prepared.csv") and "split" in data.columns:
        data = data.drop(columns=["split"])
    return data


pd.read_csv = _read_csv_fixed

if __name__ == "__main__":
    v1.main()
