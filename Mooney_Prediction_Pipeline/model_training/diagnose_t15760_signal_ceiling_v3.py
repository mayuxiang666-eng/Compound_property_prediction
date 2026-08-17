"""Timezone-safe standalone entry point for the T15760 signal diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training import diagnose_t15760_signal_ceiling as diagnostic


def causal_lab_features(frame):
    ordered = frame.sort_values(
        ["GroupProductionStart", "lab_time", "OrderID", "LabelGroupID"],
        kind="stable",
    ).copy()
    lab_events = frame.sort_values("lab_time", kind="stable")
    event_times = (
        pd.to_datetime(lab_events["lab_time"], utc=True).astype("int64").to_numpy()
    )
    event_values = lab_events["MNY"].to_numpy(dtype=float)
    latest, rolling3, ewma50, available_count = [], [], [], []
    for start in pd.to_datetime(ordered["GroupProductionStart"], utc=True):
        position = int(np.searchsorted(event_times, int(start.value), side="right"))
        values = event_values[:position]
        available_count.append(position)
        latest.append(values[-1] if len(values) else np.nan)
        rolling3.append(float(np.mean(values[-3:])) if len(values) else np.nan)
        if len(values):
            state = float(values[0])
            for value in values[1:]:
                state = 0.5 * float(value) + 0.5 * state
            ewma50.append(state)
        else:
            ewma50.append(np.nan)
    ordered["causal_lab_count"] = available_count
    ordered["causal_latest_lab"] = latest
    ordered["causal_rolling3_lab"] = rolling3
    ordered["causal_ewma50_lab"] = ewma50
    return ordered.sort_index()


if __name__ == "__main__":
    diagnostic.causal_lab_features = causal_lab_features
    diagnostic.main()
