"""Final safe entry point for the frozen-snapshot T15760 diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training import diagnose_t15760_signal_ceiling as diagnostic
from Mooney_Prediction_Pipeline.model_training.diagnose_t15760_signal_ceiling_v3 import (
    causal_lab_features,
)


_original_merge = pd.DataFrame.merge


def _strict_split_merge(self, *args, **kwargs):
    result = _original_merge(self, *args, **kwargs)
    if "split_x" in result.columns and "split_y" in result.columns:
        result["split"] = result["split_y"]
        result = result.drop(columns=["split_x", "split_y"])
    return result


if __name__ == "__main__":
    diagnostic.causal_lab_features = causal_lab_features
    pd.DataFrame.merge = _strict_split_merge
    try:
        diagnostic.main()
    finally:
        pd.DataFrame.merge = _original_merge
