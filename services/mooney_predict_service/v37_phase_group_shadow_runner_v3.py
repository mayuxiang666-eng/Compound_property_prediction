"""Final fail-closed entry point, including the all-rows-blocked case."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


def _configure_legacy_model_import() -> None:
    explicit = os.environ.get("MOONEY_MODEL_PACKAGE_DIR", "").strip()
    if explicit:
        root = str(Path(explicit).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)


_configure_legacy_model_import()

from . import v37_phase_group_shadow_runner as runner  # noqa: E402


def _complete_group_outputs(quality: pd.DataFrame, predictions: pd.DataFrame):
    expected = quality.groupby("LabelGroupID", as_index=False).agg(
        ExpectedBatchCount=("BatchNumber", "size"),
        AcceptedBatchCount=("FeatureQualityStatus", lambda values: int((values == "MODEL_INPUT_OK").sum())),
    )
    if predictions.empty:
        blocked = expected.copy()
        blocked["GroupPredictionStatus"] = "BLOCKED_INCOMPLETE_GROUP_FEATURES"
        return pd.DataFrame(columns=[
            "OrderID", "PalletID", "LabelGroupID", "CompoundName", "BatchCount",
            "GroupRawPrediction", "GroupPredictionStatus",
        ]), blocked
    merged = predictions.merge(expected, on="LabelGroupID", how="right", validate="one_to_one")
    complete = merged["ExpectedBatchCount"] == merged["AcceptedBatchCount"]
    blocked = merged.loc[~complete].copy()
    blocked["GroupPredictionStatus"] = "BLOCKED_INCOMPLETE_GROUP_FEATURES"
    valid = merged.loc[complete].copy()
    valid["GroupPredictionStatus"] = "PREDICTED_COMPLETE_GROUP_MEAN"
    return valid, blocked


runner._complete_group_outputs = _complete_group_outputs


if __name__ == "__main__":
    runner.main()
