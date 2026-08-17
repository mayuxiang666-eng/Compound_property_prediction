"""Corrected scoring entry point for the R1B batch-to-group experiment."""

from __future__ import annotations

import warnings

import pandas as pd
from pandas.errors import PerformanceWarning

from .group_metric_contract import aggregate_batch_predictions
from . import run_v371_r1b_group_atomic_batch as v1


def score_batches(model, frame: pd.DataFrame, candidate: str):
    final, stage1, stage1b, stage2 = model.predict(frame, cluster_col="material_system")
    columns = [
        column for column in (
            "OrderID", "BatchNumber", "PalletID", "LabelGroupID", "CompoundName",
            "RecipeID", "ProcessVersion", "MixerLine", "material_system", "phase_route",
            "TemplateKey", "MNY", "lab_time", "split",
        ) if column in frame.columns
    ]
    batches = frame[columns].copy()
    batches["Candidate"] = candidate
    batches["ActualMNY"] = batches["MNY"]
    batches["Stage1Prediction"] = stage1
    batches["Stage1bBias"] = stage1b
    batches["Stage2Residual"] = stage2
    batches["RawPrediction"] = final
    batches["BatchErrorAgainstSharedLab"] = batches["RawPrediction"] - batches["MNY"]
    groups = aggregate_batch_predictions(
        batches,
        group_col="LabelGroupID",
        actual_col="ActualMNY",
        prediction_col="RawPrediction",
    ).rename(columns={"ActualMNY": "MNY", "GroupRawPrediction": "RawPrediction"})
    components = batches.groupby("LabelGroupID", as_index=False).agg(
        GroupStage1Prediction=("Stage1Prediction", "mean"),
        GroupStage1bBias=("Stage1bBias", "mean"),
        GroupStage2Residual=("Stage2Residual", "mean"),
    )
    groups = groups.merge(components, on="LabelGroupID", how="left", validate="one_to_one")
    groups["Candidate"] = candidate
    groups["Error"] = groups["RawPrediction"] - groups["MNY"]
    return batches, groups


def main() -> None:
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    v1.score_batches = score_batches
    v1.main()


if __name__ == "__main__":
    main()
