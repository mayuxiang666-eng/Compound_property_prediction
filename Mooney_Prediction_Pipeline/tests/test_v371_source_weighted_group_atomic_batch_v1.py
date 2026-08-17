import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.model_training.v371_source_weighted_group_atomic_batch_v1 import (
    V371SourceWeightedGroupAtomicBatchModel,
)


def _frame(weights=(1.0, 0.25)):
    return pd.DataFrame({
        "OrderID": ["1", "1", "0", "0"],
        "BatchNumber": ["1", "2", "1", "2"],
        "LabelGroupID": ["1|10", "1|10", "0|10", "0|10"],
        "lab_time": ["2026-01-02"] * 2 + ["2025-01-02"] * 2,
        "MNY": [70.0] * 2 + [72.0] * 2,
        "_group_training_weight": [weights[0]] * 2 + [weights[1]] * 2,
    })


def test_source_weight_preserves_one_group_vote_times_reliability():
    model = V371SourceWeightedGroupAtomicBatchModel()
    prepared = model._prepare_batch_frame(_frame(), "MNY")
    effective = prepared.groupby("LabelGroupID")["_w_loss"].sum().sort_index()
    assert np.allclose(effective.to_numpy(), [0.25, 1.0])


def test_source_weight_must_be_constant_inside_lab_group():
    source = _frame()
    source.loc[1, "_group_training_weight"] = 0.5
    model = V371SourceWeightedGroupAtomicBatchModel()
    try:
        model._prepare_batch_frame(source, "MNY")
    except ValueError as exc:
        assert "CONFLICTING_GROUP_TRAINING_WEIGHT" in str(exc)
    else:
        raise AssertionError("conflicting group weights must fail closed")
