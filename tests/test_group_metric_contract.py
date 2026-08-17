import pandas as pd
import pytest

from Mooney_Prediction_Pipeline.model_training.group_metric_contract import (
    GROUP_METRIC_CONTRACT,
    aggregate_batch_predictions,
    assert_one_row_per_lab_group,
    metric_block,
)


def test_batch_predictions_are_averaged_once_per_lab_group():
    batches = pd.DataFrame(
        {
            "LabelGroupID": ["O1|P1", "O1|P1", "O2|P2"],
            "OrderID": ["O1", "O1", "O2"],
            "PalletID": ["P1", "P1", "P2"],
            "ActualMNY": [50.0, 50.0, 40.0],
            "RawPrediction": [48.0, 52.0, 42.0],
        }
    )
    groups = aggregate_batch_predictions(batches)
    assert GROUP_METRIC_CONTRACT.startswith("PREDICT_EACH_BATCH")
    assert len(groups) == 2
    first = groups.loc[groups["LabelGroupID"].eq("O1|P1")].iloc[0]
    assert first["GroupRawPrediction"] == 50.0
    assert first["BatchCount"] == 2
    metrics = metric_block(groups)
    assert metrics["mae"] == 1.0


def test_conflicting_lab_values_within_group_are_blocked():
    batches = pd.DataFrame(
        {
            "LabelGroupID": ["O1|P1", "O1|P1"],
            "ActualMNY": [50.0, 51.0],
            "RawPrediction": [49.0, 50.0],
        }
    )
    with pytest.raises(ValueError, match="CONFLICTING_LAB_LABEL_WITHIN_GROUP"):
        aggregate_batch_predictions(batches)


def test_group_training_frame_rejects_duplicate_label_events():
    groups = pd.DataFrame({"LabelGroupID": ["G1", "G1"], "MNY": [40.0, 40.0]})
    with pytest.raises(ValueError, match="DUPLICATE_LAB_GROUP_ROWS"):
        assert_one_row_per_lab_group(groups)
