import pandas as pd
import pytest

from Mooney_Prediction_Pipeline.model_training.stage3_group_calibration import (
    replay_group_stage3,
)


def test_stage3_uses_only_prior_available_independent_lab_groups():
    groups = pd.DataFrame(
        {
            "LabelGroupID": ["G1", "G2", "G3"],
            "CalibrationScope": ["T15760|ML09"] * 3,
            "ActualMNY": [50.0, 52.0, 51.0],
            "GroupRawPrediction": [48.0, 49.0, 50.0],
            "PredictionTime": [
                "2026-01-01 08:00", "2026-01-02 08:00", "2026-01-04 08:00"
            ],
            "FeedbackTime": [
                "2026-01-03 08:00", "2026-01-03 12:00", "2026-01-05 08:00"
            ],
        }
    )
    replay = replay_group_stage3(groups, history_groups=2)
    assert replay.loc[replay["LabelGroupID"].eq("G1"), "Stage3SupportGroupCount"].item() == 0
    assert replay.loc[replay["LabelGroupID"].eq("G2"), "Stage3SupportGroupCount"].item() == 0
    g3 = replay.loc[replay["LabelGroupID"].eq("G3")].iloc[0]
    assert g3["Stage3SupportGroupCount"] == 2
    assert g3["Stage3CandidateOffset"] == 2.5
    assert "G3" not in g3["Stage3SupportGroupIDs"]
    assert replay["Stage3Enabled"].eq(False).all()


def test_stage3_duplicate_group_is_blocked():
    groups = pd.DataFrame(
        {
            "LabelGroupID": ["G1", "G1"],
            "CalibrationScope": ["S", "S"],
            "ActualMNY": [50.0, 50.0],
            "GroupRawPrediction": [48.0, 49.0],
            "PredictionTime": ["2026-01-01", "2026-01-01"],
            "FeedbackTime": ["2026-01-02", "2026-01-02"],
        }
    )
    with pytest.raises(ValueError, match="DUPLICATE_LAB_GROUP_ROWS"):
        replay_group_stage3(groups)
