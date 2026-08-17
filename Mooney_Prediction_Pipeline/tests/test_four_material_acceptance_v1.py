import pandas as pd

from Mooney_Prediction_Pipeline.model_training.audit_four_material_acceptance_v1 import (
    _direction_metrics,
    _group_frame,
    _metrics,
)


def test_group_metric_averages_batch_predictions_once_per_lab_group():
    raw = pd.DataFrame(
        [
            {"LabelGroupID": "100|1001", "MNY": 50.0, "RawPrediction": 48.0, "OrderID": 100, "CompoundName": "C", "material_system": "CarbonBlack", "split": "oot", "lab_time": "2026-01-01"},
            {"LabelGroupID": "100|1001", "MNY": 50.0, "RawPrediction": 52.0, "OrderID": 100, "CompoundName": "C", "material_system": "CarbonBlack", "split": "oot", "lab_time": None},
            {"LabelGroupID": "100|1002", "MNY": 54.0, "RawPrediction": 55.0, "OrderID": 100, "CompoundName": "C", "material_system": "CarbonBlack", "split": "oot", "lab_time": "2026-01-02"},
        ]
    )
    groups = _group_frame(raw)
    assert len(groups) == 2
    assert groups.loc[groups["LabelGroupID"] == "100|1001", "Prediction"].iloc[0] == 50.0
    assert _metrics(groups)["groups"] == 2


def test_direction_and_significant_direction_are_separate_metrics():
    groups = pd.DataFrame(
        {
            "OrderID": [1, 1, 1],
            "LabelGroupID": ["1|1", "1|2", "1|3"],
            "MNY": [50.0, 53.0, 51.0],
            "Prediction": [50.0, 52.0, 52.0],
            "lab_time": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
        }
    )
    result = _direction_metrics(groups)
    assert result["direction_pairs"] == 2
    assert result["direction_accuracy"] == 0.5
    assert result["significant_direction_pairs_abs_actual_ge_2"] == 2
