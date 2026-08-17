import numpy as np
import pandas as pd
import pytest

from Mooney_Prediction_Pipeline.model_training.temporal_validation import (
    expanding_window_folds,
)
from Mooney_Prediction_Pipeline.model_training.v371_corrected_hybrid_v2 import (
    V371CorrectedTemporalGroupModel,
)


def _group_training_frame() -> pd.DataFrame:
    rows = []
    routes = [
        ("CarbonBlack", 0.0), ("CarbonBlack", 1.0),
        ("Silica", 0.0), ("Silica", 1.0),
    ]
    counter = 0
    for route_index, (material, oil) in enumerate(routes):
        for index in range(12):
            rows.append(
                {
                    "OrderID": f"O{counter}",
                    "PalletID": f"P{counter}",
                    "LabelGroupID": f"O{counter}|P{counter}",
                    "lab_time": pd.Timestamp("2025-01-01") + pd.Timedelta(days=counter),
                    "CompoundName": f"C{route_index}",
                    "material_system": material,
                    "is_oil_loading_present": oil,
                    "Stage4_WetMixing_Duration": 20.0 if oil else 0.0,
                    "static_feature": float(route_index + index / 10),
                    "process_feature": float(index),
                    "MNY": float(40 + route_index * 5 + index / 3),
                }
            )
            counter += 1
    return pd.DataFrame(rows)


def test_expanding_folds_never_train_on_future_rows():
    folds = expanding_window_folds(50, n_splits=5, min_train_fraction=0.2)
    validated = []
    for train, valid in folds:
        assert train.max() < valid.min()
        validated.extend(valid.tolist())
    assert validated == list(range(10, 50))


def test_r1_uses_equal_groups_and_temporal_oof():
    frame = _group_training_frame()
    model = V371CorrectedTemporalGroupModel(
        use_material_route_matrix=True,
        use_silica_subsystem=False,
        use_cb_subsystem=False,
        crossfit_splits=4,
        objective="regression_l1",
    ).fit(frame, ["static_feature"], ["process_feature"])
    audit = model.crossfit_audit_
    assert audit["all_group_weights_equal_one"] is True
    assert audit["split_mode"] == "EXPANDING_WINDOW_BY_LAB_TIME"
    assert audit["stage3_calibration_enabled"] is False
    assert audit["oof_eligible_rows"] + audit["warmup_rows"] == len(frame)
    for fold in audit["folds"]:
        assert fold["train_max_index"] < fold["valid_min_index"]
        assert pd.Timestamp(fold["train_max_time"]) < pd.Timestamp(fold["valid_min_time"])
    prediction = model.predict(frame)[0]
    assert np.isfinite(prediction).all()


def test_r1_rejects_duplicate_lab_group_rows():
    frame = _group_training_frame()
    frame.loc[1, "LabelGroupID"] = frame.loc[0, "LabelGroupID"]
    with pytest.raises(ValueError, match="DUPLICATE_LAB_GROUP_ROWS"):
        V371CorrectedTemporalGroupModel(
            use_material_route_matrix=True,
            use_silica_subsystem=False,
            use_cb_subsystem=False,
        ).fit(frame, ["static_feature"], ["process_feature"])
