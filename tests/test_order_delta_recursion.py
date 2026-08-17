import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.model_training.order_delta_recursion import recursive_order_prediction


def test_numeric_pair_ids_are_applied_to_string_normalized_groups():
    frame = pd.DataFrame(
        {
            "LabelGroupID": [101, 102, 201],
            "OrderID": [1, 1, 2],
            "PalletID": [1, 2, 1],
            "production_start": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01"]),
        }
    )
    pairs = pd.DataFrame({"to_group": [102], "DeltaPrediction": [4.0]})
    actual = recursive_order_prediction(frame, np.array([50.0, 51.0, 60.0]), pairs, beta=0.0)
    assert actual.tolist() == [50.0, 54.0, 60.0]


def test_beta_one_preserves_level_prediction():
    frame = pd.DataFrame(
        {
            "LabelGroupID": ["a", "b"],
            "OrderID": [1, 1],
            "PalletID": [1, 2],
            "production_start": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        }
    )
    pairs = pd.DataFrame({"to_group": ["b"], "DeltaPrediction": [9.0]})
    level = np.array([50.0, 52.0])
    np.testing.assert_allclose(recursive_order_prediction(frame, level, pairs, beta=1.0), level)
