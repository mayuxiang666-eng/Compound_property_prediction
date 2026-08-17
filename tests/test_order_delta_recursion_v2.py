import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.model_training.order_delta_recursion_v2 import recursive_order_prediction


def frame():
    return pd.DataFrame(
        {
            "LabelGroupID": [101, 102, 201],
            "OrderID": [1, 1, 2],
            "PalletID": [1, 2, 1],
            "production_start": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01"]),
        }
    )


def test_numeric_pair_ids_are_applied():
    pairs = pd.DataFrame({"to_group": [102], "DeltaPrediction": [4.0]})
    actual = recursive_order_prediction(frame(), np.array([50.0, 51.0, 60.0]), pairs, beta=0.0)
    assert actual.tolist() == [50.0, 54.0, 60.0]


def test_input_memory_is_not_mutated_between_beta_trials():
    pairs = pd.DataFrame({"to_group": [102], "DeltaPrediction": [4.0]})
    level = np.array([50.0, 51.0, 60.0])
    original = level.copy()
    zero = recursive_order_prediction(frame(), level, pairs, beta=0.0)
    one = recursive_order_prediction(frame(), level, pairs, beta=1.0)
    np.testing.assert_allclose(level, original)
    assert zero.tolist() == [50.0, 54.0, 60.0]
    np.testing.assert_allclose(one, original)
