import numpy as np
import pandas as pd
import pytest

from Mooney_Prediction_Pipeline.model_training.trend_metric_contract import (
    build_trend_pairs,
    evaluate_level_and_trend,
)


def _frame(prediction):
    actual = np.array([50, 53, 49, 54, 51, 55], dtype=float)
    return pd.DataFrame({
        "LabelGroupID": [f"g{i}" for i in range(len(actual))],
        "RecipeID": "R1",
        "ProcessVersion": "PV1",
        "MixerLine": "M1",
        "lab_time": pd.date_range("2026-01-01", periods=len(actual), freq="D"),
        "MNY": actual,
        "prediction": prediction,
    })


def test_perfect_level_and_trend_passes_relaxed_support():
    frame = _frame([50, 53, 49, 54, 51, 55])
    metrics, pairs = evaluate_level_and_trend(
        frame, actual_col="MNY", prediction_col="prediction", time_col="lab_time",
        segment_cols=["RecipeID", "ProcessVersion", "MixerLine"],
        minimum_pairs=5, minimum_significant_pairs=5,
    )
    assert metrics.level_mae == 0.0
    assert metrics.direction_accuracy == 1.0
    assert metrics.level_gate_passed and metrics.trend_gate_passed
    assert len(pairs) == 5


def test_flat_mean_prediction_fails_trend_even_with_finite_level_mae():
    frame = _frame([52, 52, 52, 52, 52, 52])
    metrics, _ = evaluate_level_and_trend(
        frame, actual_col="MNY", prediction_col="prediction", time_col="lab_time",
        segment_cols=["RecipeID", "ProcessVersion", "MixerLine"],
        minimum_pairs=5, minimum_significant_pairs=5,
    )
    assert metrics.delta_std_ratio == 0.0
    assert not metrics.trend_gate_passed


def test_duplicate_lab_groups_are_rejected():
    frame = _frame([50, 53, 49, 54, 51, 55])
    frame.loc[1, "LabelGroupID"] = "g0"
    with pytest.raises(ValueError, match="one row per Lab group"):
        build_trend_pairs(
            frame, actual_col="MNY", prediction_col="prediction", time_col="lab_time",
            segment_cols=["RecipeID", "ProcessVersion", "MixerLine"],
        )
