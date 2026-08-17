import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.model_training.r10b_phase_curve_features import (
    derive_phase_boundaries,
    extract_phase_curve_row,
)


def _hex(values):
    return "".join(f"{int(value):04X}" for value in values)


def test_phase_boundaries_are_monotonic_and_current_batch_only():
    temp = np.r_[np.linspace(70, 150, 120), np.linspace(150, 80, 20), np.linspace(80, 160, 100)]
    power = np.r_[np.repeat(100, 220), np.repeat(0, 20)]
    ram = np.r_[np.repeat(100, 30), np.repeat(80, 90)]
    row = pd.Series({"PrevStepValue": 70, "CurrentValue": 95, "Target_Temperature": 160})
    result = derive_phase_boundaries(row, {"temp": temp, "power": power, "ram": ram})
    assert [result[key] for key in ("t1", "t2", "t3", "t4", "t5", "t6", "t7", "total")] == sorted(
        result[key] for key in ("t1", "t2", "t3", "t4", "t5", "t6", "t7", "total")
    )


def test_short_top_only_signal_is_not_filled_into_bottom_mixing():
    temp = np.r_[np.linspace(70, 150, 120), np.linspace(150, 80, 20), np.linspace(80, 160, 100)]
    power = np.r_[np.repeat(100, 220), np.repeat(0, 20)]
    row = pd.Series(
        {
            "OrderID": "1",
            "BatchNumber": "1",
            "PrevStepValue": 70,
            "CurrentValue": 95,
            "Target_Temperature": 160,
            "Curve1": _hex(temp),
            "Curve2": _hex(power),
            "Curve5": _hex(np.repeat(50, 120)),
            "Curve6": _hex(np.repeat(30, 120)),
            "Curve7": _hex(np.r_[np.repeat(100, 30), np.repeat(80, 90)]),
        }
    )
    result = extract_phase_curve_row(row, points=4)
    assert result["phase_curve__boundary_status"] == "CURRENT_BATCH_ONLY"
    assert np.isnan(result["phase_curve__Stage6_BottomMixing__torque__mean"])
    assert result["phase_curve__Stage6_BottomMixing__torque__observed_fraction"] == 0.0
