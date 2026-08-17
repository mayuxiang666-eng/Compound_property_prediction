import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.feature_engineering.batchdata_step_boundary_builder import TIME_VARIABLE
from Mooney_Prediction_Pipeline.feature_engineering.step_aligned_curve_features import extract_step_aligned_curve_features


def _hex(values):
    return "".join(f"{int(value):04X}" for value in values)


def test_bottom_feature_uses_last_msb_seconds_not_transfer():
    length = 416
    temp = np.r_[np.repeat(100, 203), np.repeat(50, 108), np.repeat(160, 105)]
    power = np.r_[np.repeat(100, 203), np.repeat(10, 108), np.repeat(200, 105)]
    row = pd.Series(
        {
            "OrderID": "1", "BatchNumber": "1", "Curve1": _hex(temp), "Curve2": _hex(power),
            "Curve5": _hex(np.repeat(50, 203)), "Curve6": _hex(np.repeat(30, 203)),
            "Curve7": _hex(np.repeat(80, 203)),
        }
    )
    times = [13, 28, 75, 95, 111, 112, 150, 160, 180, 200, 203]
    data = pd.DataFrame(
        [{"GroupName": "AVR_MST", "StepNo": i + 1, "VariablePath": TIME_VARIABLE, "Target": value, "Value": value} for i, value in enumerate(times)]
        + [{"GroupName": "AVR_MSB", "StepNo": i + 1, "VariablePath": TIME_VARIABLE, "Target": value, "Value": value} for i, value in enumerate((75, 90, 105))]
    )
    mapping = {(group, step): f"{group}_S{step}" for group in ("AVR_MST", "AVR_MSB") for step in range(1, 12)}
    features, boundary = extract_step_aligned_curve_features(row, data, mapping, points=4)
    assert boundary.audit["MSBCurveStart"] == 311.0
    assert features["step_phase__AVR_MSB_S1__temp__mean"] == 160.0
    assert features["step_phase__Transfer__temp__mean"] == 50.0
    assert np.isnan(features["step_phase__AVR_MSB_S1__rotor__mean"])
