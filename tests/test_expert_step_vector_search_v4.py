from types import SimpleNamespace

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.optimization.expert_shadow_policy import EndpointBand
from Mooney_Prediction_Pipeline.optimization.expert_step_vector_search_v4 import (
    search_expert_step_vectors_strict_actual_phase,
)
from Mooney_Prediction_Pipeline.optimization.strict_actual_phase_response_contract import (
    response_column,
)


class _Raw:
    feature_columns = ["Stage4_WetMixing_temp_Mean"]

    def predict_raw(self, features):
        value = float(features.iloc[0]["Stage4_WetMixing_temp_Mean"])
        return SimpleNamespace(final_prediction=np.array([200.0 - value]))


class _StrictResponse:
    approved = True

    def __init__(self, include_actual=True):
        self.include_actual = include_actual

    def predict(self, vector):
        temp = float(vector.iloc[0]["AVR_MST_S4_temp"])
        row = {
            "dry_end_temp_actual_p10": 99.0,
            "dry_end_temp_actual_p50": 100.0,
            "dry_end_temp_actual_p90": 101.0,
            "post_oil_temp_actual_p10": 99.0,
            "post_oil_temp_actual_p50": 100.0,
            "post_oil_temp_actual_p90": 101.0,
            "wet_end_temp_actual_p10": temp - 0.5,
            "wet_end_temp_actual_p50": temp,
            "wet_end_temp_actual_p90": temp + 0.5,
            "wet_duration_actual_p10": 19.0,
            "wet_duration_actual_p50": 20.0,
            "wet_duration_actual_p90": 21.0,
            "discharge_temp_actual_p10": 149.0,
            "discharge_temp_actual_p50": 150.0,
            "discharge_temp_actual_p90": 151.0,
        }
        if self.include_actual:
            row[response_column("Stage4_WetMixing_temp_Mean")] = temp
        return pd.DataFrame([row])


def _steps():
    return pd.DataFrame([
        {
            "OrderID": "O",
            "BatchNumber": 1,
            "GroupName": "AVR_MST",
            "StepNo": 4,
            "time_SP": 20.0,
            "rotor_SP": 40.0,
            "ram_SP": 0.0,
            "temp_SP": 148.0,
            "mapped_phase": "Stage4_WetMixing_Ramp",
            "is_time_settable": True,
            "is_rotor_settable": True,
            "is_ram_settable": False,
            "is_temp_settable": True,
            "step_time_min": 10.0,
            "step_time_max": 40.0,
            "step_rotor_min": 5.0,
            "step_rotor_max": 50.0,
            "step_temp_min": 140.0,
            "step_temp_max": 150.0,
        }
    ])


def _bands():
    return {
        "dry_end_temp_actual": EndpointBand(100.0, 90.0, 110.0, 80.0, 120.0, 200),
        "post_oil_temp_actual": EndpointBand(100.0, 90.0, 110.0, 80.0, 115.0, 200),
        "wet_end_temp_actual": EndpointBand(149.0, 140.0, 150.0, 130.0, 150.0, 200),
        "wet_duration_actual": EndpointBand(20.0, 10.0, 30.0, 5.0, 35.0, 200),
    }


def test_v4_uses_explicit_actual_phase_response():
    result = search_expert_step_vectors_strict_actual_phase(
        _steps(),
        pd.Series({"Stage4_WetMixing_temp_Mean": 148.0}),
        _Raw(),
        _StrictResponse(),
        required_actual_phase_features=["Stage4_WetMixing_temp_Mean"],
        material_system="CarbonBlack",
        process_target_temperature_cap=150.0,
        endpoint_bands=_bands(),
        spec_center=50.0,
        spec_low=45.0,
        spec_high=55.0,
    )
    assert result.status == "MANUAL_REVIEW_ONLY"
    assert result.audit_table["strict_actual_phase_response_status"].eq(
        "ACTUAL_PHASE_RESPONSE_OK"
    ).all()
    assert result.audit_table["endpoint_substitution_allowed"].eq(False).all()


def test_v4_blocks_endpoint_only_response():
    result = search_expert_step_vectors_strict_actual_phase(
        _steps(),
        pd.Series({"Stage4_WetMixing_temp_Mean": 148.0}),
        _Raw(),
        _StrictResponse(include_actual=False),
        required_actual_phase_features=["Stage4_WetMixing_temp_Mean"],
        material_system="CarbonBlack",
        process_target_temperature_cap=150.0,
        endpoint_bands=_bands(),
        spec_center=50.0,
        spec_low=45.0,
        spec_high=55.0,
    )
    assert result.status == "DIAGNOSTIC_ONLY"
    assert "CONTROL_TRANSFER_UNAVAILABLE" in result.reason_codes
