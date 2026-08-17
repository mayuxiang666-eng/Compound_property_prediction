from __future__ import annotations

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.strict_actual_phase_response_contract import (
    ActualPhaseResponseContractError,
    overlay_actual_v37_phase_features,
    response_column,
)


def test_endpoint_temperature_cannot_replace_phase_mean() -> None:
    baseline = pd.DataFrame([{"Stage6_BottomMixing_temp_Mean": 150.0}])
    endpoint_only = pd.DataFrame([{"discharge_temp_actual_p50": 152.0}])
    try:
        overlay_actual_v37_phase_features(
            baseline, endpoint_only, ["Stage6_BottomMixing_temp_Mean"]
        )
    except ActualPhaseResponseContractError as exc:
        assert "CONTROL_TRANSFER_UNAVAILABLE" in str(exc)
    else:
        raise AssertionError("Endpoint-only response must fail closed")


def test_explicit_actual_phase_feature_can_be_overlaid() -> None:
    feature = "Stage6_BottomMixing_temp_Mean"
    baseline = pd.DataFrame([{feature: 150.0}])
    response = pd.DataFrame([{response_column(feature): 151.25}])
    result = overlay_actual_v37_phase_features(baseline, response, [feature])
    assert result.iloc[0][feature] == 151.25
