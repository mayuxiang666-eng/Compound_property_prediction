import pandas as pd

from services.mooney_predict_service.phase_group_contract import _route_fill_allowed
from services.mooney_predict_service.phase_group_inference_v3 import (
    STAGE6_RAM_COLUMNS,
    _route_fill_allowed_v3,
)


def test_stage6_ram_is_new_explicit_physical_zero_only():
    row = pd.Series({"material_system": "Silica", "is_oil_loading_present": 1})
    for column in STAGE6_RAM_COLUMNS:
        assert not _route_fill_allowed(row, column)
        assert _route_fill_allowed_v3(row, column)


def test_unrelated_missing_feature_remains_blocked():
    row = pd.Series({"material_system": "Silica", "is_oil_loading_present": 1})
    assert not _route_fill_allowed_v3(row, "Stage6_BottomMixing_temp_Mean")
