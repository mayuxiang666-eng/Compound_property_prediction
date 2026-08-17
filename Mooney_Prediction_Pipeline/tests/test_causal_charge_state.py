import pandas as pd

from Mooney_Prediction_Pipeline.feature_engineering.causal_charge_state import build_causal_charge_features


def test_current_and_future_lab_are_not_used_for_charge_state():
    groups = pd.DataFrame(
        {
            "LabelGroupID": ["g1", "g2", "g3"],
            "production_start": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-05"]),
            "lab_time": pd.to_datetime(["2026-01-02", "2026-01-06", "2026-01-07"]),
            "MNY": [50.0, 80.0, 60.0],
        }
    )
    charges = pd.DataFrame(
        {
            "LabelGroupID": ["g1", "g2", "g3"], "MaterialFamily": ["SILICA"] * 3,
            "ChargeID": ["A", "A", "A"], "ActualWeight": [100.0] * 3,
        }
    )
    result = build_causal_charge_features(groups, charges, shrinkage=0.0).set_index("LabelGroupID")
    assert result.loc["g1", "charge_state__SILICA__prior_count"] == 0.0
    assert result.loc["g2", "charge_state__SILICA__prior_count"] == 1.0
    assert result.loc["g2", "charge_state__SILICA__prior_delta"] == 0.0
    assert result.loc["g3", "charge_state__SILICA__prior_count"] == 1.0
