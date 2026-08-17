import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.model_training.run_v371_c1_history_augmented_phase_v1 import (
    add_common_static_contract,
    make_training_frame,
    phase_feature_columns,
)


def test_phase_feature_contract_excludes_dense_resampled_points():
    columns = [
        "phase_curve__Stage2_DryMixing__duration",
        "phase_curve__Stage2_DryMixing__temp__mean",
        "phase_curve__Stage2_DryMixing__temp__p00",
        "phase_curve__Stage5_PID__power__integral",
    ]
    selected = phase_feature_columns(columns)
    assert "phase_curve__Stage2_DryMixing__duration" in selected
    assert "phase_curve__Stage2_DryMixing__temp__mean" in selected
    assert "phase_curve__Stage5_PID__power__integral" in selected
    assert "phase_curve__Stage2_DryMixing__temp__p00" not in selected


def test_common_static_contract_keeps_missing_semantics_and_derives_route():
    source = pd.DataFrame({
        "weight_pct_solid_elastomer": [25.0, 0.0],
        "weight_pct_natural_rubber": [10.0, 0.0],
        "weight_pct_silica": [35.0, 0.0],
        "weight_pct_oil": [5.0, 0.0],
        "weight_pct_silian": [2.0, 0.0],
        "weight_pct_carbon_black": [0.0, 40.0],
        "CurrentValue": [100.0, np.nan],
        "PrevStepValue": [50.0, np.nan],
    })
    result = add_common_static_contract(source)
    assert result.loc[0, "silica_to_elastomer_phr_proxy"] == 140.0
    assert result.loc[0, "phase_route"] == "oil_wet"
    assert result.loc[1, "phase_route"] == "no_oil_dry"
    assert np.isnan(result.loc[1, "ratio_filler_polymer"])


def test_mixed_lab_time_formats_are_normalized_before_temporal_model():
    # Regression for pandas strict mixed-format parsing: historical seconds and
    # current fractional-seconds UTC values must remain valid in one frame.
    current = pd.DataFrame({
        "OrderID": ["1"], "BatchNumber": ["1"], "PalletID": ["10"],
        "LabelGroupID": ["1|10"], "CompoundName": ["M1-X"], "RecipeID": ["R"],
        "ProcessVersion": [1], "MixerLine": ["M"], "material_system": ["Silica"],
        "phase_route": ["oil_wet"], "TemplateKey": ["T"], "feature": [1.0],
    })
    history = current.assign(
        OrderID="0", LabelGroupID="0|10", data_source="history"
    )
    current["data_source"] = "current"
    current_labels = pd.DataFrame({
        "LabelGroupID": ["1|10"], "MNY": [70.0],
        "lab_time": ["2026-08-01 01:02:03.123+00:00"], "split": ["train"],
    })
    history_labels = pd.DataFrame({
        "LabelGroupID": ["0|10"], "MNY": [71.0],
        "lab_time": ["2025-06-01 01:02:03"],
    })
    frame, *_ = make_training_frame(
        current, history, current_labels, history_labels,
        {"s1_features": ["feature"], "s2_features": ["feature"]},
        include_history=True,
    )
    assert pd.to_datetime(frame["lab_time"], errors="coerce", utc=True).notna().all()
