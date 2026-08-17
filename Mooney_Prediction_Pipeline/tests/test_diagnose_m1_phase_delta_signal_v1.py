import pandas as pd

from Mooney_Prediction_Pipeline.model_training.diagnose_m1_phase_delta_signal_v1 import (
    build_transitions,
    phase_feature_bundles,
)


def test_transition_requires_same_split_and_exact_template():
    feature = "phase_curve__Stage5_PID__duration"
    frame = pd.DataFrame([
        {"LabelGroupID": "g1", "OrderID": "o", "RecipeID": "r", "ProcessVersion": 1,
         "MixerLine": "m", "TemplateKey": "t", "material_system": "Silica",
         "phase_route": "oil_wet", "split": "train", "lab_time": pd.Timestamp("2025-01-01", tz="UTC"),
         "MNY": 50.0, feature: 100.0},
        {"LabelGroupID": "g2", "OrderID": "o", "RecipeID": "r", "ProcessVersion": 1,
         "MixerLine": "m", "TemplateKey": "t", "material_system": "Silica",
         "phase_route": "oil_wet", "split": "train", "lab_time": pd.Timestamp("2025-01-02", tz="UTC"),
         "MNY": 54.0, feature: 110.0},
        {"LabelGroupID": "g3", "OrderID": "o", "RecipeID": "r", "ProcessVersion": 1,
         "MixerLine": "m", "TemplateKey": "t", "material_system": "Silica",
         "phase_route": "oil_wet", "split": "val", "lab_time": pd.Timestamp("2025-01-03", tz="UTC"),
         "MNY": 56.0, feature: 120.0},
    ])
    transitions = build_transitions(frame, [feature])
    assert len(transitions) == 1
    assert transitions.iloc[0]["actual_delta_mny"] == 4.0
    assert transitions.iloc[0][f"delta__{feature}"] == 10.0


def test_feature_bundles_exclude_loading_and_dense_points():
    columns = [
        "phase_curve__Stage2_DryMixing__duration",
        "phase_curve__Stage2_DryMixing__temp__p00",
        "phase_curve__Stage1_Loading__duration",
    ]
    bundles = phase_feature_bundles(columns)
    assert bundles["Stage2_DryMixing"] == ["phase_curve__Stage2_DryMixing__duration"]
    assert "phase_curve__Stage1_Loading__duration" not in bundles["DryWetPIDBottom_All"]
