import pandas as pd

from Mooney_Prediction_Pipeline.model_training.run_v371_c1_thermal_budget_h3_v1 import (
    build_train_only_nominals,
)


def test_thermal_features_use_train_nominal_and_no_oil_is_physical_zero():
    rows = []
    for index in range(12):
        rows.append({
            "LabelGroupID": f"g{index}", "RecipeID": "r", "ProcessVersion": 1,
            "MixerLine": "m", "material_system": "Silica", "phase_route": "oil_wet",
            "phase_curve__Stage3_OilLoading__temp__last": 100.0 if index < 11 else 105.0,
            "phase_curve__Stage4_WetMixing__duration": 60.0 if index < 11 else 50.0,
            "phase_curve__Stage4_WetMixing__rotor__integral": 2000.0 if index < 11 else 1800.0,
        })
    rows.append({
        "LabelGroupID": "dry", "RecipeID": "dry", "ProcessVersion": 1,
        "MixerLine": "m", "material_system": "CarbonBlack", "phase_route": "no_oil_dry",
        "phase_curve__Stage3_OilLoading__temp__last": float("nan"),
        "phase_curve__Stage4_WetMixing__duration": float("nan"),
        "phase_curve__Stage4_WetMixing__rotor__integral": float("nan"),
    })
    batch = pd.DataFrame(rows)
    groups = pd.DataFrame({
        "LabelGroupID": [f"g{i}" for i in range(12)] + ["dry"],
        "split": ["train"] * 11 + ["oot", "oot"],
    })
    augmented, _ = build_train_only_nominals(batch, groups, minimum_exact_groups=10)
    oot = augmented[augmented["LabelGroupID"].eq("g11")].iloc[0]
    assert oot["mechanism_stage3_oil_temp_excess"] == 5.0
    assert oot["mechanism_stage4_wet_duration_deficit"] == 10.0
    assert oot["mechanism_stage3_oil_temp_x_wet_duration_deficit"] == 50.0
    dry = augmented[augmented["LabelGroupID"].eq("dry")].iloc[0]
    assert dry["mechanism_oil_wet_route_applicable"] == 0.0
    assert dry["mechanism_stage3_oil_temp_excess"] == 0.0
