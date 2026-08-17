import pandas as pd

from Mooney_Prediction_Pipeline.model_training.v371_hierarchical_nominal_v1 import (
    V371HierarchicalNominalModel,
)


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(48):
        version = 1 if index < 24 else 2
        template = f"T{version}_{index // 4}"
        rows.append(
            {
                "LabelGroupID": f"O{index}|P{index}",
                "OrderID": f"O{index}",
                "PalletID": f"P{index}",
                "lab_time": pd.Timestamp("2025-01-01") + pd.Timedelta(index, unit="D"),
                "CompoundName": "M1-T15760---- 06 002",
                "RecipeID": "R15760",
                "ProcessVersion": version,
                "MixerLine": "ML06",
                "TemplateKey": template,
                "material_system": "Silica",
                "is_oil_loading_present": 1.0,
                "Stage4_WetMixing_Duration": 30.0,
                "static_feature": float(version),
                "process_feature": float(version * 100 + index % 4),
                "MNY": float(50 + version + index % 5),
            }
        )
    return pd.DataFrame(rows)


def test_hierarchical_nominal_prefers_template_then_version_then_recipe():
    frame = _frame()
    model = V371HierarchicalNominalModel(
        use_material_route_matrix=True,
        use_silica_subsystem=False,
        use_cb_subsystem=False,
        crossfit_splits=3,
        template_min_groups=4,
        recipe_version_min_groups=4,
        recipe_min_groups=4,
        compound_min_groups=4,
    ).fit(frame, ["static_feature"], ["process_feature"])

    exact = frame.iloc[[0]].copy()
    assert model.nominal_resolution(exact).iloc[0] == "TEMPLATE"

    unseen_template = exact.copy()
    unseen_template["TemplateKey"] = "UNSEEN_TEMPLATE"
    assert model.nominal_resolution(unseen_template).iloc[0] == "RECIPE_VERSION_MIXER"

    unseen_version = unseen_template.copy()
    unseen_version["ProcessVersion"] = 99
    assert model.nominal_resolution(unseen_version).iloc[0] == "RECIPE_MIXER"
