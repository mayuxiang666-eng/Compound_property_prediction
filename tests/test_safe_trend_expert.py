import pandas as pd

from Mooney_Prediction_Pipeline.model_training.safe_trend_expert import (
    attach_safe_prior_lab_anchors,
)


def _groups():
    return pd.DataFrame([
        {"LabelGroupID": "g1", "OrderID": "O1", "RecipeID": "R1", "ProcessVersion": "P1", "MixerLine": "M1", "CompoundName": "C", "material_system": "Silica", "phase_route": "oil_wet", "MNY": 50, "ProductionComplete": "2026-01-01 10:00", "lab_time": "2026-01-01 14:00", "split": "train", "TemplateKey": "T1"},
        {"LabelGroupID": "g2", "OrderID": "O1", "RecipeID": "R1", "ProcessVersion": "P1", "MixerLine": "M1", "CompoundName": "C", "material_system": "Silica", "phase_route": "oil_wet", "MNY": 52, "ProductionComplete": "2026-01-01 12:00", "lab_time": "2026-01-01 16:00", "split": "train", "TemplateKey": "T1"},
        {"LabelGroupID": "g3", "OrderID": "O1", "RecipeID": "R1", "ProcessVersion": "P1", "MixerLine": "M1", "CompoundName": "C", "material_system": "Silica", "phase_route": "oil_wet", "MNY": 49, "ProductionComplete": "2026-01-01 15:00", "lab_time": "2026-01-01 20:00", "split": "val", "TemplateKey": "T1"},
    ])


def test_anchor_uses_only_lab_available_at_production_time():
    result = attach_safe_prior_lab_anchors(_groups(), production_time_col="ProductionComplete")
    assert result.loc[result.LabelGroupID.eq("g2"), "anchor_available"].iloc[0] == 0
    g3 = result.loc[result.LabelGroupID.eq("g3")].iloc[0]
    assert g3["anchor__LabelGroupID"] == "g1"
    assert g3["anchor_scope"] == "SAME_ORDER_RECIPE_VERSION_MIXER"
    assert pd.Timestamp(g3["anchor__lab_time"]) <= pd.Timestamp(g3["ProductionComplete"])


def test_future_validation_label_cannot_anchor_training_row():
    groups = _groups()
    groups.loc[2, "ProductionComplete"] = "2025-12-31"
    groups.loc[2, "lab_time"] = "2025-12-31 01:00"
    result = attach_safe_prior_lab_anchors(groups, production_time_col="ProductionComplete")
    train = result.loc[result.split.eq("train")]
    assert not train.get("anchor__split", pd.Series(dtype=str)).eq("val").any()


def test_same_order_anchor_has_priority_over_newer_other_order_anchor():
    groups = _groups()
    other = groups.iloc[[0]].copy()
    other["LabelGroupID"] = "other"
    other["OrderID"] = "O2"
    other["ProductionComplete"] = "2026-01-01 13:00"
    other["lab_time"] = "2026-01-01 14:30"
    result = attach_safe_prior_lab_anchors(
        pd.concat([groups, other], ignore_index=True),
        production_time_col="ProductionComplete",
    )
    g3 = result.loc[result.LabelGroupID.eq("g3")].iloc[0]
    assert g3["anchor__LabelGroupID"] == "g1"
