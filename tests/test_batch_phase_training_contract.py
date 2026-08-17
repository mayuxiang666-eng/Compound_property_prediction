import pandas as pd

from Mooney_Prediction_Pipeline.model_training.batch_phase_training_contract import (
    build_batch_phase_training_frame,
)


def test_batch_contract_keeps_groups_atomic_and_one_effective_vote():
    batch = pd.DataFrame(
        {
            "OrderID": ["O1", "O1", "O2"],
            "BatchNumber": [1, 2, 1],
            "PalletID": ["P1", "P1", "P2"],
            "LabelGroupID": ["O1|P1", "O1|P1", "O2|P2"],
            "CompoundName": ["C", "C", "C"],
            "RecipeID": ["R", "R", "R"],
            "ProcessVersion": [1, 1, 1],
            "MixerLine": ["M", "M", "M"],
            "material_system": ["Silica", "Silica", "Silica"],
            "phase_route": ["oil_wet", "oil_wet", "oil_wet"],
            "TemplateKey": ["T", "T", "T"],
            "static": [1.0, 1.0, 1.0],
            "process": [1.0, None, 3.0],
        }
    )
    labels = pd.DataFrame(
        {
            "LabelGroupID": ["O1|P1", "O2|P2"],
            "MNY": [50.0, 40.0],
            "lab_time": ["2025-01-01", "2025-01-02"],
            "split": ["train", "val"],
        }
    )
    metadata = {"s1_features": ["static"], "s2_features": ["process", "process__group_std"]}
    frame, s1, s2, audit = build_batch_phase_training_frame(batch, labels, metadata)
    assert s1 == ["static"]
    assert s2 == ["process", "process__missing"]
    assert frame.groupby("LabelGroupID")["_w_loss"].sum().eq(1.0).all()
    assert audit["excluded_group_aggregate_feature_count"] == 1
