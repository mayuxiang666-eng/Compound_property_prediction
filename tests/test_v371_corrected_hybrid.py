import numpy as np
import pandas as pd
import pytest

from Mooney_Prediction_Pipeline.model_training.v371_corrected_hybrid_v1 import (
    V371CorrectedHybridModel,
)


def _training_frame() -> pd.DataFrame:
    rows = []
    routes = [
        ("CarbonBlack", 0.0),
        ("CarbonBlack", 1.0),
        ("Silica", 0.0),
        ("Silica", 1.0),
    ]
    for route_index, (material, oil) in enumerate(routes):
        for index in range(12):
            rows.append(
                {
                    "OrderID": f"{route_index}{index}",
                    "PalletID": f"P{route_index}{index}",
                    "CompoundName": f"C{route_index}",
                    "material_system": material,
                    "is_oil_loading_present": oil,
                    "Stage4_WetMixing_Duration": 20.0 if oil else 0.0,
                    "static_feature": float(route_index + index / 10),
                    "process_feature": float(index),
                    "MNY": float(40 + route_index * 5 + index / 3),
                }
            )
    return pd.DataFrame(rows)


def test_crossfit_audit_is_complete():
    frame = _training_frame()
    model = V371CorrectedHybridModel(
        use_cluster_experts=True,
        use_material_route_matrix=True,
        use_silica_subsystem=False,
        use_cb_subsystem=False,
        crossfit_splits=4,
    )
    model.fit(frame, ["static_feature"], ["process_feature"])
    assert model.crossfit_audit_["oof_stage1_complete"] is True
    assert model.crossfit_audit_["oof_stage1b_complete"] is True
    assert model.crossfit_audit_["stage2_residual_source"] == "OOF_STAGE1_PLUS_OOF_STAGE1B"


def test_unknown_expert_route_is_blocked():
    frame = _training_frame()
    model = V371CorrectedHybridModel(
        use_cluster_experts=True,
        use_material_route_matrix=True,
        use_silica_subsystem=False,
        use_cb_subsystem=False,
        crossfit_splits=4,
    ).fit(frame, ["static_feature"], ["process_feature"])
    unknown = frame.iloc[[0]].copy()
    unknown["material_system"] = "UnknownMaterial"
    with pytest.raises(ValueError, match="BLOCKED_UNKNOWN_EXPERT_ROUTE"):
        model.predict(unknown)

