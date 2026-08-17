import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.model_training.v371_group_atomic_batch_v1 import (
    V371GroupAtomicBatchModel,
)


def test_batch_model_keeps_lab_groups_atomic_and_predicts_each_batch():
    rows = []
    for group in range(48):
        material = "Silica" if group % 2 else "CarbonBlack"
        for batch in (1, 2):
            rows.append(
                {
                    "LabelGroupID": f"G{group}",
                    "OrderID": f"O{group}",
                    "PalletID": f"P{group}",
                    "BatchNumber": batch,
                    "lab_time": pd.Timestamp("2025-01-01") + pd.Timedelta(group, unit="D"),
                    "CompoundName": f"C{group % 4}",
                    "material_system": material,
                    "is_oil_loading_present": float(group % 3 == 0),
                    "Stage4_WetMixing_Duration": float(20 if group % 3 == 0 else 0),
                    "static_feature": float(group % 4),
                    "process_feature": float(group + batch),
                    "MNY": float(40 + group % 7),
                }
            )
    frame = pd.DataFrame(rows)
    model = V371GroupAtomicBatchModel(
        use_material_route_matrix=True,
        use_silica_subsystem=False,
        use_cb_subsystem=False,
        crossfit_splits=4,
    ).fit(frame, ["static_feature"], ["process_feature"])
    prediction = model.predict(frame)[0]
    assert len(prediction) == len(frame)
    assert np.isfinite(prediction).all()
    assert model.crossfit_audit_["each_group_effective_weight_one"] is True
    for fold in model.crossfit_audit_["folds"]:
        assert pd.Timestamp(fold["train_max_time"]) < pd.Timestamp(fold["valid_min_time"])
