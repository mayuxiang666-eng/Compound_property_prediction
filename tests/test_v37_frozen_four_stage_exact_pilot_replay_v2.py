from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from Mooney_Prediction_Pipeline.model_training.run_v37_frozen_four_stage_exact_pilot_replay_v2 import (
    EXPECTED_MODEL_SHA256,
    add_training_membership,
    normalise_keys,
    validate_four_stage_artifact,
)


class SilicaSubsystemPredictor:
    def __init__(self) -> None:
        self.combiner_ = Ridge(positive=True).fit(np.eye(4), np.arange(4.0))


class CarbonBlackSubsystemPredictor:
    def __init__(self) -> None:
        self.combiner_ = Ridge(positive=True).fit(np.eye(3), np.arange(3.0))


def _valid_model():
    return SimpleNamespace(
        use_material_route_matrix=True,
        use_silica_subsystem=True,
        use_cb_subsystem=True,
        shrinkage_k=5.0,
        stage2_experts_={
            ("CarbonBlack", "no_oil_dry"): CarbonBlackSubsystemPredictor(),
            ("CarbonBlack", "oil_wet"): CarbonBlackSubsystemPredictor(),
            ("Silica", "no_oil_dry"): SilicaSubsystemPredictor(),
            ("Silica", "oil_wet"): SilicaSubsystemPredictor(),
        },
    )


def _valid_calibrator():
    return type("Stage3DelayedFeedbackCalibrator", (), {})()


def test_four_stage_identity_accepts_only_documented_expert_matrix() -> None:
    calibrator = _valid_calibrator()
    calibrator.lag_k = 3
    calibrator.alpha_base = 0.3
    calibrator.max_offset_abs = 4.0
    calibrator.adaptive = True
    result = validate_four_stage_artifact(_valid_model(), calibrator, EXPECTED_MODEL_SHA256)
    assert result["identity_status"] == "V37_FOUR_STAGE_EXPERT_ARTIFACT_CONFIRMED"
    assert len(result["stage2_routes"]) == 4


def test_four_stage_identity_rejects_unified_nonexpert_candidate() -> None:
    calibrator = _valid_calibrator()
    calibrator.lag_k = 3
    calibrator.alpha_base = 0.3
    calibrator.max_offset_abs = 4.0
    calibrator.adaptive = True
    model = _valid_model()
    model.use_silica_subsystem = False
    with pytest.raises(RuntimeError, match="use_silica_subsystem"):
        validate_four_stage_artifact(model, calibrator, EXPECTED_MODEL_SHA256)


def test_normalise_keys_removes_excel_style_identifier_suffix() -> None:
    frame = pd.DataFrame({"OrderID": ["100.0"], "PalletID": ["8.0"], "BatchNumber": [3.0]})
    result = normalise_keys(frame)
    assert result.loc[0, "LabelGroupID"] == "100|8"
    assert result.loc[0, "BatchKey"] == "100|3"


def test_training_membership_is_batch_key_exact(tmp_path) -> None:
    training = pd.DataFrame({
        "OrderID": ["100", "100"], "BatchNumber": [1, 2], "MNY": [40.0, 40.0],
    })
    path = tmp_path / "training.csv"
    training.to_csv(path, index=False)
    source = normalise_keys(pd.DataFrame({
        "OrderID": ["100", "100"], "PalletID": ["P1", "P1"], "BatchNumber": [2, 3],
    }))
    result = add_training_membership(source, path)
    assert result["V37TrainingMembership"].tolist() == [
        "SEEN_IN_FROZEN_V37_TRAINING", "UNSEEN_BY_BATCH_KEY",
    ]
