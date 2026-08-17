"""Fail-closed raw V3.7 prediction adapter for Phase-to-Step recommendations."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


LAB_ONLY_COLUMNS = {"actual_mny", "current_batch_mny"}
REQUIRED_CONTEXT_COLUMNS = {"CompoundName", "material_system"}


class RawPredictionContractError(ValueError):
    """Raised when an input cannot safely enter raw recommendation prediction."""


@dataclass(frozen=True)
class RawPredictionResult:
    """Uncalibrated Stage 1/1b/2 prediction from one versioned V3.7 artifact."""

    final_prediction: np.ndarray
    stage1_prediction: np.ndarray
    stage1b_bias: np.ndarray
    stage2_residual: np.ndarray
    model_version: str
    artifact_path: str


class V37RawPredictor:
    """Loads one fitted V3.7 artifact and exposes only Stage 1/1b/2 prediction.

    Stage 3 online calibration is intentionally excluded so baseline and recommended
    arms are evaluated by the same deterministic raw model.
    """

    def __init__(self, package_dir: str | Path):
        self.package_dir = Path(package_dir)
        self.model_path = self.package_dir / "hybrid_model.joblib"
        self.metadata_path = self.package_dir / "feature_metadata.json"
        self.model = None
        self.metadata: dict[str, object] = {}

    def load(self) -> "V37RawPredictor":
        """Load and validate the model package; never construct a fallback model."""
        missing_files = [
            str(path)
            for path in (self.model_path, self.metadata_path)
            if not path.is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                "V3.7 raw prediction is unavailable because required package files "
                f"are missing: {', '.join(missing_files)}"
            )

        with self.metadata_path.open("r", encoding="utf-8") as metadata_file:
            self.metadata = json.load(metadata_file)

        s1_features = self.metadata.get("s1_features")
        s2_features = self.metadata.get("s2_features")
        if not isinstance(s1_features, list) or not isinstance(s2_features, list):
            raise RawPredictionContractError(
                "feature_metadata.json must define list-valued s1_features and s2_features."
            )

        # The production artifact was serialized with Mooney_Prediction_Pipeline as
        # the import root, so pickle resolves its classes as ``model_training.*``.
        legacy_import_root = str(self.package_dir.parents[1])
        if legacy_import_root not in sys.path:
            sys.path.insert(0, legacy_import_root)
        self.model = joblib.load(self.model_path)
        required_attributes = ("stage1_model_", "stage1b_bias_", "stage2_experts_")
        missing_attributes = [
            attribute
            for attribute in required_attributes
            if not hasattr(self.model, attribute)
        ]
        if missing_attributes or self.model.stage1_model_ is None or self.model.stage1b_bias_ is None:
            raise RawPredictionContractError(
                "The V3.7 artifact is not a fitted HybridUnifiedMooneyModel: "
                f"missing or unfitted {missing_attributes or required_attributes}."
            )
        if not self.model.stage2_experts_:
            raise RawPredictionContractError("The V3.7 artifact has no fitted Stage 2 experts.")
        return self

    @property
    def feature_columns(self) -> list[str]:
        if not self.metadata:
            raise RawPredictionContractError("Call load() before accessing the feature contract.")
        return list(dict.fromkeys(self.metadata["s1_features"] + self.metadata["s2_features"]))

    def _validate_input(self, phase_features: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RawPredictionContractError("Call load() before raw prediction.")
        if phase_features.empty:
            raise RawPredictionContractError("Raw prediction requires at least one candidate row.")

        normalized_columns = {str(column).strip().lower() for column in phase_features.columns}
        leaked_columns = sorted(LAB_ONLY_COLUMNS & normalized_columns)
        if leaked_columns:
            raise RawPredictionContractError(
                "Lab-only fields are forbidden in raw prediction input: "
                f"{', '.join(leaked_columns)}"
            )

        missing_context = sorted(REQUIRED_CONTEXT_COLUMNS - set(phase_features.columns))
        missing_features = sorted(set(self.feature_columns) - set(phase_features.columns))
        if missing_context or missing_features:
            details = []
            if missing_context:
                details.append(f"context={', '.join(missing_context)}")
            if missing_features:
                details.append(f"features={', '.join(missing_features)}")
            raise RawPredictionContractError(
                "Raw prediction input does not satisfy the versioned V3.7 schema: "
                + "; ".join(details)
            )

        validated = phase_features.copy()
        for column in self.feature_columns:
            values = pd.to_numeric(validated[column], errors="coerce")
            if values.isna().any():
                raise RawPredictionContractError(
                    f"Raw prediction feature {column!r} contains missing or non-numeric values."
                )
            validated[column] = values
        return validated

    def predict_raw(self, phase_features: pd.DataFrame) -> RawPredictionResult:
        """Predict with the fitted Stage 1/1b/2 model without calibration or fallback."""
        validated = self._validate_input(phase_features)
        final_prediction, stage1_prediction, stage1b_bias, stage2_residual = self.model.predict(
            validated,
            cluster_col="material_system",
        )
        return RawPredictionResult(
            final_prediction=np.asarray(final_prediction, dtype=float),
            stage1_prediction=np.asarray(stage1_prediction, dtype=float),
            stage1b_bias=np.asarray(stage1b_bias, dtype=float),
            stage2_residual=np.asarray(stage2_residual, dtype=float),
            model_version=str(self.metadata.get("model_version", "UNKNOWN")),
            artifact_path=str(self.model_path),
        )