"""Source-weighted extension of the deterministic group-atomic batch model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .v371_group_atomic_batch_v1 import V371GroupAtomicBatchModel


class V371SourceWeightedGroupAtomicBatchModel(V371GroupAtomicBatchModel):
    """Give every Lab group one vote times an explicit source reliability weight."""

    contract_version = "V3.7.1-R1B-SOURCE-WEIGHTED-GROUP-ATOMIC-V1"
    weight_column = "_group_training_weight"

    def _prepare_batch_frame(self, source: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if self.weight_column not in source.columns:
            raise ValueError(f"Missing source reliability column: {self.weight_column}")
        numeric = pd.to_numeric(source[self.weight_column], errors="coerce")
        if numeric.isna().any() or numeric.le(0).any() or numeric.gt(1).any():
            raise ValueError("GROUP_TRAINING_WEIGHT_MUST_BE_IN_0_TO_1")
        conflicts = source.assign(_weight=numeric).groupby("LabelGroupID")["_weight"].nunique()
        if conflicts.gt(1).any():
            raise ValueError("CONFLICTING_GROUP_TRAINING_WEIGHT_WITHIN_LAB_GROUP")

        frame = super()._prepare_batch_frame(source, target_col)
        reliability = pd.to_numeric(frame[self.weight_column], errors="raise")
        for column in ("_w_label_raw", "_w_loss", "_w_metric", "_sample_weight"):
            frame[column] = frame[column] * reliability
        effective = frame.groupby("LabelGroupID")["_w_loss"].sum()
        expected = frame.groupby("LabelGroupID")[self.weight_column].first()
        if not np.allclose(effective.to_numpy(), expected.to_numpy()):
            raise RuntimeError("SOURCE_WEIGHTED_GROUP_EFFECTIVE_WEIGHT_MISMATCH")
        self.source_weight_audit_ = {
            "groups": int(len(effective)),
            "effective_weight_sum": float(effective.sum()),
            "effective_weight_min": float(effective.min()),
            "effective_weight_max": float(effective.max()),
            "unit_weight_groups": int(np.isclose(effective, 1.0).sum()),
            "downweighted_groups": int((effective < 1.0 - 1e-12).sum()),
        }
        return frame

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self.crossfit_audit_["source_weight_contract"] = self.source_weight_audit_
        self.crossfit_audit_["each_group_effective_weight_one"] = False
        self.crossfit_audit_["group_effective_weight_policy"] = (
            "ONE_VOTE_TIMES_EXPLICIT_SOURCE_RELIABILITY_WEIGHT"
        )
        return self


__all__ = ["V371SourceWeightedGroupAtomicBatchModel"]
