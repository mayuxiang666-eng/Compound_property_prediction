"""Engineering tolerance contract for corrected Phase/MMS parity."""

from __future__ import annotations

import pandas as pd

from . import feature_parity_gate as base
from .feature_parity_gate import FeatureTolerance


TOLERANCE_VERSION = "PHASE-PARITY-TOLERANCE-V2"


def tolerance_for_feature(name: str) -> FeatureTolerance:
    lower = name.lower()
    if "duration" in lower or lower.startswith("idx_") or lower.startswith("time_"):
        return FeatureTolerance(1.0, 0.0)
    if "temp" in lower or "temperature" in lower:
        return FeatureTolerance(0.5, 0.003)
    if "rotorspeed" in lower or "rpm" in lower:
        return FeatureTolerance(0.5, 0.01)
    if lower.startswith("env_"):
        return FeatureTolerance(0.1, 0.002)
    if (
        lower.startswith("weight_pct_")
        or lower.endswith("_phr")
        or lower.startswith("supplier_")
        or lower.startswith("ratio_")
        or lower.startswith("is_")
        or lower.startswith("has_")
    ):
        return FeatureTolerance(1e-6, 1e-6)
    if lower.startswith(("stage", "phys_", "cb_", "pid_", "setting_")):
        return FeatureTolerance(0.1, 0.03)
    return FeatureTolerance(1e-6, 1e-6)


def evaluate_feature_parity_v2(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    feature_columns: list[str],
    *,
    prediction_tolerance: float = 0.1,
):
    original = base.tolerance_for_feature
    base.tolerance_for_feature = tolerance_for_feature
    try:
        summary, detail = base.evaluate_feature_parity(
            reference,
            candidate,
            feature_columns,
            prediction_tolerance=prediction_tolerance,
        )
    finally:
        base.tolerance_for_feature = original
    summary["ToleranceVersion"] = TOLERANCE_VERSION
    detail["ToleranceVersion"] = TOLERANCE_VERSION
    return summary, detail
