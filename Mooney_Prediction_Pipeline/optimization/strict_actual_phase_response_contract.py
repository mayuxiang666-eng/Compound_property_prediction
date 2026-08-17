"""Fail-closed contract between process response and V3.7 Phase inputs.

Endpoint temperatures are safety labels.  They are not interchangeable with
Phase means, integrals, or other model features.  A formal counterfactual must
name and predict every actual V3.7 feature it intends to overwrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


RESPONSE_PREFIX = "v37_actual__"


class ActualPhaseResponseContractError(ValueError):
    """The response artifact cannot safely feed V3.7."""


@dataclass(frozen=True)
class ActualPhaseResponseAudit:
    status: str
    required_features: tuple[str, ...]
    missing_response_columns: tuple[str, ...]
    nonfinite_response_columns: tuple[str, ...]


def response_column(feature: str, statistic: str = "p50") -> str:
    return f"{RESPONSE_PREFIX}{feature}__{statistic}"


def audit_actual_phase_response(
    prediction: pd.DataFrame,
    required_features: Iterable[str],
) -> ActualPhaseResponseAudit:
    if len(prediction) != 1:
        raise ActualPhaseResponseContractError("Exactly one process-response row is required")
    required = tuple(dict.fromkeys(str(feature) for feature in required_features))
    missing = tuple(feature for feature in required if response_column(feature) not in prediction.columns)
    nonfinite = []
    for feature in required:
        column = response_column(feature)
        if column not in prediction:
            continue
        value = pd.to_numeric(pd.Series([prediction.iloc[0][column]]), errors="coerce").iloc[0]
        if pd.isna(value) or not np.isfinite(float(value)):
            nonfinite.append(feature)
    status = "ACTUAL_PHASE_RESPONSE_OK" if not missing and not nonfinite else "CONTROL_TRANSFER_UNAVAILABLE"
    return ActualPhaseResponseAudit(status, required, missing, tuple(nonfinite))


def overlay_actual_v37_phase_features(
    baseline_features: pd.DataFrame,
    prediction: pd.DataFrame,
    required_features: Iterable[str],
) -> pd.DataFrame:
    """Overlay only explicitly predicted actual Phase features onto one baseline row."""
    if len(baseline_features) != 1:
        raise ActualPhaseResponseContractError("Exactly one V3.7 baseline row is required")
    audit = audit_actual_phase_response(prediction, required_features)
    if audit.status != "ACTUAL_PHASE_RESPONSE_OK":
        details = []
        if audit.missing_response_columns:
            details.append("missing=" + "|".join(audit.missing_response_columns))
        if audit.nonfinite_response_columns:
            details.append("nonfinite=" + "|".join(audit.nonfinite_response_columns))
        raise ActualPhaseResponseContractError(
            "CONTROL_TRANSFER_UNAVAILABLE_MISSING_ACTUAL_PHASE_FEATURE_RESPONSE: " + ";".join(details)
        )
    out = baseline_features.copy()
    for feature in audit.required_features:
        if feature not in out.columns:
            raise ActualPhaseResponseContractError(f"V3.7 baseline is missing required feature: {feature}")
        out.at[out.index[0], feature] = float(prediction.iloc[0][response_column(feature)])
    return out


def reject_endpoint_substitution(prediction: pd.DataFrame, required_features: Iterable[str]) -> None:
    """Explicitly reject the legacy endpoint-to-Phase-mean shortcut."""
    audit = audit_actual_phase_response(prediction, required_features)
    if audit.status != "ACTUAL_PHASE_RESPONSE_OK":
        raise ActualPhaseResponseContractError(
            "Endpoint outputs cannot substitute for V3.7 actual Phase feature responses"
        )
