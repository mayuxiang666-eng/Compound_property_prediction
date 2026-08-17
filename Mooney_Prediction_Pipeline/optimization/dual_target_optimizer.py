"""Fail-closed Mooney + energy target optimizer contract.

The optimizer ranks already-scored complete Step vectors. It deliberately does
not synthesize process response, Mooney, or energy predictions itself. Those
must be supplied by independently validated artifacts for the same candidate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_CANDIDATE_COLUMNS = {
    "candidate_id",
    "mooney_p10",
    "mooney_p50",
    "mooney_p90",
    "energy_kwh_p10",
    "energy_kwh_p50",
    "energy_kwh_p90",
    "model_path_status",
    "strict_actual_phase_response_status",
    "safety_status",
    "thermal_budget_status",
    "execution_capability_status",
    "ood_status",
    "action_cost",
    "changed_step_count",
}


@dataclass(frozen=True)
class DualTargetRequest:
    recipe_id: str
    target_mooney: float
    target_energy_kwh: float
    mooney_spec_low: float
    mooney_spec_high: float
    mooney_target_tolerance: float = 1.0
    energy_target_tolerance_kwh: float = 2.0
    strategy: str = "QUALITY_FIRST"

    def validate(self) -> None:
        values = (
            self.target_mooney,
            self.target_energy_kwh,
            self.mooney_spec_low,
            self.mooney_spec_high,
            self.mooney_target_tolerance,
            self.energy_target_tolerance_kwh,
        )
        if not self.recipe_id.strip():
            raise ValueError("recipe_id is required")
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("Dual-target request contains non-finite values")
        if self.mooney_spec_low >= self.mooney_spec_high:
            raise ValueError("Mooney spec low must be below spec high")
        if not self.mooney_spec_low <= self.target_mooney <= self.mooney_spec_high:
            raise ValueError("Target Mooney must lie inside the engineering specification")
        if self.target_energy_kwh <= 0:
            raise ValueError("Target energy must be positive")
        if self.mooney_target_tolerance <= 0 or self.energy_target_tolerance_kwh <= 0:
            raise ValueError("Target tolerances must be positive")
        if self.strategy not in {"QUALITY_FIRST", "BALANCED_TARGET"}:
            raise ValueError("strategy must be QUALITY_FIRST or BALANCED_TARGET")


@dataclass(frozen=True)
class DualModelReadiness:
    mooney_model_status: str
    mooney_model_id: str
    stage3_enabled: bool
    process_response_status: str
    strict_actual_phase_response_status: str
    energy_model_status: str
    energy_model_id: str
    energy_route_status: str
    template_status: str
    recipe_version_status: str
    source_status: str = "RDS_ACTIVE_ONLY"

    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.mooney_model_status != "RELEASE_GATES_PASSED":
            blockers.append("BLOCKED_MOONEY_MODEL_NOT_RELEASED")
        if self.stage3_enabled:
            blockers.append("BLOCKED_STAGE3_MUST_REMAIN_DISABLED")
        if self.process_response_status != "CONTROL_TRANSFER_APPROVED":
            blockers.append("CONTROL_TRANSFER_UNAVAILABLE")
        if self.strict_actual_phase_response_status != "ACTUAL_PHASE_RESPONSE_OK":
            blockers.append("BLOCKED_ENDPOINT_TO_PHASE_SUBSTITUTION")
        if self.energy_model_status != "SHADOW_RANKING_APPROVED":
            blockers.append("BLOCKED_ENERGY_MODEL_NOT_READY")
        if self.energy_route_status != "ROUTE_GATE_PASSED":
            blockers.append("BLOCKED_ENERGY_ROUTE_NOT_READY")
        if self.template_status != "EXACT_OR_APPROVED_POOLED_TEMPLATE":
            blockers.append("BLOCKED_INSUFFICIENT_TEMPLATE_HISTORY")
        if self.recipe_version_status != "CURRENT_VERSION_BOUND":
            blockers.append("BLOCKED_PROCESS_VERSION_MISMATCH")
        if self.source_status != "RDS_ACTIVE_ONLY":
            blockers.append("BLOCKED_UNAPPROVED_DATA_SOURCE")
        return tuple(dict.fromkeys(blockers))


@dataclass(frozen=True)
class DualTargetOptimizationResult:
    status: str
    selected_candidate_id: str | None
    selected_candidate: dict | None
    audit_table: pd.DataFrame
    reason_codes: tuple[str, ...]


def _candidate_reasons(row: pd.Series, request: DualTargetRequest) -> list[str]:
    reasons = []
    expected = {
        "model_path_status": "MODEL_PATH_OK",
        "strict_actual_phase_response_status": "ACTUAL_PHASE_RESPONSE_OK",
        "safety_status": "PASSED",
        "thermal_budget_status": "PASSED",
        "execution_capability_status": "PASSED",
        "ood_status": "IN_DISTRIBUTION",
    }
    codes = {
        "model_path_status": "NO_MODEL_PATH",
        "strict_actual_phase_response_status": "CONTROL_TRANSFER_UNAVAILABLE",
        "safety_status": "REJECTED_SAFETY_GATE",
        "thermal_budget_status": "REJECTED_THERMAL_BUDGET",
        "execution_capability_status": "REJECTED_EXECUTION_CAPABILITY",
        "ood_status": "REJECTED_OUT_OF_DISTRIBUTION",
    }
    for column, expected_value in expected.items():
        if str(row[column]) != expected_value:
            reasons.append(codes[column])
    if float(row["mooney_p10"]) < request.mooney_spec_low:
        reasons.append("REJECTED_MOONEY_INTERVAL_BELOW_SPEC")
    if float(row["mooney_p90"]) > request.mooney_spec_high:
        reasons.append("REJECTED_MOONEY_INTERVAL_ABOVE_SPEC")
    if float(row["energy_kwh_p90"]) <= 0:
        reasons.append("REJECTED_NONPOSITIVE_ENERGY_PREDICTION")
    if "model_response_delta" in row.index:
        response = pd.to_numeric(pd.Series([row["model_response_delta"]]), errors="coerce").iloc[0]
        if pd.isna(response) or abs(float(response)) < 0.25:
            reasons.append("REJECTED_MINIMUM_MOONEY_RESPONSE")
    return list(dict.fromkeys(reasons))


def optimize_dual_target(
    candidates: pd.DataFrame,
    request: DualTargetRequest,
    readiness: DualModelReadiness,
) -> DualTargetOptimizationResult:
    """Rank complete Step vectors only after all model and process gates pass."""

    request.validate()
    missing = sorted(REQUIRED_CANDIDATE_COLUMNS.difference(candidates.columns))
    if missing:
        raise ValueError(f"Dual-target candidates are missing columns: {missing}")
    if candidates["candidate_id"].astype(str).duplicated().any():
        raise ValueError("candidate_id must be unique")
    model_blockers = readiness.blockers()
    audit = candidates.copy()
    audit["recipe_id"] = request.recipe_id
    audit["target_mooney"] = request.target_mooney
    audit["target_energy_kwh"] = request.target_energy_kwh
    if model_blockers:
        audit["candidate_status"] = "BLOCKED_MODEL_READINESS"
        audit["reason_codes"] = json.dumps(list(model_blockers))
        audit["dual_target_score"] = np.nan
        return DualTargetOptimizationResult(
            "BLOCKED_MODEL_READINESS", None, None, audit, model_blockers
        )

    numeric_columns = [
        "mooney_p10", "mooney_p50", "mooney_p90", "energy_kwh_p10",
        "energy_kwh_p50", "energy_kwh_p90", "action_cost", "changed_step_count",
    ]
    for column in numeric_columns:
        audit[column] = pd.to_numeric(audit[column], errors="coerce")
    if audit[numeric_columns].isna().any().any():
        raise ValueError("Dual-target candidate predictions contain missing/non-numeric values")

    reason_lists = [_candidate_reasons(row, request) for _, row in audit.iterrows()]
    audit["reason_codes"] = [json.dumps(reasons) for reasons in reason_lists]
    audit["candidate_status"] = ["FEASIBLE" if not reasons else "REJECTED" for reasons in reason_lists]
    audit["mooney_target_distance"] = (
        audit["mooney_p50"] - request.target_mooney
    ).abs()
    audit["energy_target_distance_kwh"] = (
        audit["energy_kwh_p50"] - request.target_energy_kwh
    ).abs()
    audit["mooney_interval_width"] = audit["mooney_p90"] - audit["mooney_p10"]
    audit["energy_interval_width_kwh"] = audit["energy_kwh_p90"] - audit["energy_kwh_p10"]
    normalized_mooney = audit["mooney_target_distance"] / request.mooney_target_tolerance
    normalized_energy = audit["energy_target_distance_kwh"] / request.energy_target_tolerance_kwh
    uncertainty = (
        audit["mooney_interval_width"] / (2.0 * request.mooney_target_tolerance)
        + audit["energy_interval_width_kwh"] / (2.0 * request.energy_target_tolerance_kwh)
    )
    if request.strategy == "BALANCED_TARGET":
        primary = np.maximum(normalized_mooney, normalized_energy)
        secondary = normalized_mooney + normalized_energy
    else:
        primary = normalized_mooney
        secondary = normalized_energy
    audit["dual_target_primary_score"] = primary
    audit["dual_target_secondary_score"] = secondary
    audit["dual_target_score"] = primary + 0.10 * secondary + 0.05 * uncertainty

    feasible = audit.loc[audit["candidate_status"].eq("FEASIBLE")].copy()
    if feasible.empty:
        reasons = tuple(sorted({code for items in reason_lists for code in items}))
        return DualTargetOptimizationResult(
            "NO_VALID_CANDIDATE", None, None, audit, reasons or ("NO_CANDIDATES",)
        )
    feasible = feasible.sort_values(
        [
            "dual_target_primary_score", "dual_target_secondary_score",
            "energy_kwh_p90", "action_cost", "changed_step_count", "candidate_id",
        ],
        kind="stable",
    )
    selected = feasible.iloc[0]
    audit.loc[audit["candidate_id"].eq(selected["candidate_id"]), "candidate_status"] = (
        "MANUAL_REVIEW_ONLY"
    )
    return DualTargetOptimizationResult(
        "MANUAL_REVIEW_ONLY",
        str(selected["candidate_id"]),
        selected.to_dict(),
        audit,
        (),
    )


__all__ = [
    "DualTargetRequest",
    "DualModelReadiness",
    "DualTargetOptimizationResult",
    "optimize_dual_target",
]
