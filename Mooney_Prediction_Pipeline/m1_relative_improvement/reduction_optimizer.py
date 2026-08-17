"""Fail-closed B5 probabilistic reduction candidate selector."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import ReductionRecommendation, ReductionTargetRequest


REQUIRED_COLUMNS = {
    "candidate_id",
    "reduction_p10",
    "reduction_p50",
    "reduction_p90",
    "probability_meet_target",
    "max_supported_reduction",
    "model_response_delta",
    "distance_to_optimal_before",
    "distance_to_optimal_after",
    "action_cost",
    "changed_step_count",
    "model_path_status",
    "control_transfer_status",
    "causal_support_status",
    "confounding_status",
    "safety_status",
    "thermal_budget_status",
    "execution_capability_status",
    "ood_status",
}


@dataclass(frozen=True)
class RelativeModelReadiness:
    direction_magnitude_status: str
    process_response_status: str
    exact_template_status: str
    recipe_version_status: str
    stage3_enabled: bool = False
    source_status: str = "RDS_ACTIVE_ONLY"

    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.direction_magnitude_status != "B2_GATE_PASSED":
            blockers.append("BLOCKED_B2_DIRECTION_MAGNITUDE_GATE")
        if self.process_response_status != "CONTROL_TRANSFER_AVAILABLE":
            blockers.append("CONTROL_TRANSFER_UNAVAILABLE")
        if self.exact_template_status != "EXACT_OR_APPROVED_POOLED_TEMPLATE":
            blockers.append("BLOCKED_INSUFFICIENT_TEMPLATE_HISTORY")
        if self.recipe_version_status != "CURRENT_VERSION_BOUND":
            blockers.append("BLOCKED_PROCESS_VERSION_MISMATCH")
        if self.stage3_enabled:
            blockers.append("BLOCKED_STAGE3_MUST_REMAIN_DISABLED")
        if self.source_status != "RDS_ACTIVE_ONLY":
            blockers.append("BLOCKED_UNAPPROVED_DATA_SOURCE")
        return tuple(dict.fromkeys(blockers))


@dataclass
class ReductionOptimizationResult:
    recommendation: ReductionRecommendation
    audit_table: pd.DataFrame


def _candidate_reasons(row: pd.Series, request: ReductionTargetRequest) -> list[str]:
    expected = {
        "model_path_status": ("MODEL_PATH_OK", "NO_MODEL_PATH"),
        "control_transfer_status": ("CONTROL_TRANSFER_AVAILABLE", "CONTROL_TRANSFER_UNAVAILABLE"),
        "causal_support_status": ("CAUSAL_ACTION_SUPPORTED", "BLOCKED_NO_CAUSAL_ACTION_SUPPORT"),
        "confounding_status": ("NO_DOMINANT_MEASURED_CONFOUNDING", "CONFOUNDED_BY_MATERIAL"),
        "safety_status": ("PASSED", "REJECTED_SAFETY_GATE"),
        "thermal_budget_status": ("PASSED", "REJECTED_THERMAL_BUDGET"),
        "execution_capability_status": ("PASSED", "REJECTED_EXECUTION_CAPABILITY"),
        "ood_status": ("IN_DISTRIBUTION", "REJECTED_OUT_OF_DISTRIBUTION"),
    }
    reasons = [rejected for column, (allowed, rejected) in expected.items() if str(row[column]) != allowed]
    if float(row["reduction_p10"]) <= 0:
        reasons.append("REJECTED_NONPOSITIVE_REDUCTION_P10")
    if float(row["probability_meet_target"]) < request.confidence_floor:
        reasons.append("REJECTED_REDUCTION_PROBABILITY")
    if float(row["max_supported_reduction"]) < request.desired_reduction_mu:
        reasons.append("REJECTED_REQUEST_OUTSIDE_SUPPORTED_REDUCTION")
    if abs(float(row["model_response_delta"])) < 0.25:
        reasons.append("REJECTED_MINIMUM_MOONEY_RESPONSE")
    return list(dict.fromkeys(reasons))


def optimize_reduction_candidates(
    candidates: pd.DataFrame,
    request: ReductionTargetRequest,
    readiness: RelativeModelReadiness,
) -> ReductionOptimizationResult:
    request.validate()
    missing = sorted(REQUIRED_COLUMNS.difference(candidates.columns))
    if missing:
        raise ValueError(f"Reduction candidates missing columns: {missing}")
    if candidates["candidate_id"].astype(str).duplicated().any():
        raise ValueError("candidate_id must be unique")
    audit = candidates.copy()
    blockers = readiness.blockers()
    if blockers:
        audit["candidate_status"] = "BLOCKED_MODEL_READINESS"
        audit["reason_codes"] = json.dumps(list(blockers))
        return ReductionOptimizationResult(
            ReductionRecommendation(
                status="BLOCKED_MODEL_READINESS",
                recommendation_status="NOT_GENERATED",
                rejection_codes=blockers,
            ),
            audit,
        )
    numeric = [
        "reduction_p10", "reduction_p50", "reduction_p90", "probability_meet_target",
        "max_supported_reduction", "model_response_delta", "distance_to_optimal_before",
        "distance_to_optimal_after", "action_cost", "changed_step_count",
    ]
    audit[numeric] = audit[numeric].apply(pd.to_numeric, errors="coerce")
    if audit[numeric].isna().any().any() or not np.isfinite(audit[numeric]).all().all():
        raise ValueError("Reduction candidate numeric fields must be finite")
    reasons = [_candidate_reasons(row, request) for _, row in audit.iterrows()]
    audit["reason_codes"] = [json.dumps(value) for value in reasons]
    audit["candidate_status"] = ["FEASIBLE" if not value else "REJECTED" for value in reasons]
    feasible = audit[audit["candidate_status"].eq("FEASIBLE")].sort_values(
        ["distance_to_optimal_after", "action_cost", "changed_step_count", "candidate_id"], kind="stable"
    )
    if feasible.empty:
        codes = tuple(sorted({code for values in reasons for code in values})) or ("NO_CANDIDATES",)
        return ReductionOptimizationResult(
            ReductionRecommendation(status="NO_VALID_CANDIDATE", rejection_codes=codes), audit
        )
    selected = feasible.iloc[0]
    audit.loc[audit["candidate_id"].eq(selected["candidate_id"]), "candidate_status"] = "MANUAL_REVIEW_ONLY"
    recommendation = ReductionRecommendation(
        status="MANUAL_REVIEW_ONLY",
        reduction_p10=float(selected["reduction_p10"]),
        reduction_p50=float(selected["reduction_p50"]),
        reduction_p90=float(selected["reduction_p90"]),
        probability_meet_target=float(selected["probability_meet_target"]),
        max_supported_reduction=float(selected["max_supported_reduction"]),
        distance_to_optimal_before=float(selected["distance_to_optimal_before"]),
        distance_to_optimal_after=float(selected["distance_to_optimal_after"]),
        model_path_status="MODEL_PATH_OK",
        control_transfer_status="CONTROL_TRANSFER_AVAILABLE",
        causal_support_status="CAUSAL_ACTION_SUPPORTED",
        confounding_status="NO_DOMINANT_MEASURED_CONFOUNDING",
        recommendation_status="MANUAL_REVIEW_ONLY",
    )
    return ReductionOptimizationResult(recommendation, audit)


__all__ = ["RelativeModelReadiness", "ReductionOptimizationResult", "optimize_reduction_candidates"]
