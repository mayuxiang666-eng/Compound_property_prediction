"""Public fail-closed contracts for probabilistic M1 reduction requests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ReductionTargetRequest:
    recipe_id: str
    desired_reduction_mu: float
    order_id: str | None = None
    confidence_floor: float = 0.70
    strategy: str = "QUALITY_FIRST"
    recommendation_mode: str = "MANUAL_REVIEW_ONLY"

    def validate(self) -> None:
        if not str(self.recipe_id).strip():
            raise ValueError("recipe_id is required")
        if not np.isfinite(float(self.desired_reduction_mu)) or self.desired_reduction_mu <= 0:
            raise ValueError("desired_reduction_mu must be finite and positive")
        if not np.isfinite(float(self.confidence_floor)) or not 0.5 <= self.confidence_floor < 1.0:
            raise ValueError("confidence_floor must be in [0.5, 1.0)")
        if self.strategy != "QUALITY_FIRST":
            raise ValueError("Only QUALITY_FIRST is permitted while energy is audit-only")
        if self.recommendation_mode != "MANUAL_REVIEW_ONLY":
            raise ValueError("Automatic control is disabled")


@dataclass(frozen=True)
class ReductionRecommendation:
    status: str
    current_level_diagnostic: float | None = None
    reduction_p10: float | None = None
    reduction_p50: float | None = None
    reduction_p90: float | None = None
    probability_meet_target: float | None = None
    max_supported_reduction: float | None = None
    distance_to_optimal_before: float | None = None
    distance_to_optimal_after: float | None = None
    current_step_target_vector: tuple[dict, ...] = field(default_factory=tuple)
    recommended_step_target_vector: tuple[dict, ...] = field(default_factory=tuple)
    predicted_actual_phase_outputs: Mapping[str, float] = field(default_factory=dict)
    model_path_status: str = "NO_MODEL_PATH"
    control_transfer_status: str = "CONTROL_TRANSFER_UNAVAILABLE"
    causal_support_status: str = "NOT_ASSESSED"
    confounding_status: str = "NOT_ASSESSED"
    expert_gate_statuses: Mapping[str, str] = field(default_factory=dict)
    recommendation_status: str = "NOT_GENERATED"
    rejection_codes: tuple[str, ...] = field(default_factory=tuple)
    energy_status: str = "NOT_RANKED"
    stage3_calibration_enabled: bool = False
    automatic_control: str = "DISABLED"

    def audit_record(self) -> dict:
        return asdict(self)


def evaluate_dual_pilot_gate(pilot_statuses: Mapping[str, str]) -> dict[str, object]:
    """Require both named M1 pilots before any human micro-trial."""

    required = ("T15760", "B00458")
    missing = [pilot for pilot in required if pilot not in pilot_statuses]
    failed = [pilot for pilot in required if pilot_statuses.get(pilot) != "B6_GATE_PASSED"]
    passed = not missing and not failed
    return {
        "dual_pilot_gate_passed": passed,
        "micro_trial_status": "HUMAN_MICRO_TRIAL_ELIGIBLE" if passed else "BLOCKED_DUAL_PILOT_GATE",
        "missing_pilots": missing,
        "failed_pilots": failed,
    }
