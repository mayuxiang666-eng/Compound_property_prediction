"""Recipe-aware wrapper around the canonical V2 Step-vector shadow search."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from Mooney_Prediction_Pipeline.optimization import expert_step_vector_search as _core
from Mooney_Prediction_Pipeline.optimization.expert_shadow_policy_v2 import policy_reasons as base_policy_reasons
from Mooney_Prediction_Pipeline.optimization.expert_step_vector_search_v2 import ExpertVectorSearchResult
from Mooney_Prediction_Pipeline.optimization.recipe_scp_logic import (
    StepTerminationLogic,
    validate_step_vector_scp_logic,
)


def search_expert_step_vectors_recipe_aware(
    steps: pd.DataFrame,
    baseline_feature_row: pd.Series,
    raw_predictor: Any,
    response_model: Any,
    *,
    material_system: str,
    process_target_temperature_cap: float | None,
    endpoint_bands: dict[str, Any],
    spec_center: float,
    spec_low: float,
    spec_high: float,
    scp_contracts: tuple[StepTerminationLogic, ...],
    beam_width: int = 100,
    max_candidates: int = 5000,
) -> ExpertVectorSearchResult:
    """Run V2 search only when the recipe's termination contract is known."""
    baseline = _core.prepare_recommendation_vector(steps)
    baseline_reasons = validate_step_vector_scp_logic(baseline, scp_contracts)
    if baseline_reasons:
        audit = pd.DataFrame(
            [
                {
                    "status": "DIAGNOSTIC_ONLY",
                    "control_transfer_status": "BLOCKED_SCP_LOGIC",
                    "scp_logic_status": "REJECTED",
                    "reason_codes": json.dumps(list(baseline_reasons)),
                }
            ]
        )
        return ExpertVectorSearchResult(None, None, audit, "DIAGNOSTIC_ONLY", baseline_reasons)

    original_policy = _core.policy_reasons

    def recipe_policy(
        before: pd.DataFrame,
        candidate: pd.DataFrame,
        *,
        material_system: str,
        process_target_temperature_cap: float | None,
    ) -> tuple[str, ...]:
        reasons = list(
            base_policy_reasons(
                before,
                candidate,
                material_system=material_system,
                process_target_temperature_cap=process_target_temperature_cap,
            )
        )
        reasons.extend(validate_step_vector_scp_logic(candidate, scp_contracts))
        return tuple(sorted(set(reasons)))

    _core.policy_reasons = recipe_policy
    try:
        result = _core.search_expert_step_vectors(
            steps,
            baseline_feature_row,
            raw_predictor,
            response_model,
            material_system=material_system,
            process_target_temperature_cap=process_target_temperature_cap,
            endpoint_bands=endpoint_bands,
            spec_center=spec_center,
            spec_low=spec_low,
            spec_high=spec_high,
            beam_width=beam_width,
            max_candidates=max_candidates,
        )
    finally:
        _core.policy_reasons = original_policy

    audit = result.audit_table.copy()
    if not audit.empty:
        relation_summary = "|".join(
            f"{item.group_name}:{item.step_no}:{item.relation}" for item in scp_contracts
        )
        audit["scp_logic_status"] = "PASSED"
        audit["scp_logic_contracts"] = relation_summary
    return ExpertVectorSearchResult(
        result.baseline_prediction,
        result.recommended_steps,
        audit,
        result.status,
        result.reason_codes,
    )
