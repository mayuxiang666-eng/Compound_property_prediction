"""Strict actual-Phase-response entry point for Step-vector shadow search."""

from __future__ import annotations

import json
from typing import Any, Iterable

import pandas as pd

from Mooney_Prediction_Pipeline.optimization import expert_step_vector_search as _core
from Mooney_Prediction_Pipeline.optimization.expert_shadow_policy import (
    prepare_recommendation_vector,
)
from Mooney_Prediction_Pipeline.optimization.expert_step_vector_search_v2 import (
    ExpertVectorSearchResult,
)
from Mooney_Prediction_Pipeline.optimization.expert_step_vector_search_v3 import (
    search_expert_step_vectors_recipe_aware,
)
from Mooney_Prediction_Pipeline.optimization.expert_process_response import (
    flatten_step_targets,
)
from Mooney_Prediction_Pipeline.optimization.recipe_scp_logic import StepTerminationLogic
from Mooney_Prediction_Pipeline.optimization.strict_actual_phase_response_contract import (
    ActualPhaseResponseContractError,
    audit_actual_phase_response,
    overlay_actual_v37_phase_features,
)


def _baseline_frame(baseline_feature_row: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(baseline_feature_row, pd.Series):
        return pd.DataFrame([baseline_feature_row])
    if isinstance(baseline_feature_row, pd.DataFrame) and len(baseline_feature_row) == 1:
        return baseline_feature_row.copy()
    raise ValueError("Exactly one V3.7 baseline feature row is required")


def search_expert_step_vectors_strict_actual_phase(
    steps: pd.DataFrame,
    baseline_feature_row: pd.Series | pd.DataFrame,
    raw_predictor: Any,
    response_model: Any,
    *,
    required_actual_phase_features: Iterable[str],
    material_system: str,
    process_target_temperature_cap: float | None,
    endpoint_bands: dict[str, Any],
    spec_center: float,
    spec_low: float,
    spec_high: float,
    scp_contracts: tuple[StepTerminationLogic, ...] = (),
    beam_width: int = 100,
    max_candidates: int = 5000,
) -> ExpertVectorSearchResult:
    """Run search only when Target actions predict actual V3.7 Phase inputs.

    Endpoint temperatures remain safety outputs. They cannot substitute for
    Phase means, durations, integrals, torque, power, or other model features.
    The response artifact must explicitly expose ``v37_actual__<feature>__p50``
    for every required counterfactual V3.7 feature.
    """

    required = tuple(dict.fromkeys(str(feature) for feature in required_actual_phase_features))
    if not required:
        return ExpertVectorSearchResult(
            None,
            None,
            pd.DataFrame([
                {
                    "status": "DIAGNOSTIC_ONLY",
                    "control_transfer_status": "CONTROL_TRANSFER_UNAVAILABLE",
                    "strict_actual_phase_response_status": "BLOCKED_EMPTY_REQUIRED_FEATURE_SET",
                    "reason_codes": json.dumps(["CONTROL_TRANSFER_UNAVAILABLE"]),
                }
            ]),
            "DIAGNOSTIC_ONLY",
            ("CONTROL_TRANSFER_UNAVAILABLE",),
        )
    if not getattr(response_model, "approved", False):
        return ExpertVectorSearchResult(
            None, None, pd.DataFrame(), "DIAGNOSTIC_ONLY", ("CONTROL_TRANSFER_UNAVAILABLE",)
        )

    baseline = prepare_recommendation_vector(steps)
    try:
        baseline_response = response_model.predict(flatten_step_targets(baseline))
        preflight = audit_actual_phase_response(baseline_response, required)
    except (ValueError, RuntimeError, ActualPhaseResponseContractError) as exc:
        preflight = None
        detail = str(exc)
    else:
        detail = ""
    if preflight is None or preflight.status != "ACTUAL_PHASE_RESPONSE_OK":
        if preflight is not None:
            detail = (
                "missing=" + "|".join(preflight.missing_response_columns)
                + ";nonfinite=" + "|".join(preflight.nonfinite_response_columns)
            )
        audit = pd.DataFrame([
            {
                "status": "DIAGNOSTIC_ONLY",
                "control_transfer_status": "CONTROL_TRANSFER_UNAVAILABLE",
                "strict_actual_phase_response_status": "BLOCKED_MISSING_ACTUAL_PHASE_RESPONSE",
                "required_actual_phase_features": "|".join(required),
                "strict_contract_detail": detail,
                "reason_codes": json.dumps(["CONTROL_TRANSFER_UNAVAILABLE"]),
            }
        ])
        return ExpertVectorSearchResult(
            None, None, audit, "DIAGNOSTIC_ONLY", ("CONTROL_TRANSFER_UNAVAILABLE",)
        )

    baseline_frame = _baseline_frame(baseline_feature_row)
    original_adapter = _core._v37_process_features

    def strict_adapter(vector, ignored_baseline, process_prediction):
        del vector, ignored_baseline
        return overlay_actual_v37_phase_features(
            baseline_frame, process_prediction, required
        )

    _core._v37_process_features = strict_adapter
    try:
        if scp_contracts:
            result = search_expert_step_vectors_recipe_aware(
                steps,
                baseline_frame.iloc[0],
                raw_predictor,
                response_model,
                material_system=material_system,
                process_target_temperature_cap=process_target_temperature_cap,
                endpoint_bands=endpoint_bands,
                spec_center=spec_center,
                spec_low=spec_low,
                spec_high=spec_high,
                scp_contracts=scp_contracts,
                beam_width=beam_width,
                max_candidates=max_candidates,
            )
        else:
            result = _core.search_expert_step_vectors(
                steps,
                baseline_frame.iloc[0],
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
    except ActualPhaseResponseContractError as exc:
        audit = pd.DataFrame([
            {
                "status": "DIAGNOSTIC_ONLY",
                "control_transfer_status": "CONTROL_TRANSFER_UNAVAILABLE",
                "strict_actual_phase_response_status": "BLOCKED_DURING_CANDIDATE_SCORING",
                "strict_contract_detail": str(exc),
                "reason_codes": json.dumps(["CONTROL_TRANSFER_UNAVAILABLE"]),
            }
        ])
        return ExpertVectorSearchResult(
            None, None, audit, "DIAGNOSTIC_ONLY", ("CONTROL_TRANSFER_UNAVAILABLE",)
        )
    finally:
        _core._v37_process_features = original_adapter

    audit = result.audit_table.copy()
    if not audit.empty:
        audit["strict_actual_phase_response_status"] = "ACTUAL_PHASE_RESPONSE_OK"
        audit["required_actual_phase_features"] = "|".join(required)
        audit["endpoint_substitution_allowed"] = False
    return ExpertVectorSearchResult(
        result.baseline_prediction,
        result.recommended_steps,
        audit,
        result.status,
        result.reason_codes,
    )


__all__ = ["search_expert_step_vectors_strict_actual_phase"]
