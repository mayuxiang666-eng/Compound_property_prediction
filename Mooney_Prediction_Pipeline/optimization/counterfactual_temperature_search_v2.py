"""Fail-closed temperature Target search using counterfactual feature lineage."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.action_feature_lineage_v2 import (
    build_action_feature_lineage_v2,
    candidate_has_model_path_v2,
)
from Mooney_Prediction_Pipeline.optimization.counterfactual_phase_features import merge_counterfactual_features
from Mooney_Prediction_Pipeline.optimization.step_candidate_search import (
    CandidateSearchResult,
    _build_temperature_candidate,
    _prepare_steps,
    _temperature_action_options,
)
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor
from V4_2_Step_Phase_Architecture.optimization.acceptance_evaluator import AcceptanceEvaluator


def search_counterfactual_temperature_candidates_v2(
    steps: pd.DataFrame,
    baseline_feature_row: pd.Series,
    raw_predictor: V37RawPredictor,
    *,
    target_phase: str,
    expected_temp_direction: int,
    target_center: float,
    spec_lower: float,
    spec_upper: float,
    temperature_increment_c: float = 1.0,
    max_candidates: int = 100,
) -> CandidateSearchResult:
    """Search temperature Target actions and reject every disconnected action."""
    before = _prepare_steps(steps)
    required = {'is_temp_settable', 'step_temp_min', 'step_temp_max'}
    missing = required - set(before.columns)
    if missing:
        raise ValueError(f'Missing temperature contract columns: {sorted(missing)}')
    base_features = merge_counterfactual_features(before, baseline_feature_row)
    baseline_prediction = float(raw_predictor.predict_raw(base_features).final_prediction[0])
    phase_steps = before[
        before['mapped_phase'].eq(target_phase)
        & before['is_temp_settable'].fillna(False).astype(bool)
        & pd.to_numeric(before['temp_SP'], errors='coerce').fillna(0).gt(0)
    ]
    if phase_steps.empty:
        return CandidateSearchResult(baseline_prediction, None, pd.DataFrame(), 'NO_VALID_CANDIDATE', ('REJECTED_CORRECTED_PHASE_UNMAPPED',))
    evaluator = AcceptanceEvaluator()
    actions = _temperature_action_options(phase_steps, expected_temp_direction, temperature_increment_c, evaluator.single_shift_fraction)
    audits: list[dict[str, Any]] = []
    accepted: list[tuple[float, pd.DataFrame]] = []
    for row_index, delta in actions[:max_candidates]:
        candidate = _build_temperature_candidate(before, ((row_index, delta),))
        candidate_features = merge_counterfactual_features(candidate, baseline_feature_row)
        lineage = build_action_feature_lineage_v2(before, candidate, base_features, candidate_features, raw_predictor.feature_columns)
        action = lineage[0] if lineage else {}
        audit: dict[str, Any] = {
            'actions': json.dumps([{'GroupName': str(before.loc[row_index, 'GroupName']), 'StepNo': int(before.loc[row_index, 'StepNo']), 'delta_temp_c': delta}]),
            'action_parameter': 'temperature',
            'baseline_prediction': baseline_prediction,
            'lineage_json': json.dumps(lineage, sort_keys=True),
            'model_path_status': action.get('model_path_status', 'NO_MODEL_PATH'),
            'changed_v37_input_columns': json.dumps(action.get('changed_v37_input_columns', [])),
        }
        if not candidate_has_model_path_v2(lineage):
            audit.update({'recommended_prediction': None, 'model_response_delta': None, 'accepted': False, 'status': 'NO_MODEL_PATH', 'reason_codes': json.dumps(['NO_MODEL_PATH'])})
            audits.append(audit)
            continue
        recommended = float(raw_predictor.predict_raw(candidate_features).final_prediction[0])
        repeated = float(raw_predictor.predict_raw(candidate_features).final_prediction[0])
        decision = evaluator.evaluate(
            before, candidate, target_center=target_center, target_source='SPEC_CENTER',
            baseline_prediction=baseline_prediction, recommended_prediction=recommended,
            re_predicted_recommendation=repeated, reproducible=abs(recommended-repeated) < 0.1,
            spec_lower=spec_lower, spec_upper=spec_upper,
        )
        audit.update({'recommended_prediction': recommended, 'model_response_delta': recommended-baseline_prediction, 'accepted': decision.accepted, 'status': decision.status, 'reason_codes': json.dumps(decision.reason_codes)})
        audits.append(audit)
        if decision.accepted:
            accepted.append((abs(target_center-recommended), candidate))
    audit_frame = pd.DataFrame(audits)
    if not accepted:
        reasons = tuple(sorted({reason for encoded in audit_frame.get('reason_codes', pd.Series(dtype=str)) for reason in json.loads(encoded)}))
        return CandidateSearchResult(baseline_prediction, None, audit_frame, 'NO_VALID_CANDIDATE', reasons or ('NO_MODEL_PATH',))
    accepted.sort(key=lambda item: item[0])
    return CandidateSearchResult(baseline_prediction, accepted[0][1], audit_frame, 'MANUAL_REVIEW_ONLY', ())
