"""Transparent, fail-closed multi-Step duration search for V3.7 Mooney."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor
from V4_2_Step_Phase_Architecture.feature_engineering.aggregate_sp_to_phase import aggregate_steps_to_phase_features
from V4_2_Step_Phase_Architecture.optimization.acceptance_evaluator import AcceptanceEvaluator


GROUP_KEYS = ['OrderID', 'BatchNumber', 'GroupName']
STEP_KEYS = GROUP_KEYS + ['StepNo']
REQUIRED_COLUMNS = {
    *STEP_KEYS, 'mapped_phase', 'time_SP', 'rotor_SP', 'ram_SP', 'temp_SP',
    'is_time_settable', 'step_time_min', 'step_time_max',
}


@dataclass(frozen=True)
class CandidateSearchResult:
    """The selected complete Step vector and every candidate decision."""

    baseline_prediction: float
    recommended_steps: pd.DataFrame | None
    audit_table: pd.DataFrame
    status: str
    reason_codes: tuple[str, ...]


def _prepare_steps(steps: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(steps.columns)
    if missing:
        raise ValueError(f'Missing Step search columns: {sorted(missing)}')
    if {'actual_mny', 'current_batch_mny'} & {str(column).lower() for column in steps.columns}:
        raise ValueError('Lab-only fields cannot enter candidate search.')
    if steps.duplicated(STEP_KEYS).any():
        raise ValueError('Duplicate composite Step identity is forbidden.')

    out = steps.copy().sort_values(STEP_KEYS, kind='stable')
    out['dur_SP'] = pd.to_numeric(out['time_SP'], errors='coerce') - out.groupby(GROUP_KEYS)['time_SP'].shift(
        1,
        fill_value=0.0,
    )
    if out['dur_SP'].isna().any() or (out['dur_SP'] < 0).any():
        raise ValueError('time_SP must be numeric and non-decreasing within each mixer group.')
    return out


def _build_candidate(before_steps: pd.DataFrame, actions: tuple[tuple[int, float], ...]) -> pd.DataFrame:
    candidate = before_steps.copy()
    candidate['recommended_dur_SP'] = candidate['dur_SP']
    for row_index, delta in actions:
        candidate.loc[row_index, 'recommended_dur_SP'] += delta
    candidate['recommended_time_SP'] = candidate.groupby(GROUP_KEYS)['recommended_dur_SP'].cumsum()
    candidate['delta_time_SP'] = candidate['recommended_time_SP'] - candidate['time_SP']
    for parameter in ('rotor', 'ram', 'temp'):
        candidate[f'recommended_{parameter}_SP'] = candidate[f'{parameter}_SP']
    return candidate


def _build_temperature_candidate(before_steps: pd.DataFrame, actions: tuple[tuple[int, float], ...]) -> pd.DataFrame:
    candidate = before_steps.copy()
    candidate['recommended_dur_SP'] = candidate['dur_SP']
    candidate['recommended_time_SP'] = candidate['time_SP']
    candidate['delta_time_SP'] = 0.0
    candidate['recommended_rotor_SP'] = candidate['rotor_SP']
    candidate['recommended_ram_SP'] = candidate['ram_SP']
    candidate['recommended_temp_SP'] = candidate['temp_SP']
    for row_index, delta in actions:
        candidate.loc[row_index, 'recommended_temp_SP'] = float(candidate.loc[row_index, 'temp_SP']) + delta
    candidate['delta_temp_SP'] = candidate['recommended_temp_SP'] - candidate['temp_SP']
    return candidate


def _phase_features(steps: pd.DataFrame, baseline_feature_row: pd.Series) -> pd.DataFrame:
    if {'actual_mny', 'current_batch_mny'} & {str(column).lower() for column in baseline_feature_row.index}:
        raise ValueError('Lab-only fields cannot enter the V3.7 baseline feature row.')
    aggregated = aggregate_steps_to_phase_features(steps)
    if len(aggregated) != 1:
        raise ValueError('Candidate search accepts exactly one batch at a time.')
    features = baseline_feature_row.to_frame().T.copy()
    for column, value in aggregated.iloc[0].items():
        features.at[features.index[0], column] = value
    for column, value in baseline_feature_row.items():
        if column not in features.columns:
            features[column] = value
    return features


def _feature_delta(before_features: pd.DataFrame, after_features: pd.DataFrame) -> dict[str, float]:
    tracked_columns = [
        'Stage2_DryMixing_Duration',
        'Stage3_OilLoading_Duration',
        'Stage4_WetMixing_Duration',
        'Stage5_PID_Duration',
        'Stage6_BottomMixing_Duration',
        'Target_Temperature',
    ]
    deltas: dict[str, float] = {}
    for column in tracked_columns:
        if column in before_features.columns and column in after_features.columns:
            before_value = float(pd.to_numeric(before_features.iloc[0][column], errors='coerce'))
            after_value = float(pd.to_numeric(after_features.iloc[0][column], errors='coerce'))
            deltas[column] = after_value - before_value
    return deltas


def _fingerprint(candidate: pd.DataFrame) -> str:
    payload = candidate.sort_values(STEP_KEYS)[STEP_KEYS + ['recommended_dur_SP', 'recommended_time_SP']]
    encoded = payload.to_json(orient='records', double_precision=10).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:16]


def _action_options(
    before_steps: pd.DataFrame,
    phase_steps: pd.DataFrame,
    direction: int,
    increment_s: float,
    shift_fraction: float,
) -> list[tuple[int, float]]:
    actions: list[tuple[int, float]] = []
    for row_index, step in phase_steps.iterrows():
        current = float(before_steps.loc[row_index, 'dur_SP'])
        lower = float(step['step_time_min'])
        upper = float(step['step_time_max'])
        limit = min(
            (upper - lower) * shift_fraction,
            upper - current if direction > 0 else current - lower,
        )
        levels = np.arange(increment_s, limit + increment_s * 0.5, increment_s)
        actions.extend((row_index, float(direction * delta)) for delta in levels if delta <= limit + 1e-9)
    return actions


def _temperature_action_options(
    phase_steps: pd.DataFrame,
    direction: int,
    increment_c: float,
    shift_fraction: float,
) -> list[tuple[int, float]]:
    actions: list[tuple[int, float]] = []
    for row_index, step in phase_steps.iterrows():
        current = float(step['temp_SP'])
        lower = float(step['step_temp_min'])
        upper = float(step['step_temp_max'])
        limit = min(
            (upper - lower) * shift_fraction,
            upper - current if direction > 0 else current - lower,
        )
        levels = np.arange(increment_c, limit + increment_c * 0.5, increment_c)
        actions.extend((row_index, float(direction * delta)) for delta in levels if delta <= limit + 1e-9)
    return actions


def search_duration_candidates(
    steps: pd.DataFrame,
    baseline_feature_row: pd.Series,
    raw_predictor: V37RawPredictor,
    *,
    target_phase: str,
    expected_time_direction: int,
    target_center: float,
    spec_lower: float,
    spec_upper: float,
    duration_increment_s: float = 1.0,
    max_candidates: int = 100,
) -> CandidateSearchResult:
    """Search one- and two-Step duration actions using true V3.7 re-prediction.

    Each action changes one per-Step duration. The candidate then rebuilds all later
    cumulative PLC end times for the same MST/MSB group. ``baseline_feature_row``
    must come from the V3.7 BatchCurve/recipe feature builder and provide every
    non-Step feature required by the production artifact. No candidate uses Lab data.
    """
    if expected_time_direction not in {-1, 1} or duration_increment_s <= 0 or max_candidates < 1:
        raise ValueError('Invalid search direction, increment, or candidate limit.')

    before = _prepare_steps(steps)
    base_features = _phase_features(before, baseline_feature_row)
    baseline_prediction = float(raw_predictor.predict_raw(base_features).final_prediction[0])
    phase_steps = before[
        before['mapped_phase'].eq(target_phase) & before['is_time_settable'].fillna(False).astype(bool)
    ]
    if phase_steps.empty:
        return CandidateSearchResult(
            baseline_prediction, None, pd.DataFrame(), 'NO_VALID_CANDIDATE',
            ('REJECTED_CORRECTED_PHASE_UNMAPPED',),
        )

    evaluator = AcceptanceEvaluator()
    atomic_actions = _action_options(
        before, phase_steps, expected_time_direction, duration_increment_s, evaluator.single_shift_fraction,
    )
    action_sets: list[tuple[tuple[int, float], ...]] = [(action,) for action in atomic_actions]
    action_sets.extend(
        pair for pair in combinations(atomic_actions, 2) if pair[0][0] != pair[1][0]
    )

    audits: list[dict[str, Any]] = []
    accepted: list[tuple[tuple[float, float, int, str], pd.DataFrame]] = []
    for actions in action_sets[:max_candidates]:
        candidate = _build_candidate(before, actions)
        prediction_features = _phase_features(candidate, baseline_feature_row)
        recommended = float(raw_predictor.predict_raw(prediction_features).final_prediction[0])
        repeated = float(raw_predictor.predict_raw(prediction_features).final_prediction[0])
        phase_feature_delta = _feature_delta(base_features, prediction_features)
        decision = evaluator.evaluate(
            before,
            candidate,
            target_center=target_center,
            target_source='SPEC_CENTER',
            baseline_prediction=baseline_prediction,
            recommended_prediction=recommended,
            re_predicted_recommendation=repeated,
            reproducible=abs(recommended - repeated) < 0.1,
            spec_lower=spec_lower,
            spec_upper=spec_upper,
            expected_phase=target_phase,
            expected_time_direction=expected_time_direction,
        )
        fingerprint = _fingerprint(candidate)
        movement = float(sum(abs(delta) for _, delta in actions))
        gap_after = abs(target_center - recommended)
        action_description = [
            {
                'GroupName': str(before.loc[row_index, 'GroupName']),
                'StepNo': int(before.loc[row_index, 'StepNo']),
                'delta_duration_s': delta,
            }
            for row_index, delta in actions
        ]
        audits.append({
            'candidate_id': fingerprint,
            'actions': json.dumps(action_description),
            'net_phase_duration_delta_s': float(sum(delta for _, delta in actions)),
            'baseline_prediction': baseline_prediction,
            'recommended_prediction': recommended,
            're_predicted_prediction': repeated,
            'target_center': target_center,
            'gap_before': abs(target_center - baseline_prediction),
            'gap_after': gap_after,
            'accepted': decision.accepted,
            'status': decision.status,
            'pull_effective': decision.pull_effective,
            'overshoot_flag': decision.overshoot_flag,
            'reason_codes': json.dumps(decision.reason_codes),
            'phase_feature_delta_json': json.dumps(phase_feature_delta, sort_keys=True),
            'model_response_delta': recommended - baseline_prediction,
            'model_version': raw_predictor.metadata.get('model_version', 'UNKNOWN'),
        })
        if decision.accepted:
            accepted.append(((gap_after, movement, len(actions), fingerprint), candidate))

    audit_table = pd.DataFrame(audits)
    if not accepted:
        reason_codes = tuple(sorted({
            reason
            for encoded in audit_table.get('reason_codes', pd.Series(dtype=str))
            for reason in json.loads(encoded)
        }))
        return CandidateSearchResult(
            baseline_prediction, None, audit_table, 'NO_VALID_CANDIDATE',
            reason_codes or ('REJECTED_NO_FEASIBLE_DURATION_ACTION',),
        )

    accepted.sort(key=lambda item: item[0])
    return CandidateSearchResult(
        baseline_prediction,
        accepted[0][1],
        audit_table.sort_values(['accepted', 'gap_after'], ascending=[False, True]),
        'MANUAL_REVIEW_ONLY',
        (),
    )


def search_temperature_candidates(
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
    """Search one- and two-Step temperature-control actions using true V3.7 re-prediction."""
    if expected_temp_direction not in {-1, 1} or temperature_increment_c <= 0 or max_candidates < 1:
        raise ValueError('Invalid search direction, increment, or candidate limit.')

    required_temp_columns = {'is_temp_settable', 'step_temp_min', 'step_temp_max'}
    missing_temp_columns = required_temp_columns - set(steps.columns)
    if missing_temp_columns:
        raise ValueError(f'Missing temperature search columns: {sorted(missing_temp_columns)}')

    before = _prepare_steps(steps)
    base_features = _phase_features(before, baseline_feature_row)
    baseline_prediction = float(raw_predictor.predict_raw(base_features).final_prediction[0])
    phase_steps = before[
        before['mapped_phase'].eq(target_phase)
        & before['is_temp_settable'].fillna(False).astype(bool)
        & pd.to_numeric(before['temp_SP'], errors='coerce').fillna(0.0).gt(0)
    ]
    if phase_steps.empty:
        return CandidateSearchResult(
            baseline_prediction, None, pd.DataFrame(), 'NO_VALID_CANDIDATE',
            ('REJECTED_CORRECTED_PHASE_UNMAPPED',),
        )

    evaluator = AcceptanceEvaluator()
    atomic_actions = _temperature_action_options(
        phase_steps,
        expected_temp_direction,
        temperature_increment_c,
        evaluator.single_shift_fraction,
    )
    action_sets: list[tuple[tuple[int, float], ...]] = [(action,) for action in atomic_actions]
    action_sets.extend(
        pair for pair in combinations(atomic_actions, 2) if pair[0][0] != pair[1][0]
    )

    audits: list[dict[str, Any]] = []
    accepted: list[tuple[tuple[float, float, int, str], pd.DataFrame]] = []
    for actions in action_sets[:max_candidates]:
        candidate = _build_temperature_candidate(before, actions)
        prediction_features = _phase_features(candidate, baseline_feature_row)
        recommended = float(raw_predictor.predict_raw(prediction_features).final_prediction[0])
        repeated = float(raw_predictor.predict_raw(prediction_features).final_prediction[0])
        phase_feature_delta = _feature_delta(base_features, prediction_features)
        decision = evaluator.evaluate(
            before,
            candidate,
            target_center=target_center,
            target_source='SPEC_CENTER',
            baseline_prediction=baseline_prediction,
            recommended_prediction=recommended,
            re_predicted_recommendation=repeated,
            reproducible=abs(recommended - repeated) < 0.1,
            spec_lower=spec_lower,
            spec_upper=spec_upper,
        )
        fingerprint = _fingerprint(candidate)
        movement = float(sum(abs(delta) for _, delta in actions))
        gap_after = abs(target_center - recommended)
        action_description = [
            {
                'GroupName': str(before.loc[row_index, 'GroupName']),
                'StepNo': int(before.loc[row_index, 'StepNo']),
                'delta_temp_c': delta,
            }
            for row_index, delta in actions
        ]
        audits.append({
            'candidate_id': fingerprint,
            'actions': json.dumps(action_description),
            'action_parameter': 'temperature',
            'net_phase_temperature_delta_c': float(sum(delta for _, delta in actions)),
            'baseline_prediction': baseline_prediction,
            'recommended_prediction': recommended,
            're_predicted_prediction': repeated,
            'target_center': target_center,
            'gap_before': abs(target_center - baseline_prediction),
            'gap_after': gap_after,
            'accepted': decision.accepted,
            'status': decision.status,
            'pull_effective': decision.pull_effective,
            'overshoot_flag': decision.overshoot_flag,
            'reason_codes': json.dumps(decision.reason_codes),
            'phase_feature_delta_json': json.dumps(phase_feature_delta, sort_keys=True),
            'model_response_delta': recommended - baseline_prediction,
            'model_version': raw_predictor.metadata.get('model_version', 'UNKNOWN'),
        })
        if decision.accepted:
            accepted.append(((gap_after, movement, len(actions), fingerprint), candidate))

    audit_table = pd.DataFrame(audits)
    if not accepted:
        reason_codes = tuple(sorted({
            reason
            for encoded in audit_table.get('reason_codes', pd.Series(dtype=str))
            for reason in json.loads(encoded)
        }))
        return CandidateSearchResult(
            baseline_prediction, None, audit_table, 'NO_VALID_CANDIDATE',
            reason_codes or ('REJECTED_NO_FEASIBLE_TEMPERATURE_ACTION',),
        )

    accepted.sort(key=lambda item: item[0])
    return CandidateSearchResult(
        baseline_prediction,
        accepted[0][1],
        audit_table.sort_values(['accepted', 'gap_after'], ascending=[False, True]),
        'MANUAL_REVIEW_ONLY',
        (),
    )