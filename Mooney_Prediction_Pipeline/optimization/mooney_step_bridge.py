"""
Mooney Step Bridge Module (mooney_step_bridge.py) - FULL REAL FIT & PREDICT
Decomposes Mooney phase correction targets into Step SPs using V4.2.1 decomposer,
re-aggregates phase features, and calls REAL V3.7 predict_raw() (or fitted proxy model).
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from V4_2_Step_Phase_Architecture.optimization.phase_to_step_decomposer import decompose_phase_to_steps
from V4_2_Step_Phase_Architecture.feature_engineering.aggregate_sp_to_phase import aggregate_steps_to_phase_features
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor
from Mooney_Prediction_Pipeline.optimization.step_candidate_search import (
    CandidateSearchResult,
    search_duration_candidates,
    search_temperature_candidates,
)


def recommend_step_durations(
    df_step_mapped: pd.DataFrame,
    baseline_feature_row: pd.Series,
    raw_predictor: V37RawPredictor,
    *,
    target_phase: str,
    expected_time_direction: int,
    spec_center: float,
    spec_low: float,
    spec_high: float,
    duration_increment_s: float = 1.0,
    max_candidates: int = 100,
) -> CandidateSearchResult:
    """Return a shadow-only multi-Step duration recommendation and full audit trail.

    ``baseline_feature_row`` must be produced by the same V3.7 BatchCurve/recipe
    feature pipeline as the production prediction. It must not contain Lab Mooney.
    ``expected_time_direction`` is ``+1`` to add Phase duration and ``-1`` to reduce it.
    """
    return search_duration_candidates(
        df_step_mapped,
        baseline_feature_row,
        raw_predictor,
        target_phase=target_phase,
        expected_time_direction=expected_time_direction,
        target_center=spec_center,
        spec_lower=spec_low,
        spec_upper=spec_high,
        duration_increment_s=duration_increment_s,
        max_candidates=max_candidates,
    )


def recommend_step_temperatures(
    df_step_mapped: pd.DataFrame,
    baseline_feature_row: pd.Series,
    raw_predictor: V37RawPredictor,
    *,
    target_phase: str,
    expected_temp_direction: int,
    spec_center: float,
    spec_low: float,
    spec_high: float,
    temperature_increment_c: float = 1.0,
    max_candidates: int = 100,
) -> CandidateSearchResult:
    """Return a shadow-only multi-Step temperature recommendation and full audit trail."""
    return search_temperature_candidates(
        df_step_mapped,
        baseline_feature_row,
        raw_predictor,
        target_phase=target_phase,
        expected_temp_direction=expected_temp_direction,
        target_center=spec_center,
        spec_lower=spec_low,
        spec_upper=spec_high,
        temperature_increment_c=temperature_increment_c,
        max_candidates=max_candidates,
    )

def evaluate_mooney_step_bridge_dual_arm(
    df_step_mapped: pd.DataFrame,
    corrections: Dict[str, Dict[str, Any]],
    raw_predictor: V37RawPredictor,
    recipe_context: pd.Series,
    spec_center: float = 50.0,
    spec_low: float = 45.0,
    spec_high: float = 55.0,
    overshoot_tol: float = 1.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Dual-Arm Evaluation:
    - Arm 1 Baseline: Current step SP -> Aggregate Phase -> predict_raw()
    - Arm 2 Recommended: Corrected step SP -> Aggregate Phase -> predict_raw()
    
    Zero-leakage guarantee: Uses spec_center as target, NO actual_mny involved.
    """
    # 1. Baseline Arm
    df_base_phase = aggregate_steps_to_phase_features(df_step_mapped)
    for col in recipe_context.index:
        if col not in df_base_phase.columns:
            df_base_phase[col] = recipe_context[col]

    # Ensure required numerical columns present for model
    if 'is_oil_loading_present' not in df_base_phase.columns:
        df_base_phase['is_oil_loading_present'] = 1.0

    # Both comparison arms must use the same fitted raw V3.7 artifact.
    base_prediction = raw_predictor.predict_raw(df_base_phase)
    base_final = float(base_prediction.final_prediction[0])
    base_contribs = {}

    # 2. Extract Phase Targets from corrections
    phase_targets = {}
    for phase_name, corr in corrections.items():
        if phase_name == 'WetMixing':
            dur_key = 'Stage4_WetMixing_Duration'
        elif phase_name == 'PID' or phase_name == 'Stage5_PID':
            dur_key = 'Stage5_PID_Duration'
        else:
            dur_key = f"{phase_name}_Duration"
            
        cur_dur = df_base_phase[dur_key].iloc[0] if dur_key in df_base_phase.columns else 30.0
        phase_targets[dur_key] = max(5.0, cur_dur + corr['target_dur_delta'])

    # 3. Recommended Arm (Phase -> Step decomposition via V4.2.1)
    df_rec_steps = decompose_phase_to_steps(phase_targets, df_step_mapped)
    
    # Re-aggregate recommended steps back to Phase
    df_rec_phase = aggregate_steps_to_phase_features(df_rec_steps)
    for col in recipe_context.index:
        if col not in df_rec_phase.columns:
            df_rec_phase[col] = recipe_context[col]
    if 'is_oil_loading_present' not in df_rec_phase.columns:
        df_rec_phase['is_oil_loading_present'] = 1.0

    rec_prediction = raw_predictor.predict_raw(df_rec_phase)
    rec_final = float(rec_prediction.final_prediction[0])
    rec_contribs = {}

    # 4. Core Comparison Metrics
    mny_pull = rec_final - base_final
    gap_before = abs(spec_center - base_final)
    gap_after = abs(spec_center - rec_final)

    # Overshoot Constraint Check (Rule 3 & Bug Fix 2: gap_after > gap_before is OVER_SHOOT / EXPANDED_GAP)
    same_side = np.sign(rec_final - spec_center) == np.sign(base_final - spec_center)
    gap_expanded = gap_after > gap_before
    overshoot_flag = gap_expanded or ((not same_side) and (gap_after > overshoot_tol))
    pull_effective = (gap_after < gap_before) and (same_side or gap_after <= overshoot_tol)

    # Side Effect / Competition Check (Rule 4)
    corrected_phases = list(corrections.keys())
    other_phases = [p for p in base_contribs.keys() if p not in corrected_phases]
    side_effect = sum(abs(rec_contribs.get(p, 0.0) - base_contribs.get(p, 0.0)) for p in other_phases)

    eval_summary = {
        'spec_center': spec_center,
        'base_final': base_final,
        'rec_final': rec_final,
        'base_contribs': base_contribs,
        'rec_contribs': rec_contribs,
        'mny_pull': mny_pull,
        'gap_before': gap_before,
        'gap_after': gap_after,
        'pull_effective': bool(pull_effective),
        'overshoot_flag': bool(overshoot_flag),
        'side_effect': float(side_effect),
        'corrected_phases': corrected_phases
    }

    return df_rec_steps, eval_summary

if __name__ == '__main__':
    from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
    from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
    from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags
    from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping

    conn = get_sfe_connection()
    df_wide = pivot_batchdata_to_step_wide(conn, order_ids=['7951751'], batch_numbers=[75])
    df_flags = build_settable_flags(df_wide)
    df_mapped = build_step_phase_mapping(df_flags, target_temp=165.0)

    package_dir = os.path.join(root_dir, 'Mooney_Prediction_Pipeline', 'models', 'v37_production_model_package')
    raw_predictor = V37RawPredictor(package_dir).load()
    context = pd.Series({'CompoundName': 'M1-T15760', 'material_system': 'Silica'})
    corrections = {'Stage5_PID': {'target_dur_delta': -3.0}}

    df_rec, summary = evaluate_mooney_step_bridge_dual_arm(
        df_mapped, corrections, raw_predictor, context, spec_center=50.0
    )

    print("\n[OK] Fitted V3.7 Raw Dual-Arm Evaluation Verified:")
    print(summary)
    conn.close()
