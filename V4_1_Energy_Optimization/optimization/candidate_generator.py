# ============================================================================
# Step 5: Updated Route-Masked Candidate Parameter Generator (V4.1)
# ============================================================================
# Enforces exact branch-specific candidate spaces:
# - Silica_OilWet: WetMix, PID, Bottom, Target Temperature
# - Silica_NoOilDry: DryMix, PID (if exists), Bottom, Target Temperature
# - CB_OilWet: DryMix, WetMix, Bottom
# - CB_NoOilDry: DryMix, Bottom ONLY
# Never generates parameters for non-existing stages.
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

SUPPORTED_ROUTE_BRANCHES = {
    'Silica_OilWet',
    'Silica_NoOilDry',
    'CarbonBlack_OilWet',
    'CarbonBlack_NoOilDry',
}

STAGE_DURATION_PARAMS = {
    'has_drymix_stage': 'Stage2_DryMixing_Duration',
    'has_wetmix_stage': 'Stage4_WetMixing_Duration',
    'has_pid_stage': 'Stage5_PID_Duration',
    'has_bottom_stage': 'Stage6_BottomMixing_Duration',
    'has_oilloading_stage': 'Stage3_OilLoading_Duration',
}


def derive_route_stage_mask(recipe_row: pd.Series) -> dict[str, bool]:
    """Returns the effective stages for one batch without inventing absent stages."""
    material_system = recipe_row.get('material_system', 'Silica')
    phase_route = recipe_row.get('phase_route', 'OilWet')
    branch = f'{material_system}_{phase_route}'
    if branch not in SUPPORTED_ROUTE_BRANCHES:
        raise ValueError(f'Unsupported material-system and phase-route branch: {branch}')

    def has_observed_stage(duration_column: str, default: bool) -> bool:
        if duration_column not in recipe_row.index:
            return default
        value = pd.to_numeric(recipe_row.get(duration_column), errors='coerce')
        return bool(default and pd.notna(value) and float(value) > 0.0)

    is_oil_wet = phase_route == 'OilWet'
    is_silica = material_system == 'Silica'
    return {
        'has_drymix_stage': has_observed_stage('Stage2_DryMixing_Duration', branch != 'Silica_OilWet'),
        'has_wetmix_stage': has_observed_stage('Stage4_WetMixing_Duration', is_oil_wet),
        'has_pid_stage': has_observed_stage('Stage5_PID_Duration', is_silica),
        'has_bottom_stage': has_observed_stage('Stage6_BottomMixing_Duration', True),
        'has_oilloading_stage': has_observed_stage('Stage3_OilLoading_Duration', is_oil_wet),
    }


def generate_route_masked_candidates(
    recipe_row: pd.Series,
    bounds_row: pd.Series,
    n_candidates=100,
    random_state: int | None = None,
) -> list[dict]:
    """
    Generates n_candidates process parameter setpoints respecting route masking and safe bounds.
    """
    mat_system = recipe_row.get('material_system', 'Silica')
    phase_route = recipe_row.get('phase_route', 'OilWet')
    branch = f"{mat_system}_{phase_route}"
    if branch not in SUPPORTED_ROUTE_BRANCHES:
        raise ValueError(f'Unsupported material-system and phase-route branch: {branch}')
    stage_mask = derive_route_stage_mask(recipe_row)

    candidates = []
    rng = np.random.default_rng(random_state)

    # Base nominal setpoints
    nominal_dict = {}
    for param in [
        'Stage2_DryMixing_Duration',
        'Stage4_WetMixing_Duration',
        'Stage5_PID_Duration',
        'Stage6_BottomMixing_Duration',
        'Target_Temperature',
    ]:
        nom = bounds_row.get(f'{param}_nominal', recipe_row.get(param, 0.0))
        nominal_dict[param] = float(nom) if not pd.isna(nom) else 0.0

    for stage_flag, duration_param in STAGE_DURATION_PARAMS.items():
        if duration_param in nominal_dict and not stage_mask[stage_flag]:
            nominal_dict[duration_param] = 0.0

    # Include nominal as candidate #0
    candidates.append(nominal_dict.copy())

    # Parameter bounds
    bounds_dict = {}
    for param in nominal_dict.keys():
        p_min = bounds_row.get(f'{param}_min', nominal_dict[param] * 0.85)
        p_max = bounds_row.get(f'{param}_max', nominal_dict[param] * 1.15)
        if pd.isna(p_min) or p_min <= 0:
            p_min = max(0.0, nominal_dict[param] * 0.85)
        if pd.isna(p_max) or p_max <= 0:
            p_max = nominal_dict[param] * 1.15
        bounds_dict[param] = (float(p_min), float(p_max))

    # Grid / Random Sampling with Strict Branch Masking
    for _ in range(n_candidates - 1):
        cand = nominal_dict.copy()

        if stage_mask['has_drymix_stage']:
            min_d, max_d = bounds_dict['Stage2_DryMixing_Duration']
            if max_d > min_d:
                cand['Stage2_DryMixing_Duration'] = round(float(rng.uniform(min_d, max_d)), 1)

        if stage_mask['has_wetmix_stage']:
            min_w, max_w = bounds_dict['Stage4_WetMixing_Duration']
            if max_w > min_w:
                cand['Stage4_WetMixing_Duration'] = round(float(rng.uniform(min_w, max_w)), 1)

        if stage_mask['has_pid_stage']:
            min_pid, max_pid = bounds_dict['Stage5_PID_Duration']
            if max_pid > min_pid:
                cand['Stage5_PID_Duration'] = round(float(rng.uniform(min_pid, max_pid)), 1)

        if stage_mask['has_bottom_stage']:
            min_bot, max_bot = bounds_dict['Stage6_BottomMixing_Duration']
            if max_bot > min_bot:
                cand['Stage6_BottomMixing_Duration'] = round(float(rng.uniform(min_bot, max_bot)), 1)

        min_t, max_t = bounds_dict['Target_Temperature']
        if max_t > min_t:
            cand['Target_Temperature'] = round(float(rng.uniform(min_t, max_t)), 1)

        for stage_flag, duration_param in STAGE_DURATION_PARAMS.items():
            if duration_param in cand and not stage_mask[stage_flag]:
                cand[duration_param] = 0.0
        candidates.append(cand)

    return candidates
