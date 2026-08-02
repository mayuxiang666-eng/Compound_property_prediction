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

    candidates = []
    rng = np.random.default_rng(random_state)

    # Base nominal setpoints
    nominal_dict = {}
    for param in [
        'Stage2_DryMixing_Duration',
        'Stage2_DryMixing_RotorSpeed_Mean',
        'Stage4_WetMixing_Duration',
        'Stage5_PID_Duration',
        'Stage5_PID_temp_Mean',
        'Stage6_BottomMixing_Duration',
        'Target_Temperature',
    ]:
        nom = bounds_row.get(f'{param}_nominal', recipe_row.get(param, 0.0))
        nominal_dict[param] = float(nom) if not pd.isna(nom) else 0.0

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

        if branch == 'Silica_OilWet':
            # WetMix, PID, Bottom, Target Temperature
            min_w, max_w = bounds_dict['Stage4_WetMixing_Duration']
            if max_w > min_w:
                cand['Stage4_WetMixing_Duration'] = round(float(rng.uniform(min_w, max_w)), 1)

            min_pid, max_pid = bounds_dict['Stage5_PID_Duration']
            if max_pid > min_pid:
                cand['Stage5_PID_Duration'] = round(float(rng.uniform(min_pid, max_pid)), 1)

            min_bot, max_bot = bounds_dict['Stage6_BottomMixing_Duration']
            if max_bot > min_bot:
                cand['Stage6_BottomMixing_Duration'] = round(float(rng.uniform(min_bot, max_bot)), 1)

            min_t, max_t = bounds_dict['Target_Temperature']
            if max_t > min_t:
                cand['Target_Temperature'] = round(float(rng.uniform(min_t, max_t)), 1)

        elif branch == 'Silica_NoOilDry':
            # DryMix, PID if exists, Bottom, Target Temperature
            cand['Stage4_WetMixing_Duration'] = 0.0  # MASK OUT

            min_d, max_d = bounds_dict['Stage2_DryMixing_Duration']
            if max_d > min_d:
                cand['Stage2_DryMixing_Duration'] = round(float(rng.uniform(min_d, max_d)), 1)

            if nominal_dict['Stage5_PID_Duration'] > 0:
                min_pid, max_pid = bounds_dict['Stage5_PID_Duration']
                if max_pid > min_pid:
                    cand['Stage5_PID_Duration'] = round(float(rng.uniform(min_pid, max_pid)), 1)
            else:
                cand['Stage5_PID_Duration'] = 0.0

            min_bot, max_bot = bounds_dict['Stage6_BottomMixing_Duration']
            if max_bot > min_bot:
                cand['Stage6_BottomMixing_Duration'] = round(float(rng.uniform(min_bot, max_bot)), 1)

            min_t, max_t = bounds_dict['Target_Temperature']
            if max_t > min_t:
                cand['Target_Temperature'] = round(float(rng.uniform(min_t, max_t)), 1)

        elif branch == 'CarbonBlack_OilWet':
            # DryMix, WetMix, Bottom
            cand['Stage5_PID_Duration'] = 0.0  # MASK OUT PID
            cand['Stage5_PID_temp_Mean'] = 0.0

            min_d, max_d = bounds_dict['Stage2_DryMixing_Duration']
            if max_d > min_d:
                cand['Stage2_DryMixing_Duration'] = round(float(rng.uniform(min_d, max_d)), 1)

            min_w, max_w = bounds_dict['Stage4_WetMixing_Duration']
            if max_w > min_w:
                cand['Stage4_WetMixing_Duration'] = round(float(rng.uniform(min_w, max_w)), 1)

            min_bot, max_bot = bounds_dict['Stage6_BottomMixing_Duration']
            if max_bot > min_bot:
                cand['Stage6_BottomMixing_Duration'] = round(float(rng.uniform(min_bot, max_bot)), 1)

        elif branch == 'CarbonBlack_NoOilDry':
            # DryMix, Bottom ONLY
            cand['Stage4_WetMixing_Duration'] = 0.0  # MASK OUT
            cand['Stage5_PID_Duration'] = 0.0         # MASK OUT
            cand['Stage5_PID_temp_Mean'] = 0.0

            min_d, max_d = bounds_dict['Stage2_DryMixing_Duration']
            if max_d > min_d:
                cand['Stage2_DryMixing_Duration'] = round(float(rng.uniform(min_d, max_d)), 1)

            min_bot, max_bot = bounds_dict['Stage6_BottomMixing_Duration']
            if max_bot > min_bot:
                cand['Stage6_BottomMixing_Duration'] = round(float(rng.uniform(min_bot, max_bot)), 1)

        candidates.append(cand)

    return candidates
