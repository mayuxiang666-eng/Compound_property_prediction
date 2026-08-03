# ============================================================================
# Step 5: Safe Parameter Bounds Builder (V4.1)
# ============================================================================
# Computes P5-P95 and P10-P90 historical operating parameter bounds per:
# (RecipeCode, MixerID, material_system, phase_route).
# Ensures zero recommendations outside historical/engineering safe ranges.
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

CONTROLLABLE_PARAMS = [
    'Stage2_DryMixing_Duration',
    'Stage4_WetMixing_Duration',
    'Stage5_PID_Duration',
    'Stage6_BottomMixing_Duration',
    'Target_Temperature',
]


def build_safe_parameter_bounds(
    df_history: pd.DataFrame,
    percentile_lower=5,
    percentile_upper=95,
    min_silica_route_samples=20,
) -> pd.DataFrame:
    """
    Computes exact recipe-mixer-route bounds plus same-recipe-route fallback bounds.
    """
    bounds_rows = []
    base_group_cols = ['CompoundName', 'material_system', 'phase_route']
    grouping_levels = [(base_group_cols, 'SAME_RECIPE_ROUTE_ANY_MIXER')]
    if 'MixerLine' in df_history.columns:
        grouping_levels.insert(0, (base_group_cols + ['MixerLine'], 'EXACT_RECIPE_MIXER_ROUTE'))

    for group_cols, bounds_scope in grouping_levels:
        for keys, grp in df_history.groupby(group_cols):
            if len(group_cols) == 4:
                compound, material_system, route, mixer = keys
            else:
                compound, material_system, route = keys
                mixer = 'ALL'

            row = {
                'CompoundName': compound,
                'material_system': material_system,
                'phase_route': route,
                'MixerLine': mixer,
                'bounds_scope': bounds_scope,
                'sample_count': len(grp),
            }

            for param in CONTROLLABLE_PARAMS:
                if param in grp.columns:
                    vals = pd.to_numeric(grp[param], errors='coerce').dropna()
                    if len(vals) >= 5:
                        p_low = float(np.percentile(vals, percentile_lower))
                        p_high = float(np.percentile(vals, percentile_upper))
                        p_median = float(np.median(vals))
                    else:
                        p_low, p_high, p_median = 0.0, 0.0, 0.0
                else:
                    p_low, p_high, p_median = 0.0, 0.0, 0.0

                row[f'{param}_min'] = round(p_low, 2)
                row[f'{param}_max'] = round(p_high, 2)
                row[f'{param}_nominal'] = round(p_median, 2)

            bounds_rows.append(row)

    silica_history = df_history[df_history.get('material_system', pd.Series('', index=df_history.index)) == 'Silica']
    for (material_system, route), grp in silica_history.groupby(['material_system', 'phase_route']):
        if len(grp) < min_silica_route_samples:
            continue
        row = {
            'CompoundName': 'ALL',
            'material_system': material_system,
            'phase_route': route,
            'MixerLine': 'ALL',
            'bounds_scope': 'SILICA_ROUTE_GLOBAL_FALLBACK',
            'sample_count': len(grp),
        }
        for param in CONTROLLABLE_PARAMS:
            vals = pd.to_numeric(grp.get(param, pd.Series(dtype=float)), errors='coerce').dropna()
            if len(vals) >= 5:
                row[f'{param}_min'] = round(float(np.percentile(vals, percentile_lower)), 2)
                row[f'{param}_max'] = round(float(np.percentile(vals, percentile_upper)), 2)
                row[f'{param}_nominal'] = round(float(np.median(vals)), 2)
            else:
                row[f'{param}_min'] = 0.0
                row[f'{param}_max'] = 0.0
                row[f'{param}_nominal'] = 0.0
        bounds_rows.append(row)

    bounds_df = pd.DataFrame(bounds_rows)
    return bounds_df
