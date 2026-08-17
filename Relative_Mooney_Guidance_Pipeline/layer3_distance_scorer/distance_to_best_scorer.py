"""
Layer 3: Distance-to-Best Scorer
Calculates the multidimensional process distance between current batch and historical best envelope.
Embeds PE Domain Rules:
1. Step 1-2 Loading locked (adjustable_flag = False)
2. Ram locked (adjustable_flag = False)
3. Dry Mixing duration / rotor adjustable
4. Wet Mixing: lowering rotor extends duration for full mechanical dispersion
5. Silica PID: adjust MSB (lower mixer) 10s increments, avoid overreaction
Outputs: current_distance_to_best.csv
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def calculate_distance_to_best(
    df_current_batch_steps: pd.DataFrame,
    df_best_profile: pd.DataFrame,
    out_csv_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Computes distance to historical best condition for each step parameter.
    """
    results = []
    phase_distances = {}

    order_id = str(df_current_batch_steps['OrderID'].iloc[0]) if 'OrderID' in df_current_batch_steps.columns else 'UNKNOWN'
    batch_no = int(df_current_batch_steps['BatchNumber'].iloc[0]) if 'BatchNumber' in df_current_batch_steps.columns else 1
    compound = str(df_current_batch_steps.get('CompoundName', pd.Series(['M1-T15760'])).iloc[0])
    mat_sys = str(df_current_batch_steps.get('material_system', pd.Series(['Silica'])).iloc[0])

    for _, row in df_current_batch_steps.iterrows():
        grp = row['GroupName']
        step_no = int(row['StepNo'])
        phase = row.get('mapped_phase', 'Unknown')
        step_key = f"{grp}_Step{step_no}"

        # PE Rule 1: Step 1-2 Loading are locked
        is_loading = (step_no in [1, 2] and grp == 'AVR_MST') or 'Loading' in str(phase)
        
        for param in ['time_SP', 'rotor_SP', 'temp_SP', 'ram_SP']:
            if param not in row or pd.isna(row[param]):
                continue
            curr_val = float(row[param])

            # Find matching best profile
            prof_match = df_best_profile[
                (df_best_profile['GroupName'] == grp) &
                (df_best_profile['StepNo'] == step_no) &
                (df_best_profile['param_name'] == param)
            ]

            if len(prof_match) == 0:
                # Fallback to general phase profile
                prof_match = df_best_profile[
                    (df_best_profile['phase'] == phase) &
                    (df_best_profile['param_name'] == param)
                ]

            if len(prof_match) > 0:
                best_row = prof_match.iloc[0]
                p10 = float(best_row['best_p10'])
                p50 = float(best_row['best_p50'])
                p90 = float(best_row['best_p90'])
                range_low = float(best_row['best_range_low'])
                range_high = float(best_row['best_range_high'])
            else:
                p10, p50, p90 = curr_val, curr_val, curr_val
                range_low, range_high = curr_val, curr_val

            # Compute normalized distance
            scale = max(1.0, p90 - p10)
            dist_score = abs(curr_val - p50) / scale

            # Determine direction to best
            if curr_val < range_low:
                dir_to_best = 'INCREASE'
            elif curr_val > range_high:
                dir_to_best = 'DECREASE'
            else:
                dir_to_best = 'IN_BEST_WINDOW'

            # PE Rule 2: Ram is locked
            # PE Rule 3: Loading steps are locked
            if param == 'ram_SP' or is_loading:
                adjustable = False
            elif param in ['time_SP', 'rotor_SP']:
                adjustable = True
            elif param == 'temp_SP':
                adjustable = ('PID' in str(phase) or 'Dry' in str(phase))
            else:
                adjustable = False

            results.append({
                'OrderID': order_id,
                'BatchNumber': batch_no,
                'CompoundName': compound,
                'material_system': mat_sys,
                'phase': phase,
                'step_key': step_key,
                'GroupName': grp,
                'StepNo': step_no,
                'param_name': param,
                'current_value': round(curr_val, 2),
                'best_p50': round(p50, 2),
                'best_range_low': round(range_low, 2),
                'best_range_high': round(range_high, 2),
                'distance_score': round(dist_score, 3),
                'direction_to_best': dir_to_best,
                'adjustable_flag': adjustable
            })

            # Accumulate phase distance
            if adjustable:
                phase_distances[phase] = phase_distances.get(phase, 0.0) + dist_score

    df_dist = pd.DataFrame(results)

    if out_csv_path is None:
        out_csv_path = os.path.join(
            root_dir, 'Relative_Mooney_Guidance_Pipeline', 'reports', 'current_distance_to_best.csv'
        )
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_dist.to_csv(out_csv_path, index=False, encoding='utf-8-sig')

    return df_dist, phase_distances

if __name__ == '__main__':
    print("[OK] Distance-to-Best Scorer compiled.")
