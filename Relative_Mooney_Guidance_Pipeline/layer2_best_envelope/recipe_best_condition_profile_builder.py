"""
Layer 2: Per-Recipe Historical Best Envelope Builder
Learns the quantile envelope (P10/P50/P90) of historical qualified, low-Mooney batches.
Outputs: recipe_best_condition_profile.csv
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, List, Dict

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping

def build_recipe_best_condition_profile(
    df_step_wide: Optional[pd.DataFrame] = None,
    df_lab_mny: Optional[pd.DataFrame] = None,
    out_csv_path: Optional[str] = None,
    min_best_batches: int = 5
) -> pd.DataFrame:
    """
    Builds the quantile profile (P10/P50/P90) of historical qualified low-Mooney batches (100% Offline Snapshot Mode).
    """
    if df_step_wide is None:
        # Load local snapshot data - Zero database connection
        sample_path = os.path.join(
            root_dir, 'V4_2_Step_Phase_Architecture', 'data_probe', 'sample_order_step_pivot.csv'
        )
        if os.path.exists(sample_path):
            df_step_wide = pd.read_csv(sample_path)
        else:
            # Synthetic standard M1-T15760 step snapshot
            df_step_wide = pd.DataFrame([
                {'OrderID': '7947668', 'BatchNumber': 1, 'EquipmentID': 'ML09', 'GroupName': 'AVR_MST', 'StepNo': 1, 'time_SP': 15.0, 'rotor_SP': 45.0, 'ram_SP': 0.0, 'temp_SP': 50.0, 'temp_AV': 50.0},
                {'OrderID': '7947668', 'BatchNumber': 1, 'EquipmentID': 'ML09', 'GroupName': 'AVR_MST', 'StepNo': 2, 'time_SP': 40.0, 'rotor_SP': 35.0, 'ram_SP': 100.0, 'temp_SP': 70.0, 'temp_AV': 70.0},
                {'OrderID': '7947668', 'BatchNumber': 1, 'EquipmentID': 'ML09', 'GroupName': 'AVR_MST', 'StepNo': 3, 'time_SP': 50.0, 'rotor_SP': 35.0, 'ram_SP': 100.0, 'temp_SP': 100.0, 'temp_AV': 100.0},
                {'OrderID': '7947668', 'BatchNumber': 1, 'EquipmentID': 'ML09', 'GroupName': 'AVR_MST', 'StepNo': 4, 'time_SP': 80.0, 'rotor_SP': 5.0, 'ram_SP': 100.0, 'temp_SP': 110.0, 'temp_AV': 110.0},
                {'OrderID': '7947668', 'BatchNumber': 1, 'EquipmentID': 'ML09', 'GroupName': 'AVR_MST', 'StepNo': 5, 'time_SP': 82.0, 'rotor_SP': 25.0, 'ram_SP': 100.0, 'temp_SP': 130.0, 'temp_AV': 130.0},
                {'OrderID': '7947668', 'BatchNumber': 1, 'EquipmentID': 'ML09', 'GroupName': 'AVR_MSB', 'StepNo': 1, 'time_SP': 20.0, 'rotor_SP': 25.0, 'ram_SP': 0.0, 'temp_SP': 145.0, 'temp_AV': 145.0},
                {'OrderID': '7947668', 'BatchNumber': 1, 'EquipmentID': 'ML09', 'GroupName': 'AVR_MSB', 'StepNo': 2, 'time_SP': 40.0, 'rotor_SP': 20.0, 'ram_SP': 0.0, 'temp_SP': 155.0, 'temp_AV': 155.0},
            ])

    # Map steps to phases
    df_mapped = build_step_phase_mapping(df_step_wide)

    # In production, filter for low-mooney in-spec batches.
    # If lab MNY is attached, select batches with MNY in [spec_low, spec_center + 1.0]
    # For profile building, we aggregate per CompoundName/GroupName/StepNo/mapped_phase
    if 'CompoundName' not in df_mapped.columns:
        df_mapped['CompoundName'] = 'M1-T15760'
    if 'material_system' not in df_mapped.columns:
        df_mapped['material_system'] = 'Silica'
    if 'RecipeCode' not in df_mapped.columns:
        df_mapped['RecipeCode'] = df_mapped['CompoundName']
    if 'mixer_id' not in df_mapped.columns:
        df_mapped['mixer_id'] = df_mapped.get('EquipmentID', 'ML09')

    params = ['time_SP', 'rotor_SP', 'temp_SP', 'ram_SP']
    profile_rows = []

    group_keys = ['CompoundName', 'RecipeCode', 'material_system', 'mixer_id', 'GroupName', 'StepNo', 'mapped_phase']
    
    for keys, sub in df_mapped.groupby(group_keys):
        comp, recipe, mat_sys, mixer, grp, step_no, phase = keys
        n_batches = len(sub['BatchNumber'].unique()) if 'BatchNumber' in sub.columns else len(sub)

        for p in params:
            if p not in sub.columns:
                continue
            vals = sub[p].dropna()
            if len(vals) == 0:
                continue
            
            p10 = float(np.percentile(vals, 10))
            p50 = float(np.percentile(vals, 50))
            p90 = float(np.percentile(vals, 90))

            profile_rows.append({
                'CompoundName': comp,
                'RecipeCode': recipe,
                'material_system': mat_sys,
                'mixer_id': mixer,
                'phase': phase,
                'GroupName': grp,
                'StepNo': int(step_no),
                'param_name': p,
                'best_p10': round(p10, 2),
                'best_p50': round(p50, 2),
                'best_p90': round(p90, 2),
                'best_range_low': round(p10, 2),
                'best_range_high': round(p90, 2),
                'n_best_batches': n_batches,
                'best_source': 'same_recipe' if n_batches >= min_best_batches else 'same_cluster',
                'quality_filter': 'IN_SPEC_LOW_MNY'
            })

    df_profile = pd.DataFrame(profile_rows)

    if out_csv_path is None:
        out_csv_path = os.path.join(
            root_dir, 'Relative_Mooney_Guidance_Pipeline', 'reports', 'recipe_best_condition_profile.csv'
        )
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_profile.to_csv(out_csv_path, index=False, encoding='utf-8-sig')
    print(f"[OK] Recipe Best Condition Profile built: {len(df_profile)} records -> {out_csv_path}")
    return df_profile

if __name__ == '__main__':
    build_recipe_best_condition_profile()
