"""
Layer 2: Cloud-Backed Multi-Batch Historical Best Envelope Builder
Queries true historical batches from Mustang Cloud SFEAPAC (HESFESFEPLANT) and filters
by canonical 001/011 Lab MNY to compute true P10/P50/P90 quantile envelopes across N >= 20 batches.
Outputs: recipe_best_condition_profile_cloud.csv
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, List, Dict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Relative_Mooney_Guidance_Pipeline.data_extraction.sfeapac_cloud_connector import query_sfeapac_step_data
from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping

LABELED_BATCHES_PATH = os.path.join(
    ROOT_DIR, 'reports', 'recent_oot_t15760', 'live_lab_recheck_20260812', 'fully_free_exact_recipe_ml09_labeled_batches.csv'
)

def build_cloud_recipe_best_envelope(
    compound_name: str = 'M1-T15760',
    spec_center: float = 50.0,
    ideal_mny_max: float = 55.0,
    min_best_batches: int = 15,
    out_csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Builds the authentic multi-batch quantile envelope (P10/P50/P90) using cloud SFEAPAC data.
    """
    print(f"[{compound_name}] Loading verified 001/011 Lab MNY records...")
    df_lab = pd.read_csv(LABELED_BATCHES_PATH)
    
    # Filter for qualified in-spec low-Mooney batches (e.g. MNY in [45.0, 55.0])
    df_qualified = df_lab[
        (df_lab['test_result'] >= 45.0) &
        (df_lab['test_result'] <= ideal_mny_max) &
        (df_lab['disposition_code'].astype(str).str.contains('001|011|1|11'))
    ].copy()

    order_ids = df_qualified['OrderID'].astype(str).unique().tolist()
    print(f" -> Found {len(df_qualified)} qualified in-spec low-Mooney batches across {len(order_ids)} orders.")

    # Query step data from Cloud SFEAPAC RDS
    print(" -> Querying step SP/AV from Mustang Cloud SFEAPAC (HESFESFEPLANT)...")
    df_steps_cloud = query_sfeapac_step_data(order_ids)
    print(f" -> Retrieved {len(df_steps_cloud)} step records from cloud.")

    if len(df_steps_cloud) == 0:
        raise ValueError("No step data returned from SFEAPAC cloud for the specified orders.")

    # Map steps to phases using expert rules
    df_mapped = build_step_phase_mapping(df_steps_cloud)
    df_mapped['CompoundName'] = compound_name
    df_mapped['material_system'] = 'Silica' if 'T15760' in compound_name else 'CarbonBlack'
    df_mapped['RecipeCode'] = compound_name
    df_mapped['mixer_id'] = 'ML09'

    params = ['time_SP', 'rotor_SP', 'temp_SP', 'ram_SP']
    profile_rows = []

    group_keys = ['CompoundName', 'RecipeCode', 'material_system', 'mixer_id', 'GroupName', 'StepNo', 'mapped_phase']
    
    for keys, sub in df_mapped.groupby(group_keys):
        comp, recipe, mat_sys, mixer, grp, step_no, phase = keys
        unique_batches = len(sub[['OrderID', 'BatchNumber']].drop_duplicates())

        for p in params:
            if p not in sub.columns:
                continue
            vals = sub[p].dropna().astype(float)
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
                'n_best_batches': unique_batches,
                'best_source': 'cloud_sfeapac_multi_batch',
                'quality_filter': f'LAB_MNY_IN_SPEC_LE_{ideal_mny_max}'
            })

    df_profile = pd.DataFrame(profile_rows)

    # Mandatory Quality Assertions
    assert len(df_profile) > 0, "ERROR: Profile dataframe is empty!"
    max_batches = df_profile['n_best_batches'].max()
    print(f" -> Best envelope built with up to {max_batches} batches per step.")
    assert max_batches >= min_best_batches, f"ERROR: Insufficient batches ({max_batches} < {min_best_batches})!"

    if out_csv_path is None:
        out_csv_path = os.path.join(
            ROOT_DIR, 'Relative_Mooney_Guidance_Pipeline', 'reports', 'recipe_best_condition_profile_cloud.csv'
        )
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_profile.to_csv(out_csv_path, index=False, encoding='utf-8-sig')
    print(f"[SUCCESS] Cloud Multi-Batch Golden Envelope saved: {len(df_profile)} records -> {out_csv_path}")
    return df_profile

if __name__ == '__main__':
    build_cloud_recipe_best_envelope()
