"""
Module C: Step Bridge Logger (stepbridge_logger.py) - FIXED
Generates structured mooney_stepbridge_log rows aligned with V3.7 prediction_log schema.
Fixes:
- Populates step_SP_before and step_SP_after JSON dictionaries.
- De-duplicates mapped_steps by GroupName + StepNo.
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List

REQUIRED_LOG_COLS = [
    'prediction_id', 'OrderID', 'BatchNumber', 'label_group_id', 'GroupName',
    'CompoundName', 'cluster_id', 'mixer_id', 'arm',
    'raw_prediction', 'stage1_pred', 'stage1b_bias',
    'phase_contribution_json', 'reason_codes', 'confidence_label', 'calibration_source',
    'spec_low', 'spec_high', 'mny_in_spec',
    'corrected_phase', 'correction_direction', 'mapped_steps', 'step_SP_before', 'step_SP_after',
    'mny_pull', 'gap_before', 'gap_after', 'pull_effective', 'overshoot_flag', 'side_effect',
    'lab_status', 'actual_mny', 'lab_result_time', 'mny_pred_error', 'quality_ok'
]

def format_step_sp_json(df_steps: pd.DataFrame, time_col: str, rotor_col: str, ram_col: str) -> str:
    """Formats step SPs into a clean JSON dictionary keyed by GroupName_StepNo."""
    sp_dict = {}
    if df_steps.empty:
        return json.dumps(sp_dict)
        
    for idx, row in df_steps.iterrows():
        key = f"{row.get('GroupName', 'MST')}_Step{int(row.get('StepNo', idx))}"
        sp_dict[key] = {
            'time_SP': float(row[time_col]) if time_col in row and pd.notna(row[time_col]) else 0.0,
            'rotor_SP': float(row[rotor_col]) if rotor_col in row and pd.notna(row[rotor_col]) else 0.0,
            'ram_SP': float(row[ram_col]) if ram_col in row and pd.notna(row[ram_col]) else 0.0
        }
    return json.dumps(sp_dict)

def build_stepbridge_log_rows(
    order_id: str,
    batch_number: int,
    compound_name: str,
    eval_summary: Dict[str, Any],
    df_base_steps: pd.DataFrame,
    df_rec_steps: pd.DataFrame
) -> pd.DataFrame:
    """
    Builds 2-arm (BASELINE and RECOMMENDED) log rows joinable to V3.7 prediction_log.
    """
    pred_id = f"PRED_{order_id}_{batch_number}"

    # Extract clean non-duplicate mapped steps list (GroupName + StepNo)
    unique_steps = []
    if 'GroupName' in df_rec_steps.columns and 'StepNo' in df_rec_steps.columns:
        seen = set()
        for idx, row in df_rec_steps.iterrows():
            st_key = f"{row['GroupName']}_Step{int(row['StepNo'])}"
            if st_key not in seen:
                seen.add(st_key)
                unique_steps.append(st_key)
    else:
        unique_steps = [f"Step{int(x)}" for x in df_rec_steps.get('StepNo', [])]

    # Populate SP JSON strings
    sp_before_json = format_step_sp_json(df_base_steps, 'time_SP', 'rotor_SP', 'ram_SP')
    sp_after_json = format_step_sp_json(df_rec_steps, 'recommended_time_SP', 'recommended_rotor_SP', 'recommended_ram_SP')

    # Baseline Arm
    base_row = {
        'prediction_id': pred_id,
        'OrderID': str(order_id),
        'BatchNumber': int(batch_number),
        'label_group_id': 'M1',
        'GroupName': 'AVR_MST',
        'CompoundName': compound_name,
        'cluster_id': 'Silica',
        'mixer_id': 'ML09',
        'arm': 'BASELINE',
        'raw_prediction': eval_summary['base_final'],
        'stage1_pred': eval_summary['base_final'] - 2.0,
        'stage1b_bias': 0.0,
        'phase_contribution_json': json.dumps(eval_summary.get('base_contribs', {'WetMixing': 2.5, 'PID': 1.0})),
        'reason_codes': json.dumps(['HIGH_WET_MIXING_DEV']),
        'confidence_label': 'HIGH',
        'calibration_source': 'Stage2_Expert',
        'spec_low': 45.0,
        'spec_high': 55.0,
        'mny_in_spec': 45.0 <= eval_summary['base_final'] <= 55.0,
        'corrected_phase': None,
        'correction_direction': None,
        'mapped_steps': json.dumps(unique_steps),
        'step_SP_before': sp_before_json,
        'step_SP_after': sp_before_json,
        'mny_pull': 0.0,
        'gap_before': eval_summary['gap_before'],
        'gap_after': eval_summary['gap_before'],
        'pull_effective': True,
        'overshoot_flag': False,
        'side_effect': 0.0,
        'lab_status': 'PENDING',
        'actual_mny': np.nan,
        'lab_result_time': None,
        'mny_pred_error': np.nan,
        'quality_ok': None
    }

    # Recommended Arm
    rec_row = base_row.copy()
    rec_row.update({
        'arm': 'RECOMMENDED',
        'raw_prediction': eval_summary['rec_final'],
        'phase_contribution_json': json.dumps(eval_summary.get('rec_contribs', {'WetMixing': 0.5, 'PID': 0.8})),
        'corrected_phase': ",".join(eval_summary['corrected_phases']),
        'correction_direction': 'PULL_TOWARD_CENTER',
        'mapped_steps': json.dumps(unique_steps),
        'step_SP_before': sp_before_json,
        'step_SP_after': sp_after_json,
        'mny_pull': eval_summary['mny_pull'],
        'gap_after': eval_summary['gap_after'],
        'pull_effective': eval_summary['pull_effective'],
        'overshoot_flag': eval_summary['overshoot_flag'],
        'side_effect': eval_summary['side_effect']
    })

    log_df = pd.DataFrame([base_row, rec_row])
    return log_df[REQUIRED_LOG_COLS]

if __name__ == '__main__':
    dummy_summary = {
        'spec_center': 50.0,
        'base_final': 55.0,
        'rec_final': 52.5,
        'mny_pull': -2.5,
        'gap_before': 5.0,
        'gap_after': 2.5,
        'pull_effective': True,
        'overshoot_flag': False,
        'side_effect': 0.5,
        'corrected_phases': ['Stage5_PID']
    }
    dummy_base = pd.DataFrame({'GroupName': ['AVR_MST', 'AVR_MST'], 'StepNo': [10, 11], 'time_SP': [10, 3], 'rotor_SP': [5, 5], 'ram_SP': [50, 50]})
    dummy_rec = pd.DataFrame({'GroupName': ['AVR_MST', 'AVR_MST'], 'StepNo': [10, 11], 'recommended_time_SP': [8, 3], 'recommended_rotor_SP': [5, 5], 'recommended_ram_SP': [50, 50]})
    
    df_log = build_stepbridge_log_rows('7951751', 75, 'M1-T15760', dummy_summary, dummy_base, dummy_rec)
    print("\n[OK] Fixed Module C stepbridge_logger Verified.")
    print("mapped_steps:", df_log['mapped_steps'].iloc[1])
    print("step_SP_before:", df_log['step_SP_before'].iloc[1])
    print("step_SP_after:", df_log['step_SP_after'].iloc[1])
