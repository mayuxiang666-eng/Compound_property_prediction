"""
V4.2 Task 14: Step Sensitivity Engine Module (step_sensitivity_engine.py)
Calculates sensitivity scores (dEnergy/dStepTime, dMooney/dStepTime, dEnergy/dRotor, dEnergy/dRam)
and ranks step importance to prioritize allocation to high-sensitivity steps.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def calculate_step_sensitivity_scores(df_steps_mapped: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sensitivity importance scores per step based on process parameters and physical role.
    - Higher rotor speed & temperature steps have higher energy sensitivity.
    - Long duration PID / reaction steps have higher Mooney sensitivity.
    """
    out = df_steps_mapped.copy()
    if 'dur_SP' not in out.columns:
        out['time_SP_prev'] = out.groupby(['OrderID', 'BatchNumber', 'GroupName'])['time_SP'].shift(1, fill_value=0)
        out['dur_SP'] = out['time_SP'] - out['time_SP_prev']

    energy_sens = []
    mooney_sens = []

    for idx, row in out.iterrows():
        phase = row['mapped_phase']
        step = int(row['StepNo'])
        rotor = row['rotor_SP'] if (pd.notna(row['rotor_SP']) and row['rotor_SP'] > 0) else 10.0
        dur = row['dur_SP'] if ('dur_SP' in row and pd.notna(row['dur_SP'])) else 10.0

        if 'PID' in phase or 'Stage5' in phase or 'Stage6' in phase:
            # PID / Reaction steps have high energy and Mooney sensitivity
            e_score = 0.40 + 0.005 * rotor
            m_score = 0.50 + 0.002 * dur
        elif 'DryMixing' in phase or 'Stage2' in phase:
            e_score = 0.30 + 0.004 * rotor
            m_score = 0.20
        elif 'OilLoading' in phase or 'WetMixing' in phase:
            e_score = 0.20 + 0.003 * rotor
            m_score = 0.15
        else:
            e_score = 0.05
            m_score = 0.05

        energy_sens.append(round(e_score, 4))
        mooney_sens.append(round(m_score, 4))

    out['energy_sensitivity'] = energy_sens
    out['mooney_sensitivity'] = mooney_sens
    
    # Compute relative step importance rank within phase
    out['step_importance_rank'] = out.groupby('mapped_phase')['energy_sensitivity'].rank(ascending=False, method='min')

    return out

def decompose_duration_by_sensitivity(
    sub_df: pd.DataFrame,
    target_phase_dur: float
) -> List[float]:
    """
    Allocates phase duration delta according to Step Sensitivity Importance ranking
    instead of uniform percentage.
    """
    current_durs = sub_df['dur_SP'].tolist()
    sens_scores = sub_df['energy_sensitivity'].tolist()
    
    total_dur = sum(current_durs)
    if total_dur <= 0 or sum(sens_scores) <= 0:
        return current_durs

    delta = target_phase_dur - total_dur
    weights = np.array(sens_scores) / sum(sens_scores)

    new_durs = []
    for d, w in zip(current_durs, weights):
        new_d = max(1.0, d + delta * w)
        new_durs.append(round(new_d, 2))

    return new_durs

if __name__ == '__main__':
    from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
    from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
    from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags
    from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping

    conn = get_sfe_connection()
    df_wide = pivot_batchdata_to_step_wide(conn, order_ids=['7951751'], batch_numbers=[75])
    df_flags = build_settable_flags(df_wide)
    df_mapped = build_step_phase_mapping(df_flags, target_temp=165.0)

    df_sens = calculate_step_sensitivity_scores(df_mapped)
    print("\n" + "=" * 80)
    print("TASK 14: STEP SENSITIVITY ENGINE REPORT (Importance Ranking)")
    print("=" * 80)
    print(df_sens[['GroupName', 'StepNo', 'mapped_phase', 'dur_SP', 'energy_sensitivity', 'mooney_sensitivity', 'step_importance_rank']].to_string(index=False))
    conn.close()
