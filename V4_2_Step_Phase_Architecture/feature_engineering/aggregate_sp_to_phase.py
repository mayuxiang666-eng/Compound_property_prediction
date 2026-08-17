"""
V4.2 Module C: Step SP to Phase Feature Aggregation
Aggregates step-level SP values into phase-level features matching V4.1 model column names.
Calculates step durations as deltas: dur[i] = cum_time[i] - cum_time[i-1].
"""
import pandas as pd
import numpy as np

# Map V4.2 mapped_phase to V4.1 expected column prefixes
PHASE_PREFIX_MAP = {
    'Stage1_Loading': 'Stage1_Loading',
    'Stage2_DryMixing': 'Stage2_DryMixing',
    'Stage3_OilLoading': 'Stage3_OilLoading',
    'Stage4_WetMixing_Ramp': 'Stage4_WetMixing',
    'Stage5_MST_PID_Silanization': 'Stage5_PID',
}

def aggregate_steps_to_phase_features(df_mapped: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates step SP values into phase-level features compatible with V4.1 models.
    Strict Mode B policy: uses only Target (SP) values — never current-batch AV.
    """
    df = df_mapped.copy().sort_values(
        ['OrderID', 'BatchNumber', 'GroupName', 'StepNo'],
        kind='stable',
    )
    
    # Calculate step durations from cumulative time_SP per (OrderID, BatchNumber, GroupName)
    time_col = 'recommended_time_SP' if 'recommended_time_SP' in df.columns else 'time_SP'
    df['time_SP_prev'] = df.groupby(['OrderID', 'BatchNumber', 'GroupName'])[time_col].shift(1, fill_value=0)
    df['dur_SP'] = df[time_col] - df['time_SP_prev']
    if (df['dur_SP'] < 0).any():
        raise ValueError(f'{time_col} must be non-decreasing within each mixer group.')

    results = []
    group_keys = ['OrderID', 'BatchNumber']
    
    for (oid, bn), b_df in df.groupby(group_keys):
        row = {'OrderID': oid, 'BatchNumber': bn}
        
        # 1. MST Phase Aggregations
        mst = b_df[b_df['GroupName'] == 'AVR_MST']
        
        for mapped_p, v41_prefix in PHASE_PREFIX_MAP.items():
            p_steps = mst[mst['mapped_phase'] == mapped_p]
            
            # Duration sum
            dur_sum = p_steps['dur_SP'].sum() if not p_steps.empty else 0.0
            row[f'{v41_prefix}_Duration'] = float(dur_sum)
            
            # Rotor speed (duration-weighted average)
            r_set = p_steps[p_steps['is_rotor_settable'] == True]
            if not r_set.empty and r_set['dur_SP'].sum() > 0:
                weights = r_set['dur_SP'].fillna(0)
                row[f'{v41_prefix}_RotorSpeed_Mean'] = float(np.average(r_set['rotor_SP'].fillna(0), weights=weights))
            else:
                row[f'{v41_prefix}_RotorSpeed_Mean'] = 0.0
                
            # Temperature (Target_Temperature for PID phase)
            t_set = p_steps[p_steps['is_temp_settable'] == True]
            if not t_set.empty:
                t_val = float(t_set['temp_SP'].max())
                row[f'{v41_prefix}_temp_Mean'] = t_val
                if v41_prefix == 'Stage5_PID':
                    row['Target_Temperature'] = t_val
            else:
                row[f'{v41_prefix}_temp_Mean'] = 0.0
                if v41_prefix == 'Stage5_PID' and 'Target_Temperature' not in row:
                    row['Target_Temperature'] = 0.0

        # 2. MSB Phase Aggregations (Stage 6: BottomMixing)
        msb = b_df[b_df['GroupName'] == 'AVR_MSB']
        msb_dur_sum = msb['dur_SP'].sum() if not msb.empty else 0.0
        row['Stage6_BottomMixing_Duration'] = float(msb_dur_sum)
        
        msb_r_set = msb[msb['is_rotor_settable'] == True]
        if not msb_r_set.empty and msb_r_set['dur_SP'].sum() > 0:
            weights = msb_r_set['dur_SP'].fillna(0)
            row['Stage6_BottomMixing_RotorSpeed_Mean'] = float(np.average(msb_r_set['rotor_SP'].fillna(0), weights=weights))
        else:
            row['Stage6_BottomMixing_RotorSpeed_Mean'] = 0.0
            
        results.append(row)
        
    return pd.DataFrame(results)

if __name__ == '__main__':
    import os, sys
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
    from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
    from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags
    from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping

    conn = get_sfe_connection()
    df_wide = pivot_batchdata_to_step_wide(conn, order_ids=['7951751'], batch_numbers=[75])
    df_flags = build_settable_flags(df_wide)
    df_mapped = build_step_phase_mapping(df_flags, target_temp=165.0)
    df_phase = aggregate_steps_to_phase_features(df_mapped)
    
    print(f"[OK] Task 5: Step SP to Phase Aggregation verified.")
    print(df_phase.to_string(index=False))
    conn.close()
