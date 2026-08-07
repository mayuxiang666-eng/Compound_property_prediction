"""
V4.2 Module G: Output Builder
Generates engineer-executable step-level recommendation tables with audit fields.
"""
import pandas as pd
import numpy as np

REQUIRED_OUTPUT_COLUMNS = [
    'OrderID', 'BatchNumber', 'GroupName', 'StepNo',
    'mapped_phase',
    'current_time_SP', 'recommended_time_SP', 'delta_time_SP',
    'current_rotor_SP', 'recommended_rotor_SP', 'delta_rotor_SP',
    'current_ram_SP', 'recommended_ram_SP', 'delta_ram_SP',
    'current_temp_SP', 'recommended_temp_SP', 'delta_temp_SP',
    'phase_consistency_status', 'model_validation_status',
    'recommendation_status',
    'engineer_decision', 'trial_status', 'final_decision',
]

def build_step_recommendation_output(validated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the engineer-executable step recommendation table.
    """
    out = validated_df.copy()

    # Map current SP columns
    rename_map = {
        'time_SP': 'current_time_SP',
        'rotor_SP': 'current_rotor_SP',
        'ram_SP': 'current_ram_SP',
        'temp_SP': 'current_temp_SP',
    }
    out.rename(columns=rename_map, inplace=True)

    # Compute deltas if missing
    for p in ['time', 'rotor', 'ram', 'temp']:
        rec_col = f'recommended_{p}_SP'
        cur_col = f'current_{p}_SP'
        delta_col = f'delta_{p}_SP'
        if rec_col not in out.columns:
            out[rec_col] = out[cur_col]
        if delta_col not in out.columns:
            out[delta_col] = out[rec_col] - out[cur_col]

    # Audit & Engineer decision fields
    out['engineer_decision'] = 'PENDING'
    out['trial_status'] = 'NOT_STARTED'
    out['final_decision'] = 'PENDING'

    # Fill missing required columns
    for col in REQUIRED_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = None

    return out[REQUIRED_OUTPUT_COLUMNS]

if __name__ == '__main__':
    import os, sys
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
    from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
    from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags
    from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping
    from V4_2_Step_Phase_Architecture.optimization.phase_to_step_decomposer import decompose_phase_to_steps

    conn = get_sfe_connection()
    df_wide = pivot_batchdata_to_step_wide(conn, order_ids=['7951751'], batch_numbers=[75])
    df_flags = build_settable_flags(df_wide)
    df_mapped = build_step_phase_mapping(df_flags, target_temp=165.0)
    df_decomp = decompose_phase_to_steps({'Stage5_PID_Duration': 10.0}, df_mapped)
    df_decomp['phase_consistency_status'] = 'CONSISTENT'
    df_decomp['model_validation_status'] = 'PASSED'
    df_decomp['recommendation_status'] = 'RECOMMENDED'
    
    df_out = build_step_recommendation_output(df_decomp)
    print(f"[OK] Task 9: Output Builder verified. Output shape: {df_out.shape}")
    print(df_out[['GroupName', 'StepNo', 'mapped_phase', 'current_time_SP', 'recommended_time_SP', 'delta_time_SP', 'recommendation_status', 'engineer_decision']].head(5).to_string(index=False))
    conn.close()
