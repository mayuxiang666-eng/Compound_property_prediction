"""
V4.2 Module A1: Settable Flag Builder
Determines which parameters are settable (recommendable) per step and calculates SP-AV tracking errors.
"""
import pandas as pd
import numpy as np

def build_settable_flags(df_step_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Adds is_*_settable boolean flags and *_track_err tracking error columns to step-wide table.
    """
    out = df_step_wide.copy()

    # Settable Flags
    out['is_time_settable'] = out['time_SP'].notna() & (out['time_SP'] > 0)
    out['is_temp_settable'] = out['temp_SP'].notna() & (out['temp_SP'] > 0)
    out['is_rotor_settable'] = out['rotor_SP'].notna() & (out['rotor_SP'] > 0)
    out['is_ram_settable'] = out['ram_SP'].notna() & (out['ram_SP'] > 0)

    # Tracking Errors (AV - SP)
    out['time_track_err'] = np.where(out['is_time_settable'], out['time_AV'] - out['time_SP'], np.nan)
    out['temp_track_err'] = np.where(out['is_temp_settable'], out['temp_AV'] - out['temp_SP'], np.nan)
    out['rotor_track_err'] = np.where(out['is_rotor_settable'], out['rotor_AV'] - out['rotor_SP'], np.nan)
    out['ram_track_err'] = np.where(out['is_ram_settable'], out['ram_AV'] - out['ram_SP'], np.nan)

    return out

if __name__ == '__main__':
    import os, sys
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
    from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
    
    conn = get_sfe_connection()
    df_wide = pivot_batchdata_to_step_wide(conn, order_ids=['7951751'], batch_numbers=[75])
    df_flags = build_settable_flags(df_wide)
    print(f"[OK] Task 3: Settable Flag Builder verified. Columns added: {[c for c in df_flags.columns if 'is_' in c or 'track_' in c]}")
    print(df_flags[['GroupName', 'StepNo', 'is_temp_settable', 'temp_SP', 'temp_AV', 'temp_track_err']].to_string(index=False))
    conn.close()
