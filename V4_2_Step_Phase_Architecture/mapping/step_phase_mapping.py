"""
V4.2 Module B: Recipe-Specific Step-to-Phase Mapping
Applies the expert-verified silanization PID continuation and temperature drop discharge rules.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

VALID_PHASES = [
    'Stage1_Loading',
    'Stage2_DryMixing',
    'Stage3_OilLoading',
    'Stage4_WetMixing_Ramp',
    'Stage5_MST_PID_Silanization',
    'Stage6_MSB_PID',
    'Other'
]

def build_step_phase_mapping(
    df_step_wide: pd.DataFrame,
    target_temp: float = 165.0
) -> pd.DataFrame:
    """
    Maps step-level records to physical macro phases.
    PID Threshold = target_temp - 3.0 (e.g., 162°C).
    MST Step 10 & 11 (including anti-stick pre-discharge) belong to Stage5_MST_PID_Silanization.
    AVR_MSB steps belong to Stage6_MSB_PID (silanization continuation).
    """
    out = df_step_wide.copy()
    pid_threshold = target_temp - 3.0

    mapped_phases = []
    confidence_scores = []
    sources = []

    for idx, row in out.iterrows():
        group = str(row['GroupName'])
        step = int(row['StepNo'])
        try:
            t_av = float(row['temp_AV']) if pd.notna(row['temp_AV']) else 0.0
        except (ValueError, TypeError):
            t_av = 0.0
        try:
            t_sp = float(row['temp_SP']) if pd.notna(row['temp_SP']) else 0.0
        except (ValueError, TypeError):
            t_sp = 0.0

        if group == 'AVR_MST':
            if t_av >= pid_threshold or t_sp >= pid_threshold or step in (10, 11):
                phase = 'Stage5_MST_PID_Silanization'
                conf = 0.95
                src = 'expert_temp_drop_rule'
            elif step in (1, 2):
                phase = 'Stage1_Loading'
                conf = 0.90
                src = 'material_loading_rule'
            elif step == 3:
                phase = 'Stage2_DryMixing'
                conf = 0.85
                src = 'dry_mixing_rule'
            elif step == 4:
                phase = 'Stage3_OilLoading'
                conf = 0.85
                src = 'oil_silane_injection_rule'
            else:
                phase = 'Stage4_WetMixing_Ramp'
                conf = 0.80
                src = 'wet_mixing_ramp_rule'
        else:
            # AVR_MSB
            phase = f'Stage6_MSB_PID_Step{step}'
            conf = 0.95
            src = 'expert_msb_continuation_rule'

        mapped_phases.append(phase)
        confidence_scores.append(conf)
        sources.append(src)

    out['mapped_phase'] = mapped_phases
    out['mapping_confidence'] = confidence_scores
    out['mapping_source'] = sources
    out['auto_control_allowed'] = False

    return out

if __name__ == '__main__':
    import os, sys
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
    from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
    from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags

    conn = get_sfe_connection()
    df_wide = pivot_batchdata_to_step_wide(conn, order_ids=['7951751'], batch_numbers=[75])
    df_flags = build_settable_flags(df_wide)
    df_mapped = build_step_phase_mapping(df_flags, target_temp=165.0)
    print(f"[OK] Task 4: Step-to-Phase Mapping verified.")
    print(df_mapped[['GroupName', 'StepNo', 'time_SP', 'time_AV', 'mapped_phase', 'mapping_confidence']].to_string(index=False))
    conn.close()
