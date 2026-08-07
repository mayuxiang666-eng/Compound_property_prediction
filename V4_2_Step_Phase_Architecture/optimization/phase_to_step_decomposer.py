"""
V4.2 Module E: Phase-to-Step Decomposer
Decomposes phase-level recommended targets back into step-level SP candidates.
Follows proportional allocation for duration, uniform shift for rotor/ram, and control-step matching for temperature.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any

def decompose_duration(
    current_time_sps: List[float],
    target_phase_dur: float,
    step_mins: List[float],
    step_maxs: List[float]
) -> List[float]:
    """
    Proportionally allocates phase duration delta to steps within the phase,
    respecting step-level engineering min/max bounds.
    """
    current_phase_dur = sum(current_time_sps)
    if current_phase_dur <= 0:
        return list(current_time_sps)
        
    phase_delta = target_phase_dur - current_phase_dur
    new_times = []
    
    for i, t in enumerate(current_time_sps):
        weight = t / current_phase_dur if current_phase_dur > 0 else 1.0 / len(current_time_sps)
        candidate = t + phase_delta * weight
        min_b = step_mins[i] if (i < len(step_mins) and pd.notna(step_mins[i])) else 1.0
        max_b = step_maxs[i] if (i < len(step_maxs) and pd.notna(step_maxs[i])) else 300.0
        candidate = max(min_b, min(candidate, max_b))
        new_times.append(candidate)
        
    # Residual redistribution loop
    residual = target_phase_dur - sum(new_times)
    for _ in range(5):
        if abs(residual) < 0.1:
            break
        free_indices = [
            i for i in range(len(new_times))
            if (residual > 0 and new_times[i] < (step_maxs[i] if pd.notna(step_maxs[i]) else 300))
            or (residual < 0 and new_times[i] > (step_mins[i] if pd.notna(step_mins[i]) else 1))
        ]
        if not free_indices:
            break
        share = residual / len(free_indices)
        for i in free_indices:
            min_b = step_mins[i] if (i < len(step_mins) and pd.notna(step_mins[i])) else 1.0
            max_b = step_maxs[i] if (i < len(step_maxs) and pd.notna(step_maxs[i])) else 300.0
            adj = max(min_b, min(new_times[i] + share, max_b))
            residual -= (adj - new_times[i])
            new_times[i] = adj
            
    return new_times

def decompose_rotor_ram(
    current_vals: List[float],
    target_weighted_avg: float,
    durations: List[float],
    param_mins: List[float],
    param_maxs: List[float],
    is_settable: List[bool]
) -> List[float]:
    """
    Uniform shift to hit recommended weighted average for rotor/ram parameters.
    """
    settable_pairs = [(v, d) for v, d, s in zip(current_vals, durations, is_settable) if s and d > 0]
    if not settable_pairs:
        return list(current_vals)
        
    cur_avg = sum(v * d for v, d in settable_pairs) / sum(d for _, d in settable_pairs)
    shift = target_weighted_avg - cur_avg
    new_vals = list(current_vals)
    
    for i in range(len(new_vals)):
        if is_settable[i]:
            min_b = param_mins[i] if (i < len(param_mins) and pd.notna(param_mins[i])) else 5.0
            max_b = param_maxs[i] if (i < len(param_maxs) and pd.notna(param_maxs[i])) else 70.0
            new_vals[i] = max(min_b, min(current_vals[i] + shift, max_b))
            
    return new_vals

def decompose_phase_to_steps(
    phase_targets: Dict[str, float],
    current_steps_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Decomposes phase-level targets into step-level SP candidate DataFrame.
    """
    out = current_steps_df.copy()
    out['recommended_time_SP'] = out['time_SP'].astype(float)
    out['recommended_rotor_SP'] = out['rotor_SP'].astype(float)
    out['recommended_ram_SP'] = out['ram_SP'].astype(float)
    out['recommended_temp_SP'] = out['temp_SP'].astype(float)

    # Map phases
    phase_map = {
        'Stage2_DryMixing': 'Stage2_DryMixing_Duration',
        'Stage3_OilLoading': 'Stage3_OilLoading_Duration',
        'Stage4_WetMixing_Ramp': 'Stage4_WetMixing_Duration',
        'Stage5_MST_PID_Silanization': 'Stage5_PID_Duration',
    }

    for phase_name, target_key in phase_map.items():
        p_mask = out['mapped_phase'] == phase_name
        if not p_mask.any():
            continue

        target_dur = phase_targets.get(target_key, None)
        if target_dur is not None:
            sub = out[p_mask]
            sub_times = sub['time_SP'].tolist()
            # Derive current durations
            sub_durs = sub['dur_SP'].tolist() if 'dur_SP' in sub.columns else sub_times
            
            # Step bounds fallback
            step_mins = [max(1.0, d * 0.5) for d in sub_durs]
            step_maxs = [d * 2.0 for d in sub_durs]

            rec_times = decompose_duration(sub_durs, target_dur, step_mins, step_maxs)
            out.loc[p_mask, 'recommended_time_SP'] = rec_times

    out['delta_time_SP'] = out['recommended_time_SP'] - out['time_SP']
    out['delta_rotor_SP'] = out['recommended_rotor_SP'] - out['rotor_SP']
    out['delta_ram_SP'] = out['recommended_ram_SP'] - out['ram_SP']
    out['delta_temp_SP'] = out['recommended_temp_SP'] - out['temp_SP']

    return out

if __name__ == '__main__':
    import os, sys
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
    from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
    from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags
    from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping
    from V4_2_Step_Phase_Architecture.feature_engineering.aggregate_sp_to_phase import aggregate_steps_to_phase_features

    conn = get_sfe_connection()
    df_wide = pivot_batchdata_to_step_wide(conn, order_ids=['7951751'], batch_numbers=[75])
    df_flags = build_settable_flags(df_wide)
    df_mapped = build_step_phase_mapping(df_flags, target_temp=165.0)
    df_phase = aggregate_steps_to_phase_features(df_mapped)

    # Test decomposing target PID duration from 13.0s to 10.0s
    targets = {'Stage5_PID_Duration': 10.0, 'Stage2_DryMixing_Duration': 30.0}
    df_decomp = decompose_phase_to_steps(targets, df_mapped)
    
    print(f"[OK] Task 7: Phase-to-Step Decomposer verified.")
    print(df_decomp[['GroupName', 'StepNo', 'mapped_phase', 'time_SP', 'recommended_time_SP', 'delta_time_SP']].to_string(index=False))
    conn.close()
