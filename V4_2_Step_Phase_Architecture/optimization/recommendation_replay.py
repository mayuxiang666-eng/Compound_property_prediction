"""
V4.2 Task 13: Recommendation Replay & Bounds Safety Testing (recommendation_replay.py)
Replays historical recommendations on recent batches (e.g. B00458, T15760).
Verifies that 100% of decomposed step SPs fall within engineering safe bounds:
time_min <= rec_time <= time_max
rotor_min <= rec_rotor <= rotor_max
ram_min <= rec_ram <= ram_max
Target: Pass Rate > 95%.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags
from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping
from V4_2_Step_Phase_Architecture.optimization.phase_to_step_decomposer import decompose_phase_to_steps

def run_recommendation_replay(order_ids: List[str] = None, conn=None) -> Dict[str, Any]:
    """Replays step recommendations and checks engineering boundary safety."""
    if conn is None:
        conn = get_sfe_connection()
        
    if order_ids is None:
        sql = "SELECT DISTINCT TOP 10 OrderID FROM [SFEPlant].[dbo].[BatchData] WHERE GroupName IN ('AVR_MST','AVR_MSB');"
        df_o = pd.read_sql(sql, conn)
        order_ids = df_o['OrderID'].tolist()

    print(f"[Replay] Running Recommendation Replay across {len(order_ids)} orders...")
    df_wide = pivot_batchdata_to_step_wide(conn, order_ids=order_ids)
    df_flags = build_settable_flags(df_wide)
    df_mapped = build_step_phase_mapping(df_flags, target_temp=165.0)

    total_steps = 0
    passed_steps = 0
    replay_records = []

    for (oid, bn), b_df in df_mapped.groupby(['OrderID', 'BatchNumber']):
        # Simulate a 10% reduction recommendation in PID stage duration
        targets = {'Stage5_PID_Duration': 10.0}
        decomp = decompose_phase_to_steps(targets, b_df)

        for idx, row in decomp.iterrows():
            total_steps += 1
            
            t_rec = row['recommended_time_SP']
            r_rec = row['recommended_rotor_SP']
            ram_rec = row['recommended_ram_SP']

            # Engineering Bounds Check
            t_ok = 1.0 <= t_rec <= 300.0
            r_ok = (not row['is_rotor_settable']) or (5.0 <= r_rec <= 70.0)
            ram_ok = (not row['is_ram_settable']) or (5.0 <= ram_rec <= 100.0)

            is_safe = t_ok and r_ok and ram_ok
            if is_safe:
                passed_steps += 1

            replay_records.append({
                'OrderID': oid,
                'BatchNumber': bn,
                'GroupName': row['GroupName'],
                'StepNo': row['StepNo'],
                'mapped_phase': row['mapped_phase'],
                't_rec': t_rec,
                'r_rec': r_rec,
                'ram_rec': ram_rec,
                'is_safe': is_safe
            })

    pass_rate_pct = (passed_steps / total_steps * 100.0) if total_steps > 0 else 0.0

    print("\n" + "=" * 80)
    print("TASK 13: RECOMMENDATION REPLAY BOUNDARY SAFETY REPORT")
    print("=" * 80)
    print(f"Total Steps Tested:    {total_steps}")
    print(f"Passed Safety Bounds:  {passed_steps}")
    print(f"Safety Pass Rate:      {pass_rate_pct:.2f}%")
    print(f"Target Pass Rate:      > 95.00%")
    print(f"Status:                {'PASS (Pass Rate > 95%)' if pass_rate_pct >= 95.0 else 'INVESTIGATE'}")

    return {
        'total_steps': total_steps,
        'passed_steps': passed_steps,
        'pass_rate_pct': pass_rate_pct,
        'records_df': pd.DataFrame(replay_records)
    }

if __name__ == '__main__':
    conn = get_sfe_connection()
    res = run_recommendation_replay(conn=conn)
    conn.close()
