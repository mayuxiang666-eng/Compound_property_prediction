"""Counterfactual Target-SP aggregation for V3.7 recommendation scoring.

This adapter intentionally consumes ``recommended_*_SP`` when present.  The
legacy V4.2 aggregate module is retained unchanged for compatibility; new
recommendation searches use this adapter until it is migrated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PHASE_PREFIX_MAP = {
    'Stage1_Loading': 'Stage1_Loading',
    'Stage2_DryMixing': 'Stage2_DryMixing',
    'Stage3_OilLoading': 'Stage3_OilLoading',
    'Stage4_WetMixing_Ramp': 'Stage4_WetMixing',
    'Stage5_MST_PID_Silanization': 'Stage5_PID',
}


def aggregate_counterfactual_phase_features(steps: pd.DataFrame) -> pd.DataFrame:
    """Aggregate an entire candidate Step vector using Target values only."""
    df = steps.copy().sort_values(['OrderID', 'BatchNumber', 'GroupName', 'StepNo'], kind='stable')
    time_col = 'recommended_time_SP' if 'recommended_time_SP' in df else 'time_SP'
    rotor_col = 'recommended_rotor_SP' if 'recommended_rotor_SP' in df else 'rotor_SP'
    temp_col = 'recommended_temp_SP' if 'recommended_temp_SP' in df else 'temp_SP'
    ram_col = 'recommended_ram_SP' if 'recommended_ram_SP' in df else 'ram_SP'
    df['counterfactual_dur_SP'] = df[time_col] - df.groupby(['OrderID', 'BatchNumber', 'GroupName'])[time_col].shift(1, fill_value=0)
    if df['counterfactual_dur_SP'].isna().any() or (df['counterfactual_dur_SP'] < 0).any():
        raise ValueError('Candidate cumulative time_SP must be non-decreasing.')

    rows = []
    for (order_id, batch_number), batch in df.groupby(['OrderID', 'BatchNumber'], sort=False):
        row = {'OrderID': order_id, 'BatchNumber': batch_number}
        mst = batch[batch['GroupName'].eq('AVR_MST')]
        for phase, prefix in PHASE_PREFIX_MAP.items():
            phase_steps = mst[mst['mapped_phase'].eq(phase)]
            row[f'{prefix}_Duration'] = float(phase_steps['counterfactual_dur_SP'].sum())
            for flag, source, suffix in (
                ('is_rotor_settable', rotor_col, 'RotorSpeed_Mean'),
                ('is_ram_settable', ram_col, 'WayofRam_Mean'),
            ):
                active = phase_steps[phase_steps[flag].fillna(False).astype(bool)]
                if not active.empty and active['counterfactual_dur_SP'].sum() > 0:
                    row[f'{prefix}_{suffix}'] = float(np.average(active[source].fillna(0), weights=active['counterfactual_dur_SP']))
                else:
                    row[f'{prefix}_{suffix}'] = 0.0
            temp_active = phase_steps[phase_steps['is_temp_settable'].fillna(False).astype(bool)]
            temp_value = float(temp_active[temp_col].max()) if not temp_active.empty else 0.0
            row[f'{prefix}_temp_Mean'] = temp_value
            if prefix == 'Stage5_PID':
                row['Target_Temperature'] = temp_value

        msb = batch[batch['GroupName'].eq('AVR_MSB')]
        row['Stage6_BottomMixing_Duration'] = float(msb['counterfactual_dur_SP'].sum())
        for flag, source, suffix in (
            ('is_rotor_settable', rotor_col, 'RotorSpeed_Mean'),
            ('is_ram_settable', ram_col, 'WayofRam_Mean'),
        ):
            active = msb[msb[flag].fillna(False).astype(bool)]
            row[f'Stage6_BottomMixing_{suffix}'] = (
                float(np.average(active[source].fillna(0), weights=active['counterfactual_dur_SP']))
                if not active.empty and active['counterfactual_dur_SP'].sum() > 0 else 0.0
            )
        temp_active = msb[msb['is_temp_settable'].fillna(False).astype(bool)]
        row['Stage6_BottomMixing_temp_Mean'] = float(temp_active[temp_col].max()) if not temp_active.empty else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def merge_counterfactual_features(steps: pd.DataFrame, baseline_feature_row: pd.Series) -> pd.DataFrame:
    """Overlay Target-derived counterfactual Phase features onto a V3.7 baseline."""
    aggregate = aggregate_counterfactual_phase_features(steps)
    if len(aggregate) != 1:
        raise ValueError('Exactly one batch is required for counterfactual scoring.')
    output = baseline_feature_row.to_frame().T.copy()
    for column, value in aggregate.iloc[0].items():
        output.at[output.index[0], column] = value
    return output
