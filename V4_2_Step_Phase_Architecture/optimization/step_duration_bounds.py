"""Attach historical Step duration and control-parameter bounds for recommendation search."""

from __future__ import annotations

import pandas as pd


TIME_SQL = """
SELECT
    o.CompoundName,
    bd.OrderID,
    bd.BatchNumber,
    bd.EquipmentID,
    bd.GroupName,
    bd.StepNo,
    MAX(CASE WHEN bd.VariablePath='SCP-1-Step-Time-rel-s' THEN bd.Target END) AS time_SP
FROM [SFEPlant].[dbo].[BatchData] bd
JOIN [SFEPlant].[dbo].[Orders] o
    ON o.OrderID = bd.OrderID
WHERE o.CompoundName = ?
  AND bd.GroupName IN ('AVR_MST', 'AVR_MSB')
  AND bd.VariablePath = 'SCP-1-Step-Time-rel-s'
GROUP BY o.CompoundName, bd.OrderID, bd.BatchNumber, bd.EquipmentID, bd.GroupName, bd.StepNo
"""


TEMP_SQL = """
SELECT
        o.CompoundName,
        bd.EquipmentID,
        bd.GroupName,
        bd.StepNo,
        CAST(bd.Target AS FLOAT) AS temp_SP
FROM [SFEPlant].[dbo].[BatchData] bd
JOIN [SFEPlant].[dbo].[Orders] o
        ON o.OrderID = bd.OrderID
WHERE o.CompoundName = ?
    AND bd.GroupName IN ('AVR_MST', 'AVR_MSB')
    AND bd.VariablePath = 'SCP-3-Temperature-C-F'
    AND bd.Target IS NOT NULL
"""


def attach_historical_step_duration_bounds(
    df_steps: pd.DataFrame,
    compound_name: str,
    conn,
    *,
    exact_min_samples: int = 5,
    fallback_min_samples: int = 10,
    percentile_low: float = 0.05,
    percentile_high: float = 0.95,
) -> pd.DataFrame:
    """Attach per-step duration bounds from historical batches of the same compound.

    Priority:
    1. Same CompoundName + EquipmentID + GroupName + StepNo
    2. Same CompoundName + GroupName + StepNo across any equipment
    """
    history = pd.read_sql(TIME_SQL, conn, params=[compound_name])
    if history.empty:
        out = df_steps.copy()
        out['step_time_min'] = pd.NA
        out['step_time_max'] = pd.NA
        out['step_time_nominal'] = pd.NA
        out['step_time_sample_count'] = 0
        out['step_time_bounds_source'] = 'UNAVAILABLE'
        return out

    history['time_SP'] = pd.to_numeric(history['time_SP'], errors='coerce')
    history = history.dropna(subset=['time_SP']).sort_values(
        ['OrderID', 'BatchNumber', 'EquipmentID', 'GroupName', 'StepNo'],
        kind='stable',
    )
    history['dur_SP'] = history['time_SP'] - history.groupby(
        ['OrderID', 'BatchNumber', 'EquipmentID', 'GroupName']
    )['time_SP'].shift(1, fill_value=0.0)
    history = history[history['dur_SP'] >= 0].copy()

    exact = history.groupby(['EquipmentID', 'GroupName', 'StepNo']).agg(
        step_time_min=('dur_SP', lambda values: values.quantile(percentile_low)),
        step_time_max=('dur_SP', lambda values: values.quantile(percentile_high)),
        step_time_nominal=('dur_SP', 'median'),
        step_time_sample_count=('dur_SP', 'count'),
    ).reset_index()
    exact = exact[exact['step_time_sample_count'] >= exact_min_samples].copy()
    exact['step_time_bounds_source'] = 'EXACT_COMPOUND_EQUIPMENT_STEP'

    fallback = history.groupby(['GroupName', 'StepNo']).agg(
        step_time_min=('dur_SP', lambda values: values.quantile(percentile_low)),
        step_time_max=('dur_SP', lambda values: values.quantile(percentile_high)),
        step_time_nominal=('dur_SP', 'median'),
        step_time_sample_count=('dur_SP', 'count'),
    ).reset_index()
    fallback = fallback[fallback['step_time_sample_count'] >= fallback_min_samples].copy()
    fallback['step_time_bounds_source'] = 'SAME_COMPOUND_ANY_EQUIPMENT_STEP'

    out = df_steps.copy()
    out = out.merge(exact, on=['EquipmentID', 'GroupName', 'StepNo'], how='left')
    fallback_merge = fallback.rename(columns={
        'step_time_min': 'fallback_step_time_min',
        'step_time_max': 'fallback_step_time_max',
        'step_time_nominal': 'fallback_step_time_nominal',
        'step_time_sample_count': 'fallback_step_time_sample_count',
        'step_time_bounds_source': 'fallback_step_time_bounds_source',
    })
    out = out.merge(fallback_merge, on=['GroupName', 'StepNo'], how='left')

    for column in ('step_time_min', 'step_time_max', 'step_time_nominal', 'step_time_sample_count', 'step_time_bounds_source'):
        fallback_column = f'fallback_{column}'
        out[column] = out[column].combine_first(out[fallback_column])

    fallback_columns = [column for column in out.columns if column.startswith('fallback_')]
    out = out.drop(columns=fallback_columns)

    temp_history = pd.read_sql(TEMP_SQL, conn, params=[compound_name])
    if temp_history.empty:
        out['step_temp_min'] = pd.NA
        out['step_temp_max'] = pd.NA
        out['step_temp_nominal'] = pd.NA
        out['step_temp_sample_count'] = 0
        out['step_temp_bounds_source'] = 'UNAVAILABLE'
        return out

    temp_history['temp_SP'] = pd.to_numeric(temp_history['temp_SP'], errors='coerce')
    temp_history = temp_history.dropna(subset=['temp_SP']).copy()

    temp_exact = temp_history.groupby(['EquipmentID', 'GroupName', 'StepNo']).agg(
        step_temp_min=('temp_SP', lambda values: values.quantile(percentile_low)),
        step_temp_max=('temp_SP', lambda values: values.quantile(percentile_high)),
        step_temp_nominal=('temp_SP', 'median'),
        step_temp_sample_count=('temp_SP', 'count'),
    ).reset_index()
    temp_exact = temp_exact[temp_exact['step_temp_sample_count'] >= exact_min_samples].copy()
    temp_exact['step_temp_bounds_source'] = 'EXACT_COMPOUND_EQUIPMENT_STEP'

    temp_fallback = temp_history.groupby(['GroupName', 'StepNo']).agg(
        step_temp_min=('temp_SP', lambda values: values.quantile(percentile_low)),
        step_temp_max=('temp_SP', lambda values: values.quantile(percentile_high)),
        step_temp_nominal=('temp_SP', 'median'),
        step_temp_sample_count=('temp_SP', 'count'),
    ).reset_index()
    temp_fallback = temp_fallback[temp_fallback['step_temp_sample_count'] >= fallback_min_samples].copy()
    temp_fallback['step_temp_bounds_source'] = 'SAME_COMPOUND_ANY_EQUIPMENT_STEP'

    out = out.merge(temp_exact, on=['EquipmentID', 'GroupName', 'StepNo'], how='left')
    temp_fallback_merge = temp_fallback.rename(columns={
        'step_temp_min': 'fallback_step_temp_min',
        'step_temp_max': 'fallback_step_temp_max',
        'step_temp_nominal': 'fallback_step_temp_nominal',
        'step_temp_sample_count': 'fallback_step_temp_sample_count',
        'step_temp_bounds_source': 'fallback_step_temp_bounds_source',
    })
    out = out.merge(temp_fallback_merge, on=['GroupName', 'StepNo'], how='left')

    for column in ('step_temp_min', 'step_temp_max', 'step_temp_nominal', 'step_temp_sample_count', 'step_temp_bounds_source'):
        fallback_column = f'fallback_{column}'
        out[column] = out[column].combine_first(out[fallback_column])

    temp_fallback_columns = [column for column in out.columns if column.startswith('fallback_step_temp_')]
    return out.drop(columns=temp_fallback_columns)