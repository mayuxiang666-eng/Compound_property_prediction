# ============================================================================
# Step 1 & 2: Energy Label Builder & Feature Engineering Module (V4.1)
# ============================================================================
# Constructs energy labels:
# - total_kwh_per_batch (meter reading difference or power curve integral / 3600.0)
# - top_mixer_kwh (Stages 1-5) and bottom_mixer_kwh (Stage 6)
# - stage_kwh dictionary (Stages 1 to 6)
# - kwh_per_ton = total_kwh_per_batch / batch_weight_ton
# Segregates by Material System (Silica / CarbonBlack) and Phase Route (OilWet / NoOilDry).
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

from feature_engineering.clustering import cluster_silica_carbon_black

STAGE_POWER_INT_COLS = [
    'Stage1_Loading_power_Integral',
    'Stage2_DryMixing_power_Integral',
    'Stage3_OilLoading_power_Integral',
    'Stage4_WetMixing_power_Integral',
    'Stage5_PID_power_Integral',
    'Stage6_BottomMixing_power_Integral',
]

STAGE_DURATION_COLS = [
    'Stage1_Loading_Duration',
    'Stage2_DryMixing_Duration',
    'Stage3_OilLoading_Duration',
    'Stage4_WetMixing_Duration',
    'Stage5_PID_Duration',
    'Stage6_BottomMixing_Duration',
]

CONTROLLABLE_SETPOINT_COLS = [
    'Stage2_DryMixing_Duration',
    'Stage2_DryMixing_RotorSpeed_Mean',
    'Stage4_WetMixing_Duration',
    'Stage5_PID_Duration',
    'Stage5_PID_temp_Mean',
    'Stage6_BottomMixing_Duration',
    'Target_Temperature',
]


def build_energy_labels_and_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs energy labels and extracts controllable setpoint vs actual process features.
    """
    df = df_in.copy()
    df = cluster_silica_carbon_black(df)

    # 1. Determine Phase Route (OilWet vs NoOilDry)
    if 'is_oil_loading_present' in df.columns:
        has_oil = pd.to_numeric(df['is_oil_loading_present'], errors='coerce').fillna(0) > 0
    else:
        has_oil = (df['Stage3_OilLoading_Duration'].fillna(0) > 0) | (df['Stage4_WetMixing_Duration'].fillna(0) > 0)

    df['phase_route'] = np.where(has_oil, 'OilWet', 'NoOilDry')
    df['system_route_branch'] = df['material_system'] + '_' + df['phase_route']

    # 2. Compute Stage kWh
    for col in STAGE_POWER_INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    df['top_mixer_kwh'] = (
        df['Stage1_Loading_power_Integral'] +
        df['Stage2_DryMixing_power_Integral'] +
        df['Stage3_OilLoading_power_Integral'] +
        df['Stage4_WetMixing_power_Integral'] +
        df['Stage5_PID_power_Integral']
    ) / 3600.0

    df['bottom_mixer_kwh'] = df['Stage6_BottomMixing_power_Integral'] / 3600.0

    # 3. Total kWh per batch (Meter reading if present, power curve integral fallback)
    if 'meter_end_kwh' in df.columns and 'meter_start_kwh' in df.columns:
        meter_kwh = df['meter_end_kwh'] - df['meter_start_kwh']
        valid_meter = (meter_kwh > 5.0) & (meter_kwh < 200.0)
        df['total_kwh_per_batch'] = np.where(valid_meter, meter_kwh, df['top_mixer_kwh'] + df['bottom_mixer_kwh'])
        df['label_source'] = np.where(valid_meter, 'meter_reading', 'power_integral_fallback')
    else:
        df['total_kwh_per_batch'] = df['top_mixer_kwh'] + df['bottom_mixer_kwh']
        df['label_source'] = 'power_integral_fallback'

    # 4. Batch Weight in Tons & kWh/ton
    if 'batch_weight_ton' in df.columns:
        df['batch_weight_ton'] = pd.to_numeric(df['batch_weight_ton'], errors='coerce')
    else:
        # Standard industrial internal mixer volume & fill factor proxy (250L chamber -> ~0.22 - 0.28 tons per batch)
        top_fill = pd.to_numeric(df.get('Top_Fill_Factor', 70.0), errors='coerce').fillna(70.0)
        top_fill = np.where(top_fill > 1.0, top_fill / 100.0, top_fill)
        df['batch_weight_ton'] = np.clip(0.25 * (top_fill / 0.70), 0.15, 0.40)

    df['kwh_per_ton'] = df['total_kwh_per_batch'] / df['batch_weight_ton']

    # 5. Data Quality Flags
    df['label_quality_flag'] = np.where(
        (df['total_kwh_per_batch'] >= 10.0) & (df['total_kwh_per_batch'] <= 120.0),
        'VALID',
        'OUTLIER'
    )

    return df


def audit_energy_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates summary audit of constructed energy labels.
    """
    summary_rows = []
    for branch, grp in df.groupby('system_route_branch'):
        summary_rows.append({
            'system_route_branch': branch,
            'count': len(grp),
            'mean_kwh_per_batch': round(float(grp['total_kwh_per_batch'].mean()), 2),
            'std_kwh_per_batch': round(float(grp['total_kwh_per_batch'].std()), 2),
            'min_kwh_per_batch': round(float(grp['total_kwh_per_batch'].min()), 2),
            'max_kwh_per_batch': round(float(grp['total_kwh_per_batch'].max()), 2),
            'mean_kwh_per_ton': round(float(grp['kwh_per_ton'].mean()), 2),
            'std_kwh_per_ton': round(float(grp['kwh_per_ton'].std()), 2),
        })
    return pd.DataFrame(summary_rows)
