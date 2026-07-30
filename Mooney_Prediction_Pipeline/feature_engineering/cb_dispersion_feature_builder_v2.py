# ============================================================================
# Carbon Black Dispersion Kinetic Proxy Feature Builder V2 (EXPERIMENTAL CANDIDATE)
# ============================================================================
# STATUS: EXPERIMENTAL FEATURE CANDIDATE (NOT PROMOTED TO PRODUCTION)
# PRODUCTION BASELINE: cb_dispersion_feature_builder.py (CB_V1)
# REASON: A/B ablation experiment showed CB_A0 (V1) MAE=1.9108 vs CB_A4 (V2) MAE=1.9277.
# Feature importance alone is insufficient without generalization gain.
# ============================================================================

import numpy as np
import pandas as pd


def _get_series(df: pd.DataFrame, col_name: str, default_val: float = 0.0) -> pd.Series:
    """Helper to safely retrieve numeric series with default fallback."""
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors='coerce').fillna(default_val)
    return pd.Series(default_val, index=df.index, dtype=float)


def build_cb_dispersion_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs V2 physically refined dispersion work & kinetic proxies for Carbon Black compounds (EXPERIMENTAL)."""
    res = pd.DataFrame(index=df.index)

    d2_duration = _get_series(df, 'Stage2_DryMixing_Duration', 0.0)
    d2_power_mean = _get_series(df, 'Stage2_DryMixing_power_Mean', 0.0)
    d2_torque = _get_series(df, 'Stage2_DryMixing_Torque_Mean', 0.0)
    d6_torque = _get_series(df, 'Stage6_BottomMixing_Torque_Mean', 0.0)

    cb_oan = _get_series(df, 'supplier_carbon_black_structure_avg', 110.0)
    cb_stsa = _get_series(df, 'supplier_carbon_black_surface_area_avg', 85.0)

    # Feature 1: cb_dispersion_work_proxy
    res['cb_dispersion_work_proxy'] = d2_duration * np.maximum(d2_power_mean, 0.0)

    # Feature 2: cb_dispersion_completion_index (0 = little breakdown, 1 = strong breakdown)
    res['cb_dispersion_completion_index'] = (d2_torque - d6_torque) / np.maximum(d2_torque, 1.0)

    # Feature 3: cb_dispersion_difficulty & cb_effective_dispersion_energy
    oan_std = cb_oan.std() if len(cb_oan) > 1 and cb_oan.std() > 1e-5 else 1.0
    stsa_std = cb_stsa.std() if len(cb_stsa) > 1 and cb_stsa.std() > 1e-5 else 1.0
    z_oan = (cb_oan - cb_oan.mean()) / oan_std
    z_stsa = (cb_stsa - cb_stsa.mean()) / stsa_std
    raw_diff = z_oan + z_stsa
    res['cb_dispersion_difficulty'] = raw_diff - raw_diff.min() + 1.0
    res['cb_effective_dispersion_energy'] = res['cb_dispersion_work_proxy'] / (res['cb_dispersion_difficulty'] + 1e-5)

    # Feature 4: cb_normalized_power_decay
    p_start = _get_series(df, 'Stage2_DryMixing_power_Start', np.nan)
    p_end = _get_series(df, 'Stage2_DryMixing_power_End', np.nan)
    p_max = _get_series(df, 'Stage2_DryMixing_power_Max', 0.0)
    p_min = _get_series(df, 'Stage2_DryMixing_power_Min', 0.0)

    p_start_valid = np.where(pd.isna(p_start), p_max, p_start)
    p_end_valid = np.where(pd.isna(p_end), p_min, p_end)
    res['cb_normalized_power_decay'] = (p_start_valid - p_end_valid) / np.maximum(p_start_valid, 1.0)
    res['cb_normalized_power_decay'] = res['cb_normalized_power_decay'].fillna(0.0)

    # Feature 5: cb_specific_energy_ratio_dry_to_wet (retained)
    e_dry = _get_series(df, 'Stage2_DryMixing_Specific_Energy', 0.0)
    e_wet = _get_series(df, 'Stage4_WetMixing_Specific_Energy', 1.0)
    res['cb_specific_energy_ratio_dry_to_wet'] = e_dry / np.maximum(e_wet, 0.1)

    # Feature 6: cb_cumulative_shear_exposure
    d4_duration = _get_series(df, 'Stage4_WetMixing_Duration', 0.0)
    d4_power_mean = _get_series(df, 'Stage4_WetMixing_power_Mean', 0.0)
    drymix_power_integral = res['cb_dispersion_work_proxy']
    wetmix_power_integral = d4_duration * np.maximum(d4_power_mean, 0.0)
    res['cb_cumulative_shear_exposure'] = drymix_power_integral + 0.5 * wetmix_power_integral

    # Feature 7: cb_thermo_mechanical_index
    d2_temp_mean = _get_series(df, 'Stage2_DryMixing_temp_Mean', 1.0)
    res['cb_thermo_mechanical_index'] = res['cb_dispersion_work_proxy'] * np.sqrt(np.maximum(d2_temp_mean, 1.0))

    return res
