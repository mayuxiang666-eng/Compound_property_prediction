# ============================================================================
# Silica PID Feature Builder (Linear vs Arrhenius Kinetic Proxies)
# ============================================================================
# Computes both standard linear exposure proxy (Duration * Temp / Power) and
# Arrhenius-type kinetic reaction proxy:
#   rate_rel = exp(-Ea/R * (1/T_kelvin - 1/T_ref_kelvin))
#   arrhenius_exposure = duration * rate_rel * power
# ============================================================================

import numpy as np
import pandas as pd


def build_silica_pid_features(
    df: pd.DataFrame,
    t_ref_c: float = 145.0,
    t_risk_threshold_c: float = 160.0,
    ea_over_r: float = 8500.0,
) -> pd.DataFrame:
    """Constructs linear vs Arrhenius PID reaction exposure features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing Stage 5 PID measurements.
    t_ref_c : float
        Reference silanization reaction window temperature (Celsius).
    t_risk_threshold_c : float
        Thermal risk threshold temperature (Celsius).
    ea_over_r : float
        Arrhenius-type activation parameter Ea/R (Kelvin).

    Returns
    -------
    pd.DataFrame
        Dataframe containing constructed PID proxy features.
    """
    res = pd.DataFrame(index=df.index)

    # 1. Duration, Temperature, Power
    duration = pd.to_numeric(df.get('Stage5_PID_Duration', 0.0), errors='coerce').fillna(0.0)
    temp_mean = pd.to_numeric(df.get('Stage5_PID_temp_Mean', 0.0), errors='coerce').fillna(0.0)
    temp_std = pd.to_numeric(df.get('Stage5_PID_temp_Std', 0.0), errors='coerce').fillna(0.0)
    power_mean = pd.to_numeric(df.get('Stage5_PID_power_Mean', 1.0), errors='coerce').fillna(1.0)
    if (power_mean <= 0).all():
        power_mean = pd.Series(1.0, index=df.index)

    # 2. Linear Exposure Proxy (Duration * Temperature)
    res['pid_linear_exposure_proxy'] = duration * np.maximum(temp_mean, 0.0)

    # 3. Arrhenius Kinetic Exposure Proxy
    # rate_rel = exp(-Ea/R * (1/T_kelvin - 1/T_ref_kelvin))
    t_k = np.maximum(temp_mean + 273.15, 273.15)
    t_ref_k = t_ref_c + 273.15
    rel_rate = np.exp(-ea_over_r * (1.0 / t_k - 1.0 / t_ref_k))
    
    # Arrhenius Exposure Proxy = duration * rel_rate * power_factor
    res['pid_arrhenius_silanization_exposure_proxy'] = duration * rel_rate * np.maximum(power_mean, 0.1)

    # 4. Standard silanization exposure proxy (Alias for backwards compatibility)
    res['pid_silanization_exposure_proxy'] = res['pid_arrhenius_silanization_exposure_proxy']

    # 5. High-Temperature Risk Proxy
    temp_overshoot = np.maximum(temp_mean - t_risk_threshold_c, 0.0)
    res['pid_high_temperature_risk_proxy'] = duration * temp_overshoot

    # 6. Control Stability & Effort Proxy
    power_stability = pd.to_numeric(df.get('phys_power_stability_pid', 0.0), errors='coerce').fillna(0.0)
    res['pid_control_instability_proxy'] = temp_std * (1.0 + power_stability)

    # 7. Mechanical Load Response
    specific_energy = pd.to_numeric(df.get('Stage5_PID_Specific_Energy', 0.0), errors='coerce').fillna(0.0)
    torque_integral = pd.to_numeric(df.get('Stage5_PID_Torque_Integral', 0.0), errors='coerce').fillna(0.0)
    res['pid_mechanical_work_proxy'] = specific_energy
    res['pid_torque_integral_proxy'] = torque_integral

    # 8. Missingness / Quality flag
    res['pid_data_valid_flag'] = np.where(duration > 0, 1.0, 0.0)

    return res
