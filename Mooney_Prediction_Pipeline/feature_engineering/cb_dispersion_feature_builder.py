# ============================================================================
# Carbon Black Dispersion Kinetic Proxy Feature Builder
# ============================================================================
# Computes physical dispersion work, torque drop ratios, and agglomerate
# breakup kinetics for Carbon Black rubber compounds.
# ============================================================================

import numpy as np
import pandas as pd


def build_cb_dispersion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs physical dispersion work & kinetic proxies for Carbon Black compounds."""
    res = pd.DataFrame(index=df.index)

    d2_duration = pd.to_numeric(df.get('Stage2_DryMixing_Duration', 0.0), errors='coerce').fillna(0.0)
    d2_power = pd.to_numeric(df.get('Stage2_DryMixing_power_Mean', 0.0), errors='coerce').fillna(0.0)
    d2_torque = pd.to_numeric(df.get('Stage2_DryMixing_Torque_Mean', 0.0), errors='coerce').fillna(0.0)
    d6_torque = pd.to_numeric(df.get('Stage6_BottomMixing_Torque_Mean', 1.0), errors='coerce').fillna(1.0)
    if (d6_torque <= 0).all():
        d6_torque = pd.Series(1.0, index=df.index)

    cb_oan = pd.to_numeric(df.get('supplier_carbon_black_structure_avg', 110.0), errors='coerce').fillna(110.0)
    cb_stsa = pd.to_numeric(df.get('supplier_carbon_black_surface_area_avg', 85.0), errors='coerce').fillna(85.0)

    # 1. Dry Mix Shear Work Integral (Joule Proxy)
    res['cb_drymix_dispersion_work_integral'] = d2_duration * np.maximum(d2_power, 0.0)

    # 2. Dry-to-Bottom Torque Decay Ratio (Reflects network breakdown)
    res['cb_torque_drop_ratio_stage2_to_stage6'] = d2_torque / np.maximum(d6_torque, 1.0)

    # 3. CB Structure-Dispersion Index (OAN * Work / STSA)
    res['cb_structure_dispersion_index'] = (cb_oan * res['cb_drymix_dispersion_work_integral']) / np.maximum(cb_stsa, 10.0)

    # 4. Power Decay Rate Slope Proxy
    res['cb_power_decay_rate_stage2'] = pd.to_numeric(df.get('Stage2_power_decay_slope', 0.0), errors='coerce').fillna(0.0)

    # 5. Energy Allocation Ratio
    e_dry = pd.to_numeric(df.get('Stage2_DryMixing_Specific_Energy', 0.0), errors='coerce').fillna(0.0)
    e_wet = pd.to_numeric(df.get('Stage4_WetMixing_Specific_Energy', 1.0), errors='coerce').fillna(1.0)
    res['cb_specific_energy_ratio_dry_to_wet'] = e_dry / np.maximum(e_wet, 0.1)

    return res
