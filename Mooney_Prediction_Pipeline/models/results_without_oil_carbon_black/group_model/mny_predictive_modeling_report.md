# Two-Stage Group Predictive Modeling Report - Without-Oil Carbon-Black

This model uses a Two-Stage architecture to decouple static recipe baseline and dynamic process variations.

## 5-Fold Group CV Performance
- **R^2**: 0.9435
- **MAE**: 3.4781 MU
- **RMSE**: 4.3407 MU
- **Within-Compound Deviation Correlation**: -0.6620
- **Pred/Actual Std Ratio**: 0.9387

## Core Features Utilized
### Stage 1 Recipe Features:
['Top_Fill_Factor', 'Bot_Fill_Factor', 'Target_Temperature', 'weight_pct_solid_elastomer', 'weight_pct_natural_rubber', 'weight_pct_silica', 'weight_pct_oil', 'weight_pct_silian', 'weight_pct_carbon_black', 'silica_phr', 'is_oil_loading_present', 'ratio_nr_rubber', 'ratio_filler_polymer', 'ratio_oil_polymer', 'ratio_oil_filler', 'supplier_rubber_viscosity_avg', 'supplier_silica_moisture_avg', 'supplier_silica_surface_area_avg', 'supplier_carbon_black_structure_avg', 'supplier_carbon_black_surface_area_avg', 'supplier_carbon_black_moisture_avg']

### Stage 2 Core Process Features:
['phys_discharge_temp', 'phys_max_temp', 'phys_eta_app_discharge', 'Stage6_BottomMixing_Torque_Mean', 'Stage6_BottomMixing_power_Mean', 'Stage6_BottomMixing_Duration', 'Stage6_BottomMixing_Torque_Integral', 'Stage4_WetMixing_temp_Mean', 'Stage4_WetMixing_Duration', 'Stage2_DryMixing_Duration', 'Stage2_DryMixing_power_Mean', 'env_temp_mean', 'env_humidity_mean', 'I_silanization', 'I_scorch', 'time_to_sil_plateau_duration', 'top_sil_duration', 'bottom_sil_duration', 'silanization_energy_mj', 'top_avg_sil_temperature', 'bottom_avg_sil_temperature']
