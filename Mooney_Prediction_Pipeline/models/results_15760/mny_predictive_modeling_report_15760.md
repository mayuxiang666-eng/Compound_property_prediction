# Dedicated M1-T15760 Mooney Viscosity Prediction Model Report

This report summarizes the performance of the dedicated tree-based predictive model trained exclusively on compound **M1-T15760**.

## Model Details
* **Algorithm**: Stacked Ensemble (LightGBM + XGBoost + CatBoost + Random Forest + HistGBM)
* **Features**: 13 focused With-Oil physical curves features
* **Training set size**: 399 batches (after robust MAD outlier filtering)
* **Test set size**: 101 batches

## Test Set Metrics
* **R^2 Score**: **0.0857**
* **MAE (Mean Absolute Error)**: **3.6166 MNY**
* **RMSE (Root Mean Squared Error)**: **4.4176 MNY**

## Feature Importance (Top Features)
| Feature | Importance |
| :--- | :---: |
| Stage4_WetMixing_Duration | 0.109408 |
| Stage4_WetMixing_Torque_Mean | 0.042494 |
| phys_temp_integral_above_100 | 0.034674 |
| Stage3_OilLoading_temp_Std | 0.012147 |
| Stage2_DryMixing_power_Mean | 0.009959 |
| Stage4_WetMixing_power_Mean | 0.009792 |
| Stage6_BottomMixing_power_Integral | 0.008693 |
| phys_init_temp | 0.008499 |
| Stage6_BottomMixing_Torque_Integral | 0.008366 |
| MixerLine_ML07 | 0.000000 |
| MixerLine_ML09 | -0.000001 |
| MixerLine_MB03 | -0.000006 |
| Stage2_DryMixing_power_Integral | -0.001744 |
| Stage3_OilLoading_temp_Mean | -0.012755 |
| Stage6_BottomMixing_power_Mean | -0.014427 |

