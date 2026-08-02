# ============================================================================
# Step 3: Fast Vectorized Energy Prediction Model Architecture (V4.1)
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

PRE_BATCH_SETPOINT_COLS = [
    'Stage2_DryMixing_Duration',
    'Stage2_DryMixing_RotorSpeed_Mean',
    'Stage4_WetMixing_Duration',
    'Stage5_PID_Duration',
    'Stage5_PID_temp_Mean',
    'Stage6_BottomMixing_Duration',
    'Target_Temperature',
    'Top_Fill_Factor',
]

RECIPE_BASELINE_COLS = [
    'weight_pct_solid_elastomer',
    'weight_pct_natural_rubber',
    'weight_pct_silica',
    'weight_pct_oil',
    'weight_pct_silian',
    'weight_pct_carbon_black',
    'silica_phr',
    'supplier_rubber_viscosity_avg',
    'supplier_silica_surface_area_avg',
    'supplier_carbon_black_structure_avg',
]

POST_BATCH_ACTUAL_COLS = [
    'phys_power_integral',
    'phys_discharge_temp',
    'phys_max_power_drop_rate',
    'phys_avg_power',
    'phys_temp_rise_rate',
    'Stage2_DryMixing_power_Integral',
    'Stage4_WetMixing_power_Integral',
    'Stage5_PID_power_Integral',
    'Stage6_BottomMixing_power_Integral',
]


class MixingEnergyPredictionModel:
    def __init__(self, mode='mode_b'):
        self.mode = mode.lower()
        self.baseline_model = LGBMRegressor(n_estimators=40, learning_rate=0.08, max_depth=4, n_jobs=-1, random_state=42, verbose=-1)
        self.branch_experts = {
            'CarbonBlack_OilWet': LGBMRegressor(n_estimators=30, learning_rate=0.08, max_depth=3, n_jobs=-1, random_state=42, verbose=-1),
            'CarbonBlack_NoOilDry': LGBMRegressor(n_estimators=30, learning_rate=0.08, max_depth=3, n_jobs=-1, random_state=42, verbose=-1),
            'Silica_OilWet': LGBMRegressor(n_estimators=30, learning_rate=0.08, max_depth=3, n_jobs=-1, random_state=42, verbose=-1),
            'Silica_NoOilDry': LGBMRegressor(n_estimators=30, learning_rate=0.08, max_depth=3, n_jobs=-1, random_state=42, verbose=-1),
        }
        self.fitted_branch_experts: set[str] = set()
        self.feature_names = []

    def _get_feature_matrix(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        existing = [c for c in feature_cols if c in df.columns]
        mat = df[existing].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        if len(existing) < len(feature_cols):
            missing_count = len(feature_cols) - len(existing)
            mat = np.hstack([mat, np.zeros((len(df), missing_count), dtype=np.float32)])
        return mat

    @staticmethod
    def _get_training_weights(df: pd.DataFrame) -> np.ndarray | None:
        if '_w_loss' not in df.columns:
            return None
        weights = pd.to_numeric(df['_w_loss'], errors='coerce').fillna(1.0).to_numpy(dtype=float)
        return np.clip(weights, 0.05, None)

    def fit(self, df_train: pd.DataFrame, target_col='total_kwh_per_batch'):
        recipe_cols = [c for c in RECIPE_BASELINE_COLS if c in df_train.columns]
        setpoint_cols = [c for c in PRE_BATCH_SETPOINT_COLS if c in df_train.columns]

        if self.mode == 'mode_a':
            actual_cols = [c for c in POST_BATCH_ACTUAL_COLS if c in df_train.columns]
            feature_cols = recipe_cols + setpoint_cols + actual_cols
        else:
            feature_cols = recipe_cols + setpoint_cols

        self.feature_names = feature_cols
        X_tr = self._get_feature_matrix(df_train, feature_cols)
        y_tr = df_train[target_col].values
        training_weights = self._get_training_weights(df_train)

        self.baseline_model.fit(X_tr, y_tr, sample_weight=training_weights)
        base_preds = self.baseline_model.predict(X_tr)
        residuals = y_tr - base_preds

        branches = df_train['system_route_branch'].values if 'system_route_branch' in df_train.columns else np.full(len(df_train), 'Silica_OilWet')

        self.fitted_branch_experts.clear()
        for branch_name, expert in self.branch_experts.items():
            mask = (branches == branch_name)
            if np.sum(mask) >= 10:
                X_sub = X_tr[mask]
                y_sub = residuals[mask]
                branch_weights = training_weights[mask] if training_weights is not None else None
                expert.fit(X_sub, y_sub, sample_weight=branch_weights)
                self.fitted_branch_experts.add(branch_name)

        return self

    def predict(self, df_in: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_in = self._get_feature_matrix(df_in, self.feature_names)
        base_preds = self.baseline_model.predict(X_in)
        final_kwh = base_preds.copy()

        branches = df_in['system_route_branch'].values if 'system_route_branch' in df_in.columns else np.full(len(df_in), 'Silica_OilWet')

        for branch_name, expert in self.branch_experts.items():
            mask = (branches == branch_name)
            if np.any(mask) and branch_name in self.fitted_branch_experts:
                X_branch = X_in[mask]
                res_preds = expert.predict(X_branch)
                final_kwh[mask] += res_preds

        final_kwh = np.clip(final_kwh, 10.0, 120.0)
        batch_weight = df_in['batch_weight_ton'].values if 'batch_weight_ton' in df_in.columns else np.full(len(df_in), 0.25)
        batch_weight = np.where(batch_weight > 0.05, batch_weight, 0.25)

        kwh_per_ton = final_kwh / batch_weight
        return final_kwh, kwh_per_ton


def evaluate_energy_model_performance(y_true_kwh, y_pred_kwh, weight_ton, y_true_kwh_per_ton=None):
    mae_kwh = float(np.mean(np.abs(y_true_kwh - y_pred_kwh)))
    rmse_kwh = float(np.sqrt(np.mean((y_true_kwh - y_pred_kwh) ** 2)))

    ss_tot = float(np.sum((y_true_kwh - np.mean(y_true_kwh)) ** 2))
    ss_res = float(np.sum((y_true_kwh - y_pred_kwh) ** 2))
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    mape = float(np.mean(np.abs((y_true_kwh - y_pred_kwh) / np.maximum(y_true_kwh, 1.0))) * 100.0)

    pred_kwh_per_ton = y_pred_kwh / np.maximum(weight_ton, 0.05)
    if y_true_kwh_per_ton is None:
        true_kwh_per_ton = y_true_kwh / np.maximum(weight_ton, 0.05)
    else:
        true_kwh_per_ton = y_true_kwh_per_ton

    mae_kwh_per_ton = float(np.mean(np.abs(true_kwh_per_ton - pred_kwh_per_ton)))

    hit_2kwh = float(np.mean(np.abs(y_true_kwh - y_pred_kwh) <= 2.0) * 100.0)
    hit_5pct = float(np.mean(np.abs((y_true_kwh - y_pred_kwh) / np.maximum(y_true_kwh, 1.0)) <= 0.05) * 100.0)

    return {
        'MAE_kWh': round(mae_kwh, 3),
        'RMSE_kWh': round(rmse_kwh, 3),
        'R2': round(r2, 4),
        'MAPE_pct': round(mape, 2),
        'MAE_kWh_per_ton': round(mae_kwh_per_ton, 3),
        'Hit_2kWh_pct': round(hit_2kwh, 2),
        'Hit_5pct_pct': round(hit_5pct, 2),
    }
