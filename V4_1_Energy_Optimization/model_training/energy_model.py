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
    'Stage4_WetMixing_Duration',
    'Stage5_PID_Duration',
    'Stage6_BottomMixing_Duration',
    'Target_Temperature',
    'Top_Fill_Factor',
]

RECIPE_BASELINE_COLS = [
    'batch_weight_ton',
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

CATEGORICAL_CONTEXT_COLS = [
    'MixerLine',
    'material_system',
    'phase_route',
]

POST_BATCH_ACTUAL_COLS = [
    'Stage2_DryMixing_RotorSpeed_Mean',
    'Stage5_PID_temp_Mean',
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

ENERGY_DERIVED_TARGET_PROXY_COLS = [
    'total_kwh_per_batch',
    'kwh_per_ton',
    'top_mixer_kwh',
    'bottom_mixer_kwh',
]

MODE_B_FORBIDDEN_FEATURE_TOKENS = (
    'energy',
    'kwh',
    'power',
    'torque',
    'integral',
    'actual',
    'phys_',
)


def build_energy_feature_purge_audit() -> pd.DataFrame:
    """Returns the allowed/forbidden feature policy for both energy-model modes."""
    rows = []
    feature_groups = [
        (RECIPE_BASELINE_COLS, 'static_recipe_material', True, True, 'Available before production'),
        (CATEGORICAL_CONTEXT_COLS, 'pre_batch_context', True, True, 'Available before production'),
        (PRE_BATCH_SETPOINT_COLS, 'controllable_setpoint', True, True, 'Production target or setpoint'),
        (POST_BATCH_ACTUAL_COLS, 'post_batch_process_response', True, False, 'Measured stage response is unavailable at recommendation time'),
        (ENERGY_DERIVED_TARGET_PROXY_COLS, 'energy_derived_target_proxy', False, False, 'Energy target or target proxy is prohibited from model features'),
    ]
    for features, category, used_in_mode_a, used_in_mode_b, reason in feature_groups:
        for feature_name in features:
            rows.append({
                'feature_name': feature_name,
                'feature_category': category,
                'used_in_mode_a': used_in_mode_a,
                'used_in_mode_b': used_in_mode_b,
                'purge_reason': '' if used_in_mode_b else reason,
                'allowed_for_recommendation': used_in_mode_b,
            })
    return pd.DataFrame(rows)


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
        self.numeric_feature_names = []
        self.categorical_context_values: dict[str, list[str]] = {}

    def _fit_feature_schema(self, df: pd.DataFrame, numeric_feature_cols: list) -> None:
        self.numeric_feature_names = numeric_feature_cols
        self.categorical_context_values = {}
        categorical_feature_names = []

        for column in CATEGORICAL_CONTEXT_COLS:
            if column not in df.columns:
                continue
            values = df[column].fillna('__MISSING__').astype(str)
            categories = sorted(values.unique().tolist())
            self.categorical_context_values[column] = categories
            categorical_feature_names.extend(f'{column}={category}' for category in categories)

        self.feature_names = numeric_feature_cols + categorical_feature_names

    def _get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        numeric_columns = []
        for column in self.numeric_feature_names:
            if column in df.columns:
                numeric_columns.append(pd.to_numeric(df[column], errors='coerce').fillna(0.0).to_numpy())
            else:
                numeric_columns.append(np.zeros(len(df), dtype=np.float32))
        mat = np.column_stack(numeric_columns) if numeric_columns else np.empty((len(df), 0))

        for column, categories in self.categorical_context_values.items():
            values = (
                df[column].fillna('__MISSING__').astype(str)
                if column in df.columns
                else pd.Series('__MISSING__', index=df.index)
            )
            encoded = np.column_stack([(values == category).to_numpy(dtype=np.float32) for category in categories])
            mat = np.hstack([mat, encoded])
        return mat

    def transform_pre_batch_features(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the fitted, policy-governed feature matrix for downstream diagnostics."""
        if self.mode != 'mode_b':
            raise ValueError('High-energy diagnostics require the Mode B pre-batch feature schema.')
        return self._get_feature_matrix(df)

    @staticmethod
    def _get_training_weights(df: pd.DataFrame) -> np.ndarray | None:
        if '_w_loss' not in df.columns:
            return None
        weights = pd.to_numeric(df['_w_loss'], errors='coerce').fillna(1.0).to_numpy(dtype=float)
        return np.clip(weights, 0.05, None)

    @staticmethod
    def _validate_mode_b_feature_policy(feature_cols: list[str]) -> None:
        violations = [
            feature_name
            for feature_name in feature_cols
            if feature_name in POST_BATCH_ACTUAL_COLS
            or feature_name in ENERGY_DERIVED_TARGET_PROXY_COLS
            or any(token in feature_name.lower() for token in MODE_B_FORBIDDEN_FEATURE_TOKENS)
        ]
        if violations:
            raise ValueError(
                'Mode B feature policy violation: post-batch or energy-derived features are forbidden: '
                f'{sorted(violations)}'
            )

    def fit(self, df_train: pd.DataFrame, target_col='total_kwh_per_batch'):
        recipe_cols = [c for c in RECIPE_BASELINE_COLS if c in df_train.columns]
        setpoint_cols = [c for c in PRE_BATCH_SETPOINT_COLS if c in df_train.columns]

        if self.mode == 'mode_a':
            actual_cols = [c for c in POST_BATCH_ACTUAL_COLS if c in df_train.columns]
            feature_cols = recipe_cols + setpoint_cols + actual_cols
        else:
            feature_cols = recipe_cols + setpoint_cols
            self._validate_mode_b_feature_policy(feature_cols)

        self._fit_feature_schema(df_train, feature_cols)
        X_tr = self._get_feature_matrix(df_train)
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
        X_in = self._get_feature_matrix(df_in)
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
