# ============================================================================
# Silica Subsystem Residual Predictor V1 (With Data Quality Hardening & Fallback)
# ============================================================================
# Implements the specialized 4-expert architecture for Silica compounds:
#   1. PID Reaction Expert (Silanization window, exposure proxy, risk, stability)
#   2. Wet Preparation Expert (Upstream dry mix, oil loading, wet mix dynamics)
#   3. Bottom Post-Reaction Expert (Bottom mixing completion, energy/torque response)
#   4. Material Expert (Raw material properties, COA, PHR formulation ratios)
#
# Out-Of-Fold (OOF) predictions from each sub-expert are combined by a 2nd-level
# OOF combiner to prevent in-sample overfitting and fit-order bias.
# Generates auditable diagnostic hypotheses (Reason Codes) and automatic PID data
# quality fallback handling.
# ============================================================================

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold


class SilicaSubsystemPredictor:
    """Specialized 4-expert residual subsystem for Silica rubber compounds."""

    def __init__(self, n_splits: int = 5, shrinkage_alpha: float = 1.0, positive_constraint: bool = True):
        self.n_splits = n_splits
        self.shrinkage_alpha = shrinkage_alpha
        self.positive_constraint = positive_constraint
        
        self.pid_expert_ = None
        self.wet_expert_ = None
        self.bottom_expert_ = None
        self.material_expert_ = None
        self.combiner_ = None
        
        self.feature_names_pid_ = []
        self.feature_names_wet_ = []
        self.feature_names_bottom_ = []
        self.feature_names_material_ = []

    def _select_sub_features(self, X_delta: pd.DataFrame, df_meta: pd.DataFrame = None):
        """Categorize features into the 4 sub-expert feature spaces."""
        all_cols = list(X_delta.columns)
        
        # 1. PID Reaction Expert Features
        pid_cols = [
            c for c in all_cols
            if 'pid' in c.lower() or 'stage5' in c.lower() or 'silanization' in c.lower()
        ]
        
        # 2. Wet Preparation Expert Features
        wet_cols = [
            c for c in all_cols
            if 'stage2' in c.lower() or 'stage3' in c.lower() or 'stage4' in c.lower()
            or 'dry' in c.lower() or 'wet' in c.lower() or 'oil' in c.lower()
        ]
        
        # 3. Bottom Post-Reaction Expert Features
        bottom_cols = [
            c for c in all_cols
            if 'stage6' in c.lower() or 'bottom' in c.lower()
        ]
        
        # 4. Material Expert Features
        mat_cols = [
            c for c in all_cols
            if 'phr' in c.lower() or 'coa' in c.lower() or 'silica' in c.lower()
            or 'phys_' in c.lower() or 'env_' in c.lower()
        ]
        
        # Fallback if any is empty
        if not pid_cols:
            pid_cols = all_cols[:5]
        if not wet_cols:
            wet_cols = all_cols[:5]
        if not bottom_cols:
            bottom_cols = all_cols[:5]
        if not mat_cols:
            mat_cols = all_cols[:5]

        self.feature_names_pid_ = pid_cols
        self.feature_names_wet_ = wet_cols
        self.feature_names_bottom_ = bottom_cols
        self.feature_names_material_ = mat_cols

    def check_pid_data_quality(self, df_row_or_batch: pd.DataFrame) -> np.ndarray:
        """Checks if PID data quality is acceptable (non-null, non-zero, within physical limits)."""
        valid_mask = np.ones(len(df_row_or_batch), dtype=bool)
        for idx in range(len(df_row_or_batch)):
            row = df_row_or_batch.iloc[idx]
            exp_val = pd.to_numeric(row.get('pid_silanization_exposure_proxy', np.nan), errors='coerce')
            risk_val = pd.to_numeric(row.get('pid_high_temperature_risk_proxy', np.nan), errors='coerce')
            valid_flag = pd.to_numeric(row.get('pid_data_valid_flag', 1.0), errors='coerce')

            if pd.isna(exp_val) or pd.isna(risk_val) or valid_flag == 0 or (exp_val == 0 and risk_val == 0):
                valid_mask[idx] = False
        return valid_mask

    def fit(self, X_delta: pd.DataFrame, y_residual: np.ndarray, sample_weights: np.ndarray = None, df_meta: pd.DataFrame = None):
        """Fits the 4 sub-experts and learns an OOF combiner."""
        n_samples = len(X_delta)
        if sample_weights is None:
            sample_weights = np.ones(n_samples)

        self._select_sub_features(X_delta, df_meta)

        # Step 1: Out-Of-Fold (OOF) predictions
        oof_pid = np.zeros(n_samples)
        oof_wet = np.zeros(n_samples)
        oof_bottom = np.zeros(n_samples)
        oof_mat = np.zeros(n_samples)

        kf = KFold(n_splits=min(self.n_splits, max(2, n_samples // 10)), shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X_delta):
            X_tr, y_tr, w_tr = X_delta.iloc[train_idx], y_residual[train_idx], sample_weights[train_idx]
            X_val = X_delta.iloc[val_idx]

            m_pid = LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=3.0, random_state=42, verbose=-1)
            m_pid.fit(X_tr[self.feature_names_pid_], y_tr, sample_weight=w_tr)
            oof_pid[val_idx] = m_pid.predict(X_val[self.feature_names_pid_])

            m_wet = LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=3.0, random_state=42, verbose=-1)
            m_wet.fit(X_tr[self.feature_names_wet_], y_tr, sample_weight=w_tr)
            oof_wet[val_idx] = m_wet.predict(X_val[self.feature_names_wet_])

            m_bot = LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=3.0, random_state=42, verbose=-1)
            m_bot.fit(X_tr[self.feature_names_bottom_], y_tr, sample_weight=w_tr)
            oof_bottom[val_idx] = m_bot.predict(X_val[self.feature_names_bottom_])

            m_mat = LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=3.0, random_state=42, verbose=-1)
            m_mat.fit(X_tr[self.feature_names_material_], y_tr, sample_weight=w_tr)
            oof_mat[val_idx] = m_mat.predict(X_val[self.feature_names_material_])

        # Step 2: Fit 2nd-level Combiner
        X_oof = np.column_stack([oof_pid, oof_wet, oof_bottom, oof_mat])
        if self.positive_constraint:
            self.combiner_ = Ridge(alpha=self.shrinkage_alpha, fit_intercept=False, positive=True)
        else:
            self.combiner_ = LinearRegression(fit_intercept=False)
        self.combiner_.fit(X_oof, y_residual, sample_weight=sample_weights)

        # Step 3: Fit final sub-experts on FULL Silica training set
        self.pid_expert_ = LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=3.0, random_state=42, verbose=-1)
        self.pid_expert_.fit(X_delta[self.feature_names_pid_], y_residual, sample_weight=sample_weights)

        self.wet_expert_ = LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=3.0, random_state=42, verbose=-1)
        self.wet_expert_.fit(X_delta[self.feature_names_wet_], y_residual, sample_weight=sample_weights)

        self.bottom_expert_ = LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=3.0, random_state=42, verbose=-1)
        self.bottom_expert_.fit(X_delta[self.feature_names_bottom_], y_residual, sample_weight=sample_weights)

        self.material_expert_ = LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=3.0, random_state=42, verbose=-1)
        self.material_expert_.fit(X_delta[self.feature_names_material_], y_residual, sample_weight=sample_weights)

        return self

    def predict_experts(self, X_delta: pd.DataFrame) -> dict[str, np.ndarray]:
        """Returns predictions from each individual sub-expert."""
        pred_pid = self.pid_expert_.predict(X_delta[self.feature_names_pid_])
        pred_wet = self.wet_expert_.predict(X_delta[self.feature_names_wet_])
        pred_bottom = self.bottom_expert_.predict(X_delta[self.feature_names_bottom_])
        pred_mat = self.material_expert_.predict(X_delta[self.feature_names_material_])
        return {
            'pid': pred_pid,
            'wet': pred_wet,
            'bottom': pred_bottom,
            'material': pred_mat,
        }

    def predict(self, X_delta: pd.DataFrame, df_meta: pd.DataFrame = None) -> np.ndarray:
        """Returns combined residual prediction with automatic PID fallback handling."""
        preds = self.predict_experts(X_delta)

        if df_meta is not None:
            pid_ok = self.check_pid_data_quality(df_meta)
            # Apply fallback: if PID data is invalid, set PID prediction to 0
            preds['pid'] = np.where(pid_ok, preds['pid'], 0.0)

        X_test_oof = np.column_stack([preds['pid'], preds['wet'], preds['bottom'], preds['material']])
        return self.combiner_.predict(X_test_oof)

    def generate_reason_codes(self, df_meta: pd.DataFrame) -> pd.DataFrame:
        """Generates auditable diagnostic hypotheses and fallback reason codes for Silica batches."""
        primary_codes = []
        confidence_levels = []
        pid_active_flags = []

        for idx in range(len(df_meta)):
            row = df_meta.iloc[idx]
            batch_codes = []
            exp_proxy = pd.to_numeric(row.get('pid_silanization_exposure_proxy', np.nan), errors='coerce')
            risk_proxy = pd.to_numeric(row.get('pid_high_temperature_risk_proxy', np.nan), errors='coerce')
            instab_proxy = pd.to_numeric(row.get('pid_control_instability_proxy', np.nan), errors='coerce')
            valid_flag = pd.to_numeric(row.get('pid_data_valid_flag', 1.0), errors='coerce')

            is_pid_ok = not (pd.isna(exp_proxy) or pd.isna(risk_proxy) or valid_flag == 0 or (exp_proxy == 0 and risk_proxy == 0))

            if not is_pid_ok:
                batch_codes.append('PID_EXPERT_DISABLED_LOW_QUALITY')
                confidence_levels.append('LOW')
                pid_active_flags.append(False)
            else:
                pid_active_flags.append(True)
                confidence_levels.append('HIGH')
                if exp_proxy < 10.0:
                    batch_codes.append('LOW_PID_REACTION_EXPOSURE')
                if risk_proxy > 15.0:
                    batch_codes.append('HIGH_PID_THERMAL_EXPOSURE')
                if instab_proxy > 2.5:
                    batch_codes.append('PID_CONTROL_INSTABILITY')

            primary_code = batch_codes[0] if batch_codes else 'NORMAL_REACTION_STATE'
            primary_codes.append(primary_code)

        return pd.DataFrame({
            'primary_reason_code': primary_codes,
            'confidence_level': confidence_levels,
            'pid_expert_active': pid_active_flags,
        }, index=df_meta.index)
