# ============================================================================
# Carbon Black Subsystem Residual Predictor V1.1 (Strict Feature Decoupling)
# ============================================================================
# Implements specialized 3-expert architecture for Carbon Black compounds with
# STRICT, MUTUALLY-EXCLUSIVE PHYSICAL FEATURE PARTITIONING:
#   1. CB Prep Expert (Stage 1 Loading, Stage 2 Dry Mix, Stage 3 Oil, Stage 2 Power Decay)
#   2. CB Bottom Mixer Response Expert (Stage 4 Wet, Stage 6 Bottom Mix, Discharge Temp/Torque)
#   3. CB Material Expert (Formulation ratios, COA features, Ambient)
#
# Combined via a 2nd-level OOF Non-Negative Ridge Combiner.
# ============================================================================

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


class CarbonBlackSubsystemPredictor:
    """Specialized 3-expert residual subsystem for Carbon Black rubber compounds with decoupled feature spaces."""

    def __init__(self, n_splits: int = 5, shrinkage_alpha: float = 1.0):
        self.n_splits = n_splits
        self.shrinkage_alpha = shrinkage_alpha
        
        self.prep_expert_ = None
        self.bottom_expert_ = None
        self.material_expert_ = None
        self.combiner_ = None
        
        self.feature_names_prep_ = []
        self.feature_names_bottom_ = []
        self.feature_names_material_ = []

    def _select_sub_features(self, X_delta: pd.DataFrame):
        """Strictly categorizes features into 3 non-overlapping physical CB expert feature spaces."""
        all_cols = list(X_delta.columns)

        # 1. CB Prep Expert Features (Upstream Dry Mix, Loading & Oil Loading ONLY)
        prep_cols = [
            c for c in all_cols
            if ('stage1' in c.lower() or 'stage2' in c.lower() or 'stage3' in c.lower() or 'dry' in c.lower() or 'loading' in c.lower())
            and not ('stage4' in c.lower() or 'stage6' in c.lower() or 'bottom' in c.lower() or 'discharge' in c.lower())
        ]

        # 2. CB Bottom Response Expert Features (Bottom Mixing & Discharge Metrics ONLY)
        bottom_cols = [
            c for c in all_cols
            if ('stage4' in c.lower() or 'stage6' in c.lower() or 'bottom' in c.lower() or 'discharge' in c.lower() or 't_max_temp' in c.lower())
            and not ('stage1' in c.lower() or 'stage2' in c.lower() or 'stage3' in c.lower())
        ]

        # 3. CB Material Expert Features (COA, Ratios & Ambient ONLY)
        mat_cols = [
            c for c in all_cols
            if ('phr' in c.lower() or 'coa' in c.lower() or 'supplier' in c.lower() or 'lot_' in c.lower() or 'ratio_' in c.lower() or 'weight_pct' in c.lower() or 'env_' in c.lower() or 'init_temp' in c.lower())
            and not ('stage1_' in c.lower() or 'stage2_' in c.lower() or 'stage3_' in c.lower() or 'stage4_' in c.lower() or 'stage6_' in c.lower())
        ]

        if not prep_cols:
            prep_cols = [c for c in all_cols if 'stage2' in c.lower()] or all_cols[:5]
        if not bottom_cols:
            bottom_cols = [c for c in all_cols if 'stage6' in c.lower()] or all_cols[:5]
        if not mat_cols:
            mat_cols = [c for c in all_cols if 'supplier' in c.lower() or 'ratio' in c.lower()] or all_cols[:5]

        self.feature_names_prep_ = list(set(prep_cols))
        self.feature_names_bottom_ = list(set(bottom_cols))
        self.feature_names_material_ = list(set(mat_cols))

    def fit(self, X_delta: pd.DataFrame, y_residual: np.ndarray, sample_weights: np.ndarray = None):
        """Fits the 3 sub-experts and learns an OOF combiner for Carbon Black routes."""
        n_samples = len(X_delta)
        if sample_weights is None:
            sample_weights = np.ones(n_samples)

        self._select_sub_features(X_delta)

        oof_prep = np.zeros(n_samples)
        oof_bottom = np.zeros(n_samples)
        oof_mat = np.zeros(n_samples)

        kf = KFold(n_splits=min(self.n_splits, max(2, n_samples // 10)), shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X_delta):
            X_tr, y_tr, w_tr = X_delta.iloc[train_idx], y_residual[train_idx], sample_weights[train_idx]
            X_val = X_delta.iloc[val_idx]

            m_prep = LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=2.0, random_state=42, verbose=-1)
            m_prep.fit(X_tr[self.feature_names_prep_], y_tr, sample_weight=w_tr)
            oof_prep[val_idx] = m_prep.predict(X_val[self.feature_names_prep_])

            m_bot = LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=2.0, random_state=42, verbose=-1)
            m_bot.fit(X_tr[self.feature_names_bottom_], y_tr, sample_weight=w_tr)
            oof_bottom[val_idx] = m_bot.predict(X_val[self.feature_names_bottom_])

            m_mat = LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=2.0, random_state=42, verbose=-1)
            m_mat.fit(X_tr[self.feature_names_material_], y_tr, sample_weight=w_tr)
            oof_mat[val_idx] = m_mat.predict(X_val[self.feature_names_material_])

        # Step 2: Fit 2nd-level OOF Combiner
        X_oof = np.column_stack([oof_prep, oof_bottom, oof_mat])
        self.combiner_ = Ridge(alpha=self.shrinkage_alpha, fit_intercept=False, positive=True)
        self.combiner_.fit(X_oof, y_residual, sample_weight=sample_weights)

        # Step 3: Fit final sub-experts on FULL Carbon Black training set
        self.prep_expert_ = LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=2.0, random_state=42, verbose=-1)
        self.prep_expert_.fit(X_delta[self.feature_names_prep_], y_residual, sample_weight=sample_weights)

        self.bottom_expert_ = LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=2.0, random_state=42, verbose=-1)
        self.bottom_expert_.fit(X_delta[self.feature_names_bottom_], y_residual, sample_weight=sample_weights)

        self.material_expert_ = LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=2.0, random_state=42, verbose=-1)
        self.material_expert_.fit(X_delta[self.feature_names_material_], y_residual, sample_weight=sample_weights)

        return self

    def predict_experts(self, X_delta: pd.DataFrame) -> dict[str, np.ndarray]:
        """Returns predictions from each individual CB sub-expert."""
        pred_prep = self.prep_expert_.predict(X_delta[self.feature_names_prep_])
        pred_bottom = self.bottom_expert_.predict(X_delta[self.feature_names_bottom_])
        pred_mat = self.material_expert_.predict(X_delta[self.feature_names_material_])
        return {
            'prep': pred_prep,
            'bottom': pred_bottom,
            'material': pred_mat,
        }

    def predict(self, X_delta: pd.DataFrame) -> np.ndarray:
        """Returns combined residual prediction via the OOF combiner."""
        preds = self.predict_experts(X_delta)
        X_test_oof = np.column_stack([preds['prep'], preds['bottom'], preds['material']])
        return self.combiner_.predict(X_test_oof)
