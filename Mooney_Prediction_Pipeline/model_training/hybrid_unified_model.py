# ============================================================================
# V3.4 Hybrid Unified Architecture (Stages 1, 1b, 2)
# ============================================================================
# Stage 1: Shared Global Recipe Surface GBDT
# Stage 1b: Regularized Compound Bias Shrinkage
# Stage 2: Phase & Material-Routed Process-Delta Residual Experts
#   - Supports Specialized Silica Subsystem Predictor V1
#   - Supports Specialized Carbon Black Subsystem Predictor V1
# ============================================================================

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, HuberRegressor

from .bias_shrinkage import RegularizedBiasEstimator
from .effective_weighting import compute_effective_sample_weights
from .silica_subsystem import SilicaSubsystemPredictor
from .cb_subsystem import CarbonBlackSubsystemPredictor


class HybridUnifiedMooneyModel:
    """
    V3.6 Hybrid Unified Architecture for Compound Mooney Viscosity Prediction.
    Supports 2-expert Phase-Route baseline, 4-expert Material-System x Phase-Route Matrix,
    specialized Silica 4-Subexpert Subsystem Predictor, and specialized Carbon Black 3-Subexpert Subsystem Predictor.
    """
    def __init__(
        self,
        stage1_params=None,
        stage2_params=None,
        shrinkage_k=5.0,
        use_cluster_experts=True,
        use_material_route_matrix=False,
        no_oil_expert_type='lightgbm',
        use_silica_subsystem=False,
        use_cb_subsystem=False,
    ):
        self.shrinkage_k = shrinkage_k
        self.use_cluster_experts = use_cluster_experts
        self.use_material_route_matrix = use_material_route_matrix
        self.no_oil_expert_type = no_oil_expert_type
        self.use_silica_subsystem = use_silica_subsystem
        self.use_cb_subsystem = use_cb_subsystem
        
        # Default GBDT parameters
        self.stage1_params = stage1_params or {
            'n_estimators': 300,
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 15,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbose': -1
        }
        
        self.stage2_params = stage2_params or {
            'n_estimators': 180,
            'learning_rate': 0.025,
            'num_leaves': 20,
            'max_depth': 5,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.2,
            'reg_lambda': 3.0,
            'random_state': 42,
            'verbose': -1
        }
        
        self.stage1_model_ = None
        self.stage1b_bias_ = None
        self.stage2_experts_ = {}
        self.feature_names_s1_ = []
        self.feature_names_s2_ = []
        self.stage1_feature_importance_ = pd.DataFrame()
        self.process_nominals_ = pd.DataFrame()
        self.global_process_nominal_ = pd.Series(dtype=float)
        self.stage2_expert_report_ = pd.DataFrame()

    @staticmethod
    def _phase_route(df):
        oil_loading = pd.to_numeric(df.get('is_oil_loading_present', 0.0), errors='coerce').fillna(0.0) > 0
        wet_duration = pd.to_numeric(df.get('Stage4_WetMixing_Duration', 0.0), errors='coerce').fillna(0.0) > 0
        return np.where(oil_loading | wet_duration, 'oil_wet', 'no_oil_dry')

    def _get_route_keys(self, df, cluster_col):
        phase = pd.Series(self._phase_route(df), index=df.index)
        if self.use_material_route_matrix:
            mat = df[cluster_col].fillna('GLOBAL').astype(str) if cluster_col in df.columns else pd.Series('GLOBAL', index=df.index)
            return pd.Series(list(zip(mat, phase)), index=df.index)
        return phase

    def _transform_process_deltas(self, df, material_col):
        """Subtract training-fold phase/material nominal process values."""
        route = self._get_route_keys(df, material_col)
        if material_col in df.columns:
            material = df[material_col].fillna('GLOBAL').astype(str)
        else:
            material = pd.Series('GLOBAL', index=df.index)

        phase_series = pd.Series(self._phase_route(df), index=df.index, name='phase_route')
        keys = pd.MultiIndex.from_arrays([phase_series, material], names=['phase_route', 'material_system'])
        nominals = self.process_nominals_.reindex(keys)
        nominals.index = df.index
        nominals = nominals.reindex(columns=self.feature_names_s2_)
        nominals = nominals.fillna(self.global_process_nominal_)
        actual = df[self.feature_names_s2_].apply(pd.to_numeric, errors='coerce')
        return actual.subtract(nominals, axis='columns').fillna(0.0), route

    def fit(self, df_train, feature_cols_stage1, feature_cols_stage2, 
            target_col="MNY", compound_col="CompoundName", cluster_col="material_system"):
        """
        Fits all 3 stages sequentially.
        """
        self.feature_names_s1_ = list(feature_cols_stage1)
        self.feature_names_s2_ = list(feature_cols_stage2)
        
        # 1. Compute effective weights
        df_train = compute_effective_sample_weights(df_train, compound_col=compound_col)
        w_loss = df_train['_w_loss'].values
        
        X_s1 = df_train[self.feature_names_s1_]
        y = df_train[target_col].values
        
        # STAGE 1: Shared Global Recipe Surface GBDT
        print("  [Stage 1] Fitting Global Recipe GBDT Surface...")
        self.stage1_model_ = LGBMRegressor(**self.stage1_params)
        self.stage1_model_.fit(X_s1, y, sample_weight=w_loss)
        self.stage1_feature_importance_ = pd.DataFrame({
            'feature': self.feature_names_s1_,
            'importance': self.stage1_model_.feature_importances_,
        }).sort_values('importance', ascending=False, ignore_index=True)
        
        pred_s1 = self.stage1_model_.predict(X_s1)
        
        # STAGE 1b: Regularized Compound Bias Shrinkage
        print(f"  [Stage 1b] Fitting Regularized Bias Shrinkage (k={self.shrinkage_k})...")
        self.stage1b_bias_ = RegularizedBiasEstimator(
            shrinkage_k=self.shrinkage_k, 
            compound_col=compound_col, 
            cluster_col=cluster_col
        )
        self.stage1b_bias_.fit(y, pred_s1, df_train)
        pred_s1b_bias = self.stage1b_bias_.predict_bias(df_train)
        
        pred_s1_s1b = pred_s1 + pred_s1b_bias
        residuals_s1b = y - pred_s1_s1b
        
        # STAGE 2: Process residual experts
        print(f"  [Stage 2] Fitting Process Residual Experts (Matrix={self.use_material_route_matrix}, SilicaSubsystem={self.use_silica_subsystem}, CBSubsystem={self.use_cb_subsystem})...")
        df_train['_res_s1b'] = residuals_s1b
        df_train['_phase_route'] = self._phase_route(df_train)
        if cluster_col in df_train.columns:
            df_train['_material_system'] = df_train[cluster_col].fillna('GLOBAL').astype(str)
        else:
            df_train['_material_system'] = 'GLOBAL'
        self.process_nominals_ = df_train.groupby(
            ['_phase_route', '_material_system']
        )[self.feature_names_s2_].mean()
        self.process_nominals_.index = self.process_nominals_.index.set_names(['phase_route', 'material_system'])
        self.global_process_nominal_ = df_train[self.feature_names_s2_].mean()
        X_s2_delta, route = self._transform_process_deltas(df_train, cluster_col)

        expert_rows = []
        for route_key in sorted(route.unique(), key=lambda x: str(x)):
            route_mask = route == route_key
            X_route = X_s2_delta.loc[route_mask]
            y_route = df_train.loc[route_mask, '_res_s1b']
            weights_route = df_train.loc[route_mask, '_w_loss']
            
            is_no_oil = ('no_oil_dry' in route_key) if isinstance(route_key, tuple) else (route_key == 'no_oil_dry')
            is_silica = ('Silica' in route_key) if isinstance(route_key, tuple) else (df_train.loc[route_mask, '_material_system'].iloc[0] == 'Silica' if not route_mask.empty else False)
            is_cb = ('CarbonBlack' in route_key) if isinstance(route_key, tuple) else (df_train.loc[route_mask, '_material_system'].iloc[0] == 'CarbonBlack' if not route_mask.empty else False)

            if self.use_silica_subsystem and is_silica:
                # Specialized 4-Expert Silica Residual Subsystem V1
                expert = SilicaSubsystemPredictor(n_splits=5, shrinkage_alpha=1.0)
                expert.fit(X_route, y_route.values, sample_weights=weights_route.values, df_meta=df_train.loc[route_mask])
                model_type = 'silica_4expert_subsystem'
            elif self.use_cb_subsystem and is_cb:
                # Specialized 3-Expert Carbon Black Residual Subsystem V1
                expert = CarbonBlackSubsystemPredictor(n_splits=5, shrinkage_alpha=1.0)
                expert.fit(X_route, y_route.values, sample_weights=weights_route.values)
                model_type = 'cb_3expert_subsystem'
            elif route_mask.sum() < 30:
                expert = Ridge(alpha=10.0)
                expert.fit(X_route, y_route, sample_weight=weights_route)
                model_type = 'ridge_fallback'
            elif is_no_oil and self.no_oil_expert_type == 'ridge':
                expert = Ridge(alpha=5.0)
                expert.fit(X_route, y_route, sample_weight=weights_route)
                model_type = 'ridge_linear'
            elif is_no_oil and self.no_oil_expert_type == 'huber':
                expert = HuberRegressor(alpha=1.0)
                expert.fit(X_route, y_route, sample_weight=weights_route)
                model_type = 'huber_linear'
            else:
                params = self.stage2_params.copy()
                if is_no_oil:
                    params.update({'n_estimators': 180, 'learning_rate': 0.03, 'max_depth': 5, 'min_child_samples': 15, 'reg_alpha': 0.1, 'reg_lambda': 2.0})
                else:
                    params.update({'n_estimators': 150, 'learning_rate': 0.02, 'max_depth': 4, 'min_child_samples': 25, 'reg_alpha': 0.5, 'reg_lambda': 5.0})
                expert = LGBMRegressor(**params)
                expert.fit(X_route, y_route, sample_weight=weights_route)
                model_type = 'lightgbm'
                
            self.stage2_experts_[route_key] = expert
            expert_rows.append({
                'phase_route': str(route_key),
                'n_rows': int(route_mask.sum()),
                'n_label_groups': int(df_train.loc[route_mask, '_label_group_id'].nunique()),
                'model_type': model_type,
            })
        self.stage2_expert_report_ = pd.DataFrame(expert_rows)
            
        print("  Hybrid Unified Architecture fitting complete.")
        return self

    def predict(self, df_test, compound_col="CompoundName", cluster_col="material_system"):
        """
        Generates predictions combining Stage 1 + Stage 1b + Stage 2.
        """
        X_s1 = df_test[self.feature_names_s1_]
        pred_s1 = self.stage1_model_.predict(X_s1)
        pred_s1b_bias = self.stage1b_bias_.predict_bias(df_test)
        
        X_s2_delta, route = self._transform_process_deltas(df_test, cluster_col)
        pred_s2_res = np.zeros(len(df_test))
        fallback_expert = next(iter(self.stage2_experts_.values()))
        
        for route_key in route.unique():
            route_mask = route == route_key
            expert = self.stage2_experts_.get(route_key, fallback_expert)
            pred_s2_res[route_mask] = expert.predict(X_s2_delta.loc[route_mask])
            
        final_pred = pred_s1 + pred_s1b_bias + pred_s2_res
        return final_pred, pred_s1, pred_s1b_bias, pred_s2_res


if __name__ == '__main__':
    print("Hybrid Unified Model Module ready.")
