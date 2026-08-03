import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

class HybridUnifiedArchitectureModel(BaseEstimator, RegressorMixin):
    """
    Hybrid Unified Architecture for Compound Mooney Prediction (V3.2):
      - Stage 1: Unified Global Baseline Model (Recipe/Formulation features)
      - Stage 1b: Group Bias Offset Correction (CompoundName / Cold-Start bias)
      - Stage 2: Dual Cluster Residual Experts (Silica Expert vs Carbon Black Expert)
    """
    def __init__(self, alpha_ridge=1.0, n_estimators=100, learning_rate=0.05, random_state=42):
        self.alpha_ridge = alpha_ridge
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Models
        self.stage1_model = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbosity=-1
        )
        self.stage1b_compound_biases = {}
        
        self.stage2_silica_expert = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbosity=-1
        )
        self.stage2_cb_expert = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbosity=-1
        )
        
    def fit(self, df_train: pd.DataFrame, target_col: str = 'Mooney_Value'):
        y = df_train[target_col].values
        weights = df_train['_sample_weight'].values if '_sample_weight' in df_train.columns else None
        
        # --- STAGE 1: Fit Shared Global Baseline ---
        from Mooney_Prediction_Pipeline.feature_engineering.stage1_recipe_features import extract_stage1_recipe_features
        from Mooney_Prediction_Pipeline.feature_engineering.stage2_process_features import extract_stage2_process_features
        from Mooney_Prediction_Pipeline.feature_engineering.clustering import cluster_silica_carbon_black
        
        df_train = cluster_silica_carbon_black(df_train)
        X_stage1 = extract_stage1_recipe_features(df_train)
        X_stage1_numeric = X_stage1.select_dtypes(include=[np.number]).fillna(0)
        
        self.stage1_model.fit(X_stage1_numeric, y, sample_weight=weights)
        pred_stage1 = self.stage1_model.predict(X_stage1_numeric)
        
        # --- STAGE 1b: Compute Compound Residual Bias ---
        df_train['residual_s1'] = y - pred_stage1
        if 'CompoundName' in df_train.columns:
            self.stage1b_compound_biases = df_train.groupby('CompoundName')['residual_s1'].mean().to_dict()
        else:
            self.stage1b_compound_biases = {}
            
        pred_stage1b = pred_stage1 + df_train['CompoundName'].map(self.stage1b_compound_biases).fillna(0.0).values if 'CompoundName' in df_train.columns else pred_stage1
        df_train['residual_s1b'] = y - pred_stage1b
        
        # --- STAGE 2: Train Residual Cluster Experts ---
        X_stage2 = extract_stage2_process_features(df_train).select_dtypes(include=[np.number]).fillna(0)
        
        is_silica = df_train['compound_cluster'] == 'Silica'
        is_cb = df_train['compound_cluster'] == 'CarbonBlack'
        
        if is_silica.sum() > 0:
            w_sil = weights[is_silica] if weights is not None else None
            self.stage2_silica_expert.fit(X_stage2[is_silica], df_train.loc[is_silica, 'residual_s1b'], sample_weight=w_sil)
            
        if is_cb.sum() > 0:
            w_cb = weights[is_cb] if weights is not None else None
            self.stage2_cb_expert.fit(X_stage2[is_cb], df_train.loc[is_cb, 'residual_s1b'], sample_weight=w_cb)
            
        return self
        
    def predict(self, df_test: pd.DataFrame) -> np.ndarray:
        from Mooney_Prediction_Pipeline.feature_engineering.stage1_recipe_features import extract_stage1_recipe_features
        from Mooney_Prediction_Pipeline.feature_engineering.stage2_process_features import extract_stage2_process_features
        from Mooney_Prediction_Pipeline.feature_engineering.clustering import cluster_silica_carbon_black
        
        df_test = cluster_silica_carbon_black(df_test)
        
        # Stage 1
        X_stage1 = extract_stage1_recipe_features(df_test).select_dtypes(include=[np.number]).fillna(0)
        pred_s1 = self.stage1_model.predict(X_stage1)
        
        # Stage 1b
        if 'CompoundName' in df_test.columns:
            bias = df_test['CompoundName'].map(self.stage1b_compound_biases).fillna(0.0).values
        else:
            bias = np.zeros_like(pred_s1)
        pred_s1b = pred_s1 + bias
        
        # Stage 2
        X_stage2 = extract_stage2_process_features(df_test).select_dtypes(include=[np.number]).fillna(0)
        res_pred = np.zeros(len(df_test))
        
        is_silica = df_test['compound_cluster'] == 'Silica'
        is_cb = df_test['compound_cluster'] == 'CarbonBlack'
        
        if is_silica.sum() > 0:
            res_pred[is_silica] = self.stage2_silica_expert.predict(X_stage2[is_silica])
        if is_cb.sum() > 0:
            res_pred[is_cb] = self.stage2_cb_expert.predict(X_stage2[is_cb])
            
        return pred_s1b + res_pred
