import os
import sys
import shutil
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

WORKSPACE_DIR = os.getcwd()
sys.path.extend([
    WORKSPACE_DIR,
    os.path.join(WORKSPACE_DIR, 'Mooney_Prediction_Pipeline'),
    os.path.join(WORKSPACE_DIR, 'Mooney_Prediction_Pipeline', 'data_processing'),
    os.path.join(WORKSPACE_DIR, 'Mooney_Prediction_Pipeline', 'model_training'),
    os.path.join(WORKSPACE_DIR, 'Mooney_Prediction_Pipeline', 'model_analysis'),
])

recipe_cols = [
    'Top_Fill_Factor', 'Bot_Fill_Factor', 'Target_Temperature',
    'weight_pct_solid_elastomer', 'weight_pct_natural_rubber', 'weight_pct_silica',
    'weight_pct_oil', 'weight_pct_silian', 'weight_pct_carbon_black', 'silica_phr',
    'is_oil_loading_present', 'ratio_nr_rubber', 'ratio_filler_polymer',
    'supplier_rubber_viscosity_avg'
]

process_cols = [
    'phys_init_temp', 'phys_discharge_temp', 'phys_max_temp',
    'Stage2_DryMixing_Duration', 'Stage2_DryMixing_power_Mean',
    'Stage4_WetMixing_Duration', 'Stage4_WetMixing_temp_Mean',
    'Stage6_BottomMixing_Torque_Mean', 'Stage6_BottomMixing_power_Mean', 'Stage6_BottomMixing_Duration',
    'Stage6_BottomMixing_Torque_Integral', 'env_temp_mean', 'env_humidity_mean'
]

class RobustUltimate3StageModel:
    def __init__(self, recipe_cols, process_cols):
        self.recipe_cols = recipe_cols
        self.process_cols = process_cols
        self.baseline_ridge = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-2, 3, 20))
        )
        self.process_ridge = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-2, 3, 20))
        )
        self.compound_bias = {}

    def fit(self, df_train, y_train):
        X_rec = df_train[self.recipe_cols].copy()
        self.baseline_ridge.fit(X_rec, y_train)
        base_preds = self.baseline_ridge.predict(X_rec)
        
        df_tmp = df_train.copy()
        df_tmp['base_pred'] = base_preds
        df_tmp['bias'] = df_tmp['MNY'] - df_tmp['base_pred']
        self.compound_bias = df_tmp.groupby('CompoundName')['bias'].mean().to_dict()
        self.compound_nominals = df_train.groupby('CompoundName')[self.process_cols].mean().to_dict(orient='index')
        
        df_tmp['comp_bias'] = df_tmp['CompoundName'].map(self.compound_bias).fillna(0.0)
        y_stage1 = df_tmp['base_pred'] + df_tmp['comp_bias']
        residuals = df_tmp['MNY'] - y_stage1
        
        X_proc_list = []
        for idx, row in df_train.iterrows():
            c_name = row['CompoundName']
            c_nom = self.compound_nominals.get(c_name, {})
            row_delta = []
            for p_col in self.process_cols:
                nom_val = c_nom.get(p_col, row[p_col] if not pd.isna(row[p_col]) else 0.0)
                act_val = row[p_col] if not pd.isna(row[p_col]) else nom_val
                row_delta.append(act_val - nom_val)
            X_proc_list.append(row_delta)
            
        X_proc_delta = np.array(X_proc_list)
        self.process_ridge.fit(X_proc_delta, residuals)

    def predict(self, df_test, apply_stage3_aakf=True, R_meas=1.0, lab_noise_threshold=3.0):
        X_rec = df_test[self.recipe_cols].copy()
        base_preds = self.baseline_ridge.predict(X_rec)
        comp_biases = df_test['CompoundName'].map(self.compound_bias).fillna(0.0).values
        stage1_preds = base_preds + comp_biases
        
        X_proc_list = []
        for idx, row in df_test.iterrows():
            c_name = row['CompoundName']
            c_nom = self.compound_nominals.get(c_name, {})
            row_delta = []
            for p_col in self.process_cols:
                nom_val = c_nom.get(p_col, row[p_col] if not pd.isna(row[p_col]) else 0.0)
                act_val = row[p_col] if not pd.isna(row[p_col]) else nom_val
                row_delta.append(act_val - nom_val)
            X_proc_list.append(row_delta)
            
        X_proc_delta = np.array(X_proc_list)
        stage2_deltas = self.process_ridge.predict(X_proc_delta)
        raw_stage2_preds = stage1_preds + stage2_deltas
        
        if not apply_stage3_aakf or len(df_test) < 2:
            return raw_stage2_preds
            
        y_true = df_test['MNY'].values if 'MNY' in df_test.columns else None
        if y_true is None:
            return raw_stage2_preds
            
        n = len(df_test)
        stage3_preds = np.zeros(n)
        state_bias = 0.0
        P_state = 2.0
        
        for i in range(n):
            if i == 0:
                stage3_preds[i] = raw_stage2_preds[i] + state_bias
            else:
                innov = y_true[i-1] - stage3_preds[i-1]
                if abs(innov) > lab_noise_threshold:
                    K_k = 0.10
                elif abs(innov) > 1.8:
                    K_k = min(0.75, P_state / (P_state + R_meas) * 1.5)
                else:
                    K_k = P_state / (P_state + R_meas)
                    
                state_bias = state_bias + K_k * innov
                P_state = (1 - K_k) * P_state + 0.2
                stage3_preds[i] = raw_stage2_preds[i] + state_bias
                
        return stage3_preds
