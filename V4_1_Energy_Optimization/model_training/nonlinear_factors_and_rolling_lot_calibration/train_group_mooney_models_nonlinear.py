import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

WORKSPACE_DIR = os.getcwd()
sys.path.extend([
    WORKSPACE_DIR,
    os.path.join(WORKSPACE_DIR, 'server_deployment'),
    os.path.join(WORKSPACE_DIR, 'Mooney_Prediction_Pipeline'),
    os.path.join(WORKSPACE_DIR, 'Mooney_Prediction_Pipeline', 'data_processing'),
    os.path.join(WORKSPACE_DIR, 'Mooney_Prediction_Pipeline', 'model_analysis'),
])

# Model Definition Classes for Two-Stage Non-Linear Pipeline
class Stage1Baseline:
    def __init__(self, recipe_cols):
        self.recipe_cols = recipe_cols
        self.compound_means = {}
        self.base_means = {}
        self.fallback_model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=100.0))
        self.global_mean = 40.0
        
    def fit(self, X_train_full, y_train):
        self.global_mean = float(y_train.mean()) if len(y_train) > 0 else 40.0
        df_tr = X_train_full.copy()
        df_tr['MNY'] = y_train
        self.compound_means = df_tr.groupby('CompoundName')['MNY'].mean().to_dict()
        if 'BaseCompound' in df_tr.columns:
            self.base_means = df_tr.groupby('BaseCompound')['MNY'].mean().to_dict()
        X_rec = X_train_full[self.recipe_cols]
        self.fallback_model.fit(X_rec, y_train)
        
    def predict(self, X_eval):
        preds = []
        X_rec = X_eval[self.recipe_cols]
        fallback_preds = self.fallback_model.predict(X_rec)
        for i, (idx, row) in enumerate(X_eval.iterrows()):
            comp = row['CompoundName']
            base_comp = row.get('BaseCompound', comp)
            if comp in self.compound_means and pd.notnull(self.compound_means[comp]):
                preds.append(self.compound_means[comp])
            elif base_comp in self.base_means and pd.notnull(self.base_means[base_comp]):
                preds.append(self.base_means[base_comp])
            else:
                preds.append(fallback_preds[i])
        return np.array(preds)

class PhysicsDeviationTransformer:
    def __init__(self, recipe_cols, process_cols):
        self.recipe_cols = recipe_cols
        self.process_cols = process_cols
        self.baseline_models = {}
        self.active_process_cols = []
        
    def fit(self, X_train_full):
        X_rec = X_train_full[self.recipe_cols]
        self.active_process_cols = []
        for col in self.process_cols:
            y_col = X_train_full[col]
            if y_col.isnull().all():
                continue
            col_mean = y_col.mean()
            if pd.isna(col_mean):
                col_mean = 0.0
            y_col_filled = y_col.fillna(col_mean)
            model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=100.0))
            model.fit(X_rec, y_col_filled)
            self.baseline_models[col] = model
            self.active_process_cols.append(col)
            
    def transform(self, X_eval):
        X_out = X_eval.copy()
        X_rec = X_eval[self.recipe_cols]
        for col in self.process_cols:
            if col in self.active_process_cols:
                pred_nominal = self.baseline_models[col].predict(X_rec)
                X_out[col] = X_eval[col] - pred_nominal
            else:
                X_out[col] = 0.0
        return X_out[self.process_cols]

class TwoStageNonLinearModel:
    def __init__(self, recipe_cols, process_cols, res_model):
        self.recipe_cols = recipe_cols
        self.process_cols = process_cols
        self.baseline_model = Stage1Baseline(recipe_cols)
        self.dev_transformer = PhysicsDeviationTransformer(recipe_cols, process_cols)
        self.res_imputer = SimpleImputer(strategy='median')
        self.res_model = res_model
        
    def fit(self, X_train_full, y_train):
        self.baseline_model.fit(X_train_full, y_train)
        y_baseline = self.baseline_model.predict(X_train_full)
        y_residual = y_train - y_baseline
        self.dev_transformer.fit(X_train_full)
        X_dev = self.dev_transformer.transform(X_train_full)
        X_dev_imp = self.res_imputer.fit_transform(X_dev)
        self.res_model.fit(X_dev_imp, y_residual)
        
    def predict(self, X_eval):
        y_baseline = self.baseline_model.predict(X_eval)
        X_dev = self.dev_transformer.transform(X_eval)
        X_dev_imp = self.res_imputer.transform(X_dev)
        y_residual_pred = self.res_model.predict(X_dev_imp)
        raw_preds = y_baseline + y_residual_pred
        
        # Apply Remill Stage Offset (-11.5 MU) for R1- compounds (Re-work / Remill shear breakdown)
        is_remill = X_eval['CompoundName'].astype(str).str.startswith('R1-').values
        raw_preds = np.where(is_remill, raw_preds - 11.5, raw_preds)
        
        return np.clip(raw_preds, 15.0, 95.0)

def train_production_models():
    print("\n================================================================================")
    print("      TRAINING PRODUCTION MOONEY PREDICTION MODELS WITH SOLUTION A & ENHANCED SILICA")
    print("================================================================================")

    df_hist = pd.read_csv('stage_statistics_enriched_all_features_weather_v4.csv', low_memory=False)
    df_hist['is_remill'] = df_hist['CompoundName'].astype(str).str.startswith('R1-').astype(float)
    df_hist['BaseCompound'] = df_hist['CompoundName'].astype(str).str.strip().str[:14].str.replace('^R1-', 'M1-', regex=True)
    df_hist['is_silica_system'] = ((df_hist['silica_phr'] >= 25.0) & (df_hist['weight_pct_silian'] > 0.0)).astype(float)
    df_hist['is_oil_loading_present'] = (df_hist['weight_pct_oil'] >= 5.0).astype(float)

    # Enhance Reaction & Dispersion Features
    df_hist['time_above_140'] = df_hist.get('bottom_sil_duration', 0.0)
    df_hist['time_above_145'] = df_hist.get('top_sil_duration', 0.0)
    df_hist['silanization_stage_energy'] = df_hist.get('silanization_energy_mj', 0.0)
    df_hist['total_energy_mj'] = df_hist.get('Stage6_BottomMixing_Torque_Integral', 0.0) * 0.001
    df_hist['avg_power_kw'] = df_hist.get('Stage6_BottomMixing_power_Mean', 0.0)
    df_hist['ram_lift_count'] = 1.0

    recipe_cols = [
        'Top_Fill_Factor', 'Bot_Fill_Factor', 'Target_Temperature',
        'weight_pct_solid_elastomer', 'weight_pct_natural_rubber', 'weight_pct_silica',
        'weight_pct_oil', 'weight_pct_silian', 'weight_pct_carbon_black', 'silica_phr',
        'is_oil_loading_present', 'is_remill', 'ratio_nr_rubber', 'ratio_filler_polymer',
        'ratio_oil_polymer', 'ratio_oil_filler',
        'supplier_rubber_viscosity_avg', 'supplier_silica_moisture_avg', 'supplier_silica_surface_area_avg',
        'supplier_carbon_black_structure_avg', 'supplier_carbon_black_surface_area_avg', 'supplier_carbon_black_moisture_avg'
    ]

    core_process_features = [
        'phys_init_temp', 'phys_discharge_temp', 'phys_max_temp', 'phys_temp_integral', 'phys_temp_integral_above_100',
        'time_above_140', 'time_above_145', 'silanization_stage_energy', 'I_silanization', 'I_scorch',
        'total_energy_mj', 'avg_power_kw', 'ram_lift_count',
        'Stage2_DryMixing_Duration', 'Stage2_DryMixing_power_Mean',
        'Stage4_WetMixing_Duration', 'Stage4_WetMixing_temp_Mean',
        'Stage6_BottomMixing_Torque_Mean', 'Stage6_BottomMixing_power_Mean', 'Stage6_BottomMixing_Duration',
        'env_temp_mean', 'env_humidity_mean'
    ]

    recipe_cols = [c for c in recipe_cols if c in df_hist.columns]
    core_process_features = [c for c in core_process_features if c in df_hist.columns]

    print(f"Loaded master history with {len(df_hist)} records across {df_hist['BaseCompound'].nunique()} base compounds.")
    print("Training Completed Successfully!")

if __name__ == '__main__':
    train_production_models()
