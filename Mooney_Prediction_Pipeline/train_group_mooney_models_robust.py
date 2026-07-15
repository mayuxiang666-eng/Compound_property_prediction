# ----------------------------------------------------
# Mooney Prediction Pipeline V2.0 Path Bootstrap
# ----------------------------------------------------
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_DIR = os.path.dirname(PARENT_DIR)

sys.path.extend([
    PARENT_DIR,
    os.path.join(PARENT_DIR, 'data_processing'),
    os.path.join(PARENT_DIR, 'model_training'),
    os.path.join(PARENT_DIR, 'model_analysis'),
])
# ----------------------------------------------------

import re
import warnings
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

# ----------------------------------------------------
# Helper to clean feature names
# ----------------------------------------------------
def clean_feature_name(name):
    if not isinstance(name, str):
        return name
    return re.sub(r'[^\w]', '_', name.strip())

# ----------------------------------------------------
# 1. Classes for Two-Stage Modeling with Scaling
# ----------------------------------------------------

class Stage1Baseline:
    def __init__(self, recipe_cols):
        self.recipe_cols = recipe_cols
        self.compound_means = {}
        self.fallback_model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=100.0))
        self.global_mean = 0.0
        
    def fit(self, X_train_full, y_train):
        self.global_mean = y_train.mean()
        # Compute compound means
        df_tr = X_train_full.copy()
        df_tr['MNY'] = y_train
        self.compound_means = df_tr.groupby('CompoundName')['MNY'].mean().to_dict()
        
        # Fit fallback model on recipe features
        X_rec = X_train_full[self.recipe_cols]
        self.fallback_model.fit(X_rec, y_train)
        
    def predict(self, X_eval):
        preds = []
        X_rec = X_eval[self.recipe_cols]
        fallback_preds = self.fallback_model.predict(X_rec)
        
        for i, (idx, row) in enumerate(X_eval.iterrows()):
            comp = row['CompoundName']
            if comp in self.compound_means:
                preds.append(self.compound_means[comp])
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


# ----------------------------------------------------
# PyTorch Robust (Minimax) Ridge Regression
# ----------------------------------------------------
class RobustRidgeRegression:
    def __init__(self, alpha=100.0, max_iter=1000, lr=0.01):
        self.alpha = alpha
        self.max_iter = max_iter
        self.lr = lr
        self.coef_ = None
        self.intercept_ = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y, groups):
        # 1. Standardize features
        X_scaled = self.scaler.fit_transform(X)
        D = X_scaled.shape[1]
        
        # 2. Group the data points
        group_dict = {}
        for idx, g in enumerate(groups):
            if g not in group_dict:
                group_dict[g] = []
            group_dict[g].append(idx)
            
        max_size = max(len(indices) for indices in group_dict.values())
        N_groups = len(group_dict)
        
        # 3. Create padded arrays
        X_padded = np.zeros((N_groups, max_size, D))
        y_padded = np.zeros((N_groups, 1))
        
        for g_idx, (key, indices) in enumerate(group_dict.items()):
            for b_idx, idx in enumerate(indices):
                X_padded[g_idx, b_idx, :] = X_scaled[idx]
            for b_idx in range(len(indices), max_size):
                X_padded[g_idx, b_idx, :] = X_scaled[indices[-1]]
                
            y_padded[g_idx, 0] = y.iloc[indices[0]] if hasattr(y, 'iloc') else y[indices[0]]
            
        # 4. Train in PyTorch
        import torch
        import torch.optim as optim
        
        X_tensor = torch.tensor(X_padded, dtype=torch.float32)
        y_tensor = torch.tensor(y_padded, dtype=torch.float32)
        
        w = torch.zeros(D, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        
        optimizer = optim.Adam([w, b], lr=self.lr)
        
        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            preds = torch.matmul(X_tensor, w) + b
            errors = (preds - y_tensor) ** 2
            max_errors = torch.max(errors, dim=1)[0]
            loss = torch.mean(max_errors) + (self.alpha / N_groups) * torch.sum(w ** 2)
            loss.backward()
            optimizer.step()
            
        self.coef_ = w.detach().numpy()
        self.intercept_ = b.detach().numpy()[0]
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return np.dot(X_scaled, self.coef_) + self.intercept_


class TwoStageModel:
    def __init__(self, recipe_cols, process_cols, alpha=100.0, max_iter=1000, lr=0.01):
        self.recipe_cols = recipe_cols
        self.process_cols = process_cols
        self.baseline_model = Stage1Baseline(recipe_cols)
        self.dev_transformer = PhysicsDeviationTransformer(recipe_cols, process_cols)
        self.residual_model = RobustRidgeRegression(alpha=alpha, max_iter=max_iter, lr=lr)
        
    def fit(self, X_train_unagg, y_train_unagg, group_keys):
        def get_centroid(df_unagg):
            agg_dict_temp = {}
            for col in df_unagg.columns:
                if col in ['OrderID', 'PalletID']:
                    continue
                if col in self.recipe_cols or col in self.process_cols or col == 'MNY':
                    agg_dict_temp[col] = 'mean'
                elif df_unagg[col].dtype == object or isinstance(df_unagg[col].dtype, pd.CategoricalDtype):
                    agg_dict_temp[col] = 'first'
                else:
                    agg_dict_temp[col] = 'first'
            return df_unagg.groupby(['OrderID', 'PalletID']).agg(agg_dict_temp).reset_index()
            
        df_train_centroid = get_centroid(X_train_unagg)
        self.baseline_model.fit(df_train_centroid, df_train_centroid['MNY'])
        
        self.dev_transformer.fit(X_train_unagg)
        X_dev = self.dev_transformer.transform(X_train_unagg)
        
        y_baseline = self.baseline_model.predict(X_train_unagg)
        y_residual = y_train_unagg - y_baseline
        
        self.imputer = SimpleImputer(strategy='median')
        X_dev_imputed = self.imputer.fit_transform(X_dev)
        X_dev_imputed_df = pd.DataFrame(X_dev_imputed, columns=X_dev.columns, index=X_dev.index)
        
        self.residual_model.fit(X_dev_imputed_df, y_residual, group_keys)
        
    def predict(self, X_eval):
        y_baseline = self.baseline_model.predict(X_eval)
        X_dev = self.dev_transformer.transform(X_eval)
        X_dev_imputed = self.imputer.transform(X_dev)
        X_dev_imputed_df = pd.DataFrame(X_dev_imputed, columns=X_dev.columns, index=X_dev.index)
        y_residual_pred = self.residual_model.predict(X_dev_imputed_df)
        return y_baseline + y_residual_pred


# ----------------------------------------------------
# 2. Loading data and computing kinetics
# ----------------------------------------------------

def main():
    print("\n=== STEP 1: LOADING DATA & MERGING RAW TEMPERATURE CURVES ===")
    df_raw = pd.read_csv(os.path.join(WORKSPACE_DIR, 'stage_statistics_enriched.csv'), usecols=['OrderID', 'BatchNumber', 'temp'], low_memory=False)
    df_raw['OrderID'] = df_raw['OrderID'].astype(str).str.strip()
    df_raw['BatchNumber'] = df_raw['BatchNumber'].astype(int)

    df_seg = pd.read_csv(os.path.join(WORKSPACE_DIR, 'stage_statistics_enriched_all_features_weather_v4.csv'), low_memory=False)
    df_seg['OrderID'] = df_seg['OrderID'].astype(str).str.strip()
    df_seg['BatchNumber'] = df_seg['BatchNumber'].astype(int)

    # Merge temp curve column
    df_seg = pd.merge(df_seg, df_raw, on=['OrderID', 'BatchNumber'], how='left')
    print(f"Loaded {len(df_seg)} segmented batches.")

    def hex_to_series(hex_str):
        if pd.isnull(hex_str):
            return np.array([])
        hex_str = str(hex_str).replace(" ", "").replace("\n", "").replace("\r", "")
        hex_step = 4
        try:
            vals = [int(hex_str[i:i+hex_step], 16) for i in range(0, len(hex_str), hex_step)]
            return np.array(vals)
        except Exception:
            return np.array([])

    def calculate_kinetics(temp_array):
        temp_array = temp_array[(temp_array >= 20.0) & (temp_array <= 200.0)]
        if len(temp_array) == 0:
            return 0.0, 0.0
        Ea_sil = 75000.0
        R = 8.314
        T_ref_sil = 150.0 + 273.15
        T_K = temp_array + 273.15
        rate_sil = np.exp(-(Ea_sil / R) * (1.0 / T_K - 1.0 / T_ref_sil))
        active_mask_sil = temp_array >= 135.0
        I_silanization = np.sum(rate_sil[active_mask_sil])
        
        Ea_scorch = 120000.0
        T_ref_scorch = 165.0 + 273.15
        rate_scorch = np.exp(-(Ea_scorch / R) * (1.0 / T_K - 1.0 / T_ref_scorch))
        active_mask_scorch = temp_array > 165.0
        I_scorch = np.sum(rate_scorch[active_mask_scorch])
        
        return float(I_silanization), float(I_scorch)

    print("Computing silanization and scorch kinetics reaction indices on curves...")
    sil_indices = []
    scorch_indices = []
    for idx, row in df_seg.iterrows():
        hex_str = row['temp']
        temp_curve = hex_to_series(hex_str)
        I_sil, I_scorch = calculate_kinetics(temp_curve)
        sil_indices.append(I_sil)
        scorch_indices.append(I_scorch)

    df_seg['I_silanization'] = sil_indices
    df_seg['I_scorch'] = scorch_indices
    df_seg = df_seg.drop(columns=['temp'])
    print("Kinetics indices calculated successfully.")

    # ----------------------------------------------------
    # Merge Redshift database KPIs
    # ----------------------------------------------------
    print("\n=== STEP 2: MERGING REDSHIFT SILANIZATION & DRY MIXING KPIS ===")
    kpis_path = os.path.join(WORKSPACE_DIR, "Mooney_Prediction_Pipeline", "models", "hefei_silica_primus_kpis.csv")
    if os.path.exists(kpis_path):
        print("Merging database-derived KPIs...")
        df_kpi = pd.read_csv(kpis_path)
        df_kpi['OrderID_key'] = df_kpi['batch_information_fk'].apply(lambda x: str(x).split('_')[0].strip())
        df_kpi['BatchNumber_key'] = df_kpi['batch_information_fk'].apply(lambda x: int(str(x).split('_')[1]))
        df_kpi = df_kpi.drop_duplicates(subset=['OrderID_key', 'BatchNumber_key'])
        df_kpi = df_kpi.drop(columns=['batch_information_fk'])
        
        df_seg = df_seg.merge(df_kpi, left_on=['OrderID', 'BatchNumber'], right_on=['OrderID_key', 'BatchNumber_key'], how='left')
        df_seg = df_seg.drop(columns=['OrderID_key', 'BatchNumber_key'])
        print(f"Merged Redshift KPIs. Columns added: {list(df_kpi.columns[:-2])}")
    else:
        print("[WARNING] hefei_silica_primus_kpis.csv not found. Skipping merge.")

    # ----------------------------------------------------
    # Pre-Aggregation setup
    # ----------------------------------------------------
    df_seg['MNY'] = pd.to_numeric(df_seg['MNY'], errors='coerce')
    df_seg = df_seg.dropna(subset=['MNY'])
    print(f"Valid MNY rows for processing: {len(df_seg)}")

    df_seg['PalletID'] = df_seg['PalletID'].astype(str).str.strip()

    # Core recipe features (static Layer 1)
    recipe_cols = [
        'Top_Fill_Factor', 'Bot_Fill_Factor', 'Target_Temperature',
        'weight_pct_solid_elastomer', 'weight_pct_natural_rubber', 'weight_pct_silica',
        'weight_pct_oil', 'weight_pct_silian', 'weight_pct_carbon_black', 'silica_phr',
        'is_oil_loading_present', 'ratio_nr_rubber', 'ratio_filler_polymer',
        'ratio_oil_polymer', 'ratio_oil_filler',
        'supplier_rubber_viscosity_avg', 'supplier_silica_moisture_avg', 'supplier_silica_surface_area_avg',
        'supplier_carbon_black_structure_avg', 'supplier_carbon_black_surface_area_avg', 'supplier_carbon_black_moisture_avg'
    ]
    recipe_cols = [clean_feature_name(r) for r in recipe_cols]

    # Core process features to keep (only mean aggregation for step 3 IF)
    core_process_features = [
        'phys_discharge_temp', 'phys_max_temp', 'phys_eta_app_discharge',
        'Stage6_BottomMixing_Torque_Mean', 'Stage6_BottomMixing_power_Mean', 'Stage6_BottomMixing_Duration',
        'Stage6_BottomMixing_Torque_Integral', 'Stage4_WetMixing_temp_Mean', 'Stage4_WetMixing_Duration',
        'Stage2_DryMixing_Duration', 'Stage2_DryMixing_power_Mean',
        'env_temp_mean', 'env_humidity_mean',
        # Silica-specific
        'I_silanization', 'I_scorch', 'time_to_sil_plateau_duration', 'top_sil_duration',
        'bottom_sil_duration', 'silanization_energy_mj', 'top_avg_sil_temperature', 'bottom_avg_sil_temperature'
    ]
    core_process_features = [clean_feature_name(f) for f in core_process_features]

    all_cols = list(df_seg.columns)
    recipe_cols = [r for r in recipe_cols if r in all_cols]
    core_process_features = [f for f in core_process_features if f in all_cols]

    # Build aggregation dictionary
    agg_dict = {}
    for r in recipe_cols:
        agg_dict[r] = 'mean'
    agg_dict['MNY'] = 'mean'
    agg_dict['CompoundName'] = 'first'
    agg_dict['CompoundDescription'] = 'first'
    agg_dict['MixerLine'] = 'first'
    agg_dict['OrderStartTime'] = 'first'

    for f in core_process_features:
        agg_dict[f] = 'mean'

    # Group by actual PalletID to run Isolation Forest at the group level
    print("\n=== STEP 3: RUNNING GROUP LEVEL AGGREGATION FOR ANOMALY DETECTION ===")
    grouped = df_seg.groupby(['OrderID', 'PalletID']).agg(agg_dict)
    grouped = grouped.reset_index()

    group_sizes = df_seg.groupby(['OrderID', 'PalletID']).size().reset_index(name='N_batches')
    df_group = pd.merge(grouped, group_sizes, on=['OrderID', 'PalletID'], how='inner')

    # ----------------------------------------------------
    # Outlier Detection on Group Level
    # ----------------------------------------------------
    print("\n=== STEP 4: RUNNING GROUP LEVEL ISOLATION FOREST ===")
    features_for_if = [c for c in df_group.columns if c in core_process_features]
    X_if = df_group[features_for_if].copy()
    imputer = SimpleImputer(strategy='median')
    X_if_imputed = imputer.fit_transform(X_if)

    clf = IsolationForest(contamination=0.03, random_state=42, n_jobs=-1)
    preds = clf.fit_predict(X_if_imputed)

    # Clean group level data
    df_clean_groups = df_group[preds == 1].copy()
    num_discarded = np.sum(preds == -1)

    print(f"==================== GROUP LEVEL ISOLATION FOREST REPORT ==================")
    print(f"Total input groups: {len(df_group)}")
    print(f"Discarded anomalous groups: {num_discarded} ({num_discarded/len(df_group):.2%})")
    print(f"Cleaned group dataset size: {len(df_clean_groups)} rows.")
    print("=============================================================================")

    # Keep all unaggregated batches belonging to clean groups
    df_seg_clean = df_seg.merge(df_clean_groups[['OrderID', 'PalletID']], on=['OrderID', 'PalletID'], how='inner').copy()
    print(f"Unaggregated clean dataset size: {len(df_seg_clean)} rows.")

    # ----------------------------------------------------
    # 5. Split and Train Two-Stage Group Models with Robust Minimax
    # ----------------------------------------------------
    print("\n=== STEP 5: TRAINING SEPARATED TWO-STAGE GROUP MODELS WITH ROBUST MINIMAX ===")

    def run_two_stage_modeling(df_track_unagg, track_name, output_dir, track_recipe_cols, track_process_cols):
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean feature names
        df_track_unagg = df_track_unagg.rename(columns={col: clean_feature_name(col) for col in df_track_unagg.columns})
        
        available_recipe = [r for r in track_recipe_cols if r in df_track_unagg.columns]
        available_process = [p for p in track_process_cols if p in df_track_unagg.columns]
        
        # Get unique groups
        df_groups_metadata = df_track_unagg.groupby(['OrderID', 'PalletID']).first().reset_index()
        print(f"\n--- Training Two-Stage Group Model (Robust Minimax) for {track_name} ({len(df_groups_metadata)} groups, {len(df_track_unagg)} individual batches) ---")
        
        # Helper to compute centroid (mean of features) for a subset of unaggregated batches
        def get_centroid(df_unagg):
            agg_dict_temp = {}
            for col in df_unagg.columns:
                if col in ['OrderID', 'PalletID']:
                    continue
                if col in available_recipe or col in available_process or col == 'MNY':
                    agg_dict_temp[col] = 'mean'
                elif df_unagg[col].dtype == object or isinstance(df_unagg[col].dtype, pd.CategoricalDtype):
                    agg_dict_temp[col] = 'first'
                else:
                    agg_dict_temp[col] = 'first'
            return df_unagg.groupby(['OrderID', 'PalletID']).agg(agg_dict_temp).reset_index()

        # GroupKFold split on the unique groups (using OrderID as group key to prevent leakage)
        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(df_groups_metadata))
        
        df_track_unagg['group_key'] = list(zip(df_track_unagg['OrderID'], df_track_unagg['PalletID']))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df_groups_metadata, df_groups_metadata['MNY'], df_groups_metadata['OrderID'])):
            train_groups_df = df_groups_metadata.iloc[train_idx]
            val_groups_df = df_groups_metadata.iloc[val_idx]
            
            train_keys = set(zip(train_groups_df['OrderID'], train_groups_df['PalletID']))
            val_keys = set(zip(val_groups_df['OrderID'], val_groups_df['PalletID']))
            
            df_train_unagg = df_track_unagg[df_track_unagg['group_key'].isin(train_keys)].copy()
            df_val_unagg = df_track_unagg[df_track_unagg['group_key'].isin(val_keys)].copy()
            
            # --- Minimax Robust optimization for Fold ---
            ts_model = TwoStageModel(available_recipe, available_process, alpha=100.0)
            # Pass group_key to the custom fit method
            ts_model.fit(df_train_unagg, df_train_unagg['MNY'], df_train_unagg['group_key'].values)
            
            # Predict on centroid of validation groups
            df_val_centroid = get_centroid(df_val_unagg)
            val_preds = ts_model.predict(df_val_centroid)
            
            # Map predictions back to the GroupKFold indexing of df_groups_metadata
            pred_map = dict(zip(zip(df_val_centroid['OrderID'], df_val_centroid['PalletID']), val_preds))
            for i, idx_in_meta in enumerate(val_idx):
                row_meta = df_groups_metadata.iloc[idx_in_meta]
                key = (row_meta['OrderID'], row_meta['PalletID'])
                oof_preds[idx_in_meta] = pred_map[key]
                
        # Compute overall CV metrics on group level
        y_group = df_groups_metadata['MNY'].values
        final_rmse = np.sqrt(mean_squared_error(y_group, oof_preds))
        final_mae = mean_absolute_error(y_group, oof_preds)
        final_r2 = r2_score(y_group, oof_preds)
        
        # Compute Within-Compound Deviation Correlation
        df_eval = df_groups_metadata.copy()
        df_eval['Predicted_MNY'] = oof_preds
        
        comp_stats = df_eval.groupby('CompoundName').agg(
            mean_actual=('MNY', 'mean'),
            mean_pred=('Predicted_MNY', 'mean'),
            count=('MNY', 'count')
        ).reset_index()
        
        df_eval = df_eval.merge(comp_stats, on='CompoundName', how='left')
        df_eval['actual_dev'] = df_eval['MNY'] - df_eval['mean_actual']
        df_eval['pred_dev'] = df_eval['Predicted_MNY'] - df_eval['mean_pred']
        
        df_multi = df_eval[df_eval['count'] >= 3]
        dev_corr = np.nan
        if len(df_multi) > 1 and df_multi['actual_dev'].std() > 1e-5 and df_multi['pred_dev'].std() > 1e-5:
            dev_corr = df_multi['actual_dev'].corr(df_multi['pred_dev'])
            
        actual_std = df_groups_metadata['MNY'].std()
        pred_std = oof_preds.std()
        std_ratio = pred_std / actual_std if actual_std > 0 else 0.0
        
        print(f"==================== TWO-STAGE MODEL CV REPORT ({track_name} - Robust) ====================")
        print(f"Group CV R^2      : {final_r2:.4f}")
        print(f"Group CV MAE      : {final_mae:.4f} MU")
        print(f"Group CV RMSE     : {final_rmse:.4f} MU")
        print(f"Within-Compound Deviation Correlation (N={len(df_multi)}): {dev_corr:+.4f}")
        print(f"Standard Deviation Ratio (Pred/Actual)      : {std_ratio:.4f}")
        print("==================================================================================")
        
        # Fit final model on all clean training data
        final_ts_model = TwoStageModel(available_recipe, available_process, alpha=100.0)
        final_ts_model.fit(df_track_unagg, df_track_unagg['MNY'], df_track_unagg['group_key'].values)
            
        bundle_path = os.path.join(output_dir, "mooney_group_model_bundle.joblib")
        bundle = {
            'model': final_ts_model,
            'recipe_cols': available_recipe,
            'process_cols': available_process,
            'track_name': track_name,
            'cv_metrics': {
                'r2': final_r2,
                'mae': final_mae,
                'rmse': final_rmse,
                'within_compound_corr': dev_corr,
                'std_ratio': std_ratio
            }
        }
        joblib.dump(bundle, bundle_path)
        
        # Save report
        with open(os.path.join(output_dir, "mny_predictive_modeling_report.md"), 'w', encoding='utf-8') as f:
            f.write(f"# Two-Stage Group Predictive Modeling Report (Robust Minimax) - {track_name}\n\n")
            f.write(f"This model uses a Two-Stage architecture with PyTorch Minimax optimization to decouple static recipe baseline and dynamic process variations.\n\n")
            f.write(f"## 5-Fold Group CV Performance\n")
            f.write(f"- **R^2**: {final_r2:.4f}\n")
            f.write(f"- **MAE**: {final_mae:.4f} MU\n")
            f.write(f"- **RMSE**: {final_rmse:.4f} MU\n")
            f.write(f"- **Within-Compound Deviation Correlation**: {dev_corr:+.4f}\n")
            f.write(f"- **Pred/Actual Std Ratio**: {std_ratio:.4f}\n\n")
            f.write(f"## Core Features Utilized\n")
            f.write(f"### Stage 1 Recipe Features:\n")
            f.write(f"{available_recipe}\n\n")
            f.write(f"### Stage 2 Core Process Features:\n")
            f.write(f"{available_process}\n")
            
        return final_r2, final_rmse, dev_corr

    df_seg_clean['is_silica_system'] = ((df_seg_clean['silica_phr'] >= 25.0) & (df_seg_clean['weight_pct_silian'] > 0.0)).astype(float)

    sub_tracks = [
        ("With-Oil High-Silica", 1.0, 1.0, "results_with_oil_high_silica"),
        ("With-Oil Carbon-Black", 1.0, 0.0, "results_with_oil_carbon_black"),
        ("Without-Oil High-Silica", 0.0, 1.0, "results_without_oil_high_silica"),
        ("Without-Oil Carbon-Black", 0.0, 0.0, "results_without_oil_carbon_black")
    ]

    results = {}
    for track_name, is_oil, is_silica, folder_name in sub_tracks:
        df_sub = df_seg_clean[(df_seg_clean['is_oil_loading_present'] == is_oil) & (df_seg_clean['is_silica_system'] == is_silica)].copy()
        
        unique_groups = df_sub.groupby(['OrderID', 'PalletID']).ngroups
        if unique_groups < 40:
            print(f"\n[Warning] Too few sub-track groups ({unique_groups}). Falling back to parent track.")
            df_sub = df_seg_clean[df_seg_clean['is_oil_loading_present'] == is_oil].copy()
            track_name = "With-Oil Fallback" if is_oil == 1.0 else "Without-Oil Fallback"
            
        r2, rmse, dev_corr = run_two_stage_modeling(
            df_sub, 
            track_name, 
            os.path.join(WORKSPACE_DIR, "Mooney_Prediction_Pipeline", "models", folder_name, "group_model_robust"),
            recipe_cols,
            core_process_features
        )
        results[track_name] = {'R2': r2, 'RMSE': rmse, 'DevCorr': dev_corr}
        
    print("\n" + "="*60)
    print("      ALL TWO-STAGE GROUP ROBUST MODELS TRAINED SUCCESSFULLY")
    print("="*60)
    for k, v in results.items():
        print(f"  {k:25} | CV R2: {v['R2']:.4f} | CV RMSE: {v['RMSE']:.4f} MU | DevCorr: {v['DevCorr']:+.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
