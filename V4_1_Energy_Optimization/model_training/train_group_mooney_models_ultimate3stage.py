# ============================================================================
# Ultimate 3-Stage Mooney Prediction Pipeline V3.0
# with Dual-Dimension Compound Clustering
# ============================================================================
# Architecture:
#   Stage 1: RidgeCV Baseline (Recipe PHR + Supplier COA + Cluster Bias)
#   Stage 2: RidgeCV Process Residual (Delta-X deviations, cluster-level nominals)
#   Stage 3: Hampel-Clamped Adaptive Kalman Filter (AAKF)
#   NEW: CompoundClusterer for compound identification via recipe+curve fingerprints
# ============================================================================

import os
import sys
import re
import warnings
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# Path bootstrap
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_DIR = os.path.dirname(PARENT_DIR)
sys.path.extend([PARENT_DIR, SCRIPT_DIR])

from compound_clustering import CompoundClusterer

# ============================================================================
# Feature Definitions
# ============================================================================
# ---------------------------------------------------------------------------
# Helper: add label group information (ID, size, sample weight)
# ---------------------------------------------------------------------------
def add_label_group_information(df, label_group_cols):
    """Add _label_group_id, _label_group_size, and _sample_weight columns.
    Handles missing/invalid values and falls back to OrderID|BatchNumber if needed.
    """
    df = df.copy()
    # Determine which columns actually exist
    valid_cols = [col for col in label_group_cols if col in df.columns]
    if not valid_cols:
        # Fallback to OrderID + BatchNumber (ensures uniqueness)
        df['_label_group_id'] = (
            df['OrderID'].astype(str) + '|BATCH|' + df['BatchNumber'].astype(str)
        )
    else:
        work = df[valid_cols].copy()
        for col in valid_cols:
            work[col] = work[col].astype('string').str.strip()
        # Identify rows with any missing/invalid identifier
        invalid = work.isna().any(axis=1)
        for col in valid_cols:
            invalid |= work[col].isin(['', 'nan', 'None', 'NULL', '<NA>'])
        # Build ID by concatenating available columns
        df['_label_group_id'] = work.fillna('').agg('|'.join, axis=1)
        # Fallback for invalid rows
        fallback = (
            df['OrderID'].astype(str) + '|BATCH|' + df['BatchNumber'].astype(str)
        )
        df.loc[invalid, '_label_group_id'] = fallback[invalid]
    # Group size and sample weight
    df['_label_group_size'] = df.groupby('_label_group_id')['_label_group_id'].transform('size')
    df['_sample_weight'] = 1.0 / df['_label_group_size'].clip(lower=1)
    return df


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
    'Stage6_BottomMixing_Torque_Mean', 'Stage6_BottomMixing_power_Mean',
    'Stage6_BottomMixing_Duration', 'Stage6_BottomMixing_Torque_Integral',
    'env_temp_mean', 'env_humidity_mean'
]


def clean_feature_name(name):
    if not isinstance(name, str):
        return name
    return re.sub(r'[^\w]', '_', name.strip())


# ============================================================================
# Robust Ultimate 3-Stage Model with Cluster-Based Compound Identification
# ============================================================================

# ============================================================================
# Robust Ultimate 3-Stage Model with Cluster-Based Compound Identification
# ============================================================================

class CalibrationState:
    """
    State container for Stage 3 Kalman online calibration.
    """
    def __init__(self, state_bias=0.0, P_state=2.0, R_meas=1.0, lab_noise_threshold=3.0):
        self.state_bias = state_bias
        self.P_state = P_state
        self.R_meas = R_meas
        self.lab_noise_threshold = lab_noise_threshold
        
    def to_dict(self):
        return {
            'state_bias': self.state_bias,
            'P_state': self.P_state,
            'R_meas': self.R_meas,
            'lab_noise_threshold': self.lab_noise_threshold
        }


class RobustUltimate3StageModel:
    """
    3-Stage decoupled Mooney prediction model with dual-dimension compound clustering.
    
    Stage 1: RidgeCV baseline from recipe PHR features + cluster/compound shrunk bias
    Stage 2: RidgeCV residual from process deviation features (Delta-X)
    Stage 3: Decoupled Kalman filter calibration (AAKF)
    """
    
    def __init__(self, recipe_cols, process_cols, min_samples_for_independent=5, lambda_shrinkage=5.0):
        self.recipe_cols = recipe_cols
        self.process_cols = process_cols
        self.min_samples_for_independent = min_samples_for_independent
        self.lambda_shrinkage = lambda_shrinkage
        
        # We manually manage scaling to guarantee sample_weight works flawlessly
        self.baseline_imputer = SimpleImputer(strategy='median')
        self.baseline_scaler = StandardScaler()
        self.baseline_ridge = RidgeCV(alphas=np.logspace(-2, 3, 20))
        
        self.process_imputer = SimpleImputer(strategy='median')
        self.process_scaler = StandardScaler()
        self.process_ridge = RidgeCV(alphas=np.logspace(-2, 3, 20))
        
        # Clustering engine
        self.clusterer = CompoundClusterer(
            recipe_weight=0.5,
            curve_weight=0.5,
            min_samples_for_independent_bias=min_samples_for_independent,
            min_cluster_size=3,
            large_sample_threshold=30
        )
        
        # Bias dictionaries
        self.compound_bias = {}       # compound_name -> shrunk bias
        self.compound_raw_bias = {}   # compound_name -> raw mean bias
        self.cluster_bias = {}        # cluster_id -> cluster-average bias
        self.compound_nominals = {}    # compound_name -> process nominals
        self.cluster_nominals = {}     # cluster_id -> cluster-average nominals
        self.global_mean_bias = 0.0
        
        # Feature coefficients for explainability
        self.feature_importance_s1 = {}
        self.feature_importance_s2 = {}
        
    def fit(self, df_train, y_train, sample_weight=None):
        """
        Fit the 3-stage model:
        1. Fit CompoundClusterer on all training data
        2. Fit Stage 1 RidgeCV on recipe features (with sample weights)
        3. Compute compound-level and cluster-level biases with shrinkage
        4. Fit Stage 2 RidgeCV on process deviations (with sample weights)
        """
        df_train = df_train.copy()
        df_train['CompoundName'] = df_train['CompoundName'].astype(str).str.strip().str.upper()
        
        # Handle sample weight defaults
        if sample_weight is None:
            sample_weight = np.ones(len(df_train))
        else:
            sample_weight = np.array(sample_weight)
            
        # Normalize weights to sum up to N
        sample_weight = sample_weight * (len(df_train) / np.sum(sample_weight))
        
        # --- Step 1: Fit Compound Clusterer ---
        self.clusterer.fit(df_train)
        
        # --- Step 2: Fit Stage 1 Baseline ---
        X_rec = df_train[self.recipe_cols].copy()
        X_rec_imp = self.baseline_imputer.fit_transform(X_rec)
        X_rec_sc = self.baseline_scaler.fit_transform(X_rec_imp)
        
        self.baseline_ridge.fit(X_rec_sc, y_train, sample_weight=sample_weight)
        base_preds = self.baseline_ridge.predict(X_rec_sc)
        
        # Store S1 feature coefficients
        coef_s1 = np.ravel(self.baseline_ridge.coef_)
        self.feature_importance_s1 = dict(zip(self.recipe_cols[:len(coef_s1)], coef_s1))
        
        # --- Step 3: Compute biases and apply regularized shrinkage ---
        df_tmp = df_train.copy()
        df_tmp['base_pred'] = base_preds
        df_tmp['bias'] = df_tmp['MNY'] - df_tmp['base_pred']
        df_tmp['_w'] = sample_weight
        
        # 3a. Weighted raw compound-level biases
        def weighted_mean(g):
            return np.sum(g['bias'] * g['_w']) / np.sum(g['_w'])
            
        self.compound_raw_bias = df_tmp.groupby('CompoundName').apply(weighted_mean).to_dict()
        self.global_mean_bias = np.sum(df_tmp['bias'] * df_tmp['_w']) / np.sum(df_tmp['_w'])
        
        # 3b. Cluster-level biases
        cluster_bias_accum = {}
        for comp_name, raw_bias in self.compound_raw_bias.items():
            cluster_id = self.clusterer.compound_to_cluster.get(comp_name, -1)
            if cluster_id not in cluster_bias_accum:
                cluster_bias_accum[cluster_id] = []
            count = self.clusterer.compound_sample_counts.get(comp_name, 1)
            cluster_bias_accum[cluster_id].append((raw_bias, count))
            
        for cid, bias_list in cluster_bias_accum.items():
            total_weight = sum(c for _, c in bias_list)
            if total_weight > 0:
                self.cluster_bias[cid] = sum(b * c for b, c in bias_list) / total_weight
            else:
                self.cluster_bias[cid] = np.mean([b for b, _ in bias_list])
                
        # 3c. Regularized shrinkage formula: shrunk_bias = (n/(n+lambda))*raw_bias + (lambda/(n+lambda))*cluster_bias
        for comp_name, raw_bias in self.compound_raw_bias.items():
            n_samples = self.clusterer.compound_sample_counts.get(comp_name, 1)
            cluster_id = self.clusterer.compound_to_cluster.get(comp_name, -1)
            c_bias = self.cluster_bias.get(cluster_id, self.global_mean_bias)
            
            w_comp = n_samples / (n_samples + self.lambda_shrinkage)
            self.compound_bias[comp_name] = w_comp * raw_bias + (1.0 - w_comp) * c_bias
            
        # --- Step 4: Compute process nominals ---
        # 4a. Compound-level nominals
        for comp_name in df_train['CompoundName'].unique():
            comp_data = df_train[df_train['CompoundName'] == comp_name]
            self.compound_nominals[comp_name] = comp_data[self.process_cols].mean().to_dict()
            
        # 4b. Cluster-level nominals
        for cluster_id in self.cluster_bias.keys():
            members = [c for c, cid in self.clusterer.compound_to_cluster.items() if cid == cluster_id]
            member_data = df_train[df_train['CompoundName'].isin(members)]
            if len(member_data) > 0:
                self.cluster_nominals[cluster_id] = member_data[self.process_cols].mean().to_dict()
                
        # --- Step 5: Fit Stage 2 on process deviations ---
        y_stage1 = base_preds + df_train['CompoundName'].map(self.compound_bias).fillna(self.global_mean_bias).values
        residuals = y_train - y_stage1
        
        X_proc_list = []
        for idx, row in df_train.iterrows():
            c_name = row['CompoundName']
            c_nom = self._get_nominals(c_name)
            row_delta = []
            for p_col in self.process_cols:
                nom_val = c_nom.get(p_col, row[p_col] if not pd.isna(row[p_col]) else 0.0)
                act_val = row[p_col] if not pd.isna(row[p_col]) else nom_val
                row_delta.append(act_val - nom_val)
            X_proc_list.append(row_delta)
            
        X_proc_delta = np.array(X_proc_list)
        X_proc_imp = self.process_imputer.fit_transform(X_proc_delta)
        X_proc_sc = self.process_scaler.fit_transform(X_proc_imp)
        
        self.process_ridge.fit(X_proc_sc, residuals, sample_weight=sample_weight)
        
        # Store S2 feature coefficients
        coef_s2 = np.ravel(self.process_ridge.coef_)
        self.feature_importance_s2 = dict(zip(self.process_cols[:len(coef_s2)], coef_s2))
        
        print(f"  Stage 2 RidgeCV selected alpha: {self.process_ridge.alpha_:.4f}")
        
    def _get_nominals(self, compound_name, row_features=None):
        """Get process nominals for a compound, falling back to cluster-level matching."""
        if compound_name in self.compound_nominals:
            return self.compound_nominals[compound_name]
        cluster_id, _ = self.clusterer.get_cluster_for_compound(compound_name, row_features)
        return self.cluster_nominals.get(cluster_id, {})
        
    def _get_bias(self, compound_name, row_features=None):
        """Get bias for a compound, handling cold-start via cluster matching."""
        if compound_name in self.compound_bias:
            return self.compound_bias[compound_name]
        cluster_id, _ = self.clusterer.get_cluster_for_compound(compound_name, row_features)
        return self.cluster_bias.get(cluster_id, self.global_mean_bias)
        
    def predict_raw(self, df_test):
        """
        Stage 1 + Stage 2 pure offline physical prediction (no lab MNY feedback).
        """
        df_test = df_test.copy()
        df_test['CompoundName'] = df_test['CompoundName'].astype(str).str.strip().str.upper()
        
        # Stage 1: Baseline + bias
        X_rec = df_test[self.recipe_cols].copy()
        X_rec_imp = self.baseline_imputer.transform(X_rec)
        X_rec_sc = self.baseline_scaler.transform(X_rec_imp)
        base_preds = self.baseline_ridge.predict(X_rec_sc)
        
        biases = []
        for idx, row in df_test.iterrows():
            biases.append(self._get_bias(row['CompoundName'], df_test.loc[[idx]]))
        stage1_preds = base_preds + np.array(biases)
        
        # Stage 2: Process deviations
        X_proc_list = []
        for idx, row in df_test.iterrows():
            c_name = row['CompoundName']
            c_nom = self._get_nominals(c_name, df_test.loc[[idx]])
            row_delta = []
            for p_col in self.process_cols:
                nom_val = c_nom.get(p_col, row[p_col] if not pd.isna(row[p_col]) else 0.0)
                act_val = row[p_col] if not pd.isna(row[p_col]) else nom_val
                row_delta.append(act_val - nom_val)
            X_proc_list.append(row_delta)
            
        X_proc_delta = np.array(X_proc_list)
        X_proc_imp = self.process_imputer.transform(X_proc_delta)
        X_proc_sc = self.process_scaler.transform(X_proc_imp)
        stage2_deltas = self.process_ridge.predict(X_proc_sc)
        
        raw_preds = stage1_preds + stage2_deltas
        return raw_preds
        
    def predict_online(self, df_test, calibration_state=None):
        """
        Stage 1 + Stage 2 prediction adjusted by Stage 3 online calibration bias.
        """
        raw_preds = self.predict_raw(df_test)
        if calibration_state is None:
            return raw_preds
        return raw_preds + calibration_state.state_bias
        
    def update_calibration_state(self, actual_mny, raw_pred_mny, calibration_state):
        """
        Update Kalman calibration state given a newly released Lab MNY.
        """
        if pd.isna(actual_mny) or pd.isna(raw_pred_mny):
            return calibration_state
            
        innov = actual_mny - (raw_pred_mny + calibration_state.state_bias)
        
        # Hampel filter clamp & Adaptive tracking logic
        if abs(innov) > calibration_state.lab_noise_threshold:
            K_k = 0.10
        elif abs(innov) > 1.8:
            K_k = min(0.75, calibration_state.P_state / (calibration_state.P_state + calibration_state.R_meas) * 1.5)
        else:
            K_k = calibration_state.P_state / (calibration_state.P_state + calibration_state.R_meas)
            
        new_bias = calibration_state.state_bias + K_k * innov
        new_P = (1.0 - K_k) * calibration_state.P_state + 0.2
        
        return CalibrationState(
            state_bias=new_bias,
            P_state=new_P,
            R_meas=calibration_state.R_meas,
            lab_noise_threshold=calibration_state.lab_noise_threshold
        )
        
    def predict(self, df_test, apply_stage3_aakf=True, R_meas=1.0, lab_noise_threshold=3.0):
        """
        Standard batch prediction mimicking sequential online tracking.
        """
        raw_preds = self.predict_raw(df_test)
        if not apply_stage3_aakf or len(df_test) < 2 or 'MNY' not in df_test.columns:
            return raw_preds
            
        y_true = df_test['MNY'].values
        n = len(df_test)
        online_preds = np.zeros(n)
        
        state = CalibrationState(R_meas=R_meas, lab_noise_threshold=lab_noise_threshold)
        
        for i in range(n):
            online_preds[i] = raw_preds[i] + state.state_bias
            # Update state with the lab feedback for next batch
            state = self.update_calibration_state(y_true[i], raw_preds[i], state)
            
        return online_preds


# ============================================================================
# Visualization: t-SNE Cluster Map
# ============================================================================

def plot_cluster_tsne(clusterer, output_path, track_name):
    """Generate t-SNE visualization of compound clusters."""
    names, fused_matrix, labels, sample_counts = clusterer.get_fused_fingerprints_for_viz()
    
    if len(names) < 5:
        print(f"  [Skip] Too few compounds ({len(names)}) for t-SNE visualization.")
        return
    
    perplexity = min(30, max(5, len(names) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(fused_matrix)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap('tab20', max(len(unique_labels), 1))
    
    for i, cid in enumerate(unique_labels):
        mask = [j for j, l in enumerate(labels) if l == cid]
        sizes = [min(200, max(20, sample_counts.get(names[j], 1) * 2)) for j in mask]
        
        if cid == -1:
            ax.scatter(coords[mask, 0], coords[mask, 1], c='gray', s=sizes,
                      marker='x', alpha=0.6, label='Noise (unclustered)')
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)], s=sizes,
                      alpha=0.7, edgecolors='white', linewidth=0.5,
                      label=f'Cluster {cid} ({len(mask)} compounds)')
    
    # Annotate large compounds
    for j, name in enumerate(names):
        count = sample_counts.get(name, 0)
        if count >= 100:
            short_name = name[:14] if len(name) > 14 else name
            ax.annotate(short_name, (coords[j, 0], coords[j, 1]),
                       fontsize=6, alpha=0.8, ha='center', va='bottom')
    
    ax.set_title(f'Compound Cluster Map (t-SNE) — {track_name}\n'
                f'Recipe Fingerprint (14D) + Curve Shape Fingerprint (22D) → HDBSCAN',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(loc='best', fontsize=7, ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Cluster t-SNE map saved to: {output_path}")


# ============================================================================
# Main Training & Evaluation Pipeline
# ============================================================================

def main():
    t_start = time.time()
    
    # --- Load Data ---
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    data_path = os.path.join(WORKSPACE_DIR, 'data', 'stage_statistics_enriched_all_features_weather_v4.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(WORKSPACE_DIR, 'stage_statistics_enriched_all_features_weather_v4.csv')
    
    df = pd.read_csv(data_path, low_memory=False)
    df['MNY'] = pd.to_numeric(df['MNY'], errors='coerce')
    df = df.dropna(subset=['MNY'])
    df['CompoundName'] = df['CompoundName'].astype(str).str.strip().str.upper()
    df['OrderID'] = df['OrderID'].astype(str).str.strip()
    
    print(f"Loaded {len(df)} batches, {df['CompoundName'].nunique()} unique compounds.")
    
    # --- 4-Track Split ---
    df['is_silica_system'] = ((df['silica_phr'] >= 25.0) & (df['weight_pct_silian'] > 0.0)).astype(float)
    
    sub_tracks = [
        ("Carbon Black - With Oil",    1.0, 0.0, "results_with_oil"),
        ("Carbon Black - No Oil",      0.0, 0.0, "results_without_oil"),
        ("Silica - With Oil",          1.0, 1.0, "results_silica_with_oil"),
        ("Silica - No Oil",            0.0, 1.0, "results_silica_without_oil"),
    ]
    
    all_results = {}
    
    for track_name, is_oil, is_silica, folder_name in sub_tracks:
        print(f"\n{'='*70}")
        print(f"TRACK: {track_name}")
        print(f"{'='*70}")
        
        df_track = df[(df['is_oil_loading_present'] == is_oil) & (df['is_silica_system'] == is_silica)].copy()
        
        output_dir = os.path.join(WORKSPACE_DIR, folder_name)
        os.makedirs(output_dir, exist_ok=True)
            
        if len(df_track) < 30:
            print(f"  [SKIP] Only {len(df_track)} batches in this track.")
            continue
        
        n_compounds = df_track['CompoundName'].nunique()
        print(f"  Batches: {len(df_track)}, Compounds: {n_compounds}")
        
        # Filter features to available columns
        avail_recipe = [c for c in recipe_cols if c in df_track.columns]
        avail_process = [c for c in process_cols if c in df_track.columns]
        
        # --- Group 5-Fold CV (OrderID groups — comparable to prior models) ---
        print(f"\n  Running Group 5-Fold CV (OrderID groups — known compounds)...")
        # Pre-process for label grouping
        lab_id_cols = ["MNY_SampleID", "LabSampleID", "SampleID", "TestSampleID"]
        available_lab_id = [col for col in lab_id_cols if col in df_track.columns]
        label_group_cols = available_lab_id[:1] if available_lab_id else ["OrderID", "PalletID"]
        df_track = add_label_group_information(df_track, label_group_cols)
        
        gkf = GroupKFold(n_splits=5)
        groups_label = df_track['_label_group_id'].values
        
        oof_preds = np.full(len(df_track), np.nan)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df_track, df_track['MNY'], groups_label)):
            df_train_fold = df_track.iloc[train_idx].copy()
            df_val_fold = df_track.iloc[val_idx].copy()
            
            model_fold = RobustUltimate3StageModel(avail_recipe, avail_process)
            model_fold.fit(df_train_fold, df_train_fold['MNY'].values)
            
            val_preds = model_fold.predict(df_val_fold, apply_stage3_aakf=False)
            oof_preds[val_idx] = val_preds
            
            fold_mae = mean_absolute_error(df_val_fold['MNY'].values, val_preds)
            fold_r2 = r2_score(df_val_fold['MNY'].values, val_preds)
            fold_metrics.append({'fold': fold+1, 'mae': fold_mae, 'r2': fold_r2})
            print(f"    Fold {fold+1}: MAE={fold_mae:.3f} MU, R²={fold_r2:.4f}")
        
        # Overall CV metrics (known compounds)
        valid_mask = ~np.isnan(oof_preds)
        y_true_cv = df_track['MNY'].values[valid_mask]
        y_pred_cv = oof_preds[valid_mask]
        
        cv_r2 = r2_score(y_true_cv, y_pred_cv)
        cv_rmse = np.sqrt(mean_squared_error(y_true_cv, y_pred_cv))
        cv_mae = mean_absolute_error(y_true_cv, y_pred_cv)
        cv_corr, _ = pearsonr(y_true_cv, y_pred_cv)
        
        # --- Cold-Start Evaluation (CompoundName groups — unseen compounds) ---
        print(f"\n  Running Cold-Start Evaluation (CompoundName groups — unseen compounds)...")
        gkf_cold = GroupKFold(n_splits=5)
        groups_compound = df_track['CompoundName'].values
        
        oof_cold = np.full(len(df_track), np.nan)
        for fold, (train_idx, val_idx) in enumerate(gkf_cold.split(df_track, df_track['MNY'], groups_compound)):
            df_train_fold = df_track.iloc[train_idx].copy()
            df_val_fold = df_track.iloc[val_idx].copy()
            
            model_cold = RobustUltimate3StageModel(avail_recipe, avail_process)
            model_cold.fit(df_train_fold, df_train_fold['MNY'].values)
            
            val_preds_cold = model_cold.predict(df_val_fold, apply_stage3_aakf=False)
            oof_cold[val_idx] = val_preds_cold
        
        valid_cold = ~np.isnan(oof_cold)
        cold_r2 = r2_score(df_track['MNY'].values[valid_cold], oof_cold[valid_cold])
        cold_mae = mean_absolute_error(df_track['MNY'].values[valid_cold], oof_cold[valid_cold])
        cold_corr, _ = pearsonr(df_track['MNY'].values[valid_cold], oof_cold[valid_cold])
        
        print(f"\n  {'='*60}")
        print(f"  RESULTS — {track_name}")
        print(f"  {'='*60}")
        print(f"  [Known Compounds] R²={cv_r2:.4f}  MAE={cv_mae:.3f} MU  Corr={cv_corr:+.4f}")
        print(f"  [Cold Start]      R²={cold_r2:.4f}  MAE={cold_mae:.3f} MU  Corr={cold_corr:+.4f}")
        print(f"  {'='*60}")
        
        all_results[track_name] = {
            'N': len(df_track), 'R2': cv_r2, 'RMSE': cv_rmse,
            'MAE': cv_mae, 'Corr': cv_corr, 'n_compounds': n_compounds,
            'cold_R2': cold_r2, 'cold_MAE': cold_mae, 'cold_Corr': cold_corr
        }
        
        # --- Fit Final Model on All Data ---
        print(f"\n  Fitting final model on all {len(df_track)} batches...")
        final_model = RobustUltimate3StageModel(avail_recipe, avail_process)
        final_model.fit(df_track, df_track['MNY'].values, sample_weight=df_track['_sample_weight'].values)
        
        # --- Save Model Bundle ---
        bundle = {
            'model': final_model,
            'recipe_cols': avail_recipe,
            'process_cols': avail_process,
            'track_name': track_name,
            'cv_metrics': {
                'r2': cv_r2, 'mae': cv_mae, 'rmse': cv_rmse, 'corr': cv_corr
            },
            'cluster_stats': final_model.clusterer.cluster_stats_,
            'silhouette_score': final_model.clusterer.silhouette_score_,
        }
        bundle_path = os.path.join(output_dir, 'mooney_ultimate3stage_cluster_bundle.joblib')
        joblib.dump(bundle, bundle_path)
        print(f"  Model bundle saved: {bundle_path}")
        # ---------------------------------------------------------------------------
        # BIAS REPORT EXPORT
        # ---------------------------------------------------------------------------
        bias_rows = []
        for comp, raw_bias in final_model.compound_raw_bias.items():
            n_samples = final_model.clusterer.compound_sample_counts.get(comp, 1)
            total_weight = sum(1.0 / sz for sz in final_model.clusterer.compound_sample_counts.get(comp, 1) if isinstance(sz, int))
            # Effective sample size is sum of sample weights for this compound
            eff_sample_size = df_track[df_track['CompoundName'] == comp]['_sample_weight'].sum()
            shrinkage_factor = eff_sample_size / (eff_sample_size + final_model.lambda_shrinkage)
            cluster_id = final_model.clusterer.compound_to_cluster.get(comp, -1)
            cluster_bias = final_model.cluster_bias.get(cluster_id, final_model.global_mean_bias)
            shrunk_bias = shrinkage_factor * raw_bias + (1 - shrinkage_factor) * cluster_bias
            bias_rows.append({
                'CompoundName': comp,
                'raw_sample_count': n_samples,
                'unique_label_group_count': len(df_track[df_track['CompoundName'] == comp]['_label_group_id'].unique()),
                'total_sample_weight': eff_sample_size,
                'raw_compound_bias': raw_bias,
                'cluster_bias': cluster_bias,
                'shrinkage_factor': shrinkage_factor,
                'shrunk_compound_bias': shrunk_bias,
                'effective_weight': eff_sample_size * shrinkage_factor,
                'bias_source': 'independent' if final_model.clusterer.should_use_independent_bias(comp) else 'cluster',
                'cluster_id': cluster_id
            })
        bias_df = pd.DataFrame(bias_rows)
        bias_path = os.path.join(output_dir, 'compound_bias_report.csv')
        bias_df.to_csv(bias_path, index=False, encoding='utf-8-sig')
        print(f"  Compound bias report saved: {bias_path}")
        
        # --- Save Cluster Statistics CSV ---
        if final_model.clusterer.cluster_stats_ is not None:
            cluster_csv = os.path.join(output_dir, 'compound_cluster_stats.csv')
            final_model.clusterer.cluster_stats_.to_csv(cluster_csv, index=False, encoding='utf-8-sig')
            print(f"  Cluster stats saved: {cluster_csv}")
        
        # --- Save Compound-to-Cluster Mapping CSV ---
        mapping_rows = []
        for comp_name, cluster_id in final_model.clusterer.compound_to_cluster.items():
            count = final_model.clusterer.compound_sample_counts.get(comp_name, 0)
            bias_type = 'independent' if final_model.clusterer.should_use_independent_bias(comp_name) else 'cluster'
            bias_val = final_model.compound_bias.get(comp_name, 0.0)
            mapping_rows.append({
                'CompoundName': comp_name,
                'cluster_id': cluster_id,
                'sample_count': count,
                'bias_type': bias_type,
                'bias_value': round(bias_val, 4)
            })
        mapping_df = pd.DataFrame(mapping_rows).sort_values(['cluster_id', 'sample_count'], ascending=[True, False])
        mapping_csv = os.path.join(output_dir, 'compound_to_cluster_mapping.csv')
        mapping_df.to_csv(mapping_csv, index=False, encoding='utf-8-sig')
        print(f"  Compound-to-cluster mapping saved: {mapping_csv}")
        
        # --- t-SNE Visualization ---
        tsne_path = os.path.join(output_dir, 'cluster_tsne_map.png')
        plot_cluster_tsne(final_model.clusterer, tsne_path, track_name)
        
        # --- Save CV Predictions CSV ---
        cv_df = df_track[valid_mask][['OrderID', 'CompoundName', 'MNY']].copy()
        cv_df['Predicted_MNY'] = y_pred_cv
        cv_df['Residual'] = cv_df['MNY'] - cv_df['Predicted_MNY']
        cv_csv = os.path.join(output_dir, 'cv_predictions.csv')
        cv_df.to_csv(cv_csv, index=False, encoding='utf-8-sig')
        # ---------------------------------------------------------------------------
        # EXPLANATION REPORT (basic version)
        # ---------------------------------------------------------------------------
        expl_rows = []
        for idx, row in df_track.iterrows():
            # Stage 1 prediction
            X_rec = row[final_model.recipe_cols]
            X_rec_imp = final_model.baseline_imputer.transform([X_rec])
            X_rec_sc = final_model.baseline_scaler.transform(X_rec_imp)
            s1_pred = final_model.baseline_ridge.predict(X_rec_sc)[0]
            # Bias
            bias = final_model._get_bias(row['CompoundName'], df_track.loc[[idx]])
            # Stage 2 contribution (process delta)
            c_nom = final_model._get_nominals(row['CompoundName'], df_track.loc[[idx]])
            delta = []
            for p_col in final_model.process_cols:
                nom_val = c_nom.get(p_col, row[p_col] if not pd.isna(row[p_col]) else 0.0)
                act_val = row[p_col] if not pd.isna(row[p_col]) else nom_val
                delta.append(act_val - nom_val)
            X_proc_imp = final_model.process_imputer.transform([delta])
            X_proc_sc = final_model.process_scaler.transform(X_proc_imp)
            s2_pred = final_model.process_ridge.predict(X_proc_sc)[0]
            # Online bias (Stage 3) – simulate sequential Kalman state assuming previous labs are unavailable
            # For simplicity we use zero state bias for each batch here
            s3_pred = 0.0
            pred_mny = s1_pred + bias + s2_pred + s3_pred
            # Top process features by absolute coefficient impact
            coeffs = final_model.feature_importance_s2
            impacts = {p: abs(coeffs.get(p,0))*abs(delta[i]) for i,p in enumerate(final_model.process_cols)}
            top3 = sorted(impacts, key=impacts.get, reverse=True)[:3]
            expl_rows.append({
                'OrderID': row['OrderID'],
                'CompoundName': row['CompoundName'],
                'BatchIdx': idx,
                'Predicted_MNY': pred_mny,
                'Stage1_Contribution': s1_pred,
                'Stage2_Contribution': s2_pred,
                'Stage3_Contribution': s3_pred,
                'TopProcessFeatures': ",".join(top3),
                'confidence_score': 100.0,  # placeholder
                'confidence_level': 'High',
                'prediction_interval_lower': pred_mny - 5.0,
                'prediction_interval_upper': pred_mny + 5.0,
                'calibration_source': 'none',
                'calibration_age': np.nan,
                'P_state': np.nan,
                'Kalman_gain': np.nan,
                'confidence_reason_codes': ''
            })
        expl_df = pd.DataFrame(expl_rows)
        expl_path = os.path.join(output_dir, 'explanation_report.csv')
        expl_df.to_csv(expl_path, index=False, encoding='utf-8-sig')
        print(f"  Explanation report saved: {expl_path}")
        
        # --- Feature Importance Report ---
        ridge_stage1 = final_model.baseline_ridge[-1]
        ridge_stage2 = final_model.process_ridge[-1]
        
        coef_s1 = np.ravel(ridge_stage1.coef_)
        coef_s2 = np.ravel(ridge_stage2.coef_)
        
        # Handle potential length mismatch (imputer may drop constant columns)
        feat_names_s1 = avail_recipe[:len(coef_s1)] if len(coef_s1) <= len(avail_recipe) else \
                        avail_recipe + [f'unknown_{i}' for i in range(len(coef_s1) - len(avail_recipe))]
        feat_names_s2 = avail_process[:len(coef_s2)] if len(coef_s2) <= len(avail_process) else \
                        avail_process + [f'unknown_{i}' for i in range(len(coef_s2) - len(avail_process))]
        
        imp_s1 = pd.DataFrame({
            'Feature': feat_names_s1[:len(coef_s1)],
            'Coefficient': coef_s1[:len(feat_names_s1)],
            'AbsCoeff': np.abs(coef_s1[:len(feat_names_s1)])
        }).sort_values('AbsCoeff', ascending=False)
        
        imp_s2 = pd.DataFrame({
            'Feature': feat_names_s2[:len(coef_s2)],
            'Coefficient': coef_s2[:len(feat_names_s2)],
            'AbsCoeff': np.abs(coef_s2[:len(feat_names_s2)])
        }).sort_values('AbsCoeff', ascending=False)
        
        imp_s1.to_csv(os.path.join(output_dir, 'feature_importance_stage1_recipe.csv'),
                      index=False, encoding='utf-8-sig')
        imp_s2.to_csv(os.path.join(output_dir, 'feature_importance_stage2_process.csv'),
                      index=False, encoding='utf-8-sig')
        print(f"  Feature importance CSVs saved.")
    
    # --- Final Summary ---
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  ALL MODELS TRAINED SUCCESSFULLY (elapsed: {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"\n  [Known Compounds] OrderID-grouped CV (same compound seen in training)")
    print(f"  {'Track':<30s} | {'N':>6s} | {'Comp':>5s} | {'R²':>7s} | {'MAE':>8s} | {'Corr':>7s}")
    print("  " + "-"*78)
    for track, m in all_results.items():
        print(f"  {track:<30s} | {m['N']:>6d} | {m['n_compounds']:>5d} | {m['R2']:>7.4f} | {m['MAE']:>7.3f} MU | {m['Corr']:>+7.4f}")
    
    print(f"\n  [Cold Start] CompoundName-grouped CV (entire compound held out)")
    print(f"  {'Track':<30s} | {'N':>6s} | {'Comp':>5s} | {'R²':>7s} | {'MAE':>8s} | {'Corr':>7s}")
    print("  " + "-"*78)
    for track, m in all_results.items():
        print(f"  {track:<30s} | {m['N']:>6d} | {m['n_compounds']:>5d} | {m['cold_R2']:>7.4f} | {m['cold_MAE']:>7.3f} MU | {m['cold_Corr']:>+7.4f}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
