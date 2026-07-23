import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Set working directories
WORKSPACE_DIR = os.getcwd()

print("=== STEP 1: LOADING DATA & COMPUTING KINETICS FOR WITH-OIL CARBON-BLACK ===")

# Load raw temperature curves for kinetics
df_raw = pd.read_csv(os.path.join(WORKSPACE_DIR, 'stage_statistics_enriched.csv'), usecols=['OrderID', 'BatchNumber', 'temp'], low_memory=False)
df_raw['OrderID'] = df_raw['OrderID'].astype(str).str.strip()
df_raw['BatchNumber'] = df_raw['BatchNumber'].astype(int)

# Load full segmented dataset
df_seg = pd.read_csv(os.path.join(WORKSPACE_DIR, 'stage_statistics_enriched_all_features_weather_v4.csv'), low_memory=False)
df_seg['OrderID'] = df_seg['OrderID'].astype(str).str.strip()
df_seg['BatchNumber'] = df_seg['BatchNumber'].astype(int)
df_seg['PalletID'] = df_seg['PalletID'].astype(str).str.strip()

# Merge temp curve
df_seg = pd.merge(df_seg, df_raw, on=['OrderID', 'BatchNumber'], how='left')

# Kinetics calculation
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

print("Computing kinetics reaction indices...")
sil_indices = []
scorch_indices = []
for idx, row in df_seg.iterrows():
    temp_curve = hex_to_series(row['temp'])
    I_sil, I_scorch = calculate_kinetics(temp_curve)
    sil_indices.append(I_sil)
    scorch_indices.append(I_scorch)

df_seg['I_silanization'] = sil_indices
df_seg['I_scorch'] = scorch_indices
if 'temp' in df_seg.columns:
    df_seg = df_seg.drop(columns=['temp'])

# Filter track: With-Oil Carbon-Black
df_seg['is_silica_system'] = ((df_seg['silica_phr'] >= 25.0) & (df_seg['weight_pct_silian'] > 0.0)).astype(float)
df_track = df_seg[(df_seg['is_oil_loading_present'] == 1.0) & (df_seg['is_silica_system'] == 0.0)].copy()

print(f"Track 'With-Oil Carbon-Black' loaded with {len(df_track)} batch records.")

# Clean feature names helper
import re
def clean_feature_name(name):
    if not isinstance(name, str):
        return name
    return re.sub(r'[^\w]', '_', name.strip())

df_track.columns = [clean_feature_name(c) for c in df_track.columns]

# Aggregation to Pallet Level
recipe_cols = [
    'Top_Fill_Factor', 'Bot_Fill_Factor', 'Target_Temperature',
    'weight_pct_solid_elastomer', 'weight_pct_natural_rubber', 'weight_pct_silica',
    'weight_pct_oil', 'weight_pct_silian', 'weight_pct_carbon_black', 'silica_phr',
    'is_oil_loading_present', 'ratio_nr_rubber', 'ratio_filler_polymer',
    'ratio_oil_polymer', 'ratio_oil_filler',
    'supplier_rubber_viscosity_avg', 'supplier_silica_moisture_avg', 'supplier_silica_surface_area_avg',
    'supplier_carbon_black_structure_avg', 'supplier_carbon_black_surface_area_avg', 'supplier_carbon_black_moisture_avg'
]
recipe_cols = [clean_feature_name(r) for r in recipe_cols if clean_feature_name(r) in df_track.columns]

core_process_features = [
    'phys_discharge_temp', 'phys_max_temp', 'phys_eta_app_discharge',
    'Stage6_BottomMixing_Torque_Mean', 'Stage6_BottomMixing_power_Mean', 'Stage6_BottomMixing_Duration',
    'Stage6_BottomMixing_Torque_Integral', 'Stage4_WetMixing_temp_Mean', 'Stage4_WetMixing_Duration',
    'Stage2_DryMixing_Duration', 'Stage2_DryMixing_power_Mean',
    'env_temp_mean', 'env_humidity_mean',
    'I_silanization', 'I_scorch', 'time_to_sil_plateau_duration', 'top_sil_duration',
    'bottom_sil_duration', 'silanization_energy_mj', 'top_avg_sil_temperature', 'bottom_avg_sil_temperature',
    'phys_init_temp', 'phys_temp_integral', 'phys_temp_integral_above_100'
]
core_process_features = [clean_feature_name(f) for f in core_process_features if clean_feature_name(f) in df_track.columns]

df_track['MNY'] = pd.to_numeric(df_track['MNY'], errors='coerce')
df_track = df_track.dropna(subset=['MNY'])

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

# Group by OrderID and PalletID
grouped = df_track.groupby(['OrderID', 'PalletID']).agg(agg_dict).reset_index()

# Get Batch_List and N_batches per sample
batch_info = df_track.groupby(['OrderID', 'PalletID'])['BatchNumber'].apply(lambda x: ', '.join(map(str, sorted(list(x))))).reset_index(name='Batch_List')
batch_counts = df_track.groupby(['OrderID', 'PalletID']).size().reset_index(name='N_batches')

df_samples = pd.merge(grouped, batch_info, on=['OrderID', 'PalletID'])
df_samples = pd.merge(df_samples, batch_counts, on=['OrderID', 'PalletID'])

df_samples['OrderStartTime'] = pd.to_datetime(df_samples['OrderStartTime'], errors='coerce')
df_samples = df_samples.sort_values('OrderStartTime').reset_index(drop=True)

print(f"Total Pallet Samples: {len(df_samples)}")

# Split chronologically: Train 80%, Test 20% (Recent untrained test data)
n_train = int(len(df_samples) * 0.8)
train_df = df_samples.iloc[:n_train].copy().reset_index(drop=True)
test_df = df_samples.iloc[n_train:].copy().reset_index(drop=True)

print(f"Train samples count (earlier 80%): {len(train_df)} (from {train_df['OrderStartTime'].min()} to {train_df['OrderStartTime'].max()})")
print(f"Test samples count (recent 20%): {len(test_df)} (from {test_df['OrderStartTime'].min()} to {test_df['OrderStartTime'].max()})")

# === STEP 2: ISOLATION FOREST OUTLIER REJECTION ===
print("\n=== STEP 2: APPLYING ISOLATION FOREST FOR OUTLIER REJECTION ===")
features_for_if = core_process_features.copy()

imputer = SimpleImputer(strategy='median')
X_if_train = imputer.fit_transform(train_df[features_for_if])
X_if_test = imputer.transform(test_df[features_for_if])

# Fit Isolation Forest on Train set
iso_forest = IsolationForest(contamination=0.04, random_state=42, n_jobs=-1)
iso_forest.fit(X_if_train)

# Predict anomalies on Test set
test_if_preds = iso_forest.predict(X_if_test)

# Mark outliers in test set (1 = Outlier/Discarded, 0 = Clean)
test_df['Is_Outlier_Discarded'] = (test_if_preds == -1).astype(int)
n_outliers = test_df['Is_Outlier_Discarded'].sum()

print(f"Isolation Forest detected {n_outliers} anomalous samples ({n_outliers/len(test_df):.2%}) out of {len(test_df)} recent test samples.")

# === STEP 3: TRAIN TWO-STAGE NON-LINEAR MODEL ===
print("\n=== STEP 3: TRAINING TWO-STAGE NON-LINEAR MODEL ON TRAIN SET ===")

class Stage1Baseline:
    def __init__(self, recipe_cols):
        self.recipe_cols = recipe_cols
        self.compound_means = {}
        self.fallback_model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=100.0))
        self.global_mean = 0.0
        
    def fit(self, X_train_full, y_train):
        self.global_mean = y_train.mean()
        df_tr = X_train_full.copy()
        df_tr['MNY'] = y_train
        self.compound_means = df_tr.groupby('CompoundName')['MNY'].mean().to_dict()
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
        return y_baseline + y_residual_pred

# Filter train set with IsolationForest clean data
train_if_preds = iso_forest.predict(X_if_train)
train_clean_df = train_df[train_if_preds == 1].copy().reset_index(drop=True)

lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, num_leaves=15, min_child_samples=10, verbosity=-1, random_state=42)
two_stage_model = TwoStageNonLinearModel(recipe_cols, core_process_features, lgb_model)

X_train = train_clean_df.drop(columns=['MNY'])
y_train = train_clean_df['MNY']
two_stage_model.fit(X_train, y_train)

print("Two-Stage Non-Linear Model trained successfully.")

# === STEP 4: PREDICT AND COMPUTE METRICS ON RECENT TEST SET ===
print("\n=== STEP 4: EVALUATING PREDICTIONS ON RECENT TEST SET ===")

X_test = test_df.drop(columns=['MNY'])
y_test = test_df['MNY']

test_df['Predicted_MNY_Static'] = two_stage_model.predict(X_test)
test_df['Error_Static'] = test_df['Predicted_MNY_Static'] - test_df['MNY']
test_df['Abs_Error_Static'] = np.abs(test_df['Error_Static'])

# Adaptive Rolling Lot Calibration (Rolling Feedback Offset)
calibrated_preds = []
order_latest_residual = {}

for idx, row in test_df.iterrows():
    order_id = row['OrderID']
    pred_static = row['Predicted_MNY_Static']
    actual = row['MNY']
    
    if order_id in order_latest_residual:
        calib_pred = pred_static + order_latest_residual[order_id]
    else:
        calib_pred = pred_static
        
    calibrated_preds.append(calib_pred)
    # Update latest residual for this order if clean
    if row['Is_Outlier_Discarded'] == 0:
        order_latest_residual[order_id] = actual - pred_static

test_df['Predicted_MNY_Calibrated'] = calibrated_preds
test_df['Error_Calibrated'] = test_df['Predicted_MNY_Calibrated'] - test_df['MNY']
test_df['Abs_Error_Calibrated'] = np.abs(test_df['Error_Calibrated'])

# Split test results into ALL test vs CLEAN test
clean_test_df = test_df[test_df['Is_Outlier_Discarded'] == 0].copy()

def compute_metrics(df, pred_col):
    y_true = df['MNY']
    y_pred = df[pred_col]
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    corr = y_true.corr(y_pred)
    return r2, rmse, mae, corr

all_r2_st, all_rmse_st, all_mae_st, all_corr_st = compute_metrics(test_df, 'Predicted_MNY_Static')
clean_r2_st, clean_rmse_st, clean_mae_st, clean_corr_st = compute_metrics(clean_test_df, 'Predicted_MNY_Static')
clean_r2_ca, clean_rmse_ca, clean_mae_ca, clean_corr_ca = compute_metrics(clean_test_df, 'Predicted_MNY_Calibrated')

print("\n" + "="*70)
print("             RECENT TEST SET EVALUATION METRICS SUMMARY")
print("="*70)
print(f"Total Test Samples          : {len(test_df)}")
print(f"Clean Test Samples          : {len(clean_test_df)} (Discarded {n_outliers} outliers)")
print("-" * 70)
print(f"All Test (Static Model)     | R2: {all_r2_st:.4f} | RMSE: {all_rmse_st:.4f} MU | MAE: {all_mae_st:.4f} MU | Corr: {all_corr_st:.4f}")
print(f"Clean Test (Static Model)   | R2: {clean_r2_st:.4f} | RMSE: {clean_rmse_st:.4f} MU | MAE: {clean_mae_st:.4f} MU | Corr: {clean_corr_st:.4f}")
print(f"Clean Test (Calibrated)     | R2: {clean_r2_ca:.4f} | RMSE: {clean_rmse_ca:.4f} MU | MAE: {clean_mae_ca:.4f} MU | Corr: {clean_corr_ca:.4f}")
print("="*70)

# === STEP 5: SAVE CSV COMPARISON FILES ===
print("\n=== STEP 5: SAVING CSV COMPARISON FILES ===")

export_cols = [
    'OrderStartTime', 'OrderID', 'PalletID', 'CompoundName', 
    'N_batches', 'Batch_List', 'MNY', 
    'Predicted_MNY_Static', 'Predicted_MNY_Calibrated',
    'Error_Static', 'Abs_Error_Static',
    'Error_Calibrated', 'Abs_Error_Calibrated',
    'Is_Outlier_Discarded'
]

rename_dict = {
    'MNY': 'Actual_MNY',
    'Predicted_MNY_Static': 'Predicted_MNY',
    'Predicted_MNY_Calibrated': 'Predicted_MNY_Calibrated',
    'Error_Static': 'Prediction_Error',
    'Abs_Error_Static': 'Absolute_Error',
    'Is_Outlier_Discarded': 'Is_Outlier_Discarded'
}

df_export_all = test_df[export_cols].rename(columns=rename_dict)
df_export_clean = clean_test_df[export_cols].rename(columns=rename_dict)

csv_path_all = os.path.join(WORKSPACE_DIR, "with_oil_carbon_black_recent_test_comparison_all.csv")
csv_path_clean = os.path.join(WORKSPACE_DIR, "with_oil_carbon_black_recent_test_comparison_clean.csv")

df_export_all.to_csv(csv_path_all, index=False, encoding='utf-8-sig')
df_export_clean.to_csv(csv_path_clean, index=False, encoding='utf-8-sig')

print(f"Exported ALL test comparison CSV ({len(df_export_all)} rows) to: {csv_path_all}")
print(f"Exported CLEAN test comparison CSV ({len(df_export_clean)} rows) to: {csv_path_clean}")

# === STEP 6: PLOT TREND VISUALIZATION ===
print("\n=== STEP 6: GENERATING TREND & PARITY VISUALIZATIONS ===")

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 1. Sequence Trend Plot
plt.figure(figsize=(16, 7), dpi=300)

sample_seq = np.arange(len(test_df))
plt.plot(sample_seq, test_df['MNY'], label='Actual Mooney (实际检测值)', color='#1f77b4', linewidth=1.8, alpha=0.85, marker='o', markersize=3)
plt.plot(sample_seq, test_df['Predicted_MNY_Static'], label='Predicted Mooney (模型预测值)', color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.85)
plt.plot(sample_seq, test_df['Predicted_MNY_Calibrated'], label='Rolling Calibrated Mooney (滚动校准预测值)', color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.9)

# Highlight discarded outliers
outlier_mask = test_df['Is_Outlier_Discarded'] == 1
plt.scatter(sample_seq[outlier_mask], test_df.loc[outlier_mask, 'MNY'], color='red', s=50, zorder=5, label='Discarded Outliers (孤立森林剔除车次/样本)')

plt.title('With-Oil Carbon-Black Track: Recent Test Data Prediction vs Actual Trend\n(含油炭黑胶最近测试集真实值与预测趋势对比)', fontsize=14, pad=12, fontweight='bold')
plt.xlabel('Recent Production Sample Sequence (最近生产样本时间序列)', fontsize=12)
plt.ylabel('Mooney Viscosity (MNY / MU)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10, loc='upper right', frameon=True)
plt.tight_layout()

trend_img_path = os.path.join(WORKSPACE_DIR, "with_oil_carbon_black_recent_trend_comparison.png")
plt.savefig(trend_img_path)
plt.close()

# 2. Parity Scatter Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# All Test Parity
sns.scatterplot(data=test_df, x='MNY', y='Predicted_MNY_Static', hue='Is_Outlier_Discarded', palette={0: '#1f77b4', 1: '#d62728'}, ax=axes[0], alpha=0.7)
lims = [min(test_df['MNY'].min(), test_df['Predicted_MNY_Static'].min()) - 2, max(test_df['MNY'].max(), test_df['Predicted_MNY_Static'].max()) + 2]
axes[0].plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ideal Parity (1:1)')
axes[0].set_xlim(lims)
axes[0].set_ylim(lims)
axes[0].set_title(f'All Test Samples (N={len(test_df)})\n$R^2$: {all_r2_st:.4f} | RMSE: {all_rmse_st:.4f} MU', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Actual Mooney (MU)', fontsize=11)
axes[0].set_ylabel('Predicted Mooney (MU)', fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.5)

# Clean Test Parity
sns.scatterplot(data=clean_test_df, x='MNY', y='Predicted_MNY_Calibrated', color='#2ca02c', ax=axes[1], alpha=0.8)
lims_clean = [min(clean_test_df['MNY'].min(), clean_test_df['Predicted_MNY_Calibrated'].min()) - 2, max(clean_test_df['MNY'].max(), clean_test_df['Predicted_MNY_Calibrated'].max()) + 2]
axes[1].plot(lims_clean, lims_clean, 'k--', alpha=0.75, zorder=0, label='Ideal Parity (1:1)')
axes[1].set_xlim(lims_clean)
axes[1].set_ylim(lims_clean)
axes[1].set_title(f'Clean Test Samples (N={len(clean_test_df)})\n$R^2$: {clean_r2_ca:.4f} | RMSE: {clean_rmse_ca:.4f} MU', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Actual Mooney (MU)', fontsize=11)
axes[1].set_ylabel('Calibrated Predicted Mooney (MU)', fontsize=11)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.suptitle('With-Oil Carbon-Black Model: Parity Plot on Recent Test Data', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

parity_img_path = os.path.join(WORKSPACE_DIR, "with_oil_carbon_black_recent_parity_plot.png")
plt.savefig(parity_img_path)
plt.close()

print(f"Saved trend chart to: {trend_img_path}")
print(f"Saved parity chart to: {parity_img_path}")

print("\n=== VALIDATION COMPLETED SUCCESSFULLY ===")
