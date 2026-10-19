"""
Comprehensive Algorithm Optimization & Deep Verification Experiment
=====================================================================
Directly investigates why static models fail (getting 48-50% pass rate)
and develops 4 optimized algorithmic architectures:
  1. Rich Multi-Step Rheological Load Feature Engineering (Torque + Power + Temp Rise)
  2. Two-Stage Cascade Model (Latent Raw Level Observer + Process Delta Engine)
  3. Dynamic Local Manifold JITL (Matching by Thermal-Mechanical Profile)
  4. Online Sparse Feedback / Anchor Calibration (Simulating Real Plant Testing)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, RidgeCV
import lightgbm as lgb
from interpret.glassbox import ExplainableBoostingRegressor

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"

# Load clean raw records
math_df = pd.read_pickle(CACHE / "mathematical_dataset.pkl")
lab = pd.read_pickle(CACHE / "lab.pkl")
strict_lab = lab[lab['LabelInPallet'] == True].rename(columns={'SampleBatch': 'BatchNumber'})

math_df['OrderID'] = math_df['OrderID'].astype(str)
math_df['BatchNumber'] = math_df['BatchNumber'].astype(int)
strict_lab['OrderID'] = strict_lab['OrderID'].astype(str)
strict_lab['BatchNumber'] = strict_lab['BatchNumber'].astype(int)

df = math_df.drop(columns=['Mooney']).merge(
    strict_lab[['OrderID', 'BatchNumber', 'Mooney']], 
    on=['OrderID', 'BatchNumber'], 
    how='inner'
)
clean = df[(df['Mooney'] >= 40) & (df['Mooney'] <= 85)].copy().sort_values(['OrderID', 'BatchNumber']).reset_index(drop=True)

# ------------------------------------------------------------------------------
# Feature Engineering: Extract Rich Rheological Resistance & Thermodynamic Rates
# ------------------------------------------------------------------------------
clean['t_MST7'] = (clean['MST_S7_Time_s'] - clean['MST_S6_Time_s']).clip(lower=0)
clean['t_MSB1'] = clean['MSB_S1_Time_s']
clean['MST7_Rpm'] = clean['MST_S7_Rpm'].fillna(40.0)
clean['T_dump'] = clean['MSB_S3_Temp_C'].fillna(clean['Curve_Temp_Max_Full_C'])
clean['T_start'] = clean['MMS_Start_Temp_C'].fillna(clean['Curve_Temp_Min_MST_C'])
clean['Energy_Target'] = clean['MMS_Total_Energy_kWh'].fillna(clean['Curve_Int_Power_Full_kWh'])

# Engineering features:
# 1. Total mechanical shear resistance proxy (Mean torque across key mastication steps)
clean['Mean_Shear_Torque'] = clean[['MST_S6_Torque_Nm', 'MST_S7_Torque_Nm', 'MST_S10_Torque_Nm', 'MSB_S2_Torque_Nm']].mean(axis=1)

# 2. Viscous dissipation heating rate (degC per second of mixing)
clean['Heating_Rate_C_s'] = (clean['T_dump'] - clean['T_start']) / (clean['Curve_Length_s'] + 1e-3)

# 3. Viscous dissipation heating per kWh (degC per kWh)
clean['Heating_Efficiency_C_kWh'] = (clean['T_dump'] - clean['T_start']) / (clean['Energy_Target'] + 1e-3)

# 4. Specific power draw rate (kW = kWh / hours)
clean['Mean_Power_kW'] = (clean['Energy_Target'] * 3600.0) / (clean['Curve_Length_s'] + 1e-3)

# 5. Lower mixer mechanical load index
clean['MSB_Load_Index'] = clean['MSB_S2_Torque_Nm'] * clean['MSB_S2_Current_A'] / 1e6

BASE_FEATS = ['t_MST7', 'MST7_Rpm', 't_MSB1', 'T_dump', 'T_start', 'MSB_S2_Torque_Nm', 'MSB_S1_TempMax_C', 'Energy_Target', 'Curve_Length_s']
ENHANCED_FEATS = BASE_FEATS + ['Mean_Shear_Torque', 'Heating_Rate_C_s', 'Heating_Efficiency_C_kWh', 'Mean_Power_kW', 'MSB_Load_Index']

clean = clean.dropna(subset=ENHANCED_FEATS + ['Mooney', 'OrderID']).reset_index(drop=True)
print(f"Total valid samples after enhanced feature engineering: N = {len(clean)} across {clean['OrderID'].nunique()} orders")

# 7:3 Split by OrderID
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
tr_idx, val_idx = next(gss.split(clean, groups=clean['OrderID']))
train_df = clean.iloc[tr_idx].copy().reset_index(drop=True)
val_df = clean.iloc[val_idx].copy().reset_index(drop=True)

y_tr = train_df['Mooney'].values
y_val = val_df['Mooney'].values

print(f"Train samples: {len(train_df)} ({train_df['OrderID'].nunique()} orders)")
print(f"Val samples  : {len(val_df)} ({val_df['OrderID'].nunique()} orders)")

# ==============================================================================
# Model 1: Old Baseline (Simple 9-feature Huber Linear)
# ==============================================================================
scaler_old = StandardScaler()
X_tr_old = scaler_old.fit_transform(train_df[BASE_FEATS])
X_val_old = scaler_old.transform(val_df[BASE_FEATS])
m_old = HuberRegressor(alpha=0.01).fit(X_tr_old, y_tr)
p_old = m_old.predict(X_val_old)

# ==============================================================================
# Model 2: Enhanced Rheological Multi-Feature EBM (Explainable Boosting Machine)
# ==============================================================================
ebm_enh = ExplainableBoostingRegressor(random_state=42, max_bins=64, interactions=10)
ebm_enh.fit(train_df[ENHANCED_FEATS], y_tr)
p_ebm_enh = ebm_enh.predict(val_df[ENHANCED_FEATS])

# ==============================================================================
# Model 3: Two-Stage Cascade Model (Latent Raw Level + Process Delta)
#   Stage 1: Latent Raw Level = predicted from mechanical load + thermal efficiency
#   Stage 2: Process Delta = sensitivity w * (x - xi)
# ==============================================================================
latent_feats = ['T_start', 'Mean_Shear_Torque', 'Heating_Rate_C_s', 'Heating_Efficiency_C_kWh', 'Mean_Power_kW', 'MSB_Load_Index']
stage1_m = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42).fit(train_df[latent_feats], y_tr)
stage1_pred_val = stage1_m.predict(val_df[latent_feats])

# Causal Delta adjustments for process variations
center_rpm = train_df['MST7_Rpm'].median()
center_t_msb = train_df['t_MSB1'].median()
center_t_mst = train_df['t_MST7'].median()
center_t_dump = train_df['T_dump'].median()

p_cascade = (stage1_pred_val 
             - 0.620 * (val_df['MST7_Rpm'] - center_rpm) 
             + 0.618 * (val_df['t_MSB1'] - center_t_msb) 
             - 0.018 * (val_df['t_MST7'] - center_t_mst) 
             - 0.102 * (val_df['T_dump'] - center_t_dump))

# ==============================================================================
# Model 4: Enhanced Stacked Ensemble (EBM + LightGBM + ExtraTrees with Tolerance Loss)
# ==============================================================================
lgb_m = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, num_leaves=12, random_state=42, verbose=-1)
lgb_m.fit(train_df[ENHANCED_FEATS], y_tr)
p_lgb = lgb_m.predict(val_df[ENHANCED_FEATS])

et_m = ExtraTreesRegressor(n_estimators=100, max_depth=6, random_state=42).fit(train_df[ENHANCED_FEATS], y_tr)
p_et = et_m.predict(val_df[ENHANCED_FEATS])

p_stacked = 0.40 * p_ebm_enh + 0.35 * p_lgb + 0.25 * p_et

# ==============================================================================
# Model 5: Plant-Realistic Damped Online Updating (Simulating Lab Feedback)
#   If an order has >= 2 batches: when 1st batch is tested, update subsequent batches!
# ==============================================================================
p_online = p_stacked.copy()
damp = 0.50 # conservative smoothing
order_bias = {}

for i, row in val_df.iterrows():
    oid = row['OrderID']
    if oid in order_bias:
        p_online[i] = p_stacked[i] + order_bias[oid]
    else:
        # First batch seen of this order in validation: records the residual for subsequent batches
        order_bias[oid] = (row['Mooney'] - p_stacked[i]) * damp

# ------------------------------------------------------------------------------
# Comparative Evaluation
# ------------------------------------------------------------------------------
models = [
    ("1. 旧版简单 9 特征线性基准 (旧基准)", p_old),
    ("2. 增强流变特征·微软纯白盒树 (EBM)", p_ebm_enh),
    ("3. 级联两阶段架构 (生胶潜变量 + 工艺因果增量)", p_cascade),
    ("4. 增强流变多模态集成模型 (Stacked EBM+LGB+ET)", p_stacked),
    ("5. ★ 车间真实抽检阻尼在线校准 (Online Damped V12)", p_online)
]

print("\n" + "=" * 115)
print(f"{'优化算法模型体系':<42s} | {'MAE':<7s} | {'<= ±3.0 MU':<11s} | {'<= ±2.0 MU':<11s} | {'<= ±1.0 MU':<11s} | {'R²':<7s}")
print("=" * 115)

for name, preds in models:
    mae = mean_absolute_error(y_val, preds)
    p3 = (np.abs(y_val - preds) <= 3.0).mean() * 100
    p2 = (np.abs(y_val - preds) <= 2.0).mean() * 100
    p1 = (np.abs(y_val - preds) <= 1.0).mean() * 100
    r2 = r2_score(y_val, preds)
    print(f"{name:<42s} | {mae:5.2f} MU | {p3:9.1f}% | {p2:9.1f}% | {p1:9.1f}% | {r2:+6.3f}")
print("=" * 115)

# Detailed Look at the troublesome orders (2288700, 2300167, 2351308)
print("\n" + "=" * 115)
print("【原严重超标订单 (2288700, 2300167 等) 优化前后点对点改善核验】")
print("=" * 115)
print(f"{'工单号':<8s} | {'车次':<5s} | {'实测门尼':<8s} | {'旧模型预估(误差)':<18s} | {'优化级联预估(误差)':<20s} | {'在线校准预估(误差)':<20s}")
print("-" * 115)

inspect_orders = ['2288700', '2300167', '2351308', '2320613', '2375629', '8067458']
for i, row in val_df[val_df['OrderID'].isin(inspect_orders)].head(15).iterrows():
    oid = row['OrderID']
    bno = int(row['BatchNumber'])
    real_m = row['Mooney']
    old_p = p_old[i]
    casc_p = p_cascade[i]
    onl_p = p_online[i]
    print(f"{oid:<8s} | #{bno:02d}   | {real_m:6.2f}MU | {old_p:5.2f}MU ({old_p-real_m:+6.2f}MU) | {casc_p:5.2f}MU ({casc_p-real_m:+6.2f}MU)   | {onl_p:5.2f}MU ({onl_p-real_m:+6.2f}MU)")
print("=" * 115)
