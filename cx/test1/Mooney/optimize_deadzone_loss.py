"""
Factory Tolerance Deadzone Loss Model Optimization
===================================================
Custom Objective Function:
  L(y, p) = 0 if |y - p| <= delta
            ( |y - p| - delta )^2 if |y - p| > delta

Gradient g and Hessian h:
  For |e| <= delta: g = 0, h = 0.01 (regularization)
  For e > delta   : g = -2 * (e - delta), h = 2
  For e < -delta  : g = -2 * (e + delta), h = 2
where e = y - p
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb
from interpret.glassbox import ExplainableBoostingRegressor

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"

# Load Datasets
math_df = pd.read_pickle(CACHE / "mathematical_dataset.pkl")
lab = pd.read_pickle(CACHE / "lab.pkl")
strict_lab = lab[lab['LabelInPallet'] == True].rename(columns={'SampleBatch': 'BatchNumber'})
math_df['OrderID'] = math_df['OrderID'].astype(str)
math_df['BatchNumber'] = math_df['BatchNumber'].astype(int)
strict_lab['OrderID'] = strict_lab['OrderID'].astype(str)
strict_lab['BatchNumber'] = strict_lab['BatchNumber'].astype(int)

df = math_df.drop(columns=['Mooney']).merge(
    strict_lab[['OrderID', 'BatchNumber', 'Mooney', 'TestStartTime']], 
    on=['OrderID', 'BatchNumber'], 
    how='inner'
)
clean = df[(df['Mooney'] >= 40) & (df['Mooney'] <= 85)].copy().sort_values(['TestStartTime', 'OrderID', 'BatchNumber']).reset_index(drop=True)

# Rich Feature Engineering
clean['t_MST7'] = (clean['MST_S7_Time_s'] - clean['MST_S6_Time_s']).clip(lower=0).fillna(30.0)
clean['t_MSB1'] = clean['MSB_S1_Time_s'].fillna(25.0)
clean['MST7_Rpm'] = clean['MST_S7_Rpm'].fillna(40.0)
clean['T_dump'] = clean['MSB_S3_Temp_C'].fillna(clean['Curve_Temp_Max_Full_C']).fillna(155.0)
clean['T_start'] = clean['MMS_Start_Temp_C'].fillna(clean['Curve_Temp_Min_MST_C']).fillna(45.0)
clean['Energy_Target'] = clean['MMS_Total_Energy_kWh'].fillna(clean['Curve_Int_Power_Full_kWh']).fillna(35.0)

# Temperature corrected shear work proxy
T_k = clean['T_start'] + 273.15
clean['Viscous_Work_Proxy'] = clean['Curve_Int_Torque_Full_Nms'].fillna(120000.0) / (clean['Curve_Length_s'] + 1e-3)
clean['Peak_Power_Rate'] = clean['Curve_Power_Max_Full_kW'].fillna(700.0) / (clean['Curve_Power_Mean_Full_kW'].fillna(350.0) + 1e-3)
clean['Thermal_Buildup'] = clean['T_dump'] - clean['T_start']
clean['Energy_Per_degC'] = clean['Energy_Target'] / (clean['Thermal_Buildup'] + 1e-3)

FEATURE_COLS = [
    't_MST7', 't_MSB1', 'MST7_Rpm', 'T_dump', 'T_start', 
    'Curve_Length_s', 'Viscous_Work_Proxy', 'Peak_Power_Rate', 
    'Thermal_Buildup', 'Energy_Per_degC'
]

X = clean[FEATURE_COLS].fillna(0.0).values
y_m = clean['Mooney'].values
y_e = clean['Energy_Target'].values

# Custom Deadzone Loss for Mooney (tolerance = 3.0 MU)
def mooney_deadzone_objective(y_true, y_pred):
    e = y_pred - y_true
    delta = 3.0
    grad = np.zeros_like(e)
    hess = np.full_like(e, 0.05) # small baseline hessian
    
    over = e > delta
    under = e < -delta
    
    grad[over] = 2.0 * (e[over] - delta)
    hess[over] = 2.0
    
    grad[under] = 2.0 * (e[under] + delta)
    hess[under] = 2.0
    
    return grad, hess

# Custom Deadzone Loss for Energy (tolerance = 2.0 kWh)
def energy_deadzone_objective(y_true, y_pred):
    e = y_pred - y_true
    delta = 2.0
    grad = np.zeros_like(e)
    hess = np.full_like(e, 0.05)
    
    over = e > delta
    under = e < -delta
    
    grad[over] = 2.0 * (e[over] - delta)
    hess[over] = 2.0
    
    grad[under] = 2.0 * (e[under] + delta)
    hess[under] = 2.0
    
    return grad, hess

# 5-Fold Cross Validation Evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Benchmark 1: Standard Huber Regressor
# Benchmark 2: Standard LightGBM (RMSE loss)
# Benchmark 3: Deadzone Loss LightGBM (Custom Factory Tolerance Loss)
# Benchmark 4: Microsoft EBM (Explainable Boosting Machine)

results = {
    'Huber_Mooney': [], 'Huber_Energy': [],
    'LGB_STD_Mooney': [], 'LGB_STD_Energy': [],
    'LGB_DZ_Mooney': [], 'LGB_DZ_Energy': [],
    'EBM_Mooney': [], 'EBM_Energy': []
}

for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
    X_tr, X_te = X[tr_idx], X[te_idx]
    ym_tr, ym_te = y_m[tr_idx], y_m[te_idx]
    ye_tr, ye_te = y_e[tr_idx], y_e[te_idx]
    
    # 1. Standard LightGBM
    lgb_std_m = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, num_leaves=12, random_state=42, verbose=-1)
    lgb_std_m.fit(X_tr, ym_tr)
    p_lgb_std_m = lgb_std_m.predict(X_te)
    
    lgb_std_e = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, num_leaves=12, random_state=42, verbose=-1)
    lgb_std_e.fit(X_tr, ye_tr)
    p_lgb_std_e = lgb_std_e.predict(X_te)
    
    # 2. Deadzone Loss LightGBM
    lgb_dz_m = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=4, num_leaves=12, random_state=42, verbose=-1, objective=mooney_deadzone_objective)
    lgb_dz_m.fit(X_tr, ym_tr)
    p_lgb_dz_m = lgb_dz_m.predict(X_te)
    
    lgb_dz_e = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=4, num_leaves=12, random_state=42, verbose=-1, objective=energy_deadzone_objective)
    lgb_dz_e.fit(X_tr, ye_tr)
    p_lgb_dz_e = lgb_dz_e.predict(X_te)
    
    # 3. Microsoft EBM (Glass-box)
    ebm_m = ExplainableBoostingRegressor(random_state=42, max_bins=64)
    ebm_m.fit(X_tr, ym_tr)
    p_ebm_m = ebm_m.predict(X_te)
    
    ebm_e = ExplainableBoostingRegressor(random_state=42, max_bins=64)
    ebm_e.fit(X_tr, ye_tr)
    p_ebm_e = ebm_e.predict(X_te)
    
    # Record metrics: (Pass Rate <= 3.0 MU, MAE)
    results['LGB_STD_Mooney'].append(((np.abs(ym_te - p_lgb_std_m) <= 3.0).mean() * 100, mean_absolute_error(ym_te, p_lgb_std_m)))
    results['LGB_DZ_Mooney'].append(((np.abs(ym_te - p_lgb_dz_m) <= 3.0).mean() * 100, mean_absolute_error(ym_te, p_lgb_dz_m)))
    results['EBM_Mooney'].append(((np.abs(ym_te - p_ebm_m) <= 3.0).mean() * 100, mean_absolute_error(ym_te, p_ebm_m)))
    
    results['LGB_STD_Energy'].append(((np.abs(ye_te - p_lgb_std_e) <= 2.0).mean() * 100, mean_absolute_error(ye_te, p_lgb_std_e)))
    results['LGB_DZ_Energy'].append(((np.abs(ye_te - p_lgb_dz_e) <= 2.0).mean() * 100, mean_absolute_error(ye_te, p_lgb_dz_e)))
    results['EBM_Energy'].append(((np.abs(ye_te - p_ebm_e) <= 2.0).mean() * 100, mean_absolute_error(ye_te, p_ebm_e)))

print("\n" + "=" * 90)
print("【算法优化对比：5 折交叉验证（554 车次真实数据全量验证）】")
print("=" * 90)
print(f"{'优化算法体系':<36s} | {'门尼合格率(<=±3.0MU)':<22s} | {'能耗合格率(<=±2.0kWh)':<22s}")
print("-" * 90)

for name, key_m, key_e in [
    ("1. 标准 LightGBM (传统 RMSE 损失)", 'LGB_STD_Mooney', 'LGB_STD_Energy'),
    ("2. 微软解释性 EBM (纯白盒样条树)", 'EBM_Mooney', 'EBM_Energy'),
    ("3. ★ 容差死区损失 LightGBM (专为±3/±2设计)", 'LGB_DZ_Mooney', 'LGB_DZ_Energy'),
]:
    p_m = np.mean([r[0] for r in results[key_m]])
    mae_m = np.mean([r[1] for r in results[key_m]])
    p_e = np.mean([r[0] for r in results[key_e]])
    mae_e = np.mean([r[1] for r in results[key_e]])
    print(f"{name:<36s} | {p_m:5.1f}% (MAE {mae_m:4.2f} MU)   | {p_e:5.1f}% (MAE {mae_e:4.2f} kWh)")

print("=" * 90)
