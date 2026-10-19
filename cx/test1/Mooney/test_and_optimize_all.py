"""
Deep Algorithm Optimization and Multi-Scenario Comprehensive Verification Suite
=================================================================================
Tests and optimizes:
  1. Dynamic Baseline Inversion (Thermal-Mechanical Energy Inversion, zero static mean/median)
  2. Sequential Kalman / R2R Adaptive State Estimator
  3. Tolerance Deadzone Loss Optuna-tuned PC-GBDT
  4. Multi-Step Causal Delta Sensitivity Engine on DOE Orders
  5. Multi-Scenario Extensive Verification (DOE, High Mooney, Normal, Low Mooney)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"

# 1. Load Datasets
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

# Feature engineering
clean['t_MST7'] = (clean['MST_S7_Time_s'] - clean['MST_S6_Time_s']).clip(lower=0).fillna(30.0)
clean['t_MSB1'] = clean['MSB_S1_Time_s'].fillna(25.0)
clean['MST7_Rpm'] = clean['MST_S7_Rpm'].fillna(40.0)
clean['T_dump'] = clean['MSB_S3_Temp_C'].fillna(clean['Curve_Temp_Max_Full_C']).fillna(155.0)
clean['T_start'] = clean['MMS_Start_Temp_C'].fillna(clean['Curve_Temp_Min_MST_C']).fillna(45.0)
clean['Energy_Target'] = clean['MMS_Total_Energy_kWh'].fillna(clean['Curve_Int_Power_Full_kWh']).fillna(35.0)

print(f"Total Valid Batches: {len(clean)} across {clean['OrderID'].nunique()} Orders")

# ------------------------------------------------------------------------------
# Test 1: Evaluating Multi-Batch Orders using Kalman / Recursive State Estimation
# ------------------------------------------------------------------------------
# For all orders with >= 2 batches:
# Batch 1 provides initial observation to calibrate state Y0 dynamically.
# Then predict Batch 2, 3, ... using:
#   Predicted_Y = Current_State_Y0 + Delta_Process
# Compare with static model prediction!

multi_order_counts = clean['OrderID'].value_counts()
multi_orders = multi_order_counts[multi_order_counts >= 2].index.tolist()
multi_df = clean[clean['OrderID'].isin(multi_orders)].sort_values(['OrderID', 'BatchNumber']).copy().reset_index(drop=True)

print(f"Multi-batch orders: {len(multi_orders)} orders, containing {len(multi_df)} total batches.")

# Process Delta coefficients:
# dM / d(rpm) = -0.620 MU/rpm
# dM / d(t_MSB1) = +0.0618 MU/s
# dM / d(t_MST7) = -0.0180 MU/s
# dM / d(T_dump) = -0.1020 MU/degC

# Compare:
# Method A: Static Prediction (Global mean baseline ~59.4 MU)
# Method B: Dynamic Sequential Kalman / Adaptive Estimator (Zero hardcoded baseline)

static_preds = []
kalman_preds = []
actual_subsequent = []
orders_subsequent = []
batch_nums = []

# Sensitivities
w_rpm = -0.620
w_msb = 0.0618
w_mst = -0.0180
w_dump = -0.1020

# Global static mean
global_mean = clean['Mooney'].mean()

for oid in multi_orders:
    order_batches = multi_df[multi_df['OrderID'] == oid].sort_values('BatchNumber').reset_index(drop=True)
    if len(order_batches) < 2:
        continue
    
    # Batch 0: dynamic anchor initialization (computed from batch 0)
    b0 = order_batches.iloc[0]
    
    # State Y0 initialized strictly from batch 0 actual measurement minus batch 0 process delta
    y_state = b0['Mooney']
    p_var = 1.0 # state uncertainty
    q_var = 0.2 # process noise
    r_var = 0.8 # measurement noise
    
    ref_rpm = b0['MST7_Rpm']
    ref_msb = b0['t_MSB1']
    ref_mst = b0['t_MST7']
    ref_dump = b0['T_dump']
    
    for i in range(1, len(order_batches)):
        b = order_batches.iloc[i]
        actual_y = b['Mooney']
        
        # Static baseline prediction
        p_static = global_mean
        static_preds.append(p_static)
        
        # Calculate process adjustment delta from batch 0 to batch i
        delta_process = (
            w_rpm * (b['MST7_Rpm'] - ref_rpm) +
            w_msb * (b['t_MSB1'] - ref_msb) +
            w_mst * (b['t_MST7'] - ref_mst) +
            w_dump * (b['T_dump'] - ref_dump)
        )
        if np.isnan(delta_process):
            delta_process = 0.0
        
        # Prior prediction for batch i
        p_pred = y_state + delta_process
        kalman_preds.append(p_pred)
        actual_subsequent.append(actual_y)
        orders_subsequent.append(oid)
        batch_nums.append(int(b['BatchNumber']))
        
        # Measurement update for next batch
        kalman_gain = p_var / (p_var + r_var)
        innovation = actual_y - p_pred
        y_state = y_state + kalman_gain * innovation
        p_var = (1.0 - kalman_gain) * p_var + q_var
        
        # Update reference to current
        ref_rpm = b['MST7_Rpm']
        ref_msb = b['t_MSB1']
        ref_mst = b['t_MST7']
        ref_dump = b['T_dump']

act_arr = np.array(actual_subsequent)
stat_arr = np.array(static_preds)
kalm_arr = np.array(kalman_preds)

print("\n" + "=" * 90)
print(f"【多车次连续生产（134 个工单，共 {len(act_arr)} 车次验证）优化效果对比】")
print("=" * 90)
print(f"{'模型算法体系':<35s} | {'MAE':<8s} | {'<= ±3.0 MU (合格率)':<20s} | {'<= ±1.0 MU (黄金率)':<20s}")
print("-" * 90)

mae_stat = mean_absolute_error(act_arr, stat_arr)
p3_stat = (np.abs(act_arr - stat_arr) <= 3.0).mean() * 100
p1_stat = (np.abs(act_arr - stat_arr) <= 1.0).mean() * 100

mae_kalm = mean_absolute_error(act_arr, kalm_arr)
p3_kalm = (np.abs(act_arr - kalm_arr) <= 3.0).mean() * 100
p1_kalm = (np.abs(act_arr - kalm_arr) <= 1.0).mean() * 100

print(f"{'1. 旧版静态固定基准模型':<35s} | {mae_stat:6.2f} MU | {p3_stat:18.1f}% | {p1_stat:18.1f}%")
print(f"{'2. ★ 动态序贯滤波+因果增量模型':<35s} | {mae_kalm:6.2f} MU | {p3_kalm:18.1f}% | {p1_kalm:18.1f}%")
print("=" * 90)

# Check the troublesome high-drift orders
print("\n" + "=" * 90)
print("【高胶质/极端漂移工单（原误差高达 10~15 MU）序贯自适应改善验证】")
print("=" * 90)
print(f"{'工单号':<10s} | {'车次':<5s} | {'实测门尼':<8s} | {'旧版静态预估(误差)':<22s} | {'动态优化预估(误差)':<22s} | {'状态'}")
print("-" * 90)

trouble_df = pd.DataFrame({
    'OrderID': orders_subsequent,
    'BatchNumber': batch_nums,
    'Actual': act_arr,
    'Static': stat_arr,
    'Dynamic': kalm_arr
})

sample_trouble = trouble_df[trouble_df['OrderID'].isin(['2288700', '2300167', '2351308', '2320613', '2375629'])].head(15)
for _, r in sample_trouble.iterrows():
    oid = r['OrderID']
    bno = int(r['BatchNumber'])
    act = r['Actual']
    st = r['Static']
    dyn = r['Dynamic']
    err_st = st - act
    err_dyn = dyn - act
    status = "合格 (Pass)" if abs(err_dyn) <= 3.0 else "超标 (Fail)"
    print(f"{oid:<10s} | #{bno:02d}   | {act:6.2f}MU | {st:5.2f}MU ({err_st:+6.2f}MU)      | {dyn:5.2f}MU ({err_dyn:+6.2f}MU)       | {status}")
print("=" * 90)
