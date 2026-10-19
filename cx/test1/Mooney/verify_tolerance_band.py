"""
Verification of Exact Tolerance Band Coverage:
Check whether Real Mooney falls inside [Predicted Mooney - 3.0, Predicted Mooney + 3.0]
and whether Real Energy falls inside [Predicted Energy - 2.0, Predicted Energy + 2.0]
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"

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

clean['t_MST7'] = (clean['MST_S7_Time_s'] - clean['MST_S6_Time_s']).clip(lower=0).fillna(30.0)
clean['t_MSB1'] = clean['MSB_S1_Time_s'].fillna(25.0)
clean['MST7_Rpm'] = clean['MST_S7_Rpm'].fillna(40.0)
clean['T_dump'] = clean['MSB_S3_Temp_C'].fillna(clean['Curve_Temp_Max_Full_C']).fillna(155.0)
clean['Energy_Target'] = clean['MMS_Total_Energy_kWh'].fillna(clean['Curve_Int_Power_Full_kWh']).fillna(35.0)

# Sensitivities
w_rpm = -0.620
w_msb = 0.0618
w_mst = -0.0180
w_dump = -0.1020

w_e_rpm = 0.180
w_e_msb = 0.205
w_e_mst = 0.115
w_e_dump = 0.085

# 1. DOE Batches: Process adjustment predictions
print("=" * 125)
print("【核验 1：工艺调整工单（预测门尼绝对值 ± 3.0 区间是否严格包含真实门尼）】")
print("=" * 125)
print(f"{'工单号':<8s} | {'车次':<5s} | {'真实门尼':<8s} | {'预测绝对门尼':<12s} | {'预测门尼±3.0区间':<20s} | {'绝对误差':<9s} | {'门尼合格?':<10s} | {'真实能耗':<8s} | {'预测能耗':<8s} | {'能耗合格?'}")
print("-" * 125)

doe_orders = ['8067458', '8101635']
for oid in doe_orders:
    sub = clean[clean['OrderID'] == oid].sort_values('BatchNumber').reset_index(drop=True)
    # Target anchor from Batch 34 in 8067458 or Batch 2 in 8101635
    b_ref = sub.iloc[0]
    if oid == '8067458' and len(sub) > 2:
        b_ref = sub[sub['BatchNumber'] == 34].iloc[0]
        
    base_m = b_ref['Mooney']
    base_e = b_ref['Energy_Target']
    
    for i in range(len(sub)):
        b = sub.iloc[i]
        d_rpm = b['MST7_Rpm'] - b_ref['MST7_Rpm']
        d_msb = b['t_MSB1'] - b_ref['t_MSB1']
        d_mst = b['t_MST7'] - b_ref['t_MST7']
        d_dump = b['T_dump'] - b_ref['T_dump']
        
        pred_m = base_m + w_rpm * d_rpm + w_msb * d_msb + w_mst * d_mst + w_dump * d_dump
        pred_e = base_e + w_e_rpm * d_rpm + w_e_msb * d_msb + w_e_mst * d_mst + w_e_dump * d_dump
        
        err_m = pred_m - b['Mooney']
        err_e = pred_e - b['Energy_Target']
        
        m_band_low = pred_m - 3.0
        m_band_high = pred_m + 3.0
        is_m_pass = (b['Mooney'] >= m_band_low) and (b['Mooney'] <= m_band_high)
        is_e_pass = abs(err_e) <= 2.0
        
        bno = int(b['BatchNumber'])
        status_m = "★ 合格 (包含)" if is_m_pass else "× 超标"
        status_e = "★ 合格 (<=2.0)" if is_e_pass else "× 超标"
        band_str = f"[{m_band_low:5.2f}, {m_band_high:5.2f}]"
        
        print(f"{oid:<8s} | #{bno:02d}   | {b['Mooney']:6.2f}MU | {pred_m:6.2f}MU      | {band_str:<20s} | {err_m:+6.2f}MU  | {status_m:<10s} | {b['Energy_Target']:6.2f}kW | {pred_e:6.2f}kW | {status_e}")

print("=" * 125)

# 2. Continuous Production Multi-batch Orders
print("\n" + "=" * 125)
print("【核验 2：极端工单连续生产（自适应更新下，预测绝对门尼 ± 3.0 是否包住真实门尼）】")
print("=" * 125)
print(f"{'工单号':<8s} | {'车次':<5s} | {'真实门尼':<8s} | {'预测绝对门尼':<12s} | {'预测门尼±3.0区间':<20s} | {'绝对误差':<9s} | {'门尼合格?':<10s} | {'真实能耗':<8s} | {'预测能耗':<8s} | {'能耗合格?'}")
print("-" * 125)

multi_order_counts = clean['OrderID'].value_counts()
multi_orders = multi_order_counts[multi_order_counts >= 2].index.tolist()

checked_samples = []
for oid in ['2288700', '2300167', '2351308', '2139521', '2320613', '2375629']:
    if oid not in clean['OrderID'].values:
        continue
    sub = clean[clean['OrderID'] == oid].sort_values('BatchNumber').reset_index(drop=True)
    if len(sub) < 2:
        continue
    b0 = sub.iloc[0]
    lot_anchor = b0['Mooney']
    e_anchor = b0['Energy_Target']
    
    for i in range(1, len(sub)):
        b = sub.iloc[i]
        d_rpm = b['MST7_Rpm'] - b0['MST7_Rpm']
        d_msb = b['t_MSB1'] - b0['t_MSB1']
        d_mst = b['t_MST7'] - b0['t_MST7']
        d_dump = b['T_dump'] - b0['T_dump']
        
        pred_m = lot_anchor + (w_rpm * d_rpm + w_msb * d_msb + w_mst * d_mst + w_dump * d_dump)
        pred_e = e_anchor + (w_e_rpm * d_rpm + w_e_msb * d_msb + w_e_mst * d_mst + w_e_dump * d_dump)
        
        err_m = pred_m - b['Mooney']
        err_e = pred_e - b['Energy_Target']
        
        m_band_low = pred_m - 3.0
        m_band_high = pred_m + 3.0
        is_m_pass = (b['Mooney'] >= m_band_low) and (b['Mooney'] <= m_band_high)
        is_e_pass = abs(err_e) <= 2.0
        
        bno = int(b['BatchNumber'])
        status_m = "★ 合格 (包含)" if is_m_pass else "× 超标"
        status_e = "★ 合格 (<=2.0)" if is_e_pass else "× 超标"
        band_str = f"[{m_band_low:5.2f}, {m_band_high:5.2f}]"
        
        print(f"{oid:<8s} | #{bno:02d}   | {b['Mooney']:6.2f}MU | {pred_m:6.2f}MU      | {band_str:<20s} | {err_m:+6.2f}MU  | {status_m:<10s} | {b['Energy_Target']:6.2f}kW | {pred_e:6.2f}kW | {status_e}")
        
        # update anchor
        lot_anchor = lot_anchor + 0.65 * (b['Mooney'] - pred_m)

print("=" * 125)
