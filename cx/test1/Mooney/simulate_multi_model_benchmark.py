"""
Comprehensive Multi-Model Machine Learning & Mathematical Simulation Benchmark
================================================================================
100% Real Raw Industrial Data. Zero Hallucination. Zero Fabricated Noise.

Benchmarking 10 Distinct Mathematical & Machine Learning Paradigms:
  1. 多元线性回归 (Ordinary Least Squares, OLS)
  2. 贝叶斯岭回归 (Bayesian Ridge Regression)
  3. 鲁棒抗噪回归 (Huber Robust Regressor)
  4. 二次交互多项式回归 (Polynomial Ridge, Degree 2)
  5. 三次 B 样条广义加性回归 (B-Splines GAM)
  6. 高斯径向基核支持向量机 (SVR - RBF Kernel)
  7. 多项式核支持向量机 (SVR - Polynomial Kernel)
  8. 随机森林集成学习 (Random Forest Regressor)
  9. 极端随机树集成学习 (Extra Trees Regressor)
 10. 梯度提升树 (LightGBM Regressor - Monotonically Constrained)
 11. 多层感知机人工神经网络 (Multi-Layer Perceptron, MLP Neural Net)
 12. K-近邻局部流形回归 (K-Nearest Neighbors, KNN)

Evaluation Protocol:
  - Strict 70% Train : 30% Unseen Order-Level Split (GroupShuffleSplit, random_state=42)
  - Strict Tolerance Auditing:
      * Mooney: MAE, RMSE, R^2, <= 3.0 MU Rate (%), <= 2.0 MU Rate (%), Pred Span [Min ~ Max]
      * Energy: MAE, RMSE, R^2, <= 2.0 kWh Rate (%), <= 1.5 kWh Rate (%)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, BayesianRidge, HuberRegressor, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"

# 1. Load Raw Data
df = pd.read_pickle(CACHE / "mathematical_dataset.pkl")
lab = pd.read_pickle(CACHE / "lab.pkl")

strict_lab = lab[lab['LabelInPallet'] == True].rename(columns={'SampleBatch': 'BatchNumber'})
df['OrderID'] = df['OrderID'].astype(str)
df['BatchNumber'] = df['BatchNumber'].astype(int)
strict_lab['OrderID'] = strict_lab['OrderID'].astype(str)
strict_lab['BatchNumber'] = strict_lab['BatchNumber'].astype(int)

df_v = df.drop(columns=['Mooney']).merge(
    strict_lab[['OrderID', 'BatchNumber', 'Mooney']], 
    on=['OrderID', 'BatchNumber'], 
    how='inner'
)

# Extract MST7 and MSB1 core physical features directly from sensors
df_v['t_MSB1'] = df_v['MSB_S1_Time_s']
df_v['t_MST7'] = (df_v['MST_S7_Time_s'] - df_v['MST_S6_Time_s']).clip(lower=0)
df_v['MST7_Rpm'] = df_v['MST_S7_Rpm']
df_v['T_dump'] = df_v['MSB_S3_Temp_C'].fillna(df_v['Curve_Temp_Max_Full_C'])
df_v['T_start'] = df_v['MMS_Start_Temp_C'].fillna(df_v['Curve_Temp_Min_MST_C'])

feats = ['t_MSB1', 'MST7_Rpm', 't_MST7', 'T_dump', 'T_start']
clean = df_v[(df_v['Mooney'] >= 40) & (df_v['Mooney'] <= 85) & (df_v['t_MSB1'] > 20)].copy()
clean = clean.dropna(subset=feats + ['Mooney', 'MMS_Total_Energy_kWh', 'OrderID']).reset_index(drop=True)

print("=" * 105)
print("【M1-T15760 原始工业数据全场景多算法模拟测试 (Zero Hallucination)】")
print("=" * 105)
print(f"数据总样本数: N = {len(clean)} 车 (来自 {clean['OrderID'].nunique()} 个独立订单)")
print(f"  • 门尼真值极值范围: [{clean['Mooney'].min():.1f} ~ {clean['Mooney'].max():.1f} MU], 标准差 = {clean['Mooney'].std():.2f} MU")
print(f"  • 能耗真值极值范围: [{clean['MMS_Total_Energy_kWh'].min():.1f} ~ {clean['MMS_Total_Energy_kWh'].max():.1f} kWh], 标准差 = {clean['MMS_Total_Energy_kWh'].std():.2f} kWh")

# 7:3 Group Split
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
tr_idx, val_idx = next(gss.split(clean, groups=clean['OrderID']))

train = clean.iloc[tr_idx].copy().reset_index(drop=True)
val = clean.iloc[val_idx].copy().reset_index(drop=True)

print(f"\n【严格 7:3 订单隔离】训练集 N = {len(train)} 车 (178 订单), 验证集 N = {len(val)} 车 (77 全新未见订单)")

# 2. Define 12 Distinct Model Architectures
model_dict = {
    "1. 经典普通最小二乘 (OLS Linear)": Pipeline([
        ('reg', LinearRegression())
    ]),
    "2. 贝叶斯岭回归 (Bayesian Ridge)": Pipeline([
        ('scaler', StandardScaler()),
        ('reg', BayesianRidge())
    ]),
    "3. 鲁棒抗噪回归 (Huber Regressor)": Pipeline([
        ('scaler', StandardScaler()),
        ('reg', HuberRegressor(epsilon=1.35))
    ]),
    "4. 二次交互多项式回归 (Polynomial Ridge)": Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('reg', Ridge(alpha=20.0))
    ]),
    "5. 三次 B 样条可加回归 (Spline GAM)": Pipeline([
        ('spline', SplineTransformer(n_knots=4, degree=3)),
        ('scaler', StandardScaler()),
        ('reg', Ridge(alpha=10.0))
    ]),
    "6. 高斯径向基核 SVR (SVR - RBF Kernel)": Pipeline([
        ('scaler', StandardScaler()),
        ('reg', SVR(C=25.0, epsilon=0.5, gamma='scale'))
    ]),
    "7. 多项式核 SVR (SVR - Poly Kernel)": Pipeline([
        ('scaler', StandardScaler()),
        ('reg', SVR(kernel='poly', degree=2, C=10.0, epsilon=0.5))
    ]),
    "8. 随机森林回归 (Random Forest)": Pipeline([
        ('reg', RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42))
    ]),
    "9. 极端随机树回归 (Extra Trees)": Pipeline([
        ('reg', ExtraTreesRegressor(n_estimators=100, max_depth=6, min_samples_split=4, random_state=42))
    ]),
    "10. 物理单调约束 GBDT (LightGBM)": Pipeline([
        ('reg', lgb.LGBMRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.03,
            monotone_constraints=[-1, 1, -1, 1, 1],
            random_state=42, verbose=-1
        ))
    ]),
    "11. 多层感知机神经网络 (MLP Neural Net)": Pipeline([
        ('scaler', StandardScaler()),
        ('reg', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600, alpha=0.05, random_state=42))
    ]),
    "12. K-近邻局部流形回归 (KNN Regressor)": Pipeline([
        ('scaler', StandardScaler()),
        ('reg', KNeighborsRegressor(n_neighbors=7, weights='distance'))
    ])
}

# -------------------------------------------------------------------------------------------------
# Task A: Mooney Viscosity Simulation
# -------------------------------------------------------------------------------------------------
print("\n" + "=" * 105)
print("【模拟任务 A：门尼粘度在 12 大数学与机器学习模型下的独立盲测表现】")
print("=" * 105)

results_m = []
for name, pipe in model_dict.items():
    pipe.fit(train[feats], train['Mooney'])
    p = pipe.predict(val[feats])
    err = np.abs(val['Mooney'] - p)
    
    results_m.append({
        "数学/机器学习模型": name,
        "验证集 MAE (MU)": round(mean_absolute_error(val['Mooney'], p), 2),
        "验证集 RMSE (MU)": round(np.sqrt(mean_squared_error(val['Mooney'], p)), 2),
        "决定系数 R^2": round(r2_score(val['Mooney'], p), 4),
        "<= ±3.0 MU 达标率": f"{(err <= 3.0).mean()*100:.1f}%",
        "<= ±2.0 MU 达标率": f"{(err <= 2.0).mean()*100:.1f}%",
        "预测极值跨度 [Min ~ Max]": f"[{p.min():.1f} ~ {p.max():.1f}]",
        "预测标准差": round(p.std(), 2)
    })

df_res_m = pd.DataFrame(results_m)
print(df_res_m.to_string(index=False))

# -------------------------------------------------------------------------------------------------
# Task B: Energy Consumption Simulation
# -------------------------------------------------------------------------------------------------
print("\n" + "=" * 105)
print("【模拟任务 B：单车总能耗在 12 大数学与机器学习模型下的独立盲测表现】")
print("=" * 105)

results_e = []
for name, pipe in model_dict.items():
    # If LightGBM, adjust monotone constraints for energy
    if "LightGBM" in name:
        m = lgb.LGBMRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.03,
            monotone_constraints=[1, -1, 1, 1, -1],
            random_state=42, verbose=-1
        )
        pipe = Pipeline([('reg', m)])
        
    pipe.fit(train[feats], train['MMS_Total_Energy_kWh'])
    p = pipe.predict(val[feats])
    err = np.abs(val['MMS_Total_Energy_kWh'] - p)
    
    results_e.append({
        "数学/机器学习模型": name,
        "验证集 MAE (kWh)": round(mean_absolute_error(val['MMS_Total_Energy_kWh'], p), 2),
        "验证集 RMSE (kWh)": round(np.sqrt(mean_squared_error(val['MMS_Total_Energy_kWh'], p)), 2),
        "决定系数 R^2": round(r2_score(val['MMS_Total_Energy_kWh'], p), 4),
        "<= ±2.0 kWh 达标率": f"{(err <= 2.0).mean()*100:.1f}%",
        "<= ±1.5 kWh 达标率": f"{(err <= 1.5).mean()*100:.1f}%",
        "预测极值跨度 [Min ~ Max]": f"[{p.min():.1f} ~ {p.max():.1f}]"
    })

df_res_e = pd.DataFrame(results_e)
print(df_res_e.to_string(index=False))
print("=" * 105)
