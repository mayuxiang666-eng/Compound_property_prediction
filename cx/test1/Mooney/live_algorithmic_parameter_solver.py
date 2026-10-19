import sys
import time
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import optimize

print("=" * 110)
print(f"【现场算法自主迭代实机运行】启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 110)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "cache" / "pure_1to1_single_batch_dataset.pkl"

if not DATA_PATH.exists():
    print(f"错误: 找不到数据集文件 {DATA_PATH}")
    sys.exit(1)

df = pd.read_pickle(DATA_PATH)
print(f"1. 成功从磁盘加载纯 1对1 单车确证数据集: {DATA_PATH.name}")
print(f"   • 样本总量: {len(df)} 行记录")
print(f"   • 字段列表: {list(df.columns)}")

feature_cols = [
    'MST_Step7_Duration_s',
    'MST_Step7_Rpm',
    'MSB_Step1_Duration_s',
    'Dump_Temp_C',
    'Start_Temp_C'
]
target_mooney = 'Real_Mooney_MU'
target_energy = 'Real_Energy_kWh'

clean_df = df.dropna(subset=feature_cols + [target_mooney, target_energy]).reset_index(drop=True)
print(f"\n2. 数据质量清洗后有效训练样本数: N = {len(clean_df)} 车")
print(f"   • 门尼真值范围: [{clean_df[target_mooney].min():.2f} ~ {clean_df[target_mooney].max():.2f} MU], 均值 = {clean_df[target_mooney].mean():.2f} MU")
print(f"   • 能耗真值范围: [{clean_df[target_energy].min():.2f} ~ {clean_df[target_energy].max():.2f} kWh], 均值 = {clean_df[target_energy].mean():.2f} kWh")

X = clean_df[feature_cols].values
y_m = clean_df[target_mooney].values
y_e = clean_df[target_energy].values
n_samples, n_features = X.shape

print("\n" + "=" * 110)
print("【第一部分：门尼粘度模型参数算法迭代求解 (L-BFGS-B 拟牛顿法)】")
print("=" * 110)
print(f"优化变量规模: 待优化参数共 {1 + n_features + n_features} 个 (基准Y0 + 5个工况中心xi + 5个因果权重w)")
print("目标函数: 工业死区容差损失 L(theta) = mean( max(0, |pred - y| - 3.0)^2 ) + L2 正则化")

# Parameter vector: theta = [Y0, xi_0..xi_4, w_0..w_4]
tol_m = 3.0
l2_reg = 0.01

iteration_records_m = []

def objective_mooney(theta):
    Y0 = theta[0]
    xi = theta[1 : 1 + n_features]
    w = theta[1 + n_features : ]
    
    # Model: y_pred = Y0 + sum_j [ w_j * (x_ij - xi_j) ]
    X_centered = X - xi
    y_pred = Y0 + X_centered @ w
    
    errors = y_pred - y_m
    abs_errors = np.abs(errors)
    deadzone_violations = np.maximum(0.0, abs_errors - tol_m)
    
    # Loss = mean deadzone squared + smooth L2 + weight regularization
    loss = np.mean(deadzone_violations ** 2) + 0.05 * np.mean(errors ** 2) + l2_reg * np.sum(w ** 2)
    return loss

def callback_mooney(xk):
    current_loss = objective_mooney(xk)
    it_num = len(iteration_records_m) + 1
    iteration_records_m.append((it_num, current_loss, xk[0], np.copy(xk[1:1+n_features]), np.copy(xk[1+n_features:])))
    print(f"  [迭代 #{it_num:02d}] 目标损失 Loss = {current_loss:8.4f} | 当前基准 Y0 = {xk[0]:6.2f} MU | 权重范数 ||w|| = {np.linalg.norm(xk[1+n_features:]):.4f}")
    time.sleep(0.05) # simulate realistic step progression for clear visual verification

# Start from unbiased standard guess
theta_init_m = np.concatenate([
    [50.0],                        # initial Y0 guess (arbitrary 50 MU)
    np.mean(X, axis=0),            # initial centers
    np.zeros(n_features)           # initial weights (zero initial prior)
])

print(f"\n[开始迭代] 初始未经优化的损失 Loss_0 = {objective_mooney(theta_init_m):.4f}")
print("-" * 110)

t_start = time.time()
res_m = optimize.minimize(
    objective_mooney,
    theta_init_m,
    method='L-BFGS-B',
    callback=callback_mooney,
    options={'maxiter': 100, 'ftol': 1e-6, 'disp': False}
)
t_elapsed = time.time() - t_start

print("-" * 110)
print(f"【门尼模型迭代收敛完毕】耗时: {t_elapsed:.2f} 秒 | 算法收敛状态: {res_m.message}")
print(f"  • 总迭代轮数: {res_m.nit} 轮")
print(f"  • 函数调用次数: {res_m.nfev} 次")
print(f"  • 最终优化达到的极小损失: {res_m.fun:.4f}")

Y0_solved = res_m.x[0]
xi_solved = res_m.x[1 : 1 + n_features]
w_solved = res_m.x[1 + n_features :]

print("\n★【算法求解出来的门尼模型核心参数】(全由优化算法生成):")
print(f"  • 算法迭代解出的全局基准门尼 Y0 = {Y0_solved:.2f} MU")
for col_name, c_val, w_val in zip(feature_cols, xi_solved, w_solved):
    print(f"    - 参数: {col_name:22s} | 工况中心 xi = {c_val:8.2f} | 因果敏感度 w = {w_val:+.4f}")


print("\n" + "=" * 110)
print("【第二部分：单车总能耗模型参数算法迭代求解】")
print("=" * 110)
print("目标函数: 单车能耗死区容差损失 (容差 <= 2.0 kWh)")

tol_e = 2.0
iteration_records_e = []

def objective_energy(theta):
    E0 = theta[0]
    xi = theta[1 : 1 + n_features]
    w = theta[1 + n_features : ]
    
    X_centered = X - xi
    e_pred = E0 + X_centered @ w
    
    errors = e_pred - y_e
    abs_errors = np.abs(errors)
    deadzone_violations = np.maximum(0.0, abs_errors - tol_e)
    loss = np.mean(deadzone_violations ** 2) + 0.05 * np.mean(errors ** 2) + l2_reg * np.sum(w ** 2)
    return loss

def callback_energy(xk):
    current_loss = objective_energy(xk)
    it_num = len(iteration_records_e) + 1
    iteration_records_e.append((it_num, current_loss, xk[0], np.copy(xk[1:1+n_features]), np.copy(xk[1+n_features:])))
    print(f"  [迭代 #{it_num:02d}] 目标损失 Loss = {current_loss:8.4f} | 当前基准 E0 = {xk[0]:6.2f} kWh | 权重范数 ||w|| = {np.linalg.norm(xk[1+n_features:]):.4f}")
    time.sleep(0.05)

theta_init_e = np.concatenate([
    [30.0],
    np.mean(X, axis=0),
    np.zeros(n_features)
])

print(f"\n[开始迭代] 初始能耗损失 Loss_0 = {objective_energy(theta_init_e):.4f}")
print("-" * 110)

t_start_e = time.time()
res_e = optimize.minimize(
    objective_energy,
    theta_init_e,
    method='L-BFGS-B',
    callback=callback_energy,
    options={'maxiter': 100, 'ftol': 1e-6, 'disp': False}
)
t_elapsed_e = time.time() - t_start_e

print("-" * 110)
print(f"【能耗模型迭代收敛完毕】耗时: {t_elapsed_e:.2f} 秒 | 算法收敛状态: {res_e.message}")
print(f"  • 总迭代轮数: {res_e.nit} 轮")
print(f"  • 最终优化达到的极小损失: {res_e.fun:.4f}")

E0_solved = res_e.x[0]
xi_e_solved = res_e.x[1 : 1 + n_features]
w_e_solved = res_e.x[1 + n_features :]

print("\n★【算法求解出来的能耗模型核心参数】(全由优化算法生成):")
print(f"  • 算法迭代解出的全局基准能耗 E0 = {E0_solved:.2f} kWh")
for col_name, c_val, w_val in zip(feature_cols, xi_e_solved, w_e_solved):
    print(f"    - 参数: {col_name:22s} | 工况中心 xi = {c_val:8.2f} | 能耗做功敏感度 w = {w_val:+.4f}")

# Save the solved parameters to JSON
out_params = {
    'solve_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'n_samples': n_samples,
    'mooney_model': {
        'Y0_baseline_MU': round(float(Y0_solved), 4),
        'centers_xi': {f: round(float(xi), 4) for f, xi in zip(feature_cols, xi_solved)},
        'sensitivities_w': {f: round(float(w), 4) for f, w in zip(feature_cols, w_solved)},
        'iterations': res_m.nit,
        'final_loss': round(float(res_m.fun), 4)
    },
    'energy_model': {
        'E0_baseline_kWh': round(float(E0_solved), 4),
        'centers_xi': {f: round(float(xi), 4) for f, xi in zip(feature_cols, xi_e_solved)},
        'sensitivities_w': {f: round(float(w), 4) for f, w in zip(feature_cols, w_e_solved)},
        'iterations': res_e.nit,
        'final_loss': round(float(res_e.fun), 4)
    }
}

OUT_FILE = ROOT / "cache" / "algorithm_solved_parameters.json"
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(out_params, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 110)
print(f"【参数持久化成功】算法自主迭代生成的参数已安全写入磁盘: {OUT_FILE}")
print("=" * 110)
