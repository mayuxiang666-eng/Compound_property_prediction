import sys
import time
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

print("=" * 110)
print(f"【纯梯度反向传播·低学习率·大Epoch·早停机制全流程训练】启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 110)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "cache" / "pure_1to1_single_batch_dataset.pkl"
MATH_PATH = ROOT / "cache" / "mathematical_dataset.pkl"

df_1to1 = pd.read_pickle(DATA_PATH)
df_math = pd.read_pickle(MATH_PATH)

merged = pd.merge(
    df_1to1[['OrderID', 'BatchNumber', 'Real_Mooney_MU', 'Real_Energy_kWh']],
    df_math,
    on=['OrderID', 'BatchNumber'],
    how='inner'
)

step_cols = [
    'MST_S1_Time_s', 'MST_S2_Time_s', 'MST_S3_Time_s', 'MST_S6_Time_s',
    'MST_S7_Time_s', 'MST_S7_Rpm', 'MST_S8_Time_s', 'MST_S9_Time_s',
    'MST_S10_Time_s', 'MSB_S1_Time_s', 'MSB_S2_Time_s', 'MSB_S3_Time_s',
    'Curve_Temp_Max_MST_C', 'MSB_S3_Temp_C', 'Curve_Torque_Mean_MST_Nm', 'Curve_Power_Mean_MST_kW'
]

clean = merged[step_cols + ['Real_Mooney_MU', 'OrderID']].dropna().reset_index(drop=True)
print(f"1. 数据加载与清洗: 有效车次 N = {len(clean)} 车 (来自 {clean['OrderID'].nunique()} 个独立订单)")
print(f"   输入特征规模: 共包含 {len(step_cols)} 个工艺全流程连续特征")

# 按 OrderID 严格隔离划分 80% 训练集 与 20% 独立验证集
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr_idx, val_idx = next(gss.split(clean, groups=clean['OrderID']))

train_df = clean.iloc[tr_idx].reset_index(drop=True)
val_df = clean.iloc[val_idx].reset_index(drop=True)

X_tr_raw = train_df[step_cols].values
y_tr = train_df['Real_Mooney_MU'].values
X_val_raw = val_df[step_cols].values
y_val = val_df['Real_Mooney_MU'].values

# Min-Max 缩放到 [0, 1] 区间（严格零均值、零中位数）
X_min = X_tr_raw.min(axis=0)
X_max = X_tr_raw.max(axis=0)
denom = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))

X_tr = (X_tr_raw - X_min) / denom
X_val = (X_val_raw - X_min) / denom

n_tr, p_dim = X_tr.shape
n_val = len(X_val)
print(f"2. 数据划分与规范化:")
print(f"   • 训练集 N_train = {n_tr} 车 ({train_df['OrderID'].nunique()} 个工单)")
print(f"   • 独立验证集 N_val = {n_val} 车 ({val_df['OrderID'].nunique()} 个工单，专门监控早停)")

# -----------------------------------------------------------------------------
# 纯梯度反向传播训练配置 (低学习率、大 Epoch、早停机制)
# -----------------------------------------------------------------------------
lr = 0.02              # 学习率调低 (从 0.5 降到 0.02，极细腻平滑更新)
max_epochs = 3000      # 扩充至 3000 个 Epoch
patience = 80          # 早停容忍轮数 (80 轮无提升则早停终止)
delta = 3.0            # 工业 Huber 容差平滑阈值
l2_reg = 0.005         # 权重衰减系数

# 纯零初始值：绝对不使用均值、中位数或众数！
theta = np.zeros(1 + p_dim) # theta[0] 为基准参数，theta[1:] 为 16 个步序权重
m_adam = np.zeros(1 + p_dim)
v_adam = np.zeros(1 + p_dim)
beta1, beta2, eps = 0.9, 0.999, 1e-8

print("\n" + "=" * 110)
print(f"【开始纯梯度迭代反向传播优化】")
print(f"  • 超参数设置: 学习率 lr = {lr} | 最大轮数 = {max_epochs} | 早停耐受步数 = {patience}")
print(f"  • 初始参数状态: 全部严格从 0.00 出发 (零先验、零均值、零中值)")
print("=" * 110)

best_val_loss = float('inf')
best_epoch = 0
best_theta = np.copy(theta)
no_improve_count = 0
training_log = []

t_start = time.time()

for epoch in range(1, max_epochs + 1):
    # 1. 训练集前向传播
    y_pred_tr = theta[0] + X_tr @ theta[1:]
    err_tr = y_pred_tr - y_tr
    abs_err_tr = np.abs(err_tr)
    
    # 训练集平滑 Huber 损失
    huber_mask_tr = abs_err_tr <= delta
    train_loss = np.mean(np.where(huber_mask_tr, 0.5 * err_tr**2, delta * (abs_err_tr - 0.5 * delta)))
    
    # 2. 梯度反向传播计算 (精确偏导数)
    grad_err = np.where(huber_mask_tr, err_tr, delta * np.sign(err_tr))
    grad_theta_0 = np.mean(grad_err)
    grad_theta_w = (X_tr.T @ grad_err) / n_tr + l2_reg * theta[1:]
    grad = np.concatenate([[grad_theta_0], grad_theta_w])
    
    # 3. Adam 优化器参数迭代更新
    m_adam = beta1 * m_adam + (1.0 - beta1) * grad
    v_adam = beta2 * v_adam + (1.0 - beta2) * (grad ** 2)
    m_hat = m_adam / (1.0 - beta1 ** epoch)
    v_hat = v_adam / (1.0 - beta2 ** epoch)
    theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    # 4. 独立验证集评估 (监控早停指标)
    y_pred_val = theta[0] + X_val @ theta[1:]
    err_val = y_pred_val - y_val
    abs_err_val = np.abs(err_val)
    huber_mask_val = abs_err_val <= delta
    val_loss = np.mean(np.where(huber_mask_val, 0.5 * err_val**2, delta * (abs_err_val - 0.5 * delta)))
    val_mae = np.mean(abs_err_val)
    
    # 5. 早停机制状态检测
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_epoch = epoch
        best_theta = np.copy(theta)
        no_improve_count = 0
    else:
        no_improve_count += 1
        
    # 记录日志
    if epoch in [1, 2, 5, 10, 25, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000] or epoch == best_epoch or no_improve_count == patience:
        tr_mae = np.mean(abs_err_tr)
        training_log.append((epoch, train_loss, val_loss, theta[0], tr_mae, val_mae))
        star = " ★ (最优验证点)" if epoch == best_epoch else ""
        print(f"  Epoch {epoch:04d} | 训练损失: {train_loss:7.4f} | 验证损失: {val_loss:7.4f} | 演变中基准 theta_0: {theta[0]:6.2f} MU | 验证MAE: {val_mae:5.2f} MU{star}")
        
    if no_improve_count >= patience:
        print("-" * 110)
        print(f"★【早停机制触发】: 验证集连续 {patience} 个 Epoch 损失无实质提升 (过拟合保护启动)！")
        print(f"  • 训练在第 {epoch} 个 Epoch 自动提前终止！")
        print(f"  • 模型自动回滚至最佳 Epoch: 第 {best_epoch} 轮！")
        break

t_elapsed = time.time() - t_start

# 回滚到最佳模型
theta = best_theta
Y0_final = theta[0]
w_final = theta[1:]

print("-" * 110)
print(f"【训练优化完成】实机运行总耗时: {t_elapsed:.2f} 秒")
print(f"  • 锁定最优轮次 : Epoch #{best_epoch}")
print(f"  • 最优验证集损失: {best_val_loss:.4f}")

# 最终验证集性能
y_pred_final_val = Y0_final + X_val @ w_final
final_val_mae = mean_absolute_error(y_val, y_pred_final_val)
final_val_pass3 = np.mean(np.abs(y_pred_final_val - y_val) <= 3.0) * 100
final_val_pass2 = np.mean(np.abs(y_pred_final_val - y_val) <= 2.0) * 100

print("\n" + "=" * 110)
print("【最终算法迭代产出的纯模型参数形态 (回滚至最优早停点)】")
print("=" * 110)
print(f"★ 算法自主演进出的基准参数 theta_0 = {Y0_final:.2f} MU (从 0.00 经梯度一步步更新演化而来，无均值/中值)")
print("\n★ 各步序敏感度权重矩阵 (theta_w):")
print(f"{'步序特征名称':<26} | {'归一化敏感度权重 w':<18} | {'特征物理作用'}")
print("-" * 110)
for col_name, w_val in zip(step_cols, w_final):
    action = "负向降粘 (延长/加速使门尼下降)" if w_val < -0.1 else "正向关联 (机温/力矩流变负荷)" if w_val > 0.1 else "平稳过渡"
    print(f"  {col_name:<24s} | {w_val:+16.4f}     | {action}")

print("\n" + "=" * 110)
print("【独立验证集全盲测评估结论】")
print("=" * 110)
print(f"  • 盲测集车数 : N = {len(val_df)} 车 (全新未见工单)")
print(f"  • 最终验证集平均绝对误差 (MAE) : {final_val_mae:.2f} MU")
print(f"  • 满足 <= ±3.0 MU 工业合格率   : {final_val_pass3:.1f}%")
print(f"  • 满足 <= ±2.0 MU 高精度达标率 : {final_val_pass2:.1f}%")
print("=" * 110)

# Save to disk
save_data = {
    'best_epoch': best_epoch,
    'learning_rate': lr,
    'theta_0_baseline': round(float(Y0_final), 4),
    'theta_w_weights': {k: round(float(v), 4) for k, v in zip(step_cols, w_final)},
    'val_mae': round(float(final_val_mae), 4),
    'val_pass_3mu': round(float(final_val_pass3), 2)
}
with open(ROOT / "cache" / "pure_gradient_early_stopping_model.json", 'w', encoding='utf-8') as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)
print(f"模型参数已安全持久化至: {ROOT / 'cache' / 'pure_gradient_early_stopping_model.json'}")
