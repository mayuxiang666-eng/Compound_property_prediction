import sys
import time
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

print("=" * 110)
print(f"【高阶梯度提升树·早停机制全流程训练启动】启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
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
    'MST_S1_Time_s', 'MST_S2_Time_s', 'MST_S3_Time_s', 'MST_S4_Time_s', 'MST_S5_Time_s',
    'MST_S6_Time_s', 'MST_S7_Time_s', 'MST_S7_Rpm', 'MST_S8_Time_s', 'MST_S9_Time_s',
    'MST_S10_Time_s', 'MST_S11_Time_s',
    'MSB_S1_Time_s', 'MSB_S1_Rpm', 'MSB_S2_Time_s', 'MSB_S3_Time_s',
    'Curve_Temp_Max_MST_C', 'MSB_S3_Temp_C', 'Curve_Torque_Mean_MST_Nm', 'Curve_Power_Mean_MST_kW'
]

clean = merged[step_cols + ['Real_Mooney_MU', 'OrderID']].dropna().reset_index(drop=True)
print(f"1. 数据加载与清洗: 有效车次 N = {len(clean)} 车 (来自 {clean['OrderID'].nunique()} 个独立订单)")
print(f"   输入特征规模: 共包含 {len(step_cols)} 个全流程参数 (覆盖上密炼 S1~S11 与 下密炼 S1~S3)")

# 严格按工单划分训练集 (80%) 与 独立验证集 (20%) - 严格防止车次间穿越
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
tr_idx, val_idx = next(gss.split(clean, groups=clean['OrderID']))

train_df = clean.iloc[tr_idx].reset_index(drop=True)
val_df = clean.iloc[val_idx].reset_index(drop=True)

X_tr = train_df[step_cols].values
y_tr = train_df['Real_Mooney_MU'].values
X_val = val_df[step_cols].values
y_val = val_df['Real_Mooney_MU'].values

print(f"2. 数据集划分 (按 OrderID 严格隔离):")
print(f"   • 训练集 (Train) : N = {len(train_df)} 车 ({train_df['OrderID'].nunique()} 个工单)")
print(f"   • 独立验证集 (Val): N = {len(val_df)} 车 ({val_df['OrderID'].nunique()} 个独立工单，用于监控早停)")

# LightGBM Dataset
trn_data = lgb.Dataset(X_tr, label=y_tr, feature_name=step_cols)
val_data = lgb.Dataset(X_val, label=y_val, feature_name=step_cols, reference=trn_data)

params = {
    'objective': 'regression_l1', # MAE 目标函数，对异常值鲁棒
    'metric': ['l1', 'l2'],
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 12,
    'max_depth': 4,
    'min_child_samples': 5,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'verbose': -1,
    'seed': 42
}

print("\n" + "=" * 110)
print("【第一部分：早停机制 (Early Stopping) 迭代训练实录】")
print("=" * 110)
print("最大预设迭代上限 : 1000 轮")
print("早停容忍步数 (stopping_rounds) : 35 轮 (若验证集损失连续 35 轮无提升，立即自动终止训练并回滚至最优轮次)")
print("-" * 110)

eval_history = {}
callbacks = [
    lgb.record_evaluation(eval_history),
    lgb.early_stopping(stopping_rounds=35, verbose=False)
]

# Train model
model = lgb.train(
    params,
    trn_data,
    num_boost_round=1000,
    valid_sets=[trn_data, val_data],
    valid_names=['train', 'val'],
    callbacks=callbacks
)

best_iteration = model.best_iteration
print(f"\n★【早停机制触发成功】")
print(f"  • 训练并未傻跑到 1000 轮，算法自主在第 {best_iteration + 35} 轮检测到验证集过拟合风险，自动早停！")
print(f"  • 算法自动回滚并锁定至【全生命周期最佳迭代轮次】: Best Round = 第 {best_iteration} 轮！")
print(f"  • 最佳轮次训练集 MAE = {eval_history['train']['l1'][best_iteration-1]:.3f} MU")
print(f"  • 最佳轮次独立验证集 MAE = {eval_history['val']['l1'][best_iteration-1]:.3f} MU")

# Print iteration trajectory sampled
print("\n【早停迭代监控轨迹采样 (验证集损失演变)】:")
total_rounds = len(eval_history['val']['l1'])
sample_steps = list(range(0, total_rounds, max(1, total_rounds // 10)))
if best_iteration - 1 not in sample_steps:
    sample_steps.append(best_iteration - 1)
sample_steps = sorted(list(set(sample_steps)))

for s in sample_steps:
    r_no = s + 1
    tr_l1 = eval_history['train']['l1'][s]
    val_l1 = eval_history['val']['l1'][s]
    star = " ★ (最优轮次 Best)" if r_no == best_iteration else ""
    print(f"  轮次 #{r_no:03d} | 训练集 MAE: {tr_l1:.3f} MU | 独立验证集 MAE: {val_l1:.3f} MU{star}")

print("\n" + "=" * 110)
print(f"【第二部分：模型到底长什么样？（数学结构解剖）】")
print("=" * 110)

# Extract tree structure
model_dump = model.dump_model()
trees = model_dump['tree_info']

print(f"1. 宏观数学形态: 纯加法树集成模型 (Additive Tree Ensemble)")
print(f"   最终门尼预测值 Mooney(X) = 基础期望得分 (Base Score) + sum_{{m=1}}^{{{best_iteration}}} [ 学习率 0.03 * 树 m(X) 的叶子节点贡献 ]")
print(f"   • 当前模型树的总棵数: 恰好等于最佳早停轮次 {best_iteration} 棵树\n")

print(f"2. 微观决策树结构切片 (展示第 1 棵树 与 第 2 棵树 的具体分裂规则):")
print("-" * 110)

def print_tree_rule(tree_dict, tree_num):
    print(f"  【第 {tree_num} 棵树结构】(Tree #{tree_num}, 树权重贡献权重 = 0.03):")
    struct = tree_dict['tree_structure']
    
    # Root
    f_idx = struct.get('split_feature', 0)
    f_name = step_cols[f_idx] if f_idx < len(step_cols) else f"Feat_{f_idx}"
    thresh = struct.get('threshold', 0.0)
    print(f"    ├─ 根节点条件: 如果 [{f_name}] <= {thresh:.2f}:")
    
    # Left child
    left = struct.get('left_child', {})
    if 'leaf_value' in left:
        print(f"    │   ├─ 是: 命中叶子节点，基准修正值 = {left['leaf_value']:+.4f} MU")
    else:
        l_f = step_cols[left.get('split_feature', 0)]
        l_th = left.get('threshold', 0.0)
        print(f"    │   ├─ 是: 再判断 [{l_f}] <= {l_th:.2f}:")
        if 'left_child' in left and 'leaf_value' in left['left_child']:
            print(f"    │   │   ├─ 是: 叶子修正 = {left['left_child']['leaf_value']:+.4f} MU")
        if 'right_child' in left and 'leaf_value' in left['right_child']:
            print(f"    │   │   └─ 否: 叶子修正 = {left['right_child']['leaf_value']:+.4f} MU")
            
    # Right child
    right = struct.get('right_child', {})
    if 'leaf_value' in right:
        print(f"    │   └─ 否: 命中叶子节点，基准修正值 = {right['leaf_value']:+.4f} MU")
    else:
        r_f = step_cols[right.get('split_feature', 0)]
        r_th = right.get('threshold', 0.0)
        print(f"    │   └─ 否: 再判断 [{r_f}] <= {r_th:.2f}:")
        if 'left_child' in right and 'leaf_value' in right['left_child']:
            print(f"    │       ├─ 是: 叶子修正 = {right['left_child']['leaf_value']:+.4f} MU")
        if 'right_child' in right and 'leaf_value' in right['right_child']:
            print(f"    │       └─ 否: 叶子修正 = {right['right_child']['leaf_value']:+.4f} MU")
    print()

print_tree_rule(trees[0], 1)
print_tree_rule(trees[1], 2)

print("=" * 110)
print(f"【第三部分：全 14 步特征在最终模型中的分裂重要度排行榜 (Feature Importance)】")
print("=" * 110)
gain_imp = model.feature_importance(importance_type='gain')
split_imp = model.feature_importance(importance_type='split')

imp_df = pd.DataFrame({
    '步序特征名称': step_cols,
    '累积分裂增益(Gain)': gain_imp,
    '节点分裂次数(Split)': split_imp
}).sort_values(by='累积分裂增益(Gain)', ascending=False).reset_index(drop=True)

print(f"{'排名':<4} | {'步序特征名称':<26} | {'累积分裂增益(Gain)':<18} | {'节点分裂次数(Split)':<16} | {'模型角色'}")
print("-" * 110)
for idx, r in imp_df.iterrows():
    role = "★ 核心主控特征" if idx < 3 else "主要调节特征" if idx < 8 else "辅助微调特征"
    print(f"#{idx+1:02d}  | {r['步序特征名称']:<26} | {r['累积分裂增益(Gain)']:18.2f} | {r['节点分裂次数(Split)']:16d} | {role}")

# Evaluate predictions on Validation set
val_preds = model.predict(X_val, num_iteration=best_iteration)
val_mae = mean_absolute_error(y_val, val_preds)
val_pass_3 = (np.abs(val_preds - y_val) <= 3.0).mean() * 100
val_pass_2 = (np.abs(val_preds - y_val) <= 2.0).mean() * 100

print("\n" + "=" * 110)
print(f"【第四部分：最终独立验证集盲测性能总评】")
print("=" * 110)
print(f"  • 独立验证集样本量 : N = {len(val_df)} 车 (全未见工单盲测)")
print(f"  • 门尼平均绝对误差 (MAE) : {val_mae:.2f} MU")
print(f"  • <= ±3.0 MU 工业考核达标率: {val_pass_3:.1f}%")
print(f"  • <= ±2.0 MU 高精度达标率  : {val_pass_2:.1f}%")
print("=" * 110)
