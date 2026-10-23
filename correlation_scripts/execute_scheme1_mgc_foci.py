import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from hyppo.independence import MGC
import xicorpy
import time

print("="*80)
print("=== 严格执行方案1：相位窗MGC独立性初筛 + FOCI多元条件相关筛选 ===")
print("="*80)

# 1. 加载标准对齐输入数据
npz_path = r"d:\AntigravityIDE\Project\data\scheme1_aligned_input.npz"
data = np.load(npz_path)

X_temp = data['temperature']     # (208, 390)
X_power = data['power']         # (208, 390)
X_ram = data['ram']             # (208, 390)
X_speed = data['speed']         # (208, 390)
X_aging = data['aging_hours']   # (208, 1)
y = data['mooney']              # (208,)
y_col = y.reshape(-1, 1)        # (208, 1)

n, T = X_temp.shape
print(f"数据加载就绪: 样本数 n = {n}, 曲线对齐长度 T = {T} 秒")

# 2. 切窗 (10 段等长窗，每窗 39 秒)
channels = {
    '温度': X_temp,
    '功率': X_power,
    '压栓': X_ram,
    '转速': X_speed
}

windows = {}
num_windows = 10
win_len = T // num_windows

for ch_name, ch_arr in channels.items():
    for w_idx in range(num_windows):
        start_col = w_idx * win_len
        end_col = (w_idx + 1) * win_len if w_idx < num_windows - 1 else T
        w_name = f"{ch_name}_w{w_idx:02d}"
        windows[w_name] = ch_arr[:, start_col:end_col]

# 停留时间作为独立窗
windows['停留时间'] = X_aging
print(f"切窗完成: 共计 {len(windows)} 个候选相位窗 (每条曲线通道 10 窗，每窗 {win_len} 秒，加停留时间 1 窗)")

# 3. MGC 单窗独立性检验初筛 (pvalue < 0.05 保留)
print("\n" + "-"*80)
print(">>> 步骤 1: 运行 MGC 单窗独立性置换检验 (reps=1000, workers=-1, random_state=0) ...")
print("-"*80)

mgc_results = {}
retained_windows = []

t0 = time.time()
for idx, (w_name, w_x) in enumerate(windows.items(), 1):
    t_w0 = time.time()
    # 距离用默认欧氏距离
    stat, pval, _ = MGC().test(w_x, y_col, reps=1000, workers=-1, random_state=0)
    mgc_results[w_name] = {'stat': float(stat), 'pvalue': float(pval)}
    is_retained = pval < 0.05
    status_str = "【保留进入FOCI】" if is_retained else "【淘汰】"
    if is_retained:
        retained_windows.append(w_name)
    print(f"[{idx:02d}/{len(windows)}] 窗: {w_name:<10} | MGC统计量: {stat:+.4f} | p值: {pval:.4f} --> {status_str} (耗时: {time.time()-t_w0:.1f}s)")

print(f"\nMGC初筛全部完成！总耗时: {time.time()-t0:.1f}秒")
print(f"初筛通过 (p < 0.05) 的保留窗数量: {len(retained_windows)} / {len(windows)}")
print("保留名单:", retained_windows)

# 4. FOCI 多元条件依赖系数迭代筛选
print("\n" + "-"*80)
print(">>> 步骤 2: 在保留窗上执行 FOCI 多元条件依赖迭代贪心搜索 ...")
print("-"*80)

selected_windows = []
selected_records = []
candidates = list(retained_windows)

round_idx = 1
while candidates:
    best_cand = None
    best_coeff = -float('inf')
    
    # 准备当前已选集合构成的 given 矩阵
    if len(selected_windows) == 0:
        given = None
    else:
        given = np.hstack([windows[s] for s in selected_windows])
        
    print(f"\n--- FOCI 第 {round_idx} 轮选择 (当前已选窗数: {len(selected_windows)}, 候选窗数: {len(candidates)}) ---")
    
    for cand in candidates:
        z = windows[cand] # 当前候选窗全部列当作多元变量，不压成均值
        # 计算给定 given 条件下，门尼 y 与当前窗 z 的条件依赖系数
        c_dep = xicorpy.compute_conditional_dependence(y, z, given)
        c_dep = float(c_dep)
        print(f"  候选窗: {cand:<10} | 形状: {z.shape} | 当前条件依赖系数: {c_dep:+.4f}")
        if c_dep > best_coeff:
            best_coeff = c_dep
            best_cand = cand
            
    print(f"--> 本轮最优候选: 【{best_cand}】，最大条件依赖系数: {best_coeff:+.4f}")
    
    # 停机准则：条件依赖系数不大于 0 就停止
    if best_coeff <= 0:
        print(f"--> [停机触发] 最优条件依赖系数 {best_coeff:+.4f} <= 0，停止筛选，其余候选窗全部丢弃！")
        break
        
    selected_windows.append(best_cand)
    candidates.remove(best_cand)
    
    selected_records.append({
        '窗名': best_cand,
        '选入顺序': round_idx,
        '选入时的条件依赖系数': best_coeff,
        'MGC统计量': mgc_results[best_cand]['stat'],
        'MGC_p值': mgc_results[best_cand]['pvalue']
    })
    
    round_idx += 1

# 5. 输出方案1标准结果表
print("\n" + "="*80)
print("=== 方案 1 最终输出结果表 (严格遵循文档规范) ===")
print("="*80)

if selected_records:
    df_out = pd.DataFrame(selected_records)
    print(df_out.to_string(index=False))
else:
    print("未有窗满足条件依赖系数 > 0 的要求。")
