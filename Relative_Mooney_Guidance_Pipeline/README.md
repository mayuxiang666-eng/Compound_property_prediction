# 相对门尼引导与逆向推荐流水线 (Relative Mooney Guidance Pipeline)

## 📌 项目定位
本模块旨在解决“单车绝对门尼回归难度大、噪声高、波动预测不足”的工程难题。
核心思路转型为：**无需精确预测绝对门尼，用户只需给出相对降幅需求，系统通过剥离原材料偏置、投影黄金工艺走廊、并基于因果敏感度逆解出 Step 级调整方案。**

## 🏗️ 架构与子模块
```
Relative_Mooney_Guidance_Pipeline/
├── contracts/                  # 相对引导请求、黄金包络与安全约束合同
├── data_extraction/            # 原材料 COA / 静态配方去混杂与批次数据提取
├── method_a_golden_corridor/   # 方案 A: 历史黄金流形投影法
├── method_b_relative_delta/    # 方案 B: 自监督过程表示与差分模型
├── method_c_causal_inverse/    # 方案 C: 因果敏感度逆向二次规划
├── validation/                 # 严格分步门禁验证器 (Gate 1 - Gate 5)
└── run_relative_guidance_experiment.py # 端到端回测主程序
```

## 🔒 核心分步门禁要求
1. **Gate 1**：原材料去混杂检验（剥离天然生胶批次门尼漂移）。
2. **Gate 2**：方案 A 黄金包络内插安全率 100%（杜绝 Uniform Shift 盲目平移）。
3. **Gate 3**：方案 C 物理敏感度单调性与防焦烧保护（门尼单调下降且排胶温度 $\le 168^\circ\text{C}$）。
4. **Gate 4**：方案 B 同订单波动捕捉度（预测 $\Delta\text{Std} / \text{实际 }\Delta\text{Std} \ge 0.70$）。
5. **Gate 5**：全流程零泄漏与生产安全审查。
