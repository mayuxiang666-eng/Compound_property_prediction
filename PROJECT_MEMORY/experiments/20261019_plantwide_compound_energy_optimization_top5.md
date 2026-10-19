# 全厂炭黑胶料 Top 5 与白炭黑胶料 Top 5 能耗潜力排行及分步工艺优化（项目记忆）

- **日期**：2026-10-19
- **作者/执行人**：AI 混炼工艺工程助手 & 工艺工程团队
- **项目阶段**：全厂混炼能耗潜力诊断、分步方差归因及批处理模块落地
- **状态**：已完成（已交付自动化引擎 `compound_energy_optimizer.py`、全厂数据表与高清可视化看板）

---

## 1. 业务背景与核心诉求

1. **痛点**：全厂混炼机台能耗存在批次间（Batch-to-Batch）显著波动。单纯看单吨比能耗绝对值高低不足以指导节能——单吨耗电高的胶料若各车次波动极小，可能已接近工艺极限；而**车次能耗差距悬殊（$P_{85} - P_{15}$ 巨大）且产量规模大的胶料，直接证明当前工艺远非最优解，存在巨大的可回收节能空间**。
2. **核心目标**：
   - 结合 BOM 原料分类与全厂 20,985 车真实一混（M1）生产数据，评估出**纯炭黑胶料 Top 5**（显式包含指定胶料 `M1-B00458` 与 `M1-B00163`）与**白炭黑胶料 Top 5**；
   - 提取 47.1 万行分步遥测（Step Telemetry），**明确标注调整第几步、调多少 RPM/秒的计算依据**；
   - 构建高内聚、可复用的自动化 Python 模块 `compound_energy_optimizer.py`，支持未来对任意胶料一键诊断。

---

## 2. 四维潜力评价模型与计算公式

### 2.1 评价指标定义
1. **填料分类（BOM Classification）**：
   关联 `MooneyIntegratedData.core.BatchMaterial`，主填料为炭黑且无/极低硅烷者归为 Carbon Black（`M1-B*`、`M1-R*`、`M1-A*`）；高白炭黑+硅烷偶联剂者归为 Silica（`M1-T*`）。
2. **车次悬殊度（Energy Spread）**：
   $$\Delta e_{85-15} = P_{85} - P_{15} \quad (\text{kWh/t})$$
   衡量同配方、同机台内部高能耗批次与历史前 15% 黄金批次的离散差距。
3. **可回收单吨空间（Recoverable Potential）**：
   $$\Delta e_{\text{rec}} = P_{50} - P_{15} \quad (\text{kWh/t})$$
   代表将中位车次拉升至黄金车次水平的节能空间。
4. **年化节电潜能（Annualized Energy Saving Potential）**：
   $$\text{Annual Saving Potential (kWh/yr)} = \Delta e_{\text{rec}} \times \left(W_{\text{20d}} \times \frac{350}{20}\right)$$

---

## 3. 全厂 Top 5 榜单成果

### 3.1 炭黑胶料 Top 5（Carbon Black）
- **CB #1: `M1-B15563R1-- 01 006`**（机台 `92-1070`，1,370车/20天，年化 6,249吨）：比能耗均值 $118.88\text{ kWh/t}$，$P_{85}-P_{15} = 11.24\text{ kWh/t}$，$P_{50}-P_{15} = 5.75\text{ kWh/t}$，年化潜能 **35,936 kWh/年**（全族 **48,192 kWh/年**）。
- **CB #2: `M1-B00460B--- 04 010`**（机台 `92-1100`，1,473车/20天，年化 10,500吨，全厂最大吨位）：比能耗均值 $115.71\text{ kWh/t}$，$P_{85}-P_{15} = 6.36\text{ kWh/t}$，$P_{50}-P_{15} = 2.95\text{ kWh/t}$，年化潜能 **31,015 kWh/年**。
- **CB #3: `M1-B00163---- 08 010`**（机台 `92-1020`，954车/20天，年化 4,461吨，重点指定）：比能耗均值 $109.05\text{ kWh/t}$，$P_{85}-P_{15} = 9.24\text{ kWh/t}$，$P_{50}-P_{15} = 4.24\text{ kWh/t}$，年化潜能 **18,929 kWh/年**。
- **CB #4: `M1-R00218W2-- 10 011`**（机台 `92-1050`，461车/20天，年化 2,138吨，悬殊度最高）：比能耗均值 $98.23\text{ kWh/t}$，$P_{85}-P_{15} = 14.63\text{ kWh/t}$，$P_{50}-P_{15} = 7.19\text{ kWh/t}$，年化潜能 **15,375 kWh/年**（全族 **40,398 kWh/年**）。
- **CB #5: `M1-B00458---- 12 012`**（机台 `92-1020`，601车/20天，年化 2,783吨，重点指定）：比能耗均值 $113.36\text{ kWh/t}$，$P_{85}-P_{15} = 7.97\text{ kWh/t}$，$P_{50}-P_{15} = 3.91\text{ kWh/t}$，年化潜能 **10,882 kWh/年**（全族 **14,803 kWh/年**）。

### 3.2 白炭黑胶料 Top 5（Silica）
- **Silica #1: `M1-T15760---- 06 002`**（机台 `92-1090`，1,754车/20天，年化 8,468吨）：比能耗均值 $129.25\text{ kWh/t}$，$P_{85}-P_{15} = 10.35\text{ kWh/t}$，$P_{50}-P_{15} = 5.05\text{ kWh/t}$，年化潜能 **42,802 kWh/年**。
- **Silica #2: `M1-T20153W--- 01 001`**（机台 `92-1090`，1,030车/20天，年化 4,829吨）：比能耗均值 $145.57\text{ kWh/t}$，$P_{85}-P_{15} = 14.70\text{ kWh/t}$，$P_{50}-P_{15} = 7.14\text{ kWh/t}$，年化潜能 **34,496 kWh/年**。
- **Silica #3: `M1-T09170---- 23 006`**（机台 `92-1030`，867车/20天，年化 3,964吨）：比能耗均值 $156.40\text{ kWh/t}$，$P_{85}-P_{15} = 12.87\text{ kWh/t}$，$P_{50}-P_{15} = 6.82\text{ kWh/t}$，年化潜能 **27,049 kWh/年**。
- **Silica #4: `M1-T33025W3-- 03 005`**（机台 `92-1070`，420车/20天，年化 1,994吨，全厂悬殊度极值）：比能耗均值 $142.83\text{ kWh/t}$，$P_{85}-P_{15} = 16.77\text{ kWh/t}$，$P_{50}-P_{15} = 9.09\text{ kWh/t}$，年化潜能 **18,135 kWh/年**。
- **Silica #5: `M1-T17771---- 05 008`**（机台 `92-1030`，589车/20天，年化 2,731吨）：比能耗均值 $145.10\text{ kWh/t}$，$P_{85}-P_{15} = 11.36\text{ kWh/t}$，$P_{50}-P_{15} = 5.59\text{ kWh/t}$，年化潜能 **15,277 kWh/年**（全族 **26,352 kWh/年**）。

---

## 4. 调整步号与参数幅度的计算依据

### 4.1 调整第几步的计算依据（方差贡献度 $W_i$ 准则）
$$W_i = \frac{\bar{E}_{i, P_{85}} - \bar{E}_{i, P_{15}}}{\sum_k \max(0, \bar{E}_{k, P_{85}} - \bar{E}_{k, P_{15}})} \times 100\%$$
- 严禁调整 Step 1 ~ Step 4（投料、吃粉包覆初期）；
- 锁定 Step $\ge 5$ 中平均功率最高、方差贡献最大（$W_i \ge 13\% \sim 52\%$）的高速分散升温步。

### 4.2 MST 降多少 RPM 的计算依据（流变功率缩放定律）
$$\frac{\Delta P_i}{P_i} \approx 1.10 \times \left(\frac{\Delta N_i}{N_0}\right), \qquad \frac{\Delta t_i}{t_i} \approx -0.35 \times \left(\frac{\Delta N_i}{N_0}\right)$$
- 基速 $\ge 35\text{ RPM}$ 时下调 $-5\text{ RPM}$（相对降幅 $-10\% \sim -14\%$，功率降低 $-76 \sim -220\text{ kW}$，单步时间仅延长 $+1.0 \sim 2.0\text{ s}$，单车净节电 $-1.32 \sim -2.27\text{ kWh/t}$）；
- 基速处于中速 $25\text{ RPM}$（如 `T20153W` Step 10）时等比例下调 $-3\text{ RPM}$（相对降幅 $-12\%$）。

### 4.3 MSB 减多少秒的计算依据（边际功率积分速率）
$$e_{\text{sec}} = \frac{\bar{P}_{\text{MSB}}}{3600}\text{ (kWh/s)}, \qquad e_{\text{ton,sec}} = \frac{\bar{P}_{\text{MSB}}}{3.6 \times W_{\text{batch}}}\text{ (kWh/t/s)}$$
$$\Delta t_{\text{MSB}} = -\text{round}\left(\frac{\Delta E_{\text{target}}}{e_{\text{ton,sec}}}\right)$$
- 若 `MSB Step 1` $\ge 30\text{ s}$：直接缩短 $-8\text{ s} \sim -10\text{ s}$；
- 若 `MSB Step 1` 仅 $20\text{ s}$ 而 `MSB Step 2` 长达 $30\sim 65\text{ s}$ 且方差占比高：采用“`MSB Step 1` 减 4~5s + `MSB Step 2` 减 5s”的分摊缩时方案。

### 4.4 白炭黑专属总体 PID 时长（上密炼 MST + 下密炼 MSB）Hard Rule 刚性门禁（2026-10-19 12:57 更新）
$$t_{\text{PID,total}} = t_{\text{PID,MST}} + t_{\text{PID,MSB}} \in \left[t_{\min}(T_{\text{target}}),\; t_{\max}(T_{\text{target}})\right] \quad (\text{仅对白炭黑 Silica 生效，炭黑豁免})$$
- **Target 165°C**：**`80 – 140 s`**
- **Target 155°C**：**`130 – 230 s`**
- **Target 142°C**：**`270 – 330 s`**
- **合规审计与节电结论**：
  1. **白炭黑 Top 5（全部 Target 165°C，硬门禁 `80–140 s`）**：调整后总体 PID 时长分别为 `T15760` (`100.9s`)、`T20153W` (`113.5s`)、`T09170` (`111.7s`)、`T33025W3` (`109.6s`)、`T17771` (`123.1s`)，**100% 满足 `[80, 140] s`**，且将 `T17771` P85 批次从超标的 `141.3s` 拉回安全区，单吨节电 **`-5.00 ~ -6.38 kWh/t`**；
  2. **炭黑 Top 5（豁免白炭黑 PID 硬规则）**：全额执行 `MST -5 RPM + MSB -9s/-10s` 双联优化，预期节电分别为 `B15563R1` (`-4.29 kWh/t`)、`B00460B` (`-5.36 kWh/t`)、`B00163` (`-5.83 kWh/t`)、`R00218W2` (`-3.53 kWh/t`)、`B00458` (`-5.38 kWh/t`)。

---

## 5. 交付产物与存储路径

1. **执行主程序**：`compound_energy_optimizer.py`
2. **详细工程报告**：`reports/compound_energy_optimization_auto/Top5_CarbonBlack_and_Silica_Energy_Optimization_Report.md` 与 `docs/Top5_CarbonBlack_and_Silica_Energy_Optimization_Report.md`
3. **诊断结果数据表**：
   - `reports/compound_energy_optimization_auto/plant_wide_energy_potential_ranking.csv`
   - `reports/compound_energy_optimization_auto/top_compounds_step_diagnostics_summary.csv`（含 8 列 PID Hard Rule 审计字段）
4. **高分辨率看板图表**：
   - `reports/compound_energy_optimization_auto/top5_cb_and_silica_energy_potential_dashboard.png`
   - `reports/compound_energy_optimization_auto/b00458_b00163_and_top_compounds_step_diagnosis.png`
5. **原始数据与抓取脚本**：
   - `scratch/top_compounds_steps_raw.csv`（47.1 万行分步遥测）
   - `scratch/compound_bom_classification.csv`
   - `scratch/all_compounds_energy_ranking.csv`
   - `scratch/fetch_top_compounds_step_telemetry.py`
   - `scratch/rank_compounds_potential.py`
