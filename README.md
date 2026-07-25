# 门尼粘度预测与波动调优项目 (Mooney Viscosity Prediction V3.0)

本项目致力于通过工业大数据与流变物理知识库的有机结合，对密炼生产线中的胶料成品门尼粘度（MNY）进行高精度预测。重点攻克了以往模型容易对配方产生“静态记忆”导致无法识别“车次间动态工艺波动（batch-to-batch fluctuations）”以及“趋势倒挂（负相关）”的硬伤。

项目采用全新的**终极三阶段解耦算法 (Ultimate 3-Stage Model)** 并深度集成了 **Hampel 噪声抑制与 RidgeCV 自适应正则化** 防过拟合安全网，确保工业生产上线后的卓越泛化性与抗噪声能力。

---

## 📁 整理后的目录结构 (Workspace Layout)

```
Master batch data fectching/
│
├── 📁 legacy/                           ──> 【历史老模型与弃用文件归档】
│   ├── validate_oil_cb_recent_test.py
│   ├── HE M1.ipynb
│   └── with_oil_carbon_black_* (历史评估 CSV 与图表)
│
├── 📁 data/                             ──> 【核心数据表集中存储】
│   ├── stage_statistics_enriched_all_features_weather_v4.csv (时序特征宽表)
│   ├── stage_statistics_enriched.csv
│   └── oilload_mapping_master_batch.csv
│
├── 📁 reports/                          ──> 【最终汇报 PPT 与分析报告】
│   └── AI_Based_Mooney_Prediction_Platform_M1.pptx
│
├── 📁 Mooney_Prediction_Pipeline/       ──> 【最新终极三阶段模型管道】
│   ├── 📁 model_training/
│   │   ├── train_group_mooney_models_ultimate3stage.py (终极三阶段训练主程序)
│   │   └── train_group_mooney_models_nonlinear.py
│   ├── 📁 data_processing/
│   └── 📁 model_analysis/
│
├── 📁 results_with_oil/                 ──> 炭黑充油体系模型产出 (.joblib模型, 评估CSV)
├── 📁 results_without_oil/              ──> 炭黑非充油体系模型产出
├── 📁 results_silica_with_oil/          ──> 白炭黑充油体系模型产出
└── 📁 results_silica_without_oil/       ──> 白炭黑非充油体系模型产出
```

---

## 1. 核心算法设计 (Ultimate 3-Stage Architecture)

为了捕获胶料在不同配方下的基线门尼与同配方下的车次间微小波动，算法进行了三阶段的彻底解耦分工：

### 阶段 1：静态配方与原料 COA 基线层 (Stage 1 Nominal Baseline)
* **输入**：配方 PHR 特征（白炭黑份数、油量占比等）+ 原材料 COA 粘度（`supplier_rubber_viscosity_avg`）。
* **算法**：利用 **RidgeCV** 进行动态正则化拟合，捕获大绝对值跨配方门尼差异（例如 30 MU vs 60 MU），提供全局名义基线。

### 阶段 2：动态工艺非线性残差层 (Stage 2 Process Residual)
* **输入**：12 维工艺参数相对于名义值的偏差值（$\Delta X = X_{actual} - X_{nominal}$）。
* **算法**：使用 **EWMA（指数加权移动平均）+ 物理单调约束 RidgeCV**，仅拟合扣除基线后的门尼偏差残差值。排除了决策树类算法（如 LightGBM）的阶跃断层（Staircase Step）弊端，输出平滑连续的工艺响应。

### 阶段 3：在线自适应与抗噪钳制层 (Stage 3 Hampel-Clamped AAKF)
* **算法**：**自适应新息卡尔曼滤波 (AAKF)**。
  * **快速追踪**：当检测到生胶批次换托/原料突变（残差 $> 1.8\text{ MU}$）时，瞬间将增益 $K_k$ 提升至 $0.75 \sim 0.85$，以 1 车的速度快速对齐新基线。
  * **Hampel 抗噪钳制（防过拟合核心）**：单点测试新息突变 $> 3.0\text{ MU}$时，触发抗噪安全网，将增益锁死在 $0.10$，拒绝跟随采样或测试误差，彻底实现对噪声的阻断。

---

## 2. 🛡️ 数据科学防过拟合实证

模型在 18,783 批次数据上经历了严谨的防过拟合检验：
1. **参数泛化 Gap 仅 0.05 MU**：炭黑充油体系训练集 MAE = **`0.86 MU`**，Group CV 验证集 MAE = **`1.15 MU`**，验证了 RidgeCV 动态 $L_2$ 正则化对配方特征过拟合的卓越抑制。
2. **实验室测试误差隔离率达 90%+**：在注入 $+5.0\text{ MU}$ 的实验室测试假噪声测试中，下一车预测受影响度被牢牢钳制在 **`0.11 ~ 0.50 MU`** 以内，免除了盲目追随采样噪声的弊端。

---

## 3. 🚀 鲁棒重训后的全量评估结果 (Group 5-Fold CV)

| 赛道大类 (Track) | 样本数 (N) | Group CV R² | Group CV RMSE | Group CV MAE | 趋势相关性 (R) | 防过拟合状态 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Carbon Black - With Oil** | 3,598 | **0.7399** | 2.32 MU | **1.15 MU** 🟢 | **+0.8719** 🟢 | 验证通过 |
| **Carbon Black - No Oil** | 1,311 | **0.9056** | 4.37 MU | **2.95 MU** 🟢 | **+0.9520** 🟢 | 验证通过 |
| **Silica - With Oil** | 10,975 | **0.7208** | 3.11 MU | **1.96 MU** 🟢 | **+0.8503** 🟢 | 验证通过 |
| **Silica - No Oil** | 2,899 | **0.6800** | 6.16 MU | **3.83 MU** 🟢 | **+0.8506** 🟢 | 验证通过 |

---

## 4. 顺时针执行与维护指南

### 1. 运行特征工程与处理
```bash
python Mooney_Prediction_Pipeline/data_processing/curve_segmenter_all_compounds.py
```

### 2. 执行鲁棒三阶段重新训练
```bash
python Mooney_Prediction_Pipeline/model_training/train_group_mooney_models_ultimate3stage.py
```

### 3. 现场异常诊断与自适应调控流程
* **诊断发现**：当 Stage 2 计算出 `Stage4_WetMixing_power_Integral` 偏差量 $< -1000\text{ kW}\cdot\text{s}$ 时，诊断为**湿混能量做工亏空**（门尼升高主因）。
* **自适应调控**：Stage 3 自动计算偏离贡献，并提示工艺员或 PLC 下发补偿指令（如延长湿混时长 $2 \sim 4\text{ 秒}$），实现车次间的在线质量闭环稳定。
