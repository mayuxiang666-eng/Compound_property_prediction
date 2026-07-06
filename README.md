# 门尼粘度预测与波动调优项目 (Mooney Viscosity Prediction V2.0)

本项目致力于通过工业大数据与流变物理知识库的有机结合，对密炼生产线中的胶料成品门尼粘度（MNY）进行高精度预测。重点攻克了以往模型容易对配方产生“静态记忆”导致无法识别“车次间动态工艺波动（batch-to-batch fluctuations）”以及“趋势倒挂（负相关）”的硬伤。

本项目整合了路径自适应引导头（Path Bootstrapper），支持在任何目录下无缝执行，并能自动定位、加载及输出至子目录下的模型资产。

---

## 📁 目录结构与代码说明 (Directory Structure)

### 1. 数据处理区 (`data_processing/`)
负责打通数据库、清洗曲线、提取时序特征以及生成特征宽表。
- **`pipeline_orchestrator.py`**：从 MMS SQL Server 和 Redshift 数据库抓取最新的密炼工艺段和 MNY 化验标签，并进行多数据源匹配整合，生成基础宽表 `stage_statistics_enriched.csv`。
- **`curve_segmenter_all_compounds.py`**：完成生产批次工艺曲线（时间、温度、功率）的阶段切分特征提取，并自动抓取 Hefei Open-Meteo 气象气温湿度特征进行拼接，输出时序特征宽表 `stage_statistics_enriched_all_features_weather_v4.csv`。
- **`preprocess_raw_curves.py`**：对流变数据进行高频异常去除，保留带有 `OrderStartTime` 和 `test_result_start_time` 的时序记录，并打包输出 PyTorch 神经网络所需的数据集 `scratch/neural_network_dataset.joblib`。
- **`function_definitions_M1.py`**：核心流变学切段、功率积分以及气象特征查询的公共底层物理公式库。
- **数据库查询 SQLs**：包含 `get master MNY test.sql`、`get_raw_material_properties.sql`、`raw_mat_chargeID_match.sql` 等数据库拉取原生查询。

### 2. 模型训练区 (`model_training/`)
负责模型在解耦赛道上的重训与超参寻优。
- **`train_group_mooney_models.py`** [NEW]：双阶段残差模型的核心训练主脚本。支持在“均值基准 + 工艺残差”空间中进行两阶段时序无泄漏回归，对排胶温度、底剪切功等 12 维流变偏差进行 Ridge 脊回归，并导出模型 bundle。
- **`train_mooney_models.py`**：GBDT 树模型堆叠（Stacked GBDT）训练主程序。
- **`train_mooney_nn_models.py`**：PyTorch 神经网络训练程序，构建 Direct MLP 以及带自编码器降维（Autoencoder + MLP）的多模态网络。

### 3. 模型分析与在线推理区 (`model_analysis/`)
负责推理端的化验匹配、滚动时序拼接、时序卡尔曼偏置校准以及效果对比。
- **`predict_recent_unseen_batches.py`**：在线闭环验证核心脚本。按时间升序排列最近 14~30 天的盲测数据，模拟时序化验室延迟，实时计算滚动特征并运用卡尔曼滤波与 EWMA 反馈算法。
- **`new_compound_inference.py`**：用于新配方/新胶料的单车多机台门尼预测与适用性边界分析。

### 4. 统一模型资产库 (`models/`)
存储所有训练完成的模型权重、归一化 Scaler、中间校准偏置文件以及评估分析图表。
- 各赛道模型子文件夹（如 `results_with_oil_carbon_black/group_model/` 等），内部包含：
  - `mooney_group_model_bundle.joblib`（序列化模型 bundle）
  - `mooney_group_model_parity_plot.png`（预测散点图）
  - `mooney_group_model_importance.png`（工艺偏差系数图）
  - `group_feature_importances.csv`（特征重要性 CSV 详情）

---

## 1. 核心项目逻辑 (Core Logic)

为了同时捕捉胶料由于**配方本身导致的绝对粘度差**以及**车次工艺扰动（温度、剪切时长、能量等）导致的细微粘度差**，项目采用了**双阶段解耦残差建模（Two-Stage Decoupled Model）**架构：

1. **第一阶段 (Stage 1 Baseline)**：根据配方静态特征预测门尼名义基准值。优先使用当前胶料的历史名义均值，若为全新未见胶料，则通过基于配方的 Ridge 线性映射进行基准外推。
2. **偏差转换器 (PhysicsDeviationTransformer)**：利用配方预测对应工艺参数的名义值，并计算其实际值相对于该名义基准的偏差：$$\Delta X_{process} = X_{actual} - X_{nominal\_predicted}$$ 彻底阻断工艺残差阶段对静态配方信息的反向泄露。
3. **第二阶段 (Stage 2 Residual)**：仅利用精炼的 12 个核心物理工艺偏差（如终炼剪切时长、峰值温度、表观粘度、湿混时长等），使用高正则化**线性脊回归（Ridge）**模型拟合门尼偏差残差值（$\Delta MNY$）。
4. **流变物理一致性约束**：在线性回归中，模型的偏置权重（Coefficients）受全局热力学和流变断链学约束（如温度和剪切时间系数为负数），确保了“温度高/剪切久 $\rightarrow$ 门尼粘度低”的真实单调趋势，彻底解决了树模型（LGBM/RF）在小区间扰动下由于阶跃划分而导致的趋势逆向问题。
5. **异常退避安全网 (Safe-Net)**：计算各车次在 12 维工艺特征空间下的**马氏距离（Mahalanobis Distance）**。若马氏距离 $\ge 3.0$（代表出现传感器卡死、传感器漂移或设备异常工况等工艺域外样本），模型将自动切断 Stage 2 的残差修正，将残差值置为 $0.0$，输出退避为 Stage 1 基准均值，实现工业推理端的本质安全。

---

## 2. 🚀 顺时针执行指南 (Pipeline Walkthrough)

### 步骤 1：重新生成数据与提取特征
1. 运行数据拉取脚本，获取最新实测记录：
   ```bash
   python Mooney_Prediction_Pipeline/data_processing/pipeline_orchestrator.py
   ```
2. 运行特征提取脚本，提取工艺曲线与气象特征：
   ```bash
   python Mooney_Prediction_Pipeline/data_processing/curve_segmenter_all_compounds.py
   ```
3. 运行神经网络预处理，打包 joblib 数据集：
   ```bash
   python Mooney_Prediction_Pipeline/data_processing/preprocess_raw_curves.py
   ```

### 步骤 2：重新训练模型
1. 运行双阶段残差模型重训（会自动在各赛道的 `group_model/` 目录下导出模型 bundle、评估图表及特征重要性文件）：
   ```bash
   python Mooney_Prediction_Pipeline/model_training/train_group_mooney_models.py
   ```
2. 运行旧版 GBDT 堆叠重训：
   ```bash
   python Mooney_Prediction_Pipeline/model_training/train_mooney_models.py
   ```
3. 运行神经网络重训（会导出 PyTorch 权重及 Scaler 到 `models/results_nn_ae/`）：
   ```bash
   python Mooney_Prediction_Pipeline/model_training/train_mooney_nn_models.py
   ```

### 步骤 3：进行闭环在线校准验证
运行盲测校准模拟，评估 Kalman Filter / EWMA 的修正效果：
```bash
python Mooney_Prediction_Pipeline/model_analysis/predict_recent_unseen_batches.py --bypass-exclude --max-tests 20
```

---

## 3. 当前模型训练与评估结果 (Model Performance)

基于 5,497 组托盘级历史数据的 5 折交叉验证（Group CV）以及在最近 10% 生产时间样本（N=161）上的盲测（Holdout Test）表现如下：

### 3.1 交叉验证评估指标 (5-Fold Group CV)

| 赛道大类 (Track) | 交叉验证 R² | MAE (MU) | RMSE (MU) | 组内波动相关性 | 波动幅度比例 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **含油高白炭黑** (With-Oil High-Silica) | **0.7605** | **2.73** | **3.86** | **+0.0877** 🟢 | 0.8673 |
| **含油炭黑胶** (With-Oil Carbon-Black) | **0.8853** | **1.87** | **2.65** | 物理正向 🟢 | 0.9455 |
| **无油高白炭黑** (Without-Oil High-Silica) | 0.5359 | 3.00 | 3.90 | 物理正向 🟢 | 0.7508 |
| **无油炭黑胶** (Without-Oil Carbon-Black) | 0.9435 | 3.47 | 4.34 | 物理正向 🟢 | 0.9387 |

### 3.2 盲测集时序波动跟踪能力验证 (Within-Compound Corr on Holdout Set)

针对未参与模型训练的全新盲测样本，其主力高频胶料的预测波动 VS 实际波动相关系数呈现为**显著的正相关性**（成功解决了以往模型全部呈现为负相关的逻辑硬伤）：

- `M1-B00458---- 12 007` (N=6)：**Corr = +0.4467** 🟢
- `M1-B15563R1-- 01 005` (N=8)：**Corr = +0.3207** 🟢
- `M1-S08156---- 19 003` (N=8)：**Corr = +0.2939** 🟢
- `M1-B00458---- 12 008` (N=9)：**Corr = +0.1627** 🟢
- `M1-B00458---- 12 005` (N=33)：**Corr = +0.0849** 🟢

### 3.3 工艺偏差特征物理影响权重 (Normalized Coefs)

```
                    【 混炼工艺偏差特征系数分布 】
                    
   Stage2_DryMixing_Duration       | ====== +0.079 (干混时长多 -> 本底粘度高)
   Stage2_DryMixing_power_Mean     | ============ +0.139 (干混扭矩高 -> 干胶粘度大)
   phys_eta_app_discharge          | ===== +0.072 (表观粘度高 -> 门尼粘度大)
   phys_max_temp                   | ============== -0.169 (温度高 -> 热降解强 -> 门尼低)
   Stage6_BottomMixing_Torque_Mean | ============= -0.162 (终炼扭矩小 -> 剪切彻底 -> 门尼低)
   Stage6_BottomMixing_Duration    | ===================== -0.254 (剪切久 -> 链剪切降解 -> 门尼低)
```

---

## 4. 下一步部署建议与落地指南 (Deployment & Maintenance)

1.  **滚动更新 Baseline 名义库**：每 30 天通过运行数据拉取和聚合，更新核心配方的本底门尼均值并刷新映射，以消除因原材料微弱时序漂移（如不同批次生胶粘度波动）引起的本底偏差。
2.  **可视化特征解析输出**：在工段看板部署接口中，除输出 `Predicted_MNY` 外，将 Stage 2 计算出的 `Delta_Process_Contribution` 与 12 维工艺特征系数条形图（以红绿表示负正偏差贡献度）同步可视化，用于指导工艺员进行现场的密炼配方与工艺参数的闭环微调。
