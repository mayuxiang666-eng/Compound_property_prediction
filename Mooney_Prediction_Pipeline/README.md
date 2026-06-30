# Mooney Viscosity 预测模型 V2.0 生产级集成管道

本文件夹整合并归类了 Mooney 预测模型 V2.0 的全部核心生产代码与已训练的模型权重文件。目录分为：数据拉取与处理、模型训练与超参优化、在线推理与闭环验证，以及统一的模型资产夹。

所有 Python 脚本均已注入路径自适应引导头（Path Bootstrapper），支持在任何目录下无缝执行，并能自动定位、加载及输出至子目录下的模型资产。

---

## 📁 目录结构与脚本归类

### 1. 数据处理区 (`data_processing/`)
负责打通数据库、清洗曲线、提取时序特征以及生成特征宽表。
- **`pipeline_orchestrator.py`**  
  *说明：* 从 MMS SQL Server 和 Redshift 数据库抓取最新的密炼工艺段和 MNY 化验标签，并进行多数据源匹配整合，生成基础宽表 `stage_statistics_enriched.csv`。
- **`curve_segmenter_all_compounds.py`**  
  *说明：* 完成生产批次工艺曲线（时间、温度、功率）的阶段切分特征提取，并自动抓取 Hefei Open-Meteo 气象气温湿度特征进行拼接，输出时序特征宽表 `stage_statistics_enriched_all_features_weather_v4.csv`。
- **`preprocess_raw_curves.py`**  
  *说明：* 对流变数据进行高频异常去除，保留带有 `OrderStartTime` 和 `test_result_start_time` 的时序记录，并打包输出 PyTorch 神经网络所需的数据集 `scratch/neural_network_dataset.joblib`。
- **`function_definitions_M1.py`**  
  *说明：* 核心流变学切段、功率积分以及气象特征查询的公共底层物理公式库。
- **`*.sql`**  
  *说明：* 包含 `get master MNY test.sql`、`get_raw_material_properties.sql`、`raw_mat_chargeID_match.sql` 等数据库拉取原生查询。
- **`credentials.json`**  
  *说明：* 本地 SQL 数据库和 Datamart 连接认证配置。

### 2. 模型训练区 (`model_training/`)
负责模型在双轨（有油/无油）和专用轨道上的重训与超参寻优。
- **`train_mooney_models.py`**  
  *说明：* GBDT 树模型堆叠（Stacked GBDT）训练主程序。支持在“均值基准 + 工艺残差”空间中进行两阶段时序无泄漏回归，通过 Optuna 自动调整 LightGBM/XGBoost/CatBoost 权重与超参。
- **`train_mooney_nn_models.py`**  
  *说明：* PyTorch 神经网络训练程序。在 PyTorch 中构建 Direct MLP 以及带自编码器降维（Autoencoder + MLP）的多模态网络。

### 3. 模型分析与在线推理区 (`model_analysis/`)
负责推理端的化验匹配、滚动时序拼接、时序卡尔曼偏置校准以及效果对比。
- **`predict_recent_unseen_batches.py`**  
  *说明：* 在线闭环验证核心脚本。按时间升序排列最近 14~30 天的盲测数据，模拟时序化验室延迟，实时计算无泄漏滚动特征，并运用**配方家族级（Family-level）卡尔曼滤波（Kalman Filter）与 EWMA 在线反馈算法**动态更新系统漂移。
- **`new_compound_inference.py`**  
  *说明：* 用于新配方/新胶料的单车多机台门尼预测与适用性（Reliability/Distance）边界分析推理接口。
- **`predict_latest_sfe_order.py`**  
  *说明：* 从运行中的车次中实时提取密炼特征并生成模型预测输入。
- **`validate_recent_ce_gap.py`**  
  *说明：* 提供流变样本号到物理车次（Batch Number）和密炼控制曲线的物理映射校验支持。
- **`compare_unified_vs_separated.py`**  
  *说明：* 提供单轨（Unified）大模型与双轨分离（Separated）模型的 R² 与 MAE 对比评估。
- **`compare_model_performance_by_compound.py`**  
  *说明：* 提供不同配方家族、不同机台维度的 RMSE 详细热力对比分析。

### 4. 统一模型资产库 (`models/`)
存储所有训练完成的模型权重、归一化 Scaler、中间校准偏置文件以及评估分析图表。
- **`results_with_oil/`**：有油 GBDT 堆叠模型 bundle。
- **`results_without_oil/`**：无油 GBDT 堆叠模型 bundle。
- **`results_15760/`**：专为高 Silica 高波动胶料 M1-T15760 建立的局部强化模型 bundle。
- **`results_nn_ae/`**：PyTorch 神经网络模型权重（AE、MLP、AE_MLP）及流变/配方归一化参数。
- **`results_m2_analysis/`**：保存自动计算出的 Few-shot 静态校准偏差 `calibration_biases.json`。
- **`results_model_comparison/`**：各版本模型离线对比评估报告和分析热力图。

---

## 🚀 顺时针执行指南 (Pipeline Walkthrough)

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
1. 运行 GBDT 堆叠重训（会自动在 `models/results_with_oil/` 和 `models/results_without_oil/` 目录下导出模型）：
   ```bash
   python Mooney_Prediction_Pipeline/model_training/train_mooney_models.py
   ```
2. 运行神经网络重训（会导出 PyTorch 权重及 Scaler 到 `models/results_nn_ae/`）：
   ```bash
   python Mooney_Prediction_Pipeline/model_training/train_mooney_nn_models.py
   ```

### 步骤 3：进行闭环在线校准验证
运行盲测校准模拟，评估 Kalman Filter / EWMA 的修正效果（结果会输出至 `results_recent_validation.csv`）：
```bash
python Mooney_Prediction_Pipeline/model_analysis/predict_recent_unseen_batches.py --bypass-exclude --max-tests 20
```
