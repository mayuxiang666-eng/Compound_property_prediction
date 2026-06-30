# 胶料门尼粘度 (MNY) 预测模型开发与评估报告 (With-Oil Carbon-Black 胶料模型)
    
本报告详细说明了针对 **With-Oil Carbon-Black** 胶料数据集进行门尼粘度（MNY）回归建模的评估结果。

---

## 1. 建模数据概况

- **数据集类型**：With-Oil Carbon-Black 胶料数据（is_oil_loading_present = 1.0）
- **总车次行数**：1016 车
- **异常值清理**：使用**孤立森林 (Isolation Forest) 算法**对温度、功率、扭矩和持续时间等核心曲线特征进行多维异常诊断。
- **清洗后带有门尼粘度标签的车次**：1003 车
- **特征工程预处理**：
  1. 使用中位数填充缺失值。
  2. 进行标准差标准化（StandardScaler）。
    3. 先执行物理可解释特征审计，剔除高缺失、高单一值、高共线和低优先级特征。
    4. 使用方差过滤（VarianceThreshold，阈值=0.01）移除噪声/恒定特征，过滤后剩余 36 个特征进行训练。

### 1.1 可解释特征筛选

- 候选特征数：36
- 进入紧凑特征集：36
- 最终进入模型训练：36
- 特征审计明细：`interpretable_feature_audit.csv`

---

## 2. 模型选择与交叉验证 (Model Comparison)

使用 **5折交叉验证 (5-Fold CV)** 在训练集上评估了多种回归算法。性能比较如下：

| 模型算法 | 平均 CV $R^2$ (决定系数) | 平均 CV MAE (平均绝对误差) | 平均 CV RMSE (均方根误差) |
| :--- | :---: | :---: | :---: |
| Ridge Regression | -0.0537 | 1.4213 | 1.9733 |
| Random Forest | 0.0656 | 1.3046 | 1.8593 |
| Gradient Boosting | -0.0231 | 1.3883 | 1.9455 |
| XGBoost (Baseline) | -0.0842 | 1.4291 | 2.0016 |
| LightGBM (Huber) | 0.0648 | 1.3059 | 1.8595 |
| HistGradientBoosting | 0.0170 | 1.3597 | 1.9065 |
| CatBoost (Baseline) | 0.0280 | 1.3469 | 1.8961 |
| LightGBM (Tuned) | 0.0978 | 1.2792 | 1.8253 |
| XGBoost (Tuned) | 0.0985 | 1.3017 | 1.8254 |

*注：$R^2$ 越接近 1.0 预测拟合效果越好。MAE 代表门尼粘度预测值与实测值的绝对平均偏差值。*

---

## 3. Optuna 超参数优化与测试集最终表现

最优模型为 **Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF)**，其最优超参数集为：
- `best_params`: `LGBM: {'n_estimators': 718, 'max_depth': 4, 'learning_rate': 0.011430941831248929, 'num_leaves': 58, 'min_child_samples': 33, 'min_split_gain': 0.052849766446526836, 'subsample': 0.6981703399524762, 'colsample_bytree': 0.7231216619528729, 'reg_alpha': 0.0009307156257795905, 'reg_lambda': 14.599848002132797} | XGB: {'n_estimators': 667, 'max_depth': 4, 'learning_rate': 0.008328437996752626, 'subsample': 0.8763869015386214, 'colsample_bytree': 0.6171129592372265, 'min_child_weight': 39.03496194375423, 'gamma': 1.5206390955645817, 'reg_alpha': 0.00010805299440918792, 'reg_lambda': 15.36481567275946}`

### 3.1 测试集（Held-out Test Set, 20% 未参训数据）的最终表现：
- **测试集决定系数 $R^2$**：**0.9439**
- **测试集平均绝对误差 (MAE)**：**1.5545**
- **测试集均方根误差 (RMSE)**：**2.2010**

### 3.2 预测效果 Parity 对齐图
![Mooney预测Parity图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_with_oil_carbon_black/mooney_model_parity_plot.png)

---

## 4. 特征贡献度分析 (Feature Importance)

以下是经过调优后的 Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF) 模型在预测门尼粘度时，排名前 15 位的核心物理特征：

| 排名 | 特征名称 | 贡献权重 (Importance) | 物理流变学解释 |
| :---: | :--- | :---: | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.0624 | 流变/物理参数描述 |
| 2 | `roll_mny_mean_3b_same_comp` | 0.0612 | 流变/物理参数描述 |
| 3 | `roll_resid_mean_10b_mixer` | 0.0561 | 流变/物理参数描述 |
| 4 | `Stage6_BottomMixing_Torque_Integral` | 0.0464 | Stage6阶段扭矩累积积分（功历史） |
| 5 | `phys_temp_integral_above_100` | 0.0451 | 总混炼温度-时间积分（热历史） |
| 6 | `Stage3_OilLoading_temp_Std` | 0.0402 | 流变/物理参数描述 |
| 7 | `Stage6_BottomMixing_power_Mean` | 0.0398 | 流变/物理参数描述 |
| 8 | `roll_mny_std_10b_same_comp` | 0.0385 | 流变/物理参数描述 |
| 9 | `Stage2_DryMixing_power_Integral` | 0.0383 | 流变/物理参数描述 |
| 10 | `Stage3_OilLoading_temp_Mean` | 0.0383 | 流变/物理参数描述 |
| 11 | `supplier_carbon_black_structure_avg` | 0.0373 | 流变/物理参数描述 |
| 12 | `Stage6_BottomMixing_power_Integral` | 0.0347 | 流变/物理参数描述 |
| 13 | `supplier_rubber_viscosity_avg` | 0.0344 | 供应商生胶出厂粘度平均值 |
| 14 | `Stage2_DryMixing_power_Mean` | 0.0323 | 流变/物理参数描述 |
| 15 | `Stage4_WetMixing_Duration` | 0.0315 | Stage4阶段持续时间（秒） |


### 4.2 核心贡献度图解
![Mooney特征重要性图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_with_oil_carbon_black/mooney_model_feature_importance.png)

---

## 5. 工艺调整解释表

下表将模型重要性、单调相关趋势和密炼工艺含义放在一起，用于预测完成后的工艺诊断。注意：该方向代表训练数据中的统计趋势，实际调整必须结合胶料体系、配方窗口和现场约束确认。

| 排名 | 特征名称 | 模型重要性 | Spearman趋势 | 趋势解释 | 工艺含义/调整方向 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.0624 | 0.2148 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 2 | `roll_mny_mean_3b_same_comp` | 0.0612 | 0.1504 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 3 | `roll_resid_mean_10b_mixer` | 0.0561 | 0.2128 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 4 | `Stage6_BottomMixing_Torque_Integral` | 0.0464 | -0.1235 | 该特征升高时，模型/数据倾向于 MNY 降低 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 5 | `phys_temp_integral_above_100` | 0.0451 | -0.1203 | 该特征升高时，模型/数据倾向于 MNY 降低 | 高温积分代表热反应历史，白炭黑/硅烷体系中过高可能导致反应过度，过低可能反应不足。 |
| 6 | `Stage3_OilLoading_temp_Std` | 0.0402 | -0.0465 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 7 | `Stage6_BottomMixing_power_Mean` | 0.0398 | 0.0620 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 8 | `roll_mny_std_10b_same_comp` | 0.0385 | -0.0021 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 9 | `Stage2_DryMixing_power_Integral` | 0.0383 | 0.0021 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 10 | `Stage3_OilLoading_temp_Mean` | 0.0383 | -0.0827 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 11 | `supplier_carbon_black_structure_avg` | 0.0373 | -0.0085 | 单调方向较弱，更多体现交互或分群作用 | 炭黑结构值决定了吸留橡胶比例与填充网络刚性，吸油值偏高会直接导致混炼门尼升高，需核对该批次炭黑指标。 |
| 12 | `Stage6_BottomMixing_power_Integral` | 0.0347 | -0.1220 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 13 | `supplier_rubber_viscosity_avg` | 0.0344 | -0.0024 | 单调方向较弱，更多体现交互或分群作用 | 生胶供应商粘度是上游遗传因素，偏高时不要只调密炼参数，应同步看原材料批次。 |
| 14 | `Stage2_DryMixing_power_Mean` | 0.0323 | 0.0033 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 15 | `Stage4_WetMixing_Duration` | 0.0315 | -0.0039 | 单调方向较弱，更多体现交互或分群作用 | 这是阶段时间特征，适合结合该阶段温度、扭矩和能量共同判断节拍是否偏离。 |


完整明细见：`process_adjustment_guidance.csv`


---

## 6. 各配方系列 (Compound Family) 的预测表现

以下是测试集中前 15 个主要胶料配方系列的表现统计（按 RMSE 降序排列）：

| 排名 | 配方系列 | 测试车次 | 真实门尼均值 | 预测门尼均值 | 偏差 (Bias) | MAE | RMSE | $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `M1-R00037W9` | 2 | 54.48 | 46.61 | -7.8676 | 7.8676 | 7.9351 | -1.1917 |
| 2 | `M1-A06441R1` | 1 | 54.05 | 61.49 | 7.4428 | 7.4428 | 7.4428 | N/A |
| 3 | `M1-T00011-M` | 1 | 41.72 | 35.29 | -6.4287 | 6.4287 | 6.4287 | N/A |
| 4 | `M1-A00268R2` | 2 | 47.14 | 42.88 | -4.2573 | 4.2573 | 5.5232 | -37.9482 |
| 5 | `M1-B00163XV` | 1 | 35.88 | 31.68 | -4.1962 | 4.1962 | 4.1962 | N/A |
| 6 | `M1-A00205` | 1 | 37.74 | 41.79 | 4.0531 | 4.0531 | 4.0531 | N/A |
| 7 | `M1-B00163` | 2 | 39.20 | 40.36 | 1.1538 | 3.7272 | 3.9017 | -0.0739 |
| 8 | `M1-B00163-M` | 4 | 30.76 | 28.36 | -2.3907 | 3.1159 | 3.4895 | -0.8327 |
| 9 | `M1-A00517B3` | 2 | 78.93 | 80.46 | 1.5279 | 2.9551 | 3.3267 | -0.4849 |
| 10 | `M1-B00163XM` | 1 | 31.19 | 34.21 | 3.0158 | 3.0158 | 3.0158 | N/A |
| 11 | `M1-A44517` | 1 | 69.53 | 66.55 | -2.9799 | 2.9799 | 2.9799 | N/A |
| 12 | `M1-B15563` | 6 | 41.83 | 41.00 | -0.8288 | 2.5412 | 2.9317 | -1.4286 |
| 13 | `M1-B00163R4` | 2 | 31.86 | 30.24 | -1.6170 | 2.2946 | 2.8071 | -0.5225 |
| 14 | `M1-B00460B` | 18 | 54.11 | 54.28 | 0.1739 | 2.2450 | 2.5924 | -0.0405 |
| 15 | `M1-T00011B2` | 9 | 42.65 | 42.77 | 0.1212 | 2.1304 | 2.5570 | 0.2585 |

*注：对于仅有1个测试样本或方差为0的系列，$R^2$ 显示为 N/A。*
