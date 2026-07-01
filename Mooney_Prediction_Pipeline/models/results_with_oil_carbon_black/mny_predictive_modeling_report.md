# 胶料门尼粘度 (MNY) 预测模型开发与评估报告 (With-Oil Carbon-Black 胶料模型)
    
本报告详细说明了针对 **With-Oil Carbon-Black** 胶料数据集进行门尼粘度（MNY）回归建模的评估结果。

---

## 1. 建模数据概况

- **数据集类型**：With-Oil Carbon-Black 胶料数据（is_oil_loading_present = 1.0）
- **总车次行数**：730 车
- **异常值清理**：使用**孤立森林 (Isolation Forest) 算法**对温度、功率、扭矩和持续时间等核心曲线特征进行多维异常诊断。
- **清洗后带有门尼粘度标签的车次**：728 车
- **特征工程预处理**：
  1. 使用中位数填充缺失值。
  2. 进行标准差标准化（StandardScaler）。
    3. 先执行物理可解释特征审计，剔除高缺失、高单一值、高共线和低优先级特征。
    4. 使用方差过滤（VarianceThreshold，阈值=0.01）移除噪声/恒定特征，过滤后剩余 35 个特征进行训练。

### 1.1 可解释特征筛选

- 候选特征数：35
- 进入紧凑特征集：35
- 最终进入模型训练：35
- 特征审计明细：`interpretable_feature_audit.csv`

---

## 2. 模型选择与交叉验证 (Model Comparison)

使用 **5折交叉验证 (5-Fold CV)** 在训练集上评估了多种回归算法。性能比较如下：

| 模型算法 | 平均 CV $R^2$ (决定系数) | 平均 CV MAE (平均绝对误差) | 平均 CV RMSE (均方根误差) |
| :--- | :---: | :---: | :---: |
| Ridge Regression | -0.0394 | 1.3651 | 1.8683 |
| Random Forest | 0.0938 | 1.2705 | 1.7436 |
| Gradient Boosting | 0.0456 | 1.2993 | 1.7852 |
| XGBoost (Baseline) | -0.0188 | 1.3404 | 1.8423 |
| LightGBM (Huber) | 0.0750 | 1.2877 | 1.7628 |
| HistGradientBoosting | 0.0309 | 1.3265 | 1.8004 |
| CatBoost (Baseline) | 0.0424 | 1.3089 | 1.7902 |
| LightGBM (Tuned) | 0.0920 | 1.2462 | 1.7448 |
| XGBoost (Tuned) | 0.0965 | 1.2463 | 1.7411 |

*注：$R^2$ 越接近 1.0 预测拟合效果越好。MAE 代表门尼粘度预测值与实测值的绝对平均偏差值。*

---

## 3. Optuna 超参数优化与测试集最终表现

最优模型为 **Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF)**，其最优超参数集为：
- `best_params`: `LGBM: {'n_estimators': 832, 'max_depth': 6, 'learning_rate': 0.010812875196624305, 'num_leaves': 57, 'min_child_samples': 42, 'min_split_gain': 1.0462485185003123, 'subsample': 0.6899333325087741, 'colsample_bytree': 0.7470192712819486, 'reg_alpha': 0.008289919407648314, 'reg_lambda': 1.6297453199705119} | XGB: {'n_estimators': 357, 'max_depth': 5, 'learning_rate': 0.005349246442872605, 'subsample': 0.8184669903174201, 'colsample_bytree': 0.6485165464679071, 'min_child_weight': 39.142302365938285, 'gamma': 0.6087153900144324, 'reg_alpha': 0.849420831358656, 'reg_lambda': 2.0991872523448714}`

### 3.1 测试集（Held-out Test Set, 20% 未参训数据）的最终表现：
- **测试集决定系数 $R^2$**：**0.8121**
- **测试集平均绝对误差 (MAE)**：**1.9823**
- **测试集均方根误差 (RMSE)**：**3.4362**

### 3.2 预测效果 Parity 对齐图
![Mooney预测Parity图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_with_oil_carbon_black/mooney_model_parity_plot.png)

---

## 4. 特征贡献度分析 (Feature Importance)

以下是经过调优后的 Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF) 模型在预测门尼粘度时，排名前 15 位的核心物理特征：

| 排名 | 特征名称 | 贡献权重 (Importance) | 物理流变学解释 |
| :---: | :--- | :---: | :--- |
| 1 | `Stage6_BottomMixing_Torque_Integral` | 0.0843 | Stage6阶段扭矩累积积分（功历史） |
| 2 | `roll_mny_mean_3b_same_comp` | 0.0741 | 流变/物理参数描述 |
| 3 | `roll_resid_mean_10b_mixer` | 0.0727 | 流变/物理参数描述 |
| 4 | `roll_resid_mean_5b_family` | 0.0687 | 流变/物理参数描述 |
| 5 | `Stage3_OilLoading_temp_Mean` | 0.0510 | 流变/物理参数描述 |
| 6 | `Stage3_OilLoading_temp_Std` | 0.0493 | 流变/物理参数描述 |
| 7 | `Stage6_BottomMixing_Torque_Mean` | 0.0441 | Stage6阶段扭矩平均值 |
| 8 | `Stage6_BottomMixing_power_Mean` | 0.0424 | 流变/物理参数描述 |
| 9 | `roll_mny_std_10b_same_comp` | 0.0413 | 流变/物理参数描述 |
| 10 | `Stage4_WetMixing_power_Mean` | 0.0412 | 流变/物理参数描述 |
| 11 | `Stage6_BottomMixing_power_Integral` | 0.0384 | 流变/物理参数描述 |
| 12 | `phys_init_temp` | 0.0327 | 流变学计算特征 |
| 13 | `phys_temp_integral_above_100` | 0.0312 | 总混炼温度-时间积分（热历史） |
| 14 | `weight_pct_natural_rubber` | 0.0275 | 配方中该组分的重量百分比 |
| 15 | `supplier_silica_surface_area_avg` | 0.0265 | 流变/物理参数描述 |


### 4.2 核心贡献度图解
![Mooney特征重要性图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_with_oil_carbon_black/mooney_model_feature_importance.png)

---

## 5. 工艺调整解释表

下表将模型重要性、单调相关趋势和密炼工艺含义放在一起，用于预测完成后的工艺诊断。注意：该方向代表训练数据中的统计趋势，实际调整必须结合胶料体系、配方窗口和现场约束确认。

| 排名 | 特征名称 | 模型重要性 | Spearman趋势 | 趋势解释 | 工艺含义/调整方向 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `Stage6_BottomMixing_Torque_Integral` | 0.0843 | -0.1978 | 该特征升高时，模型/数据倾向于 MNY 降低 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 2 | `roll_mny_mean_3b_same_comp` | 0.0741 | 0.1820 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 3 | `roll_resid_mean_10b_mixer` | 0.0727 | 0.2341 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 4 | `roll_resid_mean_5b_family` | 0.0687 | 0.1967 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 5 | `Stage3_OilLoading_temp_Mean` | 0.0510 | -0.0335 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 6 | `Stage3_OilLoading_temp_Std` | 0.0493 | -0.0465 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 7 | `Stage6_BottomMixing_Torque_Mean` | 0.0441 | -0.0539 | 单调方向较弱，更多体现交互或分群作用 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 8 | `Stage6_BottomMixing_power_Mean` | 0.0424 | 0.1061 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 9 | `roll_mny_std_10b_same_comp` | 0.0413 | 0.0040 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 10 | `Stage4_WetMixing_power_Mean` | 0.0412 | 0.0593 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 11 | `Stage6_BottomMixing_power_Integral` | 0.0384 | -0.1761 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 12 | `phys_init_temp` | 0.0327 | -0.0533 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 13 | `phys_temp_integral_above_100` | 0.0312 | -0.1441 | 该特征升高时，模型/数据倾向于 MNY 降低 | 高温积分代表热反应历史，白炭黑/硅烷体系中过高可能导致反应过度，过低可能反应不足。 |
| 14 | `weight_pct_natural_rubber` | 0.0275 | 0.0219 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 15 | `supplier_silica_surface_area_avg` | 0.0265 | -0.0674 | 单调方向较弱，更多体现交互或分群作用 | 白炭黑比表面积决定了偶联活性位点与填料网络强弱，异常偏高会增强粒子自凝聚进而推高门尼，应核对批次检测。 |


完整明细见：`process_adjustment_guidance.csv`


---

## 6. 各配方系列 (Compound Family) 的预测表现

以下是测试集中前 15 个主要胶料配方系列的表现统计（按 RMSE 降序排列）：

| 排名 | 配方系列 | 测试车次 | 真实门尼均值 | 预测门尼均值 | 偏差 (Bias) | MAE | RMSE | $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `M1-B00460-M` | 1 | 60.71 | 40.56 | -20.1467 | 20.1467 | 20.1467 | N/A |
| 2 | `M1-R00037XE` | 1 | 59.49 | 40.66 | -18.8341 | 18.8341 | 18.8341 | N/A |
| 3 | `M1-B00163XM` | 2 | 29.84 | 39.35 | 9.5054 | 9.5054 | 9.5726 | -49.6543 |
| 4 | `M1-R00037W9` | 3 | 55.17 | 52.32 | -2.8515 | 6.5112 | 8.0326 | -5.7238 |
| 5 | `M1-B00163XV` | 3 | 34.64 | 34.73 | 0.0883 | 6.2300 | 6.6588 | -5.9127 |
| 6 | `M1-T00011-A` | 1 | 36.68 | 40.65 | 3.9668 | 3.9668 | 3.9668 | N/A |
| 7 | `M1-B00163R4` | 2 | 27.58 | 30.49 | 2.9088 | 2.9088 | 3.3468 | -2.9686 |
| 8 | `M1-A00205W3` | 3 | 38.37 | 41.09 | 2.7177 | 2.7177 | 2.8812 | -8.0651 |
| 9 | `M1-T00011B2` | 12 | 42.28 | 41.88 | -0.4079 | 2.0993 | 2.4620 | 0.0186 |
| 10 | `M1-B00460B` | 13 | 54.94 | 54.74 | -0.1999 | 1.8433 | 2.2206 | 0.5198 |
| 11 | `M1-T00011XV` | 2 | 38.66 | 40.62 | 1.9568 | 1.9568 | 2.0011 | -13.8085 |
| 12 | `M1-B15563` | 2 | 38.81 | 40.66 | 1.8453 | 1.8453 | 1.9097 | -21.2334 |
| 13 | `M1-B00163` | 2 | 29.95 | 30.46 | 0.5097 | 1.4491 | 1.5361 | 0.1064 |
| 14 | `M1-S08156B1` | 8 | 26.96 | 26.01 | -0.9544 | 1.1869 | 1.4810 | -0.4470 |
| 15 | `M1-B00458` | 75 | 40.24 | 40.42 | 0.1772 | 1.1883 | 1.4570 | 0.2370 |

*注：对于仅有1个测试样本或方差为0的系列，$R^2$ 显示为 N/A。*
