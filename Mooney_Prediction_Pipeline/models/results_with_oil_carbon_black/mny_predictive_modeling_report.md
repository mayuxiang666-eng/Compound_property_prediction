# 胶料门尼粘度 (MNY) 预测模型开发与评估报告 (With-Oil Carbon-Black 胶料模型)
    
本报告详细说明了针对 **With-Oil Carbon-Black** 胶料数据集进行门尼粘度（MNY）回归建模的评估结果。

---

## 1. 建模数据概况

- **数据集类型**：With-Oil Carbon-Black 胶料数据（is_oil_loading_present = 1.0）
- **总车次行数**：1484 车
- **异常值清理**：使用**孤立森林 (Isolation Forest) 算法**对温度、功率、扭矩和持续时间等核心曲线特征进行多维异常诊断。
- **清洗后带有门尼粘度标签的车次**：1479 车
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
| Ridge Regression | 0.0166 | 1.4644 | 1.9592 |
| Random Forest | 0.1038 | 1.3482 | 1.8710 |
| Gradient Boosting | 0.0252 | 1.4061 | 1.9511 |
| XGBoost (Baseline) | 0.0191 | 1.4201 | 1.9574 |
| LightGBM (Huber) | 0.1194 | 1.3415 | 1.8541 |
| HistGradientBoosting | 0.0360 | 1.4285 | 1.9401 |
| CatBoost (Baseline) | 0.0854 | 1.3684 | 1.8897 |
| LightGBM (Tuned) | 0.1309 | 1.3276 | 1.8419 |
| XGBoost (Tuned) | 0.1277 | 1.3468 | 1.8457 |

*注：$R^2$ 越接近 1.0 预测拟合效果越好。MAE 代表门尼粘度预测值与实测值的绝对平均偏差值。*

---

## 3. Optuna 超参数优化与测试集最终表现

最优模型为 **Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF)**，其最优超参数集为：
- `best_params`: `LGBM: {'n_estimators': 446, 'max_depth': 6, 'learning_rate': 0.04253129403201789, 'num_leaves': 17, 'min_child_samples': 74, 'min_split_gain': 0.389988613566436, 'subsample': 0.7775590725904209, 'colsample_bytree': 0.7984867401964624, 'reg_alpha': 0.4861035488077223, 'reg_lambda': 2.2792420992919586} | XGB: {'n_estimators': 785, 'max_depth': 4, 'learning_rate': 0.006218166489642157, 'subsample': 0.6821778418173812, 'colsample_bytree': 0.6848926791410658, 'min_child_weight': 38.87450154897765, 'gamma': 3.013242227518378, 'reg_alpha': 0.06147335332064465, 'reg_lambda': 35.22279046883403}`

### 3.1 测试集（Held-out Test Set, 20% 未参训数据）的最终表现：
- **测试集决定系数 $R^2$**：**0.7595**
- **测试集平均绝对误差 (MAE)**：**1.8676**
- **测试集均方根误差 (RMSE)**：**3.3061**

### 3.2 预测效果 Parity 对齐图
![Mooney预测Parity图](file:///c:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_with_oil_carbon_black/mooney_model_parity_plot.png)

---

## 4. 特征贡献度分析 (Feature Importance)

以下是经过调优后的 Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF) 模型在预测门尼粘度时，排名前 15 位的核心物理特征：

| 排名 | 特征名称 | 贡献权重 (Importance) | 物理流变学解释 |
| :---: | :--- | :---: | :--- |
| 1 | `roll_resid_mean_10b_mixer` | 0.0700 | 流变/物理参数描述 |
| 2 | `roll_resid_mean_5b_family` | 0.0684 | 流变/物理参数描述 |
| 3 | `roll_mny_mean_3b_same_comp` | 0.0669 | 流变/物理参数描述 |
| 4 | `supplier_rubber_viscosity_avg` | 0.0533 | 供应商生胶出厂粘度平均值 |
| 5 | `roll_mny_std_10b_same_comp` | 0.0519 | 流变/物理参数描述 |
| 6 | `Stage6_BottomMixing_Torque_Mean` | 0.0502 | Stage6阶段扭矩平均值 |
| 7 | `Stage6_BottomMixing_Torque_Integral` | 0.0433 | Stage6阶段扭矩累积积分（功历史） |
| 8 | `Stage6_BottomMixing_power_Integral` | 0.0402 | 流变/物理参数描述 |
| 9 | `phys_temp_integral_above_100` | 0.0383 | 总混炼温度-时间积分（热历史） |
| 10 | `Stage6_BottomMixing_power_Mean` | 0.0359 | 流变/物理参数描述 |
| 11 | `Stage3_OilLoading_temp_Std` | 0.0357 | 流变/物理参数描述 |
| 12 | `Stage4_WetMixing_power_Mean` | 0.0341 | 流变/物理参数描述 |
| 13 | `Stage2_DryMixing_power_Mean` | 0.0322 | 流变/物理参数描述 |
| 14 | `Stage2_DryMixing_power_Integral` | 0.0318 | 流变/物理参数描述 |
| 15 | `Stage3_OilLoading_temp_Mean` | 0.0295 | 流变/物理参数描述 |


### 4.2 核心贡献度图解
![Mooney特征重要性图](file:///c:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_with_oil_carbon_black/mooney_model_feature_importance.png)

---

## 5. 工艺调整解释表

下表将模型重要性、单调相关趋势和密炼工艺含义放在一起，用于预测完成后的工艺诊断。注意：该方向代表训练数据中的统计趋势，实际调整必须结合胶料体系、配方窗口和现场约束确认。

| 排名 | 特征名称 | 模型重要性 | Spearman趋势 | 趋势解释 | 工艺含义/调整方向 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `roll_resid_mean_10b_mixer` | 0.0700 | 0.2602 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 2 | `roll_resid_mean_5b_family` | 0.0684 | 0.2465 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 3 | `roll_mny_mean_3b_same_comp` | 0.0669 | 0.2025 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 4 | `supplier_rubber_viscosity_avg` | 0.0533 | 0.0554 | 单调方向较弱，更多体现交互或分群作用 | 生胶供应商粘度是上游遗传因素，偏高时不要只调密炼参数，应同步看原材料批次。 |
| 5 | `roll_mny_std_10b_same_comp` | 0.0519 | -0.0675 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 6 | `Stage6_BottomMixing_Torque_Mean` | 0.0502 | -0.0636 | 单调方向较弱，更多体现交互或分群作用 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 7 | `Stage6_BottomMixing_Torque_Integral` | 0.0433 | -0.1031 | 该特征升高时，模型/数据倾向于 MNY 降低 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 8 | `Stage6_BottomMixing_power_Integral` | 0.0402 | -0.0949 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 9 | `phys_temp_integral_above_100` | 0.0383 | -0.0968 | 该特征升高时，模型/数据倾向于 MNY 降低 | 高温积分代表热反应历史，白炭黑/硅烷体系中过高可能导致反应过度，过低可能反应不足。 |
| 10 | `Stage6_BottomMixing_power_Mean` | 0.0359 | -0.0014 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 11 | `Stage3_OilLoading_temp_Std` | 0.0357 | -0.0737 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 12 | `Stage4_WetMixing_power_Mean` | 0.0341 | 0.0101 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 13 | `Stage2_DryMixing_power_Mean` | 0.0322 | 0.0631 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 14 | `Stage2_DryMixing_power_Integral` | 0.0318 | 0.0351 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 15 | `Stage3_OilLoading_temp_Mean` | 0.0295 | 0.0257 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |


完整明细见：`process_adjustment_guidance.csv`


---

## 6. 各配方系列 (Compound Family) 的预测表现

以下是测试集中前 15 个主要胶料配方系列的表现统计（按 RMSE 降序排列）：

| 排名 | 配方系列 | 测试车次 | 真实门尼均值 | 预测门尼均值 | 偏差 (Bias) | MAE | RMSE | $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `M1-A44517` | 1 | 63.92 | 41.00 | -22.9212 | 22.9212 | 22.9212 | N/A |
| 2 | `M1-B00460XP` | 1 | 59.21 | 40.75 | -18.4500 | 18.4500 | 18.4500 | N/A |
| 3 | `M1-B00460B` | 17 | 55.89 | 51.73 | -4.1682 | 4.9192 | 7.6706 | -7.7964 |
| 4 | `M1-R00037W9` | 4 | 56.58 | 54.22 | -2.3624 | 5.0444 | 7.2087 | -5.4294 |
| 5 | `M1-B00163XM` | 1 | 28.50 | 35.42 | 6.9246 | 6.9246 | 6.9246 | N/A |
| 6 | `M1-A00205` | 1 | 37.74 | 43.50 | 5.7559 | 5.7559 | 5.7559 | N/A |
| 7 | `M1-T01004` | 3 | 40.73 | 35.43 | -5.3004 | 5.3004 | 5.5707 | -11.2850 |
| 8 | `M1-A03391` | 1 | 47.19 | 43.11 | -4.0830 | 4.0830 | 4.0830 | N/A |
| 9 | `M1-T00011-M` | 5 | 40.57 | 40.52 | -0.0503 | 3.5499 | 3.8268 | -0.6006 |
| 10 | `M1-B00458RIL` | 1 | 40.50 | 36.77 | -3.7316 | 3.7316 | 3.7316 | N/A |
| 11 | `M1-B00163XV` | 1 | 35.88 | 32.36 | -3.5205 | 3.5205 | 3.5205 | N/A |
| 12 | `M1-B15563R1` | 3 | 38.42 | 38.96 | 0.5352 | 3.2199 | 3.2707 | -0.4314 |
| 13 | `M1-T05073XV` | 1 | 58.10 | 54.97 | -3.1271 | 3.1271 | 3.1271 | N/A |
| 14 | `M1-B00163R4` | 5 | 32.64 | 31.89 | -0.7568 | 2.2613 | 2.9801 | 0.3933 |
| 15 | `M1-T00011XV` | 2 | 37.75 | 40.65 | 2.9033 | 2.9033 | 2.9722 | -16.0405 |

*注：对于仅有1个测试样本或方差为0的系列，$R^2$ 显示为 N/A。*
