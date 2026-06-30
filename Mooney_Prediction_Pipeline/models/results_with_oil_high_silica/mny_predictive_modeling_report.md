# 胶料门尼粘度 (MNY) 预测模型开发与评估报告 (With-Oil High-Silica 胶料模型)
    
本报告详细说明了针对 **With-Oil High-Silica** 胶料数据集进行门尼粘度（MNY）回归建模的评估结果。

---

## 1. 建模数据概况

- **数据集类型**：With-Oil High-Silica 胶料数据（is_oil_loading_present = 1.0）
- **总车次行数**：2439 车
- **异常值清理**：使用**孤立森林 (Isolation Forest) 算法**对温度、功率、扭矩和持续时间等核心曲线特征进行多维异常诊断。
- **清洗后带有门尼粘度标签的车次**：2160 车
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
| Ridge Regression | 0.2035 | 2.3809 | 3.1479 |
| Random Forest | 0.2709 | 2.2445 | 3.0126 |
| Gradient Boosting | 0.2448 | 2.2845 | 3.0657 |
| XGBoost (Baseline) | 0.1635 | 2.4240 | 3.2253 |
| LightGBM (Huber) | 0.2547 | 2.2603 | 3.0448 |
| HistGradientBoosting | 0.2200 | 2.3385 | 3.1157 |
| CatBoost (Baseline) | 0.2435 | 2.3021 | 3.0676 |
| LightGBM (Tuned) | 0.2535 | 2.2740 | 3.0477 |
| XGBoost (Tuned) | 0.2885 | 2.2206 | 2.9757 |

*注：$R^2$ 越接近 1.0 预测拟合效果越好。MAE 代表门尼粘度预测值与实测值的绝对平均偏差值。*

---

## 3. Optuna 超参数优化与测试集最终表现

最优模型为 **Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF)**，其最优超参数集为：
- `best_params`: `LGBM: {'n_estimators': 545, 'max_depth': 6, 'learning_rate': 0.04350174496293259, 'num_leaves': 7, 'min_child_samples': 47, 'min_split_gain': 1.0420852612259477, 'subsample': 0.667374783884946, 'colsample_bytree': 0.5137229419248488, 'reg_alpha': 0.0010363397675026372, 'reg_lambda': 17.84885321949675} | XGB: {'n_estimators': 677, 'max_depth': 6, 'learning_rate': 0.009618604722782897, 'subsample': 0.6559174463197828, 'colsample_bytree': 0.450618665256212, 'min_child_weight': 3.533760838349414, 'gamma': 2.163995002549626, 'reg_alpha': 0.009161176915463915, 'reg_lambda': 2.3775297453615707}`

### 3.1 测试集（Held-out Test Set, 20% 未参训数据）的最终表现：
- **测试集决定系数 $R^2$**：**0.8290**
- **测试集平均绝对误差 (MAE)**：**2.4059**
- **测试集均方根误差 (RMSE)**：**3.4528**

### 3.2 预测效果 Parity 对齐图
![Mooney预测Parity图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_with_oil_high_silica/mooney_model_parity_plot.png)

---

## 4. 特征贡献度分析 (Feature Importance)

以下是经过调优后的 Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF) 模型在预测门尼粘度时，排名前 15 位的核心物理特征：

| 排名 | 特征名称 | 贡献权重 (Importance) | 物理流变学解释 |
| :---: | :--- | :---: | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.1391 | 流变/物理参数描述 |
| 2 | `roll_resid_mean_10b_mixer` | 0.0883 | 流变/物理参数描述 |
| 3 | `supplier_silica_surface_area_avg` | 0.0605 | 流变/物理参数描述 |
| 4 | `roll_mny_mean_3b_same_comp` | 0.0505 | 流变/物理参数描述 |
| 5 | `roll_mny_std_10b_same_comp` | 0.0442 | 流变/物理参数描述 |
| 6 | `Stage4_WetMixing_power_Mean` | 0.0440 | 流变/物理参数描述 |
| 7 | `phys_init_temp` | 0.0354 | 流变学计算特征 |
| 8 | `phys_temp_integral_above_100` | 0.0333 | 总混炼温度-时间积分（热历史） |
| 9 | `Stage6_BottomMixing_Torque_Integral` | 0.0329 | Stage6阶段扭矩累积积分（功历史） |
| 10 | `Stage3_OilLoading_temp_Mean` | 0.0325 | 流变/物理参数描述 |
| 11 | `Stage6_BottomMixing_power_Integral` | 0.0304 | 流变/物理参数描述 |
| 12 | `Stage6_BottomMixing_Torque_Mean` | 0.0300 | Stage6阶段扭矩平均值 |
| 13 | `weight_pct_silian` | 0.0283 | 配方中该组分的重量百分比 |
| 14 | `Stage3_OilLoading_temp_Std` | 0.0274 | 流变/物理参数描述 |
| 15 | `Stage6_BottomMixing_power_Mean` | 0.0268 | 流变/物理参数描述 |


### 4.2 核心贡献度图解
![Mooney特征重要性图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_with_oil_high_silica/mooney_model_feature_importance.png)

---

## 5. 工艺调整解释表

下表将模型重要性、单调相关趋势和密炼工艺含义放在一起，用于预测完成后的工艺诊断。注意：该方向代表训练数据中的统计趋势，实际调整必须结合胶料体系、配方窗口和现场约束确认。

| 排名 | 特征名称 | 模型重要性 | Spearman趋势 | 趋势解释 | 工艺含义/调整方向 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.1391 | 0.4067 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 2 | `roll_resid_mean_10b_mixer` | 0.0883 | 0.3739 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 3 | `supplier_silica_surface_area_avg` | 0.0605 | 0.1245 | 该特征升高时，模型/数据倾向于 MNY 升高 | 白炭黑比表面积决定了偶联活性位点与填料网络强弱，异常偏高会增强粒子自凝聚进而推高门尼，应核对批次检测。 |
| 4 | `roll_mny_mean_3b_same_comp` | 0.0505 | 0.2069 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 5 | `roll_mny_std_10b_same_comp` | 0.0442 | 0.0522 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 6 | `Stage4_WetMixing_power_Mean` | 0.0440 | -0.0107 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 7 | `phys_init_temp` | 0.0354 | 0.0727 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 8 | `phys_temp_integral_above_100` | 0.0333 | -0.0173 | 单调方向较弱，更多体现交互或分群作用 | 高温积分代表热反应历史，白炭黑/硅烷体系中过高可能导致反应过度，过低可能反应不足。 |
| 9 | `Stage6_BottomMixing_Torque_Integral` | 0.0329 | 0.0306 | 单调方向较弱，更多体现交互或分群作用 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 10 | `Stage3_OilLoading_temp_Mean` | 0.0325 | -0.0820 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 11 | `Stage6_BottomMixing_power_Integral` | 0.0304 | 0.0315 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 12 | `Stage6_BottomMixing_Torque_Mean` | 0.0300 | 0.0362 | 单调方向较弱，更多体现交互或分群作用 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 13 | `weight_pct_silian` | 0.0283 | 0.0240 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 14 | `Stage3_OilLoading_temp_Std` | 0.0274 | -0.0399 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 15 | `Stage6_BottomMixing_power_Mean` | 0.0268 | 0.0754 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |


完整明细见：`process_adjustment_guidance.csv`


---

## 6. 各配方系列 (Compound Family) 的预测表现

以下是测试集中前 15 个主要胶料配方系列的表现统计（按 RMSE 降序排列）：

| 排名 | 配方系列 | 测试车次 | 真实门尼均值 | 预测门尼均值 | 偏差 (Bias) | MAE | RMSE | $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `M1-T17771X6` | 1 | 32.55 | 53.33 | 20.7829 | 20.7829 | 20.7829 | N/A |
| 2 | `M1-T15611W2` | 1 | 32.77 | 52.69 | 19.9232 | 19.9232 | 19.9232 | N/A |
| 3 | `M1-T19012` | 1 | 112.97 | 104.75 | -8.2209 | 8.2209 | 8.2209 | N/A |
| 4 | `M1-T15192R` | 3 | 67.55 | 64.99 | -2.5541 | 6.1205 | 7.8348 | -1.1261 |
| 5 | `M1-T33025XP` | 1 | 46.89 | 53.90 | 7.0043 | 7.0043 | 7.0043 | N/A |
| 6 | `M1-T33025XA` | 1 | 46.57 | 53.34 | 6.7690 | 6.7690 | 6.7690 | N/A |
| 7 | `M1-T09170WZ` | 1 | 59.18 | 52.95 | -6.2313 | 6.2313 | 6.2313 | N/A |
| 8 | `M1-T15899` | 19 | 49.20 | 48.65 | -0.5482 | 4.4081 | 5.3494 | -0.1308 |
| 9 | `M1-T17732` | 3 | 58.14 | 61.74 | 3.5966 | 3.6141 | 4.7993 | -1.8491 |
| 10 | `M1-T15192W6` | 2 | 57.98 | 53.48 | -4.4960 | 4.4960 | 4.6056 | -383.0921 |
| 11 | `M1-T20153` | 2 | 72.44 | 70.17 | -2.2708 | 3.7011 | 4.3422 | -0.4114 |
| 12 | `M1-T19065` | 11 | 50.78 | 50.94 | 0.1601 | 3.7124 | 4.1760 | -0.1079 |
| 13 | `M1-T01139XZ` | 1 | 57.36 | 53.19 | -4.1692 | 4.1692 | 4.1692 | N/A |
| 14 | `M1-T15554` | 2 | 33.03 | 29.57 | -3.4641 | 3.4641 | 4.1526 | -13.2515 |
| 15 | `M1-T02128W5` | 3 | 60.67 | 64.70 | 4.0250 | 4.0250 | 4.0392 | -13.0981 |

*注：对于仅有1个测试样本或方差为0的系列，$R^2$ 显示为 N/A。*
