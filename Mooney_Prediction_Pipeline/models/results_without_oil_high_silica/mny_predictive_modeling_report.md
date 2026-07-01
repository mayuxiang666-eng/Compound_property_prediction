# 胶料门尼粘度 (MNY) 预测模型开发与评估报告 (Without-Oil High-Silica 胶料模型)
    
本报告详细说明了针对 **Without-Oil High-Silica** 胶料数据集进行门尼粘度（MNY）回归建模的评估结果。

---

## 1. 建模数据概况

- **数据集类型**：Without-Oil High-Silica 胶料数据（is_oil_loading_present = 0.0）
- **总车次行数**：463 车
- **异常值清理**：使用**孤立森林 (Isolation Forest) 算法**对温度、功率、扭矩和持续时间等核心曲线特征进行多维异常诊断。
- **清洗后带有门尼粘度标签的车次**：454 车
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
| Ridge Regression | -0.0128 | 2.5444 | 3.2922 |
| Random Forest | 0.1231 | 2.3671 | 3.0600 |
| Gradient Boosting | -0.0006 | 2.5440 | 3.2596 |
| XGBoost (Baseline) | -0.0618 | 2.6529 | 3.3515 |
| LightGBM (Huber) | 0.1123 | 2.3469 | 3.0794 |
| HistGradientBoosting | 0.0794 | 2.4608 | 3.1328 |
| CatBoost (Baseline) | 0.0957 | 2.3928 | 3.1142 |
| LightGBM (Tuned) | 0.0751 | 2.4010 | 3.1506 |
| XGBoost (Tuned) | 0.1252 | 2.3593 | 3.0608 |

*注：$R^2$ 越接近 1.0 预测拟合效果越好。MAE 代表门尼粘度预测值与实测值的绝对平均偏差值。*

---

## 3. Optuna 超参数优化与测试集最终表现

最优模型为 **Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF)**，其最优超参数集为：
- `best_params`: `LGBM: {'n_estimators': 408, 'max_depth': 6, 'learning_rate': 0.03342176486520334, 'num_leaves': 47, 'min_child_samples': 35, 'min_split_gain': 1.4966143672906531, 'subsample': 0.7498153399483386, 'colsample_bytree': 0.746268417420161, 'reg_alpha': 0.0251730217141128, 'reg_lambda': 5.738299439899682} | XGB: {'n_estimators': 256, 'max_depth': 2, 'learning_rate': 0.01914579674036364, 'subsample': 0.7034755599252758, 'colsample_bytree': 0.4613651539574399, 'min_child_weight': 3.338416672333195, 'gamma': 2.6764454019679347, 'reg_alpha': 0.0025041728097637893, 'reg_lambda': 2.7021859358977447}`

### 3.1 测试集（Held-out Test Set, 20% 未参训数据）的最终表现：
- **测试集决定系数 $R^2$**：**0.5675**
- **测试集平均绝对误差 (MAE)**：**2.8199**
- **测试集均方根误差 (RMSE)**：**3.6596**

### 3.2 预测效果 Parity 对齐图
![Mooney预测Parity图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_without_oil_high_silica/mooney_model_parity_plot.png)

---

## 4. 特征贡献度分析 (Feature Importance)

以下是经过调优后的 Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF) 模型在预测门尼粘度时，排名前 15 位的核心物理特征：

| 排名 | 特征名称 | 贡献权重 (Importance) | 物理流变学解释 |
| :---: | :--- | :---: | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.0897 | 流变/物理参数描述 |
| 2 | `roll_mny_mean_3b_same_comp` | 0.0740 | 流变/物理参数描述 |
| 3 | `Stage2_DryMixing_temp_Std` | 0.0731 | 流变/物理参数描述 |
| 4 | `Stage5_PID_power_Integral` | 0.0646 | 流变/物理参数描述 |
| 5 | `roll_resid_mean_10b_mixer` | 0.0627 | 流变/物理参数描述 |
| 6 | `roll_mny_std_10b_same_comp` | 0.0529 | 流变/物理参数描述 |
| 7 | `phys_temp_rise_rate` | 0.0470 | 流变学计算特征 |
| 8 | `Stage1_Loading_WayofRam_Mean` | 0.0408 | 流变/物理参数描述 |
| 9 | `Stage6_BottomMixing_power_Mean` | 0.0385 | 流变/物理参数描述 |
| 10 | `supplier_silica_moisture_avg` | 0.0355 | 流变/物理参数描述 |
| 11 | `Stage2_DryMixing_WayofRam_Std` | 0.0337 | 流变/物理参数描述 |
| 12 | `supplier_carbon_black_structure_avg` | 0.0322 | 流变/物理参数描述 |
| 13 | `Stage2_DryMixing_power_Integral` | 0.0306 | 流变/物理参数描述 |
| 14 | `phys_init_temp` | 0.0299 | 流变学计算特征 |
| 15 | `weight_pct_silian` | 0.0283 | 配方中该组分的重量百分比 |


### 4.2 核心贡献度图解
![Mooney特征重要性图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_without_oil_high_silica/mooney_model_feature_importance.png)

---

## 5. 工艺调整解释表

下表将模型重要性、单调相关趋势和密炼工艺含义放在一起，用于预测完成后的工艺诊断。注意：该方向代表训练数据中的统计趋势，实际调整必须结合胶料体系、配方窗口和现场约束确认。

| 排名 | 特征名称 | 模型重要性 | Spearman趋势 | 趋势解释 | 工艺含义/调整方向 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.0897 | 0.2428 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 2 | `roll_mny_mean_3b_same_comp` | 0.0740 | 0.2394 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 3 | `Stage2_DryMixing_temp_Std` | 0.0731 | -0.1303 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 4 | `Stage5_PID_power_Integral` | 0.0646 | -0.1154 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 5 | `roll_resid_mean_10b_mixer` | 0.0627 | 0.2014 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 6 | `roll_mny_std_10b_same_comp` | 0.0529 | -0.1141 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 7 | `phys_temp_rise_rate` | 0.0470 | -0.0981 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 8 | `Stage1_Loading_WayofRam_Mean` | 0.0408 | -0.0342 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 9 | `Stage6_BottomMixing_power_Mean` | 0.0385 | -0.0076 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 10 | `supplier_silica_moisture_avg` | 0.0355 | -0.0228 | 单调方向较弱，更多体现交互或分群作用 | 白炭黑含水率是硅烷化偶联反应的关键催化剂，过低或过高均会干扰反应程度从而引起门尼异常，需结合硅烷、温升等排查。 |
| 11 | `Stage2_DryMixing_WayofRam_Std` | 0.0337 | -0.0586 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 12 | `supplier_carbon_black_structure_avg` | 0.0322 | 0.1460 | 该特征升高时，模型/数据倾向于 MNY 升高 | 炭黑结构值决定了吸留橡胶比例与填充网络刚性，吸油值偏高会直接导致混炼门尼升高，需核对该批次炭黑指标。 |
| 13 | `Stage2_DryMixing_power_Integral` | 0.0306 | -0.0029 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 14 | `phys_init_temp` | 0.0299 | 0.0518 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 15 | `weight_pct_silian` | 0.0283 | 0.0157 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |


完整明细见：`process_adjustment_guidance.csv`


---

## 6. 各配方系列 (Compound Family) 的预测表现

以下是测试集中前 15 个主要胶料配方系列的表现统计（按 RMSE 降序排列）：

| 排名 | 配方系列 | 测试车次 | 真实门尼均值 | 预测门尼均值 | 偏差 (Bias) | MAE | RMSE | $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `M1-T19808` | 1 | 48.33 | 61.96 | 13.6268 | 13.6268 | 13.6268 | N/A |
| 2 | `M1-T15215` | 2 | 73.20 | 79.62 | 6.4145 | 6.4145 | 6.8532 | -6.7296 |
| 3 | `M1-T25088` | 9 | 58.50 | 58.37 | -0.1296 | 3.2759 | 3.9973 | 0.0605 |
| 4 | `M1-T26104` | 8 | 60.24 | 62.73 | 2.4901 | 3.0967 | 3.6677 | -0.9992 |
| 5 | `M1-T25045` | 22 | 60.40 | 61.24 | 0.8417 | 2.7469 | 3.4541 | 0.2357 |
| 6 | `M1-T13127` | 16 | 62.51 | 62.67 | 0.1603 | 3.0401 | 3.4360 | 0.1735 |
| 7 | `M1-T11923` | 1 | 50.97 | 47.57 | -3.4038 | 3.4038 | 3.4038 | N/A |
| 8 | `M1-T14885` | 2 | 69.21 | 66.19 | -3.0200 | 3.0200 | 3.2468 | -0.9252 |
| 9 | `M1-T16734` | 5 | 64.70 | 62.57 | -2.1323 | 2.5200 | 3.2402 | 0.4033 |
| 10 | `M1-T25045-V` | 1 | 62.27 | 59.37 | -2.9030 | 2.9030 | 2.9030 | N/A |
| 11 | `M1-T14885B3` | 2 | 75.69 | 73.30 | -2.3855 | 2.3855 | 2.5153 | -11.9116 |
| 12 | `M1-T01102W2` | 6 | 58.30 | 58.07 | -0.2260 | 1.3162 | 1.8489 | -0.0943 |
| 13 | `M1-T11347W4` | 9 | 55.16 | 55.23 | 0.0618 | 1.3287 | 1.7974 | 0.5954 |
| 14 | `M1-T25045-M` | 1 | 58.92 | 58.59 | -0.3305 | 0.3305 | 0.3305 | N/A |

*注：对于仅有1个测试样本或方差为0的系列，$R^2$ 显示为 N/A。*
