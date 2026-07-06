# 胶料门尼粘度 (MNY) 预测模型开发与评估报告 (With-Oil High-Silica 胶料模型)
    
本报告详细说明了针对 **With-Oil High-Silica** 胶料数据集进行门尼粘度（MNY）回归建模的评估结果。

---

## 1. 建模数据概况

- **数据集类型**：With-Oil High-Silica 胶料数据（is_oil_loading_present = 1.0）
- **总车次行数**：3034 车
- **异常值清理**：使用**孤立森林 (Isolation Forest) 算法**对温度、功率、扭矩和持续时间等核心曲线特征进行多维异常诊断。
- **清洗后带有门尼粘度标签的车次**：3022 车
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
| Ridge Regression | 0.1829 | 2.2203 | 2.9738 |
| Random Forest | 0.2245 | 2.1470 | 2.8974 |
| Gradient Boosting | 0.1979 | 2.1848 | 2.9463 |
| XGBoost (Baseline) | 0.1494 | 2.2726 | 3.0342 |
| LightGBM (Huber) | 0.2378 | 2.1327 | 2.8721 |
| HistGradientBoosting | 0.1938 | 2.2163 | 2.9531 |
| CatBoost (Baseline) | 0.1740 | 2.2151 | 2.9898 |
| LightGBM (Tuned) | 0.2273 | 2.1430 | 2.8918 |
| XGBoost (Tuned) | 0.2435 | 2.1295 | 2.8615 |

*注：$R^2$ 越接近 1.0 预测拟合效果越好。MAE 代表门尼粘度预测值与实测值的绝对平均偏差值。*

---

## 3. Optuna 超参数优化与测试集最终表现

最优模型为 **Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF)**，其最优超参数集为：
- `best_params`: `LGBM: {'n_estimators': 452, 'max_depth': 5, 'learning_rate': 0.03832345181319785, 'num_leaves': 50, 'min_child_samples': 31, 'min_split_gain': 1.926564000108046, 'subsample': 0.9143524716543098, 'colsample_bytree': 0.8120072640264038, 'reg_alpha': 0.0006383652459906377, 'reg_lambda': 11.368114692864342} | XGB: {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.022977578897095407, 'subsample': 0.8682550933254385, 'colsample_bytree': 0.6526122716531488, 'min_child_weight': 7.171308962208152, 'gamma': 4.1136389661318615, 'reg_alpha': 0.0003834445658852348, 'reg_lambda': 1.8713075914014283}`

### 3.1 测试集（Held-out Test Set, 20% 未参训数据）的最终表现：
- **测试集决定系数 $R^2$**：**0.6840**
- **测试集平均绝对误差 (MAE)**：**2.5701**
- **测试集均方根误差 (RMSE)**：**4.2774**

### 3.2 预测效果 Parity 对齐图
![Mooney预测Parity图](file:///c:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_with_oil_high_silica/mooney_model_parity_plot.png)

---

## 4. 特征贡献度分析 (Feature Importance)

以下是经过调优后的 Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF) 模型在预测门尼粘度时，排名前 15 位的核心物理特征：

| 排名 | 特征名称 | 贡献权重 (Importance) | 物理流变学解释 |
| :---: | :--- | :---: | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.1754 | 流变/物理参数描述 |
| 2 | `roll_resid_mean_10b_mixer` | 0.0678 | 流变/物理参数描述 |
| 3 | `Stage4_WetMixing_Torque_Mean` | 0.0467 | Stage4阶段扭矩平均值 |
| 4 | `supplier_silica_surface_area_avg` | 0.0465 | 流变/物理参数描述 |
| 5 | `phys_temp_integral_above_100` | 0.0384 | 总混炼温度-时间积分（热历史） |
| 6 | `Stage2_DryMixing_power_Mean` | 0.0377 | 流变/物理参数描述 |
| 7 | `Stage3_OilLoading_temp_Std` | 0.0363 | 流变/物理参数描述 |
| 8 | `Stage4_WetMixing_power_Mean` | 0.0354 | 流变/物理参数描述 |
| 9 | `roll_mny_mean_3b_same_comp` | 0.0353 | 流变/物理参数描述 |
| 10 | `roll_mny_std_10b_same_comp` | 0.0347 | 流变/物理参数描述 |
| 11 | `Stage3_OilLoading_temp_Mean` | 0.0343 | 流变/物理参数描述 |
| 12 | `supplier_silica_moisture_avg` | 0.0317 | 流变/物理参数描述 |
| 13 | `Stage6_BottomMixing_Torque_Mean` | 0.0303 | Stage6阶段扭矩平均值 |
| 14 | `supplier_rubber_viscosity_avg` | 0.0285 | 供应商生胶出厂粘度平均值 |
| 15 | `Stage2_DryMixing_power_Integral` | 0.0265 | 流变/物理参数描述 |


### 4.2 核心贡献度图解
![Mooney特征重要性图](file:///c:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_with_oil_high_silica/mooney_model_feature_importance.png)

---

## 5. 工艺调整解释表

下表将模型重要性、单调相关趋势和密炼工艺含义放在一起，用于预测完成后的工艺诊断。注意：该方向代表训练数据中的统计趋势，实际调整必须结合胶料体系、配方窗口和现场约束确认。

| 排名 | 特征名称 | 模型重要性 | Spearman趋势 | 趋势解释 | 工艺含义/调整方向 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.1754 | 0.4148 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 2 | `roll_resid_mean_10b_mixer` | 0.0678 | 0.3649 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 3 | `Stage4_WetMixing_Torque_Mean` | 0.0467 | -0.0260 | 单调方向较弱，更多体现交互或分群作用 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 4 | `supplier_silica_surface_area_avg` | 0.0465 | 0.1174 | 该特征升高时，模型/数据倾向于 MNY 升高 | 白炭黑比表面积决定了偶联活性位点与填料网络强弱，异常偏高会增强粒子自凝聚进而推高门尼，应核对批次检测。 |
| 5 | `phys_temp_integral_above_100` | 0.0384 | -0.0165 | 单调方向较弱，更多体现交互或分群作用 | 高温积分代表热反应历史，白炭黑/硅烷体系中过高可能导致反应过度，过低可能反应不足。 |
| 6 | `Stage2_DryMixing_power_Mean` | 0.0377 | -0.0432 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 7 | `Stage3_OilLoading_temp_Std` | 0.0363 | -0.0512 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 8 | `Stage4_WetMixing_power_Mean` | 0.0354 | -0.0402 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 9 | `roll_mny_mean_3b_same_comp` | 0.0353 | 0.2018 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 10 | `roll_mny_std_10b_same_comp` | 0.0347 | 0.0380 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 11 | `Stage3_OilLoading_temp_Mean` | 0.0343 | -0.0518 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 12 | `supplier_silica_moisture_avg` | 0.0317 | -0.0678 | 单调方向较弱，更多体现交互或分群作用 | 白炭黑含水率是硅烷化偶联反应的关键催化剂，过低或过高均会干扰反应程度从而引起门尼异常，需结合硅烷、温升等排查。 |
| 13 | `Stage6_BottomMixing_Torque_Mean` | 0.0303 | -0.0148 | 单调方向较弱，更多体现交互或分群作用 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 14 | `supplier_rubber_viscosity_avg` | 0.0285 | 0.1724 | 该特征升高时，模型/数据倾向于 MNY 升高 | 生胶供应商粘度是上游遗传因素，偏高时不要只调密炼参数，应同步看原材料批次。 |
| 15 | `Stage2_DryMixing_power_Integral` | 0.0265 | 0.0145 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |


完整明细见：`process_adjustment_guidance.csv`


---

## 6. 各配方系列 (Compound Family) 的预测表现

以下是测试集中前 15 个主要胶料配方系列的表现统计（按 RMSE 降序排列）：

| 排名 | 配方系列 | 测试车次 | 真实门尼均值 | 预测门尼均值 | 偏差 (Bias) | MAE | RMSE | $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `M1-T19012X1` | 1 | 110.97 | 53.19 | -57.7793 | 57.7793 | 57.7793 | N/A |
| 2 | `M1-T17771X6` | 1 | 32.55 | 53.00 | 20.4477 | 20.4477 | 20.4477 | N/A |
| 3 | `M1-T18035` | 3 | 29.19 | 36.63 | 7.4476 | 9.4890 | 14.7219 | -84.0906 |
| 4 | `M1-T09170X3` | 2 | 57.90 | 52.09 | -5.8108 | 9.4039 | 11.0544 | -0.6034 |
| 5 | `M1-T01139B` | 3 | 63.90 | 58.14 | -5.7588 | 7.9179 | 9.0215 | -13.1453 |
| 6 | `M1-T17771B1` | 3 | 38.35 | 43.30 | 4.9491 | 7.5624 | 8.0071 | -0.6720 |
| 7 | `M1-T20153` | 4 | 69.29 | 76.43 | 7.1430 | 7.1430 | 7.3330 | -22.8855 |
| 8 | `M1-T33025XP` | 1 | 46.89 | 54.20 | 7.3026 | 7.3026 | 7.3026 | N/A |
| 9 | `M1-T17732` | 5 | 65.10 | 60.64 | -4.4652 | 6.0807 | 7.0376 | -0.9838 |
| 10 | `M1-T10035B6` | 1 | 32.33 | 38.09 | 5.7630 | 5.7630 | 5.7630 | N/A |
| 11 | `M1-T16455` | 1 | 44.87 | 50.48 | 5.6111 | 5.6111 | 5.6111 | N/A |
| 12 | `M1-T19065` | 10 | 51.33 | 50.88 | -0.4532 | 3.6622 | 4.7990 | -0.6140 |
| 13 | `M1-T15192R` | 10 | 61.23 | 60.45 | -0.7760 | 3.6534 | 4.3353 | -0.6365 |
| 14 | `M1-T15899` | 23 | 47.84 | 49.71 | 1.8646 | 3.4352 | 4.2559 | -0.6379 |
| 15 | `M1-T15760` | 74 | 57.19 | 56.88 | -0.3052 | 3.3165 | 4.1815 | 0.2202 |

*注：对于仅有1个测试样本或方差为0的系列，$R^2$ 显示为 N/A。*
