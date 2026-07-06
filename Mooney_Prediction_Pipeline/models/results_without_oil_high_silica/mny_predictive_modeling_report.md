# 胶料门尼粘度 (MNY) 预测模型开发与评估报告 (Without-Oil High-Silica 胶料模型)
    
本报告详细说明了针对 **Without-Oil High-Silica** 胶料数据集进行门尼粘度（MNY）回归建模的评估结果。

---

## 1. 建模数据概况

- **数据集类型**：Without-Oil High-Silica 胶料数据（is_oil_loading_present = 0.0）
- **总车次行数**：802 车
- **异常值清理**：使用**孤立森林 (Isolation Forest) 算法**对温度、功率、扭矩和持续时间等核心曲线特征进行多维异常诊断。
- **清洗后带有门尼粘度标签的车次**：797 车
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
| Ridge Regression | -0.0840 | 2.6829 | 3.4487 |
| Random Forest | 0.0662 | 2.4869 | 3.1983 |
| Gradient Boosting | -0.0193 | 2.6131 | 3.3429 |
| XGBoost (Baseline) | -0.0619 | 2.6639 | 3.4095 |
| LightGBM (Huber) | 0.0185 | 2.5479 | 3.2799 |
| HistGradientBoosting | -0.0714 | 2.6767 | 3.4252 |
| CatBoost (Baseline) | 0.0018 | 2.5934 | 3.3062 |
| LightGBM (Tuned) | 0.0447 | 2.5207 | 3.2388 |
| XGBoost (Tuned) | 0.0669 | 2.4976 | 3.1995 |

*注：$R^2$ 越接近 1.0 预测拟合效果越好。MAE 代表门尼粘度预测值与实测值的绝对平均偏差值。*

---

## 3. Optuna 超参数优化与测试集最终表现

最优模型为 **Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF)**，其最优超参数集为：
- `best_params`: `LGBM: {'n_estimators': 899, 'max_depth': 4, 'learning_rate': 0.04785022361807594, 'num_leaves': 7, 'min_child_samples': 51, 'min_split_gain': 2.9173359487587667, 'subsample': 0.9335174670152472, 'colsample_bytree': 0.7903150075085079, 'reg_alpha': 0.00028575292040511146, 'reg_lambda': 40.58197593637454} | XGB: {'n_estimators': 414, 'max_depth': 3, 'learning_rate': 0.006961382460175267, 'subsample': 0.9148275219608636, 'colsample_bytree': 0.8464355321097602, 'min_child_weight': 19.9083381221902, 'gamma': 4.32743065947476, 'reg_alpha': 0.008761958481137106, 'reg_lambda': 41.8451677903281}`

### 3.1 测试集（Held-out Test Set, 20% 未参训数据）的最终表现：
- **测试集决定系数 $R^2$**：**0.5271**
- **测试集平均绝对误差 (MAE)**：**3.0806**
- **测试集均方根误差 (RMSE)**：**3.8151**

### 3.2 预测效果 Parity 对齐图
![Mooney预测Parity图](file:///c:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_without_oil_high_silica/mooney_model_parity_plot.png)

---

## 4. 特征贡献度分析 (Feature Importance)

以下是经过调优后的 Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF) 模型在预测门尼粘度时，排名前 15 位的核心物理特征：

| 排名 | 特征名称 | 贡献权重 (Importance) | 物理流变学解释 |
| :---: | :--- | :---: | :--- |
| 1 | `roll_mny_mean_3b_same_comp` | 0.1364 | 流变/物理参数描述 |
| 2 | `Stage5_PID_power_Integral` | 0.0929 | 流变/物理参数描述 |
| 3 | `roll_resid_mean_10b_mixer` | 0.0821 | 流变/物理参数描述 |
| 4 | `supplier_silica_surface_area_avg` | 0.0657 | 流变/物理参数描述 |
| 5 | `Stage1_Loading_WayofRam_Mean` | 0.0471 | 流变/物理参数描述 |
| 6 | `supplier_rubber_viscosity_avg` | 0.0421 | 供应商生胶出厂粘度平均值 |
| 7 | `supplier_carbon_black_structure_avg` | 0.0421 | 流变/物理参数描述 |
| 8 | `Stage2_DryMixing_temp_Std` | 0.0370 | 流变/物理参数描述 |
| 9 | `roll_resid_mean_5b_family` | 0.0336 | 流变/物理参数描述 |
| 10 | `Stage1_Loading_RotorSpeed_Integral` | 0.0324 | Stage1阶段剪切历史积分（转子总圈数） |
| 11 | `Stage5_PID_RotorSpeed_Integral` | 0.0297 | Stage5阶段剪切历史积分（转子总圈数） |
| 12 | `weight_pct_silian` | 0.0284 | 配方中该组分的重量百分比 |
| 13 | `roll_mny_std_10b_same_comp` | 0.0257 | 流变/物理参数描述 |
| 14 | `supplier_silica_moisture_avg` | 0.0247 | 流变/物理参数描述 |
| 15 | `Stage5_PID_WayofRam_Std` | 0.0235 | 流变/物理参数描述 |


### 4.2 核心贡献度图解
![Mooney特征重要性图](file:///c:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_without_oil_high_silica/mooney_model_feature_importance.png)

---

## 5. 工艺调整解释表

下表将模型重要性、单调相关趋势和密炼工艺含义放在一起，用于预测完成后的工艺诊断。注意：该方向代表训练数据中的统计趋势，实际调整必须结合胶料体系、配方窗口和现场约束确认。

| 排名 | 特征名称 | 模型重要性 | Spearman趋势 | 趋势解释 | 工艺含义/调整方向 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `roll_mny_mean_3b_same_comp` | 0.1364 | 0.1644 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 2 | `Stage5_PID_power_Integral` | 0.0929 | -0.1260 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 3 | `roll_resid_mean_10b_mixer` | 0.0821 | 0.2122 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 4 | `supplier_silica_surface_area_avg` | 0.0657 | 0.1048 | 该特征升高时，模型/数据倾向于 MNY 升高 | 白炭黑比表面积决定了偶联活性位点与填料网络强弱，异常偏高会增强粒子自凝聚进而推高门尼，应核对批次检测。 |
| 5 | `Stage1_Loading_WayofRam_Mean` | 0.0471 | -0.0355 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 6 | `supplier_rubber_viscosity_avg` | 0.0421 | 0.1474 | 该特征升高时，模型/数据倾向于 MNY 升高 | 生胶供应商粘度是上游遗传因素，偏高时不要只调密炼参数，应同步看原材料批次。 |
| 7 | `supplier_carbon_black_structure_avg` | 0.0421 | 0.1182 | 该特征升高时，模型/数据倾向于 MNY 升高 | 炭黑结构值决定了吸留橡胶比例与填充网络刚性，吸油值偏高会直接导致混炼门尼升高，需核对该批次炭黑指标。 |
| 8 | `Stage2_DryMixing_temp_Std` | 0.0370 | -0.0880 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 9 | `roll_resid_mean_5b_family` | 0.0336 | 0.1473 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 10 | `Stage1_Loading_RotorSpeed_Integral` | 0.0324 | -0.0483 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 11 | `Stage5_PID_RotorSpeed_Integral` | 0.0297 | -0.0448 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 12 | `weight_pct_silian` | 0.0284 | 0.0222 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 13 | `roll_mny_std_10b_same_comp` | 0.0257 | -0.0015 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 14 | `supplier_silica_moisture_avg` | 0.0247 | -0.0577 | 单调方向较弱，更多体现交互或分群作用 | 白炭黑含水率是硅烷化偶联反应的关键催化剂，过低或过高均会干扰反应程度从而引起门尼异常，需结合硅烷、温升等排查。 |
| 15 | `Stage5_PID_WayofRam_Std` | 0.0235 | -0.0090 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |


完整明细见：`process_adjustment_guidance.csv`


---

## 6. 各配方系列 (Compound Family) 的预测表现

以下是测试集中前 15 个主要胶料配方系列的表现统计（按 RMSE 降序排列）：

| 排名 | 配方系列 | 测试车次 | 真实门尼均值 | 预测门尼均值 | 偏差 (Bias) | MAE | RMSE | $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `M1-T15215` | 1 | 67.79 | 60.60 | -7.1877 | 7.1877 | 7.1877 | N/A |
| 2 | `M1-T25088` | 8 | 59.70 | 58.70 | -0.9981 | 4.6464 | 5.4154 | -0.2454 |
| 3 | `M1-T01102W2` | 4 | 55.26 | 59.12 | 3.8575 | 3.9840 | 5.0451 | -2.6584 |
| 4 | `M1-T16734` | 11 | 63.11 | 60.81 | -2.2962 | 3.2080 | 4.4443 | -0.2187 |
| 5 | `M1-T25045` | 27 | 62.05 | 61.88 | -0.1730 | 3.6376 | 4.4408 | 0.0740 |
| 6 | `M1-T26104` | 9 | 60.32 | 62.07 | 1.7491 | 3.4460 | 4.0596 | -0.1598 |
| 7 | `M1-T14885B3` | 14 | 72.13 | 73.16 | 1.0254 | 3.0604 | 3.7649 | 0.1081 |
| 8 | `M1-T14885` | 7 | 65.09 | 66.23 | 1.1382 | 2.7269 | 3.2921 | -0.2022 |
| 9 | `M1-T13127` | 51 | 62.12 | 62.53 | 0.4045 | 2.7345 | 3.2631 | 0.1510 |
| 10 | `M1-T11347W4` | 15 | 55.45 | 55.52 | 0.0761 | 2.1815 | 2.4792 | 0.2791 |
| 11 | `M1-T25045-M` | 3 | 58.03 | 58.27 | 0.2390 | 2.2082 | 2.3477 | 0.0351 |
| 12 | `M1-T25045-V` | 3 | 57.99 | 59.92 | 1.9293 | 1.9293 | 2.2143 | -14.7887 |

*注：对于仅有1个测试样本或方差为0的系列，$R^2$ 显示为 N/A。*
