# 胶料门尼粘度 (MNY) 预测模型开发与评估报告 (Without-Oil High-Silica 胶料模型)
    
本报告详细说明了针对 **Without-Oil High-Silica** 胶料数据集进行门尼粘度（MNY）回归建模的评估结果。

---

## 1. 建模数据概况

- **数据集类型**：Without-Oil High-Silica 胶料数据（is_oil_loading_present = 0.0）
- **总车次行数**：660 车
- **异常值清理**：使用**孤立森林 (Isolation Forest) 算法**对温度、功率、扭矩和持续时间等核心曲线特征进行多维异常诊断。
- **清洗后带有门尼粘度标签的车次**：565 车
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
| Ridge Regression | -0.0406 | 2.7696 | 3.5311 |
| Random Forest | 0.0780 | 2.5661 | 3.3185 |
| Gradient Boosting | 0.0112 | 2.7101 | 3.4431 |
| XGBoost (Baseline) | -0.0885 | 2.7755 | 3.6173 |
| LightGBM (Huber) | 0.0346 | 2.6393 | 3.3917 |
| HistGradientBoosting | -0.0036 | 2.6870 | 3.4633 |
| CatBoost (Baseline) | 0.0508 | 2.5968 | 3.3638 |
| LightGBM (Tuned) | 0.0442 | 2.6491 | 3.3875 |
| XGBoost (Tuned) | 0.0755 | 2.5959 | 3.3304 |

*注：$R^2$ 越接近 1.0 预测拟合效果越好。MAE 代表门尼粘度预测值与实测值的绝对平均偏差值。*

---

## 3. Optuna 超参数优化与测试集最终表现

最优模型为 **Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF)**，其最优超参数集为：
- `best_params`: `LGBM: {'n_estimators': 785, 'max_depth': 3, 'learning_rate': 0.015572598877754269, 'num_leaves': 36, 'min_child_samples': 69, 'min_split_gain': 0.32700761377093235, 'subsample': 0.9420159183468965, 'colsample_bytree': 0.5935184429826071, 'reg_alpha': 0.00012267832224099664, 'reg_lambda': 2.3556084786281177} | XGB: {'n_estimators': 310, 'max_depth': 4, 'learning_rate': 0.007641031406669726, 'subsample': 0.7902025121815949, 'colsample_bytree': 0.6174576708322543, 'min_child_weight': 9.575700512453572, 'gamma': 1.5318732775580504, 'reg_alpha': 0.055763880693077264, 'reg_lambda': 39.58477591968949}`

### 3.1 测试集（Held-out Test Set, 20% 未参训数据）的最终表现：
- **测试集决定系数 $R^2$**：**0.7032**
- **测试集平均绝对误差 (MAE)**：**3.3730**
- **测试集均方根误差 (RMSE)**：**4.9709**

### 3.2 预测效果 Parity 对齐图
![Mooney预测Parity图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_without_oil_high_silica/mooney_model_parity_plot.png)

---

## 4. 特征贡献度分析 (Feature Importance)

以下是经过调优后的 Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF) 模型在预测门尼粘度时，排名前 15 位的核心物理特征：

| 排名 | 特征名称 | 贡献权重 (Importance) | 物理流变学解释 |
| :---: | :--- | :---: | :--- |
| 1 | `roll_mny_mean_3b_same_comp` | 0.0932 | 流变/物理参数描述 |
| 2 | `roll_resid_mean_10b_mixer` | 0.0661 | 流变/物理参数描述 |
| 3 | `roll_resid_mean_5b_family` | 0.0577 | 流变/物理参数描述 |
| 4 | `supplier_carbon_black_structure_avg` | 0.0495 | 流变/物理参数描述 |
| 5 | `Stage5_PID_WayofRam_Std` | 0.0466 | 流变/物理参数描述 |
| 6 | `Stage6_BottomMixing_power_Mean` | 0.0462 | 流变/物理参数描述 |
| 7 | `Stage2_DryMixing_power_Integral` | 0.0432 | 流变/物理参数描述 |
| 8 | `Stage2_DryMixing_temp_Std` | 0.0421 | 流变/物理参数描述 |
| 9 | `roll_mny_std_10b_same_comp` | 0.0417 | 流变/物理参数描述 |
| 10 | `weight_pct_silica` | 0.0376 | 配方中该组分的重量百分比 |
| 11 | `Stage6_BottomMixing_Torque_Mean` | 0.0367 | Stage6阶段扭矩平均值 |
| 12 | `Stage1_Loading_RotorSpeed_Integral` | 0.0356 | Stage1阶段剪切历史积分（转子总圈数） |
| 13 | `phys_temp_rise_rate` | 0.0336 | 流变学计算特征 |
| 14 | `Stage1_Loading_WayofRam_Mean` | 0.0324 | 流变/物理参数描述 |
| 15 | `weight_pct_solid_elastomer` | 0.0288 | 配方中该组分的重量百分比 |


### 4.2 核心贡献度图解
![Mooney特征重要性图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_without_oil_high_silica/mooney_model_feature_importance.png)

---

## 5. 工艺调整解释表

下表将模型重要性、单调相关趋势和密炼工艺含义放在一起，用于预测完成后的工艺诊断。注意：该方向代表训练数据中的统计趋势，实际调整必须结合胶料体系、配方窗口和现场约束确认。

| 排名 | 特征名称 | 模型重要性 | Spearman趋势 | 趋势解释 | 工艺含义/调整方向 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `roll_mny_mean_3b_same_comp` | 0.0932 | 0.2036 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 2 | `roll_resid_mean_10b_mixer` | 0.0661 | 0.2373 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 3 | `roll_resid_mean_5b_family` | 0.0577 | 0.1869 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 4 | `supplier_carbon_black_structure_avg` | 0.0495 | 0.1984 | 该特征升高时，模型/数据倾向于 MNY 升高 | 炭黑结构值决定了吸留橡胶比例与填充网络刚性，吸油值偏高会直接导致混炼门尼升高，需核对该批次炭黑指标。 |
| 5 | `Stage5_PID_WayofRam_Std` | 0.0466 | 0.0875 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 6 | `Stage6_BottomMixing_power_Mean` | 0.0462 | 0.0022 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 7 | `Stage2_DryMixing_power_Integral` | 0.0432 | -0.0598 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 8 | `Stage2_DryMixing_temp_Std` | 0.0421 | -0.0793 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 9 | `roll_mny_std_10b_same_comp` | 0.0417 | -0.1305 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 10 | `weight_pct_silica` | 0.0376 | 0.0449 | 单调方向较弱，更多体现交互或分群作用 | 白炭黑含量提高会增强填料网络和反应需求，需配合硅烷、温度积分和 PID 时间解释。 |
| 11 | `Stage6_BottomMixing_Torque_Mean` | 0.0367 | -0.0632 | 单调方向较弱，更多体现交互或分群作用 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 12 | `Stage1_Loading_RotorSpeed_Integral` | 0.0356 | -0.0111 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 13 | `phys_temp_rise_rate` | 0.0336 | -0.0214 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 14 | `Stage1_Loading_WayofRam_Mean` | 0.0324 | -0.0932 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 15 | `weight_pct_solid_elastomer` | 0.0288 | -0.0168 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |


完整明细见：`process_adjustment_guidance.csv`


---

## 6. 各配方系列 (Compound Family) 的预测表现

以下是测试集中前 15 个主要胶料配方系列的表现统计（按 RMSE 降序排列）：

| 排名 | 配方系列 | 测试车次 | 真实门尼均值 | 预测门尼均值 | 偏差 (Bias) | MAE | RMSE | $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `M1-T18191` | 4 | 99.91 | 89.83 | -10.0799 | 12.1155 | 17.2159 | -9.0957 |
| 2 | `M1-T15215` | 1 | 70.74 | 77.29 | 6.5532 | 6.5532 | 6.5532 | N/A |
| 3 | `M1-T13127` | 14 | 64.06 | 61.62 | -2.4461 | 4.3796 | 4.8596 | -0.3198 |
| 4 | `M1-T01102W2` | 5 | 55.97 | 57.72 | 1.7538 | 3.9772 | 4.7544 | 0.2826 |
| 5 | `M1-T14885` | 9 | 66.38 | 66.94 | 0.5580 | 3.7840 | 4.1844 | 0.2193 |
| 6 | `M1-T25045` | 29 | 62.06 | 61.55 | -0.5101 | 3.2128 | 4.1494 | -0.1033 |
| 7 | `M1-T26104` | 9 | 61.97 | 63.24 | 1.2685 | 3.0117 | 3.8408 | 0.0883 |
| 8 | `M1-T25045-M` | 1 | 60.14 | 57.10 | -3.0406 | 3.0406 | 3.0406 | N/A |
| 9 | `M1-T11347W4` | 10 | 55.52 | 56.37 | 0.8552 | 2.0839 | 2.6650 | 0.2512 |
| 10 | `M1-T25045-V` | 5 | 60.79 | 58.87 | -1.9281 | 2.1101 | 2.5958 | -0.2543 |
| 11 | `M1-T25088` | 8 | 58.64 | 58.06 | -0.5883 | 2.2326 | 2.5581 | -0.4302 |
| 12 | `M1-T11923` | 1 | 44.82 | 47.10 | 2.2780 | 2.2780 | 2.2780 | N/A |
| 13 | `M1-T14885B3` | 5 | 71.53 | 71.85 | 0.3129 | 2.0089 | 2.1812 | 0.6930 |
| 14 | `M1-T16734` | 8 | 61.45 | 62.12 | 0.6684 | 1.5621 | 2.1312 | 0.7537 |

*注：对于仅有1个测试样本或方差为0的系列，$R^2$ 显示为 N/A。*
