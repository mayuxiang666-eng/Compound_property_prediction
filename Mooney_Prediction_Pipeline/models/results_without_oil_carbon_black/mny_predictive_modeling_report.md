# 胶料门尼粘度 (MNY) 预测模型开发与评估报告 (Without-Oil Carbon-Black 胶料模型)
    
本报告详细说明了针对 **Without-Oil Carbon-Black** 胶料数据集进行门尼粘度（MNY）回归建模的评估结果。

---

## 1. 建模数据概况

- **数据集类型**：Without-Oil Carbon-Black 胶料数据（is_oil_loading_present = 0.0）
- **总车次行数**：603 车
- **异常值清理**：使用**孤立森林 (Isolation Forest) 算法**对温度、功率、扭矩和持续时间等核心曲线特征进行多维异常诊断。
- **清洗后带有门尼粘度标签的车次**：558 车
- **特征工程预处理**：
  1. 使用中位数填充缺失值。
  2. 进行标准差标准化（StandardScaler）。
    3. 先执行物理可解释特征审计，剔除高缺失、高单一值、高共线和低优先级特征。
    4. 使用方差过滤（VarianceThreshold，阈值=0.01）移除噪声/恒定特征，过滤后剩余 38 个特征进行训练。

### 1.1 可解释特征筛选

- 候选特征数：38
- 进入紧凑特征集：38
- 最终进入模型训练：38
- 特征审计明细：`interpretable_feature_audit.csv`

---

## 2. 模型选择与交叉验证 (Model Comparison)

使用 **5折交叉验证 (5-Fold CV)** 在训练集上评估了多种回归算法。性能比较如下：

| 模型算法 | 平均 CV $R^2$ (决定系数) | 平均 CV MAE (平均绝对误差) | 平均 CV RMSE (均方根误差) |
| :--- | :---: | :---: | :---: |
| Ridge Regression | -0.0705 | 1.7235 | 2.3914 |
| Random Forest | -0.0693 | 1.6921 | 2.3866 |
| Gradient Boosting | -0.2599 | 1.8547 | 2.5725 |
| XGBoost (Baseline) | -0.1801 | 1.8719 | 2.5004 |
| LightGBM (Huber) | -0.0813 | 1.7290 | 2.4028 |
| HistGradientBoosting | -0.1733 | 1.8529 | 2.4895 |
| CatBoost (Baseline) | -0.1212 | 1.7983 | 2.4322 |
| LightGBM (Tuned) | -0.0187 | 1.5478 | 2.3397 |
| XGBoost (Tuned) | 0.0148 | 1.6005 | 2.2978 |

*注：$R^2$ 越接近 1.0 预测拟合效果越好。MAE 代表门尼粘度预测值与实测值的绝对平均偏差值。*

---

## 3. Optuna 超参数优化与测试集最终表现

最优模型为 **Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF)**，其最优超参数集为：
- `best_params`: `LGBM: {'n_estimators': 408, 'max_depth': 3, 'learning_rate': 0.005445266055373836, 'num_leaves': 55, 'min_child_samples': 126, 'min_split_gain': 0.1709428415821007, 'subsample': 0.9065871174551354, 'colsample_bytree': 0.4685774913476361, 'reg_alpha': 0.0020666197911756593, 'reg_lambda': 3.7835403902491445} | XGB: {'n_estimators': 254, 'max_depth': 6, 'learning_rate': 0.008723586453731714, 'subsample': 0.7479247801134282, 'colsample_bytree': 0.4864166702586976, 'min_child_weight': 3.8005638131563555, 'gamma': 2.454468552866685, 'reg_alpha': 0.1330132157878189, 'reg_lambda': 14.309386575582156}`

### 3.1 测试集（Held-out Test Set, 20% 未参训数据）的最终表现：
- **测试集决定系数 $R^2$**：**0.8070**
- **测试集平均绝对误差 (MAE)**：**4.1784**
- **测试集均方根误差 (RMSE)**：**7.2902**

### 3.2 预测效果 Parity 对齐图
![Mooney预测Parity图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_without_oil_carbon_black/mooney_model_parity_plot.png)

---

## 4. 特征贡献度分析 (Feature Importance)

以下是经过调优后的 Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF) 模型在预测门尼粘度时，排名前 15 位的核心物理特征：

| 排名 | 特征名称 | 贡献权重 (Importance) | 物理流变学解释 |
| :---: | :--- | :---: | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.1168 | 流变/物理参数描述 |
| 2 | `Stage5_PID_power_Integral` | 0.0689 | 流变/物理参数描述 |
| 3 | `phys_temp_integral_above_100` | 0.0619 | 总混炼温度-时间积分（热历史） |
| 4 | `Stage1_Loading_WayofRam_Mean` | 0.0587 | 流变/物理参数描述 |
| 5 | `roll_mny_mean_3b_same_comp` | 0.0577 | 流变/物理参数描述 |
| 6 | `phys_temp_rise_rate` | 0.0542 | 流变学计算特征 |
| 7 | `Stage6_BottomMixing_Torque_Mean` | 0.0536 | Stage6阶段扭矩平均值 |
| 8 | `roll_resid_mean_10b_mixer` | 0.0480 | 流变/物理参数描述 |
| 9 | `Stage6_BottomMixing_power_Mean` | 0.0421 | 流变/物理参数描述 |
| 10 | `Stage5_PID_RotorSpeed_Integral` | 0.0399 | Stage5阶段剪切历史积分（转子总圈数） |
| 11 | `Stage5_PID_WayofRam_Std` | 0.0365 | 流变/物理参数描述 |
| 12 | `phys_init_temp` | 0.0359 | 流变学计算特征 |
| 13 | `Stage2_DryMixing_temp_Std` | 0.0354 | 流变/物理参数描述 |
| 14 | `Stage2_DryMixing_WayofRam_Std` | 0.0346 | 流变/物理参数描述 |
| 15 | `roll_mny_std_10b_same_comp` | 0.0313 | 流变/物理参数描述 |


### 4.2 核心贡献度图解
![Mooney特征重要性图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_without_oil_carbon_black/mooney_model_feature_importance.png)

---

## 5. 工艺调整解释表

下表将模型重要性、单调相关趋势和密炼工艺含义放在一起，用于预测完成后的工艺诊断。注意：该方向代表训练数据中的统计趋势，实际调整必须结合胶料体系、配方窗口和现场约束确认。

| 排名 | 特征名称 | 模型重要性 | Spearman趋势 | 趋势解释 | 工艺含义/调整方向 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.1168 | -0.0801 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 2 | `Stage5_PID_power_Integral` | 0.0689 | 0.0308 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 3 | `phys_temp_integral_above_100` | 0.0619 | -0.0312 | 单调方向较弱，更多体现交互或分群作用 | 高温积分代表热反应历史，白炭黑/硅烷体系中过高可能导致反应过度，过低可能反应不足。 |
| 4 | `Stage1_Loading_WayofRam_Mean` | 0.0587 | -0.0279 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 5 | `roll_mny_mean_3b_same_comp` | 0.0577 | 0.0149 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 6 | `phys_temp_rise_rate` | 0.0542 | 0.0378 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 7 | `Stage6_BottomMixing_Torque_Mean` | 0.0536 | 0.0190 | 单调方向较弱，更多体现交互或分群作用 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 8 | `roll_resid_mean_10b_mixer` | 0.0480 | 0.0104 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 9 | `Stage6_BottomMixing_power_Mean` | 0.0421 | -0.0085 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 10 | `Stage5_PID_RotorSpeed_Integral` | 0.0399 | 0.0252 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 11 | `Stage5_PID_WayofRam_Std` | 0.0365 | 0.0376 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 12 | `phys_init_temp` | 0.0359 | 0.0403 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 13 | `Stage2_DryMixing_temp_Std` | 0.0354 | -0.0307 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 14 | `Stage2_DryMixing_WayofRam_Std` | 0.0346 | -0.0270 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 15 | `roll_mny_std_10b_same_comp` | 0.0313 | -0.0099 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |


完整明细见：`process_adjustment_guidance.csv`


---

## 6. 各配方系列 (Compound Family) 的预测表现

以下是测试集中前 15 个主要胶料配方系列的表现统计（按 RMSE 降序排列）：

| 排名 | 配方系列 | 测试车次 | 真实门尼均值 | 预测门尼均值 | 偏差 (Bias) | MAE | RMSE | $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `M1-T00274` | 1 | 84.99 | 53.56 | -31.4327 | 31.4327 | 31.4327 | N/A |
| 2 | `R1-B00458RRL` | 2 | 24.02 | 39.05 | 15.0283 | 15.8118 | 21.8143 | -358.8216 |
| 3 | `R1-T19012` | 3 | 79.41 | 61.23 | -18.1720 | 18.1720 | 19.5095 | -28.0723 |
| 4 | `R1-T00011XS` | 1 | 35.53 | 54.21 | 18.6830 | 18.6830 | 18.6830 | N/A |
| 5 | `R1-T30087` | 1 | 69.29 | 53.30 | -15.9874 | 15.9874 | 15.9874 | N/A |
| 6 | `R1-T20153W` | 6 | 75.00 | 68.43 | -6.5657 | 6.7405 | 11.8345 | -12.3514 |
| 7 | `R1-T14150` | 2 | 49.50 | 53.93 | 4.4369 | 5.2827 | 6.8988 | -0.8999 |
| 8 | `R1-T16734` | 4 | 64.02 | 62.25 | -1.7698 | 5.2957 | 6.3108 | -1.2343 |
| 9 | `M1-R00218W2` | 3 | 84.90 | 89.26 | 4.3553 | 4.8564 | 6.2170 | -0.8583 |
| 10 | `R1-T13127` | 6 | 61.01 | 64.39 | 3.3864 | 5.6216 | 6.0645 | -0.6836 |
| 11 | `R1-T25045` | 10 | 60.49 | 57.54 | -2.9449 | 4.0496 | 5.0981 | -0.7506 |
| 12 | `M1-R00218W1` | 5 | 70.36 | 73.56 | 3.1984 | 3.1984 | 4.8560 | -0.7116 |
| 13 | `R1-T26104` | 1 | 62.33 | 57.67 | -4.6629 | 4.6629 | 4.6629 | N/A |
| 14 | `R1-T20153` | 3 | 71.00 | 70.71 | -0.2881 | 3.4500 | 3.5717 | -0.0566 |
| 15 | `R1-T01139-M` | 1 | 53.33 | 49.81 | -3.5200 | 3.5200 | 3.5200 | N/A |

*注：对于仅有1个测试样本或方差为0的系列，$R^2$ 显示为 N/A。*
