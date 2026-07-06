# 胶料门尼粘度 (MNY) 预测模型开发与评估报告 (Without-Oil Carbon-Black 胶料模型)
    
本报告详细说明了针对 **Without-Oil Carbon-Black** 胶料数据集进行门尼粘度（MNY）回归建模的评估结果。

---

## 1. 建模数据概况

- **数据集类型**：Without-Oil Carbon-Black 胶料数据（is_oil_loading_present = 0.0）
- **总车次行数**：476 车
- **异常值清理**：使用**孤立森林 (Isolation Forest) 算法**对温度、功率、扭矩和持续时间等核心曲线特征进行多维异常诊断。
- **清洗后带有门尼粘度标签的车次**：470 车
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
| Ridge Regression | -0.1857 | 1.8425 | 2.5653 |
| Random Forest | -0.0270 | 1.7149 | 2.3941 |
| Gradient Boosting | -0.1799 | 1.8799 | 2.5701 |
| XGBoost (Baseline) | -0.2458 | 1.9547 | 2.6441 |
| LightGBM (Huber) | -0.0980 | 1.7305 | 2.4743 |
| HistGradientBoosting | -0.1713 | 1.8729 | 2.5614 |
| CatBoost (Baseline) | -0.1019 | 1.7990 | 2.4733 |
| LightGBM (Tuned) | -0.0265 | 1.6205 | 2.3923 |
| XGBoost (Tuned) | -0.0375 | 1.6297 | 2.4058 |

*注：$R^2$ 越接近 1.0 预测拟合效果越好。MAE 代表门尼粘度预测值与实测值的绝对平均偏差值。*

---

## 3. Optuna 超参数优化与测试集最终表现

最优模型为 **Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF)**，其最优超参数集为：
- `best_params`: `LGBM: {'n_estimators': 704, 'max_depth': 4, 'learning_rate': 0.0205805199651302, 'num_leaves': 22, 'min_child_samples': 65, 'min_split_gain': 1.5932968077056802, 'subsample': 0.9443810722657874, 'colsample_bytree': 0.6575626239808249, 'reg_alpha': 0.13646044662933265, 'reg_lambda': 97.1890020868072} | XGB: {'n_estimators': 206, 'max_depth': 4, 'learning_rate': 0.0051372887009059585, 'subsample': 0.7977551153697758, 'colsample_bytree': 0.6467907976756703, 'min_child_weight': 30.41543810386967, 'gamma': 3.6003587531181873, 'reg_alpha': 0.00011093602491605325, 'reg_lambda': 90.85392449335801}`

### 3.1 测试集（Held-out Test Set, 20% 未参训数据）的最终表现：
- **测试集决定系数 $R^2$**：**0.7383**
- **测试集平均绝对误差 (MAE)**：**4.9127**
- **测试集均方根误差 (RMSE)**：**7.9711**

### 3.2 预测效果 Parity 对齐图
![Mooney预测Parity图](file:///c:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_without_oil_carbon_black/mooney_model_parity_plot.png)

---

## 4. 特征贡献度分析 (Feature Importance)

以下是经过调优后的 Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF) 模型在预测门尼粘度时，排名前 15 位的核心物理特征：

| 排名 | 特征名称 | 贡献权重 (Importance) | 物理流变学解释 |
| :---: | :--- | :---: | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.2997 | 流变/物理参数描述 |
| 2 | `roll_mny_mean_3b_same_comp` | 0.0728 | 流变/物理参数描述 |
| 3 | `Stage5_PID_WayofRam_Std` | 0.0590 | 流变/物理参数描述 |
| 4 | `roll_mny_std_10b_same_comp` | 0.0569 | 流变/物理参数描述 |
| 5 | `Stage6_BottomMixing_power_Mean` | 0.0423 | 流变/物理参数描述 |
| 6 | `Stage5_PID_power_Integral` | 0.0419 | 流变/物理参数描述 |
| 7 | `roll_resid_mean_10b_mixer` | 0.0409 | 流变/物理参数描述 |
| 8 | `phys_temp_rise_rate` | 0.0386 | 流变学计算特征 |
| 9 | `Stage1_Loading_RotorSpeed_Integral` | 0.0369 | Stage1阶段剪切历史积分（转子总圈数） |
| 10 | `Stage2_DryMixing_temp_Std` | 0.0363 | 流变/物理参数描述 |
| 11 | `Stage2_DryMixing_WayofRam_Std` | 0.0321 | 流变/物理参数描述 |
| 12 | `Stage6_BottomMixing_Torque_Mean` | 0.0291 | Stage6阶段扭矩平均值 |
| 13 | `Stage5_PID_RotorSpeed_Integral` | 0.0291 | Stage5阶段剪切历史积分（转子总圈数） |
| 14 | `Stage2_DryMixing_power_Integral` | 0.0281 | 流变/物理参数描述 |
| 15 | `phys_init_temp` | 0.0278 | 流变学计算特征 |


### 4.2 核心贡献度图解
![Mooney特征重要性图](file:///c:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_without_oil_carbon_black/mooney_model_feature_importance.png)

---

## 5. 工艺调整解释表

下表将模型重要性、单调相关趋势和密炼工艺含义放在一起，用于预测完成后的工艺诊断。注意：该方向代表训练数据中的统计趋势，实际调整必须结合胶料体系、配方窗口和现场约束确认。

| 排名 | 特征名称 | 模型重要性 | Spearman趋势 | 趋势解释 | 工艺含义/调整方向 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.2997 | -0.1853 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 2 | `roll_mny_mean_3b_same_comp` | 0.0728 | -0.0455 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 3 | `Stage5_PID_WayofRam_Std` | 0.0590 | 0.0778 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 4 | `roll_mny_std_10b_same_comp` | 0.0569 | -0.0052 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 5 | `Stage6_BottomMixing_power_Mean` | 0.0423 | -0.0425 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 6 | `Stage5_PID_power_Integral` | 0.0419 | 0.0157 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 7 | `roll_resid_mean_10b_mixer` | 0.0409 | -0.0747 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 8 | `phys_temp_rise_rate` | 0.0386 | 0.0284 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 9 | `Stage1_Loading_RotorSpeed_Integral` | 0.0369 | -0.0187 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 10 | `Stage2_DryMixing_temp_Std` | 0.0363 | -0.0235 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 11 | `Stage2_DryMixing_WayofRam_Std` | 0.0321 | -0.0265 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 12 | `Stage6_BottomMixing_Torque_Mean` | 0.0291 | 0.0059 | 单调方向较弱，更多体现交互或分群作用 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 13 | `Stage5_PID_RotorSpeed_Integral` | 0.0291 | 0.0091 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 14 | `Stage2_DryMixing_power_Integral` | 0.0281 | 0.0017 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 15 | `phys_init_temp` | 0.0278 | 0.0175 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |


完整明细见：`process_adjustment_guidance.csv`


---

## 6. 各配方系列 (Compound Family) 的预测表现

以下是测试集中前 15 个主要胶料配方系列的表现统计（按 RMSE 降序排列）：

| 排名 | 配方系列 | 测试车次 | 真实门尼均值 | 预测门尼均值 | 偏差 (Bias) | MAE | RMSE | $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `R1-X20814T` | 1 | 85.22 | 57.25 | -27.9693 | 27.9693 | 27.9693 | N/A |
| 2 | `R1-T00011XZ` | 1 | 31.26 | 57.16 | 25.9021 | 25.9021 | 25.9021 | N/A |
| 3 | `M1-H16435Y` | 1 | 32.57 | 57.16 | 24.5886 | 24.5886 | 24.5886 | N/A |
| 4 | `M1-A03405R3` | 3 | 39.53 | 57.27 | 17.7428 | 17.7428 | 17.8333 | -108.5574 |
| 5 | `R1-T01139` | 1 | 41.66 | 57.09 | 15.4270 | 15.4270 | 15.4270 | N/A |
| 6 | `M1-C10430` | 3 | 29.55 | 37.53 | 7.9785 | 10.4391 | 14.9775 | -77.8725 |
| 7 | `R1-T19012` | 5 | 81.84 | 85.32 | 3.4781 | 9.3705 | 9.8412 | -2.6517 |
| 8 | `R1-A00205X4` | 1 | 66.96 | 57.26 | -9.6989 | 9.6989 | 9.6989 | N/A |
| 9 | `R1-T00011B2` | 9 | 31.95 | 35.60 | 3.6506 | 4.7523 | 8.2681 | -12.4256 |
| 10 | `R1-T14150` | 3 | 46.24 | 50.90 | 4.6606 | 4.6606 | 7.1536 | -47.3336 |
| 11 | `R1-R00218W9` | 1 | 50.58 | 57.03 | 6.4548 | 6.4548 | 6.4548 | N/A |
| 12 | `M1-R00218W2` | 1 | 89.74 | 84.56 | -5.1843 | 5.1843 | 5.1843 | N/A |
| 13 | `R1-T14885B3` | 3 | 71.09 | 71.27 | 0.1808 | 4.3132 | 5.1280 | -2.2383 |
| 14 | `R1-T20153` | 3 | 65.82 | 66.92 | 1.0992 | 4.3313 | 4.5164 | -1202.0314 |
| 15 | `R1-T26104` | 2 | 60.08 | 56.40 | -3.6767 | 3.6767 | 4.3762 | -2.7830 |

*注：对于仅有1个测试样本或方差为0的系列，$R^2$ 显示为 N/A。*
