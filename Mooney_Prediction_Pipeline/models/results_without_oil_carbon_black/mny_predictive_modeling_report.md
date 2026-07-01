# 胶料门尼粘度 (MNY) 预测模型开发与评估报告 (Without-Oil Carbon-Black 胶料模型)
    
本报告详细说明了针对 **Without-Oil Carbon-Black** 胶料数据集进行门尼粘度（MNY）回归建模的评估结果。

---

## 1. 建模数据概况

- **数据集类型**：Without-Oil Carbon-Black 胶料数据（is_oil_loading_present = 0.0）
- **总车次行数**：285 车
- **异常值清理**：使用**孤立森林 (Isolation Forest) 算法**对温度、功率、扭矩和持续时间等核心曲线特征进行多维异常诊断。
- **清洗后带有门尼粘度标签的车次**：278 车
- **特征工程预处理**：
  1. 使用中位数填充缺失值。
  2. 进行标准差标准化（StandardScaler）。
    3. 先执行物理可解释特征审计，剔除高缺失、高单一值、高共线和低优先级特征。
    4. 使用方差过滤（VarianceThreshold，阈值=0.01）移除噪声/恒定特征，过滤后剩余 34 个特征进行训练。

### 1.1 可解释特征筛选

- 候选特征数：34
- 进入紧凑特征集：34
- 最终进入模型训练：34
- 特征审计明细：`interpretable_feature_audit.csv`

---

## 2. 模型选择与交叉验证 (Model Comparison)

使用 **5折交叉验证 (5-Fold CV)** 在训练集上评估了多种回归算法。性能比较如下：

| 模型算法 | 平均 CV $R^2$ (决定系数) | 平均 CV MAE (平均绝对误差) | 平均 CV RMSE (均方根误差) |
| :--- | :---: | :---: | :---: |
| Ridge Regression | -0.2514 | 1.9636 | 2.8292 |
| Random Forest | -0.1267 | 1.8762 | 2.6831 |
| Gradient Boosting | -0.2573 | 2.0052 | 2.8273 |
| XGBoost (Baseline) | -0.2349 | 2.0152 | 2.7973 |
| LightGBM (Huber) | -0.0831 | 1.8013 | 2.6303 |
| HistGradientBoosting | -0.2553 | 2.0821 | 2.8089 |
| CatBoost (Baseline) | -0.1198 | 1.8991 | 2.6673 |
| LightGBM (Tuned) | -0.0186 | 1.6312 | 2.5506 |
| XGBoost (Tuned) | -0.0258 | 1.6356 | 2.5596 |

*注：$R^2$ 越接近 1.0 预测拟合效果越好。MAE 代表门尼粘度预测值与实测值的绝对平均偏差值。*

---

## 3. Optuna 超参数优化与测试集最终表现

最优模型为 **Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF)**，其最优超参数集为：
- `best_params`: `LGBM: {'n_estimators': 316, 'max_depth': 2, 'learning_rate': 0.018964838969847304, 'num_leaves': 60, 'min_child_samples': 51, 'min_split_gain': 2.585475591182841, 'subsample': 0.6914332209589549, 'colsample_bytree': 0.7685317590385838, 'reg_alpha': 0.00019684322096762618, 'reg_lambda': 44.75474340792476} | XGB: {'n_estimators': 206, 'max_depth': 6, 'learning_rate': 0.005074584437814601, 'subsample': 0.6546194600186628, 'colsample_bytree': 0.843867620700801, 'min_child_weight': 39.661135884489646, 'gamma': 2.6118971263306316, 'reg_alpha': 0.008227680463943095, 'reg_lambda': 91.77250240641062}`

### 3.1 测试集（Held-out Test Set, 20% 未参训数据）的最终表现：
- **测试集决定系数 $R^2$**：**0.8326**
- **测试集平均绝对误差 (MAE)**：**4.1747**
- **测试集均方根误差 (RMSE)**：**6.4510**

### 3.2 预测效果 Parity 对齐图
![Mooney预测Parity图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_without_oil_carbon_black/mooney_model_parity_plot.png)

---

## 4. 特征贡献度分析 (Feature Importance)

以下是经过调优后的 Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF) 模型在预测门尼粘度时，排名前 15 位的核心物理特征：

| 排名 | 特征名称 | 贡献权重 (Importance) | 物理流变学解释 |
| :---: | :--- | :---: | :--- |
| 1 | `roll_mny_mean_3b_same_comp` | 0.0608 | 流变/物理参数描述 |
| 2 | `Stage1_Loading_WayofRam_Mean` | 0.0563 | 流变/物理参数描述 |
| 3 | `roll_resid_mean_10b_mixer` | 0.0547 | 流变/物理参数描述 |
| 4 | `roll_resid_mean_5b_family` | 0.0507 | 流变/物理参数描述 |
| 5 | `roll_mny_std_10b_same_comp` | 0.0502 | 流变/物理参数描述 |
| 6 | `phys_temp_integral_above_100` | 0.0463 | 总混炼温度-时间积分（热历史） |
| 7 | `Stage6_BottomMixing_power_Mean` | 0.0451 | 流变/物理参数描述 |
| 8 | `phys_temp_rise_rate` | 0.0383 | 流变学计算特征 |
| 9 | `Stage5_PID_RotorSpeed_Integral` | 0.0361 | Stage5阶段剪切历史积分（转子总圈数） |
| 10 | `Stage2_DryMixing_WayofRam_Std` | 0.0357 | 流变/物理参数描述 |
| 11 | `Stage2_DryMixing_temp_Std` | 0.0337 | 流变/物理参数描述 |
| 12 | `Stage6_BottomMixing_Torque_Mean` | 0.0328 | Stage6阶段扭矩平均值 |
| 13 | `Stage2_DryMixing_power_Integral` | 0.0323 | 流变/物理参数描述 |
| 14 | `phys_init_temp` | 0.0303 | 流变学计算特征 |
| 15 | `Stage5_PID_power_Integral` | 0.0297 | 流变/物理参数描述 |


### 4.2 核心贡献度图解
![Mooney特征重要性图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_without_oil_carbon_black/mooney_model_feature_importance.png)

---

## 5. 工艺调整解释表

下表将模型重要性、单调相关趋势和密炼工艺含义放在一起，用于预测完成后的工艺诊断。注意：该方向代表训练数据中的统计趋势，实际调整必须结合胶料体系、配方窗口和现场约束确认。

| 排名 | 特征名称 | 模型重要性 | Spearman趋势 | 趋势解释 | 工艺含义/调整方向 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `roll_mny_mean_3b_same_comp` | 0.0608 | -0.0665 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 2 | `Stage1_Loading_WayofRam_Mean` | 0.0563 | -0.0652 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 3 | `roll_resid_mean_10b_mixer` | 0.0547 | -0.0881 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 4 | `roll_resid_mean_5b_family` | 0.0507 | -0.1447 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 5 | `roll_mny_std_10b_same_comp` | 0.0502 | -0.0233 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 6 | `phys_temp_integral_above_100` | 0.0463 | 0.0079 | 单调方向较弱，更多体现交互或分群作用 | 高温积分代表热反应历史，白炭黑/硅烷体系中过高可能导致反应过度，过低可能反应不足。 |
| 7 | `Stage6_BottomMixing_power_Mean` | 0.0451 | -0.0244 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 8 | `phys_temp_rise_rate` | 0.0383 | 0.0169 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 9 | `Stage5_PID_RotorSpeed_Integral` | 0.0361 | 0.0195 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 10 | `Stage2_DryMixing_WayofRam_Std` | 0.0357 | -0.0382 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 11 | `Stage2_DryMixing_temp_Std` | 0.0337 | -0.0236 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 12 | `Stage6_BottomMixing_Torque_Mean` | 0.0328 | -0.0158 | 单调方向较弱，更多体现交互或分群作用 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 13 | `Stage2_DryMixing_power_Integral` | 0.0323 | 0.0320 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 14 | `phys_init_temp` | 0.0303 | 0.0001 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 15 | `Stage5_PID_power_Integral` | 0.0297 | 0.0197 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |


完整明细见：`process_adjustment_guidance.csv`


---

## 6. 各配方系列 (Compound Family) 的预测表现

以下是测试集中前 15 个主要胶料配方系列的表现统计（按 RMSE 降序排列）：

| 排名 | 配方系列 | 测试车次 | 真实门尼均值 | 预测门尼均值 | 偏差 (Bias) | MAE | RMSE | $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `R1-T30087` | 1 | 76.49 | 58.03 | -18.4558 | 18.4558 | 18.4558 | N/A |
| 2 | `M1-R00218-M` | 1 | 75.63 | 57.64 | -17.9922 | 17.9922 | 17.9922 | N/A |
| 3 | `M1-R00218W2` | 4 | 87.47 | 76.25 | -11.2244 | 11.2244 | 13.8845 | -13.7798 |
| 4 | `R1-T20153W` | 3 | 72.21 | 67.77 | -4.4390 | 7.5424 | 9.2821 | -7.0152 |
| 5 | `R1-R00218W9` | 1 | 50.58 | 57.81 | 7.2332 | 7.2332 | 7.2332 | N/A |
| 6 | `M1-R00218W1` | 4 | 67.64 | 71.23 | 3.5889 | 3.6316 | 5.3787 | -0.4466 |
| 7 | `R1-R00218W2` | 3 | 62.95 | 60.59 | -2.3656 | 5.1538 | 5.3383 | -11.8808 |
| 8 | `R1-T25045` | 3 | 59.41 | 63.24 | 3.8281 | 4.7277 | 5.2970 | -5.7799 |
| 9 | `R1-T13127` | 4 | 62.56 | 66.27 | 3.7034 | 3.7034 | 4.2214 | -3.4062 |
| 10 | `R1-T26104` | 3 | 59.55 | 58.84 | -0.7169 | 3.1922 | 3.7829 | -2.6415 |
| 11 | `R1-R00218W5` | 7 | 51.17 | 52.43 | 1.2584 | 2.0335 | 3.3623 | -2.5050 |
| 12 | `M1-C10430` | 1 | 25.30 | 28.09 | 2.7931 | 2.7931 | 2.7931 | N/A |
| 13 | `R1-R00218-M` | 1 | 55.16 | 57.82 | 2.6649 | 2.6649 | 2.6649 | N/A |
| 14 | `M1-R00218W5` | 2 | 72.26 | 73.62 | 1.3565 | 2.2788 | 2.6520 | 0.1695 |
| 15 | `R1-T00011B2` | 8 | 31.34 | 31.39 | 0.0574 | 2.1537 | 2.4685 | -0.7463 |

*注：对于仅有1个测试样本或方差为0的系列，$R^2$ 显示为 N/A。*
