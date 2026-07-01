# 胶料门尼粘度 (MNY) 预测模型开发与评估报告 (With-Oil High-Silica 胶料模型)
    
本报告详细说明了针对 **With-Oil High-Silica** 胶料数据集进行门尼粘度（MNY）回归建模的评估结果。

---

## 1. 建模数据概况

- **数据集类型**：With-Oil High-Silica 胶料数据（is_oil_loading_present = 1.0）
- **总车次行数**：1789 车
- **异常值清理**：使用**孤立森林 (Isolation Forest) 算法**对温度、功率、扭矩和持续时间等核心曲线特征进行多维异常诊断。
- **清洗后带有门尼粘度标签的车次**：1783 车
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
| Ridge Regression | 0.1772 | 2.3289 | 3.1284 |
| Random Forest | 0.2316 | 2.2029 | 3.0207 |
| Gradient Boosting | 0.2391 | 2.2123 | 3.0043 |
| XGBoost (Baseline) | 0.1528 | 2.3565 | 3.1683 |
| LightGBM (Huber) | 0.2342 | 2.2248 | 3.0179 |
| HistGradientBoosting | 0.1776 | 2.3029 | 3.1220 |
| CatBoost (Baseline) | 0.2116 | 2.2746 | 3.0609 |
| LightGBM (Tuned) | 0.2350 | 2.2314 | 3.0176 |
| XGBoost (Tuned) | 0.2396 | 2.2218 | 3.0051 |

*注：$R^2$ 越接近 1.0 预测拟合效果越好。MAE 代表门尼粘度预测值与实测值的绝对平均偏差值。*

---

## 3. Optuna 超参数优化与测试集最终表现

最优模型为 **Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF)**，其最优超参数集为：
- `best_params`: `LGBM: {'n_estimators': 826, 'max_depth': 4, 'learning_rate': 0.02231093156252502, 'num_leaves': 27, 'min_child_samples': 30, 'min_split_gain': 2.694748220588317, 'subsample': 0.87969086056401, 'colsample_bytree': 0.6935891909513648, 'reg_alpha': 0.00041106333506918104, 'reg_lambda': 10.949030695630334} | XGB: {'n_estimators': 439, 'max_depth': 2, 'learning_rate': 0.02156454687913767, 'subsample': 0.8680055071933214, 'colsample_bytree': 0.7553388717147046, 'min_child_weight': 7.198352671242651, 'gamma': 4.492257920113191, 'reg_alpha': 0.10161820966130389, 'reg_lambda': 2.438719043966601}`

### 3.1 测试集（Held-out Test Set, 20% 未参训数据）的最终表现：
- **测试集决定系数 $R^2$**：**0.7029**
- **测试集平均绝对误差 (MAE)**：**2.5927**
- **测试集均方根误差 (RMSE)**：**4.0203**

### 3.2 预测效果 Parity 对齐图
![Mooney预测Parity图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_with_oil_high_silica/mooney_model_parity_plot.png)

---

## 4. 特征贡献度分析 (Feature Importance)

以下是经过调优后的 Stacked Ensemble (Tuned LGBM + Tuned XGB + CatBoost + HistGBM + RF) 模型在预测门尼粘度时，排名前 15 位的核心物理特征：

| 排名 | 特征名称 | 贡献权重 (Importance) | 物理流变学解释 |
| :---: | :--- | :---: | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.1925 | 流变/物理参数描述 |
| 2 | `roll_resid_mean_10b_mixer` | 0.0924 | 流变/物理参数描述 |
| 3 | `supplier_silica_surface_area_avg` | 0.0516 | 流变/物理参数描述 |
| 4 | `roll_mny_std_10b_same_comp` | 0.0437 | 流变/物理参数描述 |
| 5 | `roll_mny_mean_3b_same_comp` | 0.0380 | 流变/物理参数描述 |
| 6 | `Stage6_BottomMixing_Torque_Mean` | 0.0376 | Stage6阶段扭矩平均值 |
| 7 | `Stage6_BottomMixing_Torque_Integral` | 0.0358 | Stage6阶段扭矩累积积分（功历史） |
| 8 | `supplier_rubber_viscosity_avg` | 0.0351 | 供应商生胶出厂粘度平均值 |
| 9 | `phys_temp_integral_above_100` | 0.0331 | 总混炼温度-时间积分（热历史） |
| 10 | `Stage3_OilLoading_temp_Mean` | 0.0313 | 流变/物理参数描述 |
| 11 | `Stage2_DryMixing_power_Mean` | 0.0309 | 流变/物理参数描述 |
| 12 | `Stage4_WetMixing_power_Mean` | 0.0307 | 流变/物理参数描述 |
| 13 | `Stage2_DryMixing_power_Integral` | 0.0294 | 流变/物理参数描述 |
| 14 | `Stage4_WetMixing_Torque_Mean` | 0.0263 | Stage4阶段扭矩平均值 |
| 15 | `Stage3_OilLoading_temp_Std` | 0.0246 | 流变/物理参数描述 |


### 4.2 核心贡献度图解
![Mooney特征重要性图](file:///C:/Users/uif35346/OneDrive - Continental AG/Desktop/Compound property prediction/Master batch data fectching/Mooney_Prediction_Pipeline/models/results_with_oil_high_silica/mooney_model_feature_importance.png)

---

## 5. 工艺调整解释表

下表将模型重要性、单调相关趋势和密炼工艺含义放在一起，用于预测完成后的工艺诊断。注意：该方向代表训练数据中的统计趋势，实际调整必须结合胶料体系、配方窗口和现场约束确认。

| 排名 | 特征名称 | 模型重要性 | Spearman趋势 | 趋势解释 | 工艺含义/调整方向 |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `roll_resid_mean_5b_family` | 0.1925 | 0.4307 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 2 | `roll_resid_mean_10b_mixer` | 0.0924 | 0.3963 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 3 | `supplier_silica_surface_area_avg` | 0.0516 | 0.0755 | 单调方向较弱，更多体现交互或分群作用 | 白炭黑比表面积决定了偶联活性位点与填料网络强弱，异常偏高会增强粒子自凝聚进而推高门尼，应核对批次检测。 |
| 4 | `roll_mny_std_10b_same_comp` | 0.0437 | 0.0338 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 5 | `roll_mny_mean_3b_same_comp` | 0.0380 | 0.2163 | 该特征升高时，模型/数据倾向于 MNY 升高 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 6 | `Stage6_BottomMixing_Torque_Mean` | 0.0376 | -0.0112 | 单调方向较弱，更多体现交互或分群作用 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 7 | `Stage6_BottomMixing_Torque_Integral` | 0.0358 | -0.0301 | 单调方向较弱，更多体现交互或分群作用 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 8 | `supplier_rubber_viscosity_avg` | 0.0351 | 0.2349 | 该特征升高时，模型/数据倾向于 MNY 升高 | 生胶供应商粘度是上游遗传因素，偏高时不要只调密炼参数，应同步看原材料批次。 |
| 9 | `phys_temp_integral_above_100` | 0.0331 | -0.0504 | 单调方向较弱，更多体现交互或分群作用 | 高温积分代表热反应历史，白炭黑/硅烷体系中过高可能导致反应过度，过低可能反应不足。 |
| 10 | `Stage3_OilLoading_temp_Mean` | 0.0313 | -0.1363 | 该特征升高时，模型/数据倾向于 MNY 降低 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 11 | `Stage2_DryMixing_power_Mean` | 0.0309 | -0.0571 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 12 | `Stage4_WetMixing_power_Mean` | 0.0307 | -0.0079 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 13 | `Stage2_DryMixing_power_Integral` | 0.0294 | 0.0135 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |
| 14 | `Stage4_WetMixing_Torque_Mean` | 0.0263 | -0.0106 | 单调方向较弱，更多体现交互或分群作用 | 这是流变响应特征，适合用来识别混炼中后段粘度是否异常上升。 |
| 15 | `Stage3_OilLoading_temp_Std` | 0.0246 | -0.0722 | 单调方向较弱，更多体现交互或分群作用 | 建议结合分群残差和同胶料历史分布判断其是否是真正可调工艺杠杆。 |


完整明细见：`process_adjustment_guidance.csv`


---

## 6. 各配方系列 (Compound Family) 的预测表现

以下是测试集中前 15 个主要胶料配方系列的表现统计（按 RMSE 降序排列）：

| 排名 | 配方系列 | 测试车次 | 真实门尼均值 | 预测门尼均值 | 偏差 (Bias) | MAE | RMSE | $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `M1-T19012W` | 1 | 85.31 | 52.57 | -32.7385 | 32.7385 | 32.7385 | N/A |
| 2 | `M1-T10035B6` | 1 | 32.32 | 53.51 | 21.1864 | 21.1864 | 21.1864 | N/A |
| 3 | `M1-T10035B7` | 2 | 35.16 | 51.29 | 16.1227 | 16.1227 | 16.3451 | -32.0074 |
| 4 | `M1-T33025XP` | 1 | 46.89 | 53.81 | 6.9118 | 6.9118 | 6.9118 | N/A |
| 5 | `M1-T20153` | 2 | 77.05 | 70.66 | -6.3912 | 6.3912 | 6.5053 | -9.9322 |
| 6 | `M1-T17732` | 2 | 56.67 | 62.36 | 5.6977 | 5.6977 | 6.3259 | -5.9764 |
| 7 | `M1-T20153W` | 3 | 75.48 | 81.09 | 5.6114 | 5.6114 | 6.2803 | -3.3841 |
| 8 | `M1-T15192R` | 1 | 61.03 | 66.52 | 5.4890 | 5.4890 | 5.4890 | N/A |
| 9 | `M1-T15899` | 9 | 48.51 | 47.98 | -0.5291 | 3.6631 | 4.5463 | -0.1803 |
| 10 | `M1-T15760` | 77 | 56.17 | 56.63 | 0.4615 | 3.5702 | 4.3077 | 0.2546 |
| 11 | `M1-T15760B` | 17 | 59.56 | 59.09 | -0.4791 | 2.9934 | 4.1439 | -0.0310 |
| 12 | `M1-T01139-A` | 1 | 55.73 | 52.76 | -2.9751 | 2.9751 | 2.9751 | N/A |
| 13 | `M1-T19065` | 8 | 51.01 | 50.01 | -1.0005 | 2.0836 | 2.5846 | 0.3654 |
| 14 | `M1-T02128W5` | 3 | 63.47 | 62.48 | -0.9897 | 1.7719 | 2.4550 | -0.4276 |
| 15 | `M1-T33025W3` | 77 | 46.66 | 46.25 | -0.4065 | 1.8709 | 2.3459 | 0.5809 |

*注：对于仅有1个测试样本或方差为0的系列，$R^2$ 显示为 N/A。*
