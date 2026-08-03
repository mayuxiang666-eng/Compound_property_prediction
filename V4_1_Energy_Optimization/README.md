# V4.1 Energy Optimization

## 独立学习与使用指南

本目录是一个独立整理的 V4.1 密炼能耗预测与推荐架构。它把能耗模型、候选生成、Mooney 质量约束、历史最佳模板、报告和模型结果集中在一个目录中，方便新用户从架构学习到实际运行。

> 重要原则：历史低能耗批次不是最终 setpoint 推荐。历史批次只能作为候选模板和 benchmark。最终推荐必须在当前原料、环境、批重、fill factor、mixer 和 route 条件下重新经过 Mode B 能耗模型与 Mooney 安全门禁。

---

## 1. 先理解架构

V4.1 的核心流程如下：

```text
原始批次数据
    |
    v
数据清洗、物料体系分类、OilWet/NoOilDry 路线分类
    |
    +--> Mode A: 使用批后响应字段，做能耗诊断
    |
    +--> Mode B: 只使用批前可知条件和可控 process profile
              |
              v
       历史最佳候选 cohort
              |
              v
       用历史批次提取 duration / program profile
              |
              v
       替换成当前原料、环境、批重、fill factor、mixer 条件
              |
              +--> Mode B 当前条件 + 当前 process profile
              |
              +--> Mode B 当前条件 + 历史候选 process profile
              |
              v
       计算 model_based_saving_kwh / model_based_saving_pct
              |
              v
       Mooney interval、confidence、OOD、safe bounds、route stage gates
              |
              v
       最终推荐或拒绝
```

### 1.1 三个关键概念

#### Mode A: 批后诊断

Mode A 可以使用批后响应信息，例如实际 power、torque、integral 和部分实际过程响应。它用于解释已经发生的批次或比较模型诊断能力，不能直接用于生产前推荐。

#### Mode B: 批前推荐

Mode B 只能使用推荐时已经知道或可以设置的条件，包括：

- 配方和原料信息
- supplier COA 特征
- 环境温度和湿度
- batch weight
- fill factor
- MixerLine、material system、phase route
- 可确认可控的 duration、target temperature 等 process setting

Mode B 会拒绝包含 `energy`、`kwh`、`power`、`torque`、`integral`、`actual` 等批后响应或标签代理字段的特征，防止信息泄漏。

#### 历史最佳模板

历史最佳批次需要满足低能耗分位、Mooney 在规格内、质量未拒绝、阶段数据有效、批重和 fill factor 有效等条件。模板保留历史 process profile，但会用当前生产上下文覆盖历史上下文。

---

## 2. 目录结构

```text
V4_1_Energy_Optimization/
|
+-- README.md                         本文档
|
+-- feature_engineering/              能耗标签和特征构建
|   +-- energy_label_builder.py       total kWh、kWh/ton、路线标签
|   +-- stage1_recipe_features.py     配方和 COA 特征白名单
|   +-- stage2_process_features.py    过程特征白名单
|   +-- clustering.py                 Silica / CarbonBlack 分类
|   +-- silica_pid_feature_builder.py Silica PID 特征
|   +-- cb_dispersion_feature_builder.py
|
+-- model_training/                   模型训练、验证和审计
|   +-- run_v41_energy_experiment.py  V4.1 主入口
|   +-- energy_model.py               Mode A / Mode B 能耗模型
|   +-- high_energy_specialist.py     高能耗专项模型
|   +-- hybrid_unified_model.py       Mooney 质量模型
|   +-- effective_weighting.py        训练样本权重
|   +-- split_builder.py              防泄漏数据切分
|   +-- nonlinear_factors_and_rolling_lot_calibration/
|                                      历史 Mooney 训练测试代码
|   +-- run_* / test_*                 训练、消融、验证和审计脚本
|
+-- optimization/                     推荐逻辑
|   +-- candidate_generator.py        route-aware 候选生成
|   +-- historical_candidate_builder.py
|                                      历史 cohort、模板和 uncertainty
|   +-- energy_optimizer.py           Mode B 评分和推荐 gate
|   +-- mooney_constraint_checker.py  Mooney interval / OOD 检查
|   +-- safe_bounds_builder.py        安全参数边界
|
+-- models/                           模型结果和生产包
|   +-- v37_production_model_package/
|
+-- reports/                          V4.1 运行输出
    +-- v41_energy_model/             特征审计和 Mode A/B 特征清单
    +-- v41_energy_optimization/     推荐、能耗、Mooney 和 uncertainty 报告
```

---

## 3. 学习顺序

建议按下面顺序学习，不要一开始就阅读所有历史实验脚本。

### 第一步：看主流程

阅读：

```text
model_training/run_v41_energy_experiment.py
```

重点看这些阶段：

1. 数据加载和字段兼容处理
2. `build_energy_labels_and_features`
3. `extract_stage1_recipe_features` 和 `extract_stage2_process_features`
4. train / validation / test split
5. Mode A、Mode B 训练
6. Mooney model 和 safe bounds 构建
7. `optimizer.optimize_batch`
8. CSV report export

### 第二步：看 Mode B 特征政策

阅读：

```text
model_training/energy_model.py
feature_engineering/stage1_recipe_features.py
feature_engineering/stage2_process_features.py
```

重点理解：

- `PRE_BATCH_SETPOINT_COLS`
- `RECIPE_BASELINE_COLS`
- `CATEGORICAL_CONTEXT_COLS`
- `POST_BATCH_ACTUAL_COLS`
- `MODE_B_FORBIDDEN_FEATURE_TOKENS`
- `build_energy_feature_purge_audit`

学习目标是回答：某一列数据是在批前已知、可控 setpoint、批后实际响应，还是能耗标签代理。

### 第三步：看历史模板逻辑

阅读：

```text
optimization/historical_candidate_builder.py
```

重点理解：

- Recipe + Mixer + Route 如何确定 cohort
- p10 到 p30 如何筛选低能耗历史批次
- Mooney、quality、stage、batch weight、fill factor 如何做有效性检查
- 哪些字段属于 process profile
- 哪些字段只是 current-context difference
- RotorSpeed 和 WayofRam 为什么只作为 profile evidence

### 第四步：看推荐 gate

阅读：

```text
optimization/energy_optimizer.py
optimization/mooney_constraint_checker.py
optimization/safe_bounds_builder.py
```

推荐必须同时满足以下条件：

- Mooney prediction interval 在规格内
- confidence label 达到允许阈值
- `ood_flag = False`
- `safe_bound_status = WITHIN_SAFE_BOUNDS`
- `route_stage_status = VALID_ROUTE_STAGES`
- model-based saving 达到阈值
- valid candidate count 达到阈值

### 第五步：阅读报告

建议先看：

```text
reports/v41_energy_optimization/model_normalized_candidate_predictions.csv
reports/v41_energy_optimization/shadow_recommendation_validated.csv
reports/v41_energy_optimization/recommendation_uncertainty_explanation.csv
reports/v41_energy_optimization/recipe_operating_profile_reference.csv
```

---

## 4. 环境准备

推荐使用项目根目录的虚拟环境。PowerShell 示例：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

确认 Python：

```powershell
python --version
python -c "import numpy, pandas, sklearn, lightgbm; print('dependencies ok')"
```

如果依赖没有安装，应根据项目环境安装 `numpy`、`pandas`、`scikit-learn`、`lightgbm` 等包。不要把 `.venv` 提交到 Git。

---

## 5. 运行 V4.1 完整流程

当前工作目录应是项目根目录：

```text
Master batch data fectching/
```

运行独立归档入口：

```powershell
python -u .\V4_1_Energy_Optimization\model_training\run_v41_energy_experiment.py
```

入口会：

1. 读取上一级项目数据目录中的：
   - `data/stage_statistics_enriched_all_features_weather_v4.csv`
   - 如果不存在，则尝试 `data/enriched_mny_all.csv`
2. 构建能耗标签和路线字段
3. 训练 Mode A 和 Mode B
4. 训练 Mooney 质量模型
5. 构建安全边界
6. 在 test/shadow 批次上执行历史模板和随机候选评估
7. 输出全部报告到：

```text
V4_1_Energy_Optimization/reports/v41_energy_optimization/
V4_1_Energy_Optimization/reports/v41_energy_model/
```

成功结束时，终端应看到类似：

```text
V4.1 COMPREHENSIVE ENERGY EXPERIMENTS COMPLETED SUCCESSFULLY
Total Evaluated Batches        : ...
Recommended Batches (Passed Gate): ...
Reports Exported To            : ...
```

---

## 6. 运行后如何验证

### 6.1 检查输出文件

```powershell
$out = '.\V4_1_Energy_Optimization\reports\v41_energy_optimization'
Get-ChildItem $out -File | Select-Object Name,Length,LastWriteTime
```

以下文件应重点检查：

| 文件 | 用途 |
|---|---|
| `historical_best_candidate_templates.csv` | 历史 cohort 和模板 benchmark，不是最终推荐 |
| `model_normalized_candidate_predictions.csv` | 当前上下文归一化后的模型预测和节能结果 |
| `shadow_recommendation_validated.csv` | 所有 shadow 批次的最终 gate 状态 |
| `shadow_recommendation_summary.csv` | 通过推荐 gate 的记录 |
| `shadow_recommendation_rejection_reasons.csv` | 被拒绝记录和原因 |
| `recipe_operating_profile_reference.csv` | 被模型选中的 RotorSpeed/RAM profile reference |
| `recommendation_uncertainty_explanation.csv` | 不确定性因素和工程解释 |
| `recipe_level_pilot_recommendation_window_with_uncertainty.csv` | recipe-level 聚合窗口 |
| `safe_parameter_bounds.csv` | 安全边界来源和范围 |
| `safe_bounds_coverage_summary.csv` | 各路线安全边界覆盖情况 |

### 6.2 检查关键字段

```powershell
Import-Csv (Join-Path $out 'shadow_recommendation_validated.csv') |
    Select-Object -First 5 CompoundName,selected_template_id,model_based_saving_pct,historical_best_actual_saving_pct,recommendation_status
```

建议确认：

- `recommendation_status` 是否为 `RECOMMENDED`
- `model_based_saving_pct` 是否达到工程门槛
- `historical_best_actual_saving_pct` 和模型调整后的 saving 是否同时存在
- `selected_template_id` 是否可追溯
- `safe_bound_status` 是否为 `WITHIN_SAFE_BOUNDS`
- `route_stage_status` 是否为 `VALID_ROUTE_STAGES`
- `ood_flag` 是否为 `False`

---

## 7. 如何理解 saving 数值

### 历史 benchmark saving

```text
historical_best_actual_saving_pct
```

这是当前批次实际能耗与历史最佳批次实际能耗之间的参考差异。它只代表历史现象，不能直接当作新批次可实现的节能承诺。

### 模型调整后的 saving

```text
model_based_saving_kwh = prediction_A - prediction_B
model_based_saving_pct = model_based_saving_kwh / prediction_A * 100
```

其中：

- `prediction_A`：当前条件 + 当前 process profile
- `prediction_B`：当前条件 + 历史候选 process profile

因此，最终推荐应主要依据 `model_based_saving_kwh` 和 `model_based_saving_pct`，而不是历史 benchmark saving。

如果模型调整后 saving 明显低于历史 benchmark，应该查看：

- `top_uncertainty_factor_1`
- `top_uncertainty_factor_2`
- `top_uncertainty_factor_3`
- `uncertainty_explanation_text`

---

## 8. Rotor Speed 和 RAM 的正确使用方式

当前架构不把 RotorSpeed 或 WayofRam 默认当作可直接下发的 per-stage setpoint。

### 正确做法

它们可以作为历史候选的运行 profile evidence：

```text
profile_reference_status = OPERATING_PROFILE_REFERENCE
profile_reference_type = OPERATING_PROFILE_REFERENCE
```

这些值应来自模型最终选中的历史候选，而不是简单取全历史最低能耗批次。

### 错误做法

不要：

- 把历史最低能耗批次的 rotor speed 直接写成推荐 setpoint
- 为没有真实 PLC/MES 控制接口的 RotorSpeed/RAM 生成虚假 setpoint
- 把 profile reference 当成设备控制指令
- 在没有确认控制权限、单位和阶段映射前下发这些值

只有当工艺工程师确认 RotorSpeed/RAM 是真实可控、阶段映射稳定、单位明确，并且 safe bounds 也覆盖时，才可以单独设计 setpoint recommendation。

---

## 9. 常见拒绝原因

| 拒绝原因 | 含义 | 下一步 |
|---|---|---|
| `SAFE_BOUNDS_UNAVAILABLE` | 没有同 Recipe/Mixer/Route 的安全窗口 | 先积累有效历史，不要使用默认范围 |
| `COLD_START_RECIPE_NO_HISTORY` | 新配方没有可用历史 | 只能 observation/reference，不直接 pilot |
| `REJECTED_OUT_OF_SAFE_BOUNDS` | 候选超过安全范围 | 检查 safe bounds 来源 |
| `INVALID_STAGE_FOR_ROUTE` | 候选包含当前路线不存在的阶段 | 检查 route mask 和阶段数据 |
| `MOONEY_CONFIDENCE_LOW` | Mooney 预测置信度不足 | 增加相似历史或收紧推荐范围 |
| `CANDIDATE_OUT_OF_DISTRIBUTION` | 候选偏离训练分布 | 检查原料、mixer、route 和 process profile |
| `MOONEY_INTERVAL_OUTSIDE_SPEC` | 预测区间不在规格内 | 不推荐该候选 |
| `SAVING_BELOW_THRESHOLD` | 模型调整后节能不足 | 保留为 benchmark 或 shadow observation |
| `INSUFFICIENT_VALID_CANDIDATES_COUNT` | 通过质量和安全检查的候选太少 | 检查数据质量、历史 cohort 和 safe bounds |

---

## 10. 如何修改架构

### 修改历史最佳分位

编辑：

```text
optimization/historical_candidate_builder.py
```

默认使用 p10 到 p30。修改后必须重新检查：

- cohort 样本量
- Mooney spec 通过率
- quality rejection 过滤
- stage validity
- model-adjusted saving

### 修改推荐门槛

编辑：

```text
optimization/energy_optimizer.py
```

修改门槛前要同时查看：

- 推荐数量变化
- rejection reason 分布
- Mooney interval 通过率
- OOD 数量
- safe bounds 覆盖率
- 历史 saving 与模型 saving 的差异

### 增加特征

不要直接把列加入模型。应先确认：

1. 该列在推荐时是否可获得
2. 该列是否是 setpoint、配方、环境或上下文
3. 该列是否包含批后响应或标签泄漏
4. Mode A 和 Mode B 是否应该不同处理
5. 特征审计 CSV 是否记录了决策

---

## 11. 测试和质量检查

修改 Python 后先做静态检查：

```powershell
python -m py_compile `
  .\V4_1_Energy_Optimization\model_training\run_v41_energy_experiment.py `
  .\V4_1_Energy_Optimization\model_training\energy_model.py `
  .\V4_1_Energy_Optimization\optimization\energy_optimizer.py `
  .\V4_1_Energy_Optimization\optimization\historical_candidate_builder.py
```

提交前检查空白错误：

```powershell
git diff --check
```

如果获得运行授权，再运行完整 V4.1：

```powershell
python -u .\V4_1_Energy_Optimization\model_training\run_v41_energy_experiment.py
```

不要仅因为脚本正常结束就认为推荐正确。还要检查：

- 是否发生数据泄漏
- 是否出现跨路线 safe bounds
- 是否有非活动阶段 duration
- 是否把 RotorSpeed/RAM 当成 setpoint
- 是否所有推荐都经过 Mooney、OOD 和安全边界 gate
- 报告中的 saving 是否明确区分 historical 和 model-adjusted

---

## 12. 工程上线前检查清单

在任何真实生产试验前，工程师应确认：

- [ ] 数据字段和单位已经和 MES/PLC/LIMS 对齐
- [ ] `batch_weight_ton` 和 fill factor 的来源可靠
- [ ] 原料 lot 和 supplier COA 可以追溯
- [ ] 环境温度、湿度和 material initial temperature 可用
- [ ] MixerLine、material_system、phase_route 分类正确
- [ ] 候选 process profile 在 safe bounds 内
- [ ] Mooney 预测区间在规格内
- [ ] confidence 不是 LOW
- [ ] `ood_flag` 为 False
- [ ] `route_stage_status` 为 `VALID_ROUTE_STAGES`
- [ ] model-adjusted saving 达到门槛
- [ ] valid candidate count 达到门槛
- [ ] RotorSpeed/RAM 仅作为 profile reference，除非已确认可控
- [ ] 先做 shadow 或 pilot review，再做小批量现场试验
- [ ] 现场试验有实际 kWh、Mooney、质量状态和工程备注回写

---

## 13. 一句话总结

V4.1 不是“找历史最低能耗批次并照抄参数”，而是：

> 用合格的历史低能耗批次提供可行 process profile，再把 profile 放回当前生产上下文，由 Mode B 能耗模型和 Mooney 安全门禁共同决定是否推荐。
