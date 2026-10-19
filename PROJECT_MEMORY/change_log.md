# 逐次改动登记

规则：每次修改代码、合同、数据快照、模型、报告或测试，都必须在本文件追加一条记录，并同步更新 `decision_log.md` 或 `experiment_index.csv`。本文件只追加，不覆盖历史条目。

| 时间 | 改动类型 | 文件/产物 | 内容 | 验证 | 状态 |
|---|---|---|---|---|---|
| 2026-08-13 | 知识库 | `PROJECT_MEMORY/` | 建立长期本地知识库、当前状态、合同、决策日志和实验索引 | 文件已创建 | 完成 |
| 2026-08-13 | 数据质量 | `curve_quality_contract_v1.py` | 物理无效剔除、统计异常保留、训练集分层 Isolation Forest | 6,814 批次审计 | 完成 |
| 2026-08-13 | 特征语义 | `v371_corrected_hybrid_v2.py` | V3.7.1 过程差值保留 NaN，不再用 0 伪造缺失 | Raw/IF+Flag 重训 | 完成 |
| 2026-08-13 | 模型实验 | `v371_r1_pid_curve_quality_flag_missing_safe` | 统计异常风险特征接入 C1 对照模型 | Global/T15760/B00458 OOT | 完成，未达放行 |
| 2026-08-13 | 报告工程 | `run_v371_r1_temporal_group.py` | 缺少 tabulate 时使用纯文本表格，保证报告可生成 | 报告成功生成 | 完成 |
| 2026-08-13 | 输入合同 | `build_v371_batch_pid_quality_snapshot_v1.py` | 批次过程快照移除 MNY/lab_time/split，避免 R1B 重复合并标签 | 修复 split_x/split_y 接口错误 | 完成 |
| 2026-08-13 | 评分接口 | `run_v371_r1b_group_atomic_batch_v2.py` | R1B 组评分使用已修正的 ActualMNY→MNY 包装器，避免重复目标列 | v1 评分入口暴露重复列 | 已切换 v2 |
| 2026-08-13 | 模型实验 | `v371_r1b_pid_quality` | 批次级动态 PID + 质量合同 R1B，Global OOT MAE 约 4.39，T15760 约 4.83，B00458 约 4.07 | Lab 组级评价 | 完成，未达放行 |
| 2026-08-13 | 特征工程 | `build_v371_batch_pid_mechanism_snapshot_v1.py` | 增加油后升温、Wet 热暴露、Wet 转子时间、PID 总时间等目标无关机制特征 | 待 R1B OOT 对照 | 进行中 |
| 2026-08-13 | 输入合同 | `prepare_v371_r1b_mechanism_contract_v1.py` | 机制实验继承 PID+质量组标签框架，避免新实验丢失前序合同字段 | 修复 metadata/group 缺列 | 完成 |
| 2026-08-13 | 模型实验 | `v371_r1b_pid_mechanism` | 7 个专家机制交互特征对照，Global/T15760/B00458 均未改善 | R1B OOT 指标 | 不进入主模型 |
| 2026-08-13 | 验收扩展 | `audit_four_material_acceptance_v1.py` | 在 B00458/T15760 外增加 M3B-B00460、M2-B00163、T09170、T33025 四个精确配方版本；按 LabelGroup 组内预测均值评价，并增加方向准确度和波动捕捉度 | 4 个新增配方、6 个验收配方完成组级审计；单元测试 2 项通过 | 完成，六个配方仍不放行 |
| 2026-08-13 | 评价口径修正 | `r1b_batch_predictions.csv` + `audit_four_material_acceptance_v1.py` | 发现组级导出中含连接辅助行，改用真正的一车次一行批次预测作为输入，重新计算六个配方的组级 MAE、方向和波动指标 | 重跑报告；组内每车次预测先均值、每个 Lab 组只比较一次 | 已修正，当前报告以批次级输入为准 |
| 2026-08-13 | 范围修正 | `audit_four_material_acceptance_v1.py` | 将模型/验收范围硬限制为 M1；撤销 M2/M3B 验收项，改为 M1-B00460B、M1-B15563R1，并保留 M1-B00458、M1-T15760、M1-T09170、M1-T33025 | M1 过滤后 4,100 个批次；新增炭黑版本各仅 2 个 OOT Lab 组 | M1 子集诊断完成；严格 M1-only 模型尚需重训 |
| 2026-08-13 | 模型重训 | `prepare_v371_r1b_m1_only_contract_v1.py` + `v371_r1b_m1_only` | 从云端 RDS Active 快照构建严格 M1-only 批次/组级快照，M2/M3/PB/R1 不参与 Stage1、Stage1b、Stage2 拟合 | 4,100 批次、1,197 Lab 组；训练 2,704 批次/791 组，OOT 716 批次/211 组；物理批次重复 0 | 完成，模型仍未放行 |
| 2026-08-13 | M1-only 验收 | `v371_r1b_m1_only_acceptance` | 对严格 M1-only R1B 重训模型执行六个 M1 精确配方的 MAE、方向和波动验收 | Global OOT MAE 5.188；T15760 精确版本 MAE 5.124、预测Std/实际Std 0.514 | 未达 2.5 MU/波动门槛，继续修复过程状态残差 |
| 2026-08-14 | 外部研究 | `docs/M1_only_advanced_method_research_review_20260814.md` | 调研橡胶门尼软测量、JIT/GPR、半监督、灰盒过程模型、过程约束贝叶斯优化、安全RL和MPC | 形成M1-only研究路线；未改变模型/数据库/控制状态 | 下一步先做M1 causal state + probabilistic local expert |
| 2026-08-14 | 数据源审计 | `PROJECT_MEMORY/historical_csv_source_audit_20260814.md` | 只读盘点 `data/` 历史 CSV；确认 `stage_statistics_enriched.csv` 为首选原始曲线候选源，旧 V4 派生特征禁止直接训练 | 23,740 唯一批次/7,268 Lab组；20,201 批次不在当前4,100车快照；未访问 MMS/Archive | 完成，历史标签和特征仍需隔离重建 |
| 2026-08-14 | 历史数据隔离 | `prepare_m1_local_history_quarantine_v1.py` + `local_history_quarantine_v1/` | 规范化历史 OrderID，保护当前快照与固定Val/OOT，执行M1/曲线物理门禁并缓存本地Parquet；历史Lab保持阻断 | 23,740车审计；20,126车/6,190组可重建Phase；训练可用0；未访问数据库 | 完成 |
| 2026-08-14 | Phase重建诊断 | `build_m1_local_history_phase_pilot_v1.py` + `local_history_phase_pilot_v1/` | 六个M1验收胶料各100车，按当前车次独立曲线重建V3.7 t1–t7诊断特征 | 600/600成功、0阻断、边界单调600、无MNY/跨批填补 | 完成，仅诊断 |
| 2026-08-14 | 历史标签释放 | `release_m1_local_history_labels_v1.py` + `local_history_label_release_v1/` | 仅用本地规范001/011 Lab替换旧CSV MNY，并按精确RecipeID固定Val/OOT起点执行时间门禁；过程与标签分离保存 | 验证1,128组/3,887车；安全释放385组/1,385车；旧MNY仅543组完全一致、最大差33.23MU | 完成，等待同源Phase特征重建 |
| 2026-08-14 | 同源 Phase 重建 | `run_v371_c1_history_augmented_phase_v1.py` + `v371_c1_history_augmented_phase_v1/` | 当前4,100车与历史1,385车使用同一 current-batch-only Phase 提取器；补齐订单开始前24小时天气；逐车预测后按Lab组均值评价 | 5,485车全部Phase重建成功；当前/历史批次和Label组重叠均为0；7项合同测试通过 | 完成 |
| 2026-08-14 | 模型对照 | `H0_CURRENT_ONLY_PHASE_C1` vs `H1_HISTORY_AUGMENTED_PHASE_C1` | 固定195组Val/211组OOT；Stage1+Stage1b+四路线Stage2，Stage3关闭 | OOT MAE 5.502→6.467；订单内方向42.9%→52.4%，但变化幅度比0.349→0.255；H1明显恶化 | H1拒绝，不得发布 |
| 2026-08-14 | 稳定性修复 | `group_atomic_subsystems.py` + `v371_group_atomic_batch_v1.py` | 固定子专家特征顺序，LightGBM启用deterministic和force_col_wise | 连续两次H0/H1模型SHA256及metrics CSV SHA256完全一致；5项相关测试通过 | 完成 |

后续追加格式：

```text
时间 | 改动类型 | 文件/产物 | 内容 | 验证 | 状态
```

| 2026-08-17 | 新路线合同 | `m1_relative_improvement/` | 新增概率降门尼请求、双试点门禁、Gate 0动作支持、去混杂包络、B2方向/幅度、B3过程响应和B5失败关闭接口 | 18项相关回归测试通过 | 完成 |
| 2026-08-17 | Gate 0实验 | `m1_relative_improvement_gate0_ab_v1/` | 审计T15760/B00458历史动作支持；B00458无Pallet标签交集时禁止伪造Lab组 | 1,127行；因果支持0 | 完成，B4阻断 |
| 2026-08-17 | A/B2/B3实验 | `run_m1_relative_improvement_gate0_ab_v1.py` | 1,173组静态去混杂；147/36/29对方向模型；275/186车双试点过程响应 | 两次运行5个核心产物SHA256一致；3张PNG已生成 | 完成，B2/B3未通过 |
| 2026-08-20 | 数据回填修复 | `06_v2_6_cloud_m1_incremental_backfill.sql` + `10_run_v26_cloud_m1_step_backfill.py` | 确认云端Active含约13个月、约49.4万M1实际车次；改为最多25个精确订单、五路径源端透视、断点续跑，更新Raw/Core/Mart真实Step | 两个T15760订单105车全部Step完整；10订单459车/5,721 Step耗时11.6秒；Run49全年执行中 | 进行中 |
| 2026-08-20 | 相对指导严格接管 | `m1_relative_improvement/integrated_data.py` + `run_strict_relative_guidance_v1.py` | 以MooneyIntegratedData为唯一入口重建Gate 0；修复真实24小时天气；原料覆盖只承认Material+ChargeID+typed QM；禁用旧硬编码推荐 | 1,708批/523组/22,795 Step；天气100%；原料QM 3.805%；9项测试通过 | Gate 0阻断，不生成推荐 |
| 2026-08-20 | 云端材料增量链 | `07_v27_cloud_m1_material_incremental_backfill.sql` + `11_run_v27_cloud_m1_material_backfill.py` | 精确OrderID+BatchNumber回填Active BatchMaterials；按受影响批次重建typed QM；修正WeighingTime为称量耗时语义 | 29个成功Run、666订单；重复/孤儿/伪时间戳/部分源订单均0；QM组覆盖86.848% | 数据链完成，Gate 0仍因真实QM缺测阻断 |

| 2026-08-14 | 版本语义审计 | `stage_statistics_enriched.csv` + `local_history_label_release_v1/` | 确认旧历史源没有订单真实 ProcessVersion；RecipeID 中 `_PVxx_` 仅为配方标识的一部分，不再冒充订单工艺版本 | B00458/T33025/T15760 与当前订单实际 ProcessVersion 逐项对照 | 完成；标记 `LEGACY_ORDER_PROCESS_VERSION_UNAVAILABLE` |
| 2026-08-14 | 模型工程 | `v371_source_weighted_group_atomic_batch_v1.py` + H2 runner/tests | 新增 Lab 组来源可靠度权重；历史准入采用精确 RecipeID+Mixer+Route+静态工艺合同，权重采用 180 天半衰期及逐 RecipeID 25% 上限 | 4 项新增单元测试通过；有效组权重逐组/逐配方守恒 | 完成 |
| 2026-08-14 | 模型对照 | `v371_c1_history_weighted_h2_v1/` | H2 仅放行 13/385 个历史组（均为 B00458），总有效票数 2.630；固定 211 组 OOT 与 H0 对照 | MAE 5.502→5.575；RMSE 9.004→9.200；方向 42.9%→38.1%；连续两次模型和指标 SHA256 一致 | H2 拒绝，不发布 |
| 2026-08-14 | 过程信号诊断 | `diagnose_m1_phase_delta_signal_v1.py` + `m1_phase_delta_signal_v1/` | 对同 Order/Recipe/ProcessVersion/Mixer/Template 的相邻 Lab 组构造 Phase 执行差；训练折拟合固定正则 Phase bundle，验证集选择，OOT 只评估 | 147/36/29 个 train/val/OOT 转移；2 项单元测试通过；已生成 Actual vs Prediction 图 | 完成，所有 bundle 不放行 |
| 2026-08-14 | 模型对照 | `run_v371_c1_thermal_budget_h3_v1.py` + `v371_c1_thermal_budget_h3_v1/` | 在 C1 中增加6个训练片 nominal 驱动的油后温度×Wet热预算低维特征；无油路线为物理不适用零 | Val MAE +0.0139、OOT MAE +0.0124；订单内方向不变、delta幅度比仅0.353；1项新增测试通过；已生成全览图 | H3拒绝，不发布 |
| 2026-08-14 | 方案合同 | `docs/M1大规模无标签订单半监督路线_V3_20260814.md` | 将每日大量无Lab的M1订单用于Phase-aware自监督表示、真实Lab组级监督和主动检测；当前关闭伪MNY与RL | 对照既有Teacher–Student/Representation V2失败结果和当前标签合同 | 设计完成，待无标签语料清单审计 |
# 2026-08-20 — MooneyIntegratedData V2.4 补漏审计

- 完成数据库与 pipeline 只读复核；未访问本地 MMS Archive，未修改数据库。
- 确认 M1 范围、基础胶料号关联、Lab 冲突隔离和 Weather 严格 NULL 正确。
- 确认 Core Step/Phase provenance 与实际 legacy/synthetic 来源矛盾；Core Batch 未由真实 Header 更新；Core Material 范围存在大量孤儿；QM 单值聚合混合不同检验项目且并未按重量加权。
- 确认当前仍为 TRUNCATE 全量实验流程，SQL Agent job 为 0，尚非每日幂等增量链路。
- 详细证据：`PROJECT_MEMORY/mooney_integrated_data_v24_supplemental_audit_20260820.md`。

# 2026-08-20 — MooneyIntegratedData V2.5 修复与数据库回填

- 新增V2.5 schema、真实Core回填、合肥本地时区天气和治理Mart终审脚本。
- 数据库Run 39/42/43成功；3,300批真实Header、262批真实Step、53,427行范围正确的Core Material、104,865个typed QM特征、23,088小时天气已落库。
- 固定6步代理、Core Material孤儿、合成Phase错误provenance和Mart旧Phase残留均降为0。
- Mart保持0条TrainingReady；无真实曲线/Phase的批次不放行。
- 旧破坏性V2.4回填入口默认阻断；未来H2c材料提取保存WeighingTime。
- 详细证据：`PROJECT_MEMORY/mooney_integrated_data_v25_repair_execution_20260820.md`。
# 2026-08-20 BatchTrainingReady 完整性事实核查

- 新增只读审计脚本 `Mooney_Integrated_Data_Asset/pipeline/13_audit_batch_training_ready_completeness.py`，仅连接 `MooneyIntegratedData`，未访问 MMS/Archive，未写数据库。
- 实时结果：`mart.BatchTrainingReady` 347,981 行；343,780 行有 StepCount、StepVectorJson 和 TargetVectorJson；PhaseComplete=1 为 0；任一 Phase 宽特征非空为 0；TRAINING_READY 为 0。
- 329,342 行为 `AUTHENTIC_STEP_ONLY`，阻断原因 `MISSING_AUTHENTIC_1HZ_CURVE_AND_PHASE`；14,438 行为 `RECIPE_TARGET_ONLY`。
- `core.BatchStep` 已有 4,122,265 行、覆盖 343,780 批次；`core.BatchPhase` 只有 37,202 行、覆盖 6,561 批次，但治理版 mart 主动清空未经真实曲线验收的 Phase 特征。
- 结论：不是后台尚未结束，而是 V2.6/V2.7 只完成 Step/材料层，尚未实现基于云端真实数据的 Phase 构建及 mart 宽特征回填。不得将当前表描述为“训练就绪”。
- 证据：`Mooney_Integrated_Data_Asset/reports/batch_training_ready_completeness.json`。
# 2026-08-21 Step→Phase 重构计划严格审阅

- 审阅另一 agent 的“基于412万真实Step数据的阶段聚合与训练宽表重构计划”。
- 数据库只读实证：341,420批次仍为 `M1_ROUTE_PENDING_PHASE_MAPPING`；最近1,000批次10,120个相邻 EnergyAV 全部单调非下降，确认累计能量不可直接求和；`core.BatchPhase` 九个核心数值列仍为 NOT NULL。
- 评定为 `CONDITIONAL_REVIEW_PASS / IMPLEMENTATION_BLOCKED_PENDING_P0`。
- P0包括：模板级动态Phase mapping、累计能量边界差分和3600单位换算、Step终点代理与1Hz特征分列、Phase NULL schema迁移、新建事务化V2.8 builder、重写硬编码H6。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/step_to_phase_rebuild_plan_review_20260821.md`。
- 本次只读数据库，未访问 MMS/Archive，未修改数据库。
# 2026-08-21 Recipe/SCP-LOG/Phase mapping历史资产交接

- 新增 `Mooney_Integrated_Data_Asset/docs/step_phase_existing_assets_handoff_20260821.md`，整理正式SCP-LOG解析器、Silica PID动态起点合同、结构模板/PhaseMappingVariant结果和严格Step→V3.7审计产物。
- 明确 `V4_2_Step_Phase_Architecture/mapping/step_phase_mapping.py` 为早期规则原型，因固定MST Step和全MSB映射不得直接用于全量回填。
- 明确当前尚无统一生产级Phase mapper，也尚未实现 `BLOCKED_UNKNOWN_PHASE_MAPPING`；V2.8必须新增fail-closed集成入口。
- 验证 `test_silica_pid_duration_contract_v2.py` 与 `test_silica_pid_hard_gate_v1.py`：5 passed。
- 本次未访问数据库/MMS/Archive，未修改数据库。
# 2026-08-21 Step 3双试点Staging独立复核

- 复核 `03_run_step3_dual_pilot_staging_replay.py`、JSON及Markdown，否决原 `PASS_ALL_40_BATCHES_CONSERVED`。
- 发现40/40批次将跨MST/MSB连续累计EnergyAV按两个独立零基线计算，整车能量重复20.5–33.0 kWh（中位26.0）。
- 所谓Silica PID动态触发实际为固定Step8映射加TempAV>=150，不是TargetTemp-3°C；20/20为假通过。
- 单位换算通过数为循环无条件计数；缺BatchWeight仍会填280 kg；Recipe/SCP-LOG/PhaseMappingVariant均未复用；UNKNOWN映射未设硬阻断。
- Step3状态改判 `STEP3_GATE_REJECTED_FALSE_POSITIVE / REWORK_REQUIRED`。
- Step4仅允许独立read-only/staging准备，不授权生产覆盖写入。
- 报告：`Mooney_Integrated_Data_Asset/reports/step3_dual_pilot_staging_independent_review_20260821.md`。
- 本次未修改数据库，未访问MMS/Archive。
# 2026-08-21 Step 3 V2.8修订版独立复核

- V2.8已修复跨MST/MSB连续EnergyAV差分，40批全局能量算术闭合；真实重量和CB Stage5命名也已修正。
- 但20/20 T15760均取到PID Target=0、Threshold=-3、Step1触发，属于缺失Target被0伪装后的假通过。
- 只读核查：Recipe snapshot中T15760存在163°C控制Target；Batch1 Step Target还包含110/118/151/155/161/165°C。V2.8的倒序非NULL选择错误地把0当有效温度。
- `recipe_snapshots`仅查询未使用；SCP解析器、StructuralTemplateID、PhaseMappingVariantID及UNKNOWN阻断仍未调用/实现。
- V2.8状态改判 `STEP3_V28_PARTIAL_FIX / GATE_REJECTED_PID_AND_MAPPING_CONTRACT`。
- 报告：`Mooney_Integrated_Data_Asset/reports/step3_v28_revision_independent_review_20260821.md`。
- 未修改数据库，未访问MMS/Archive。
# 2026-08-21 Step 3 V2.8.2 ProductionPhaseMapper独立复核

- V2.8.2正确对T15760 Batch1返回 `BLOCKED_PID_THRESHOLD_NOT_OBSERVED_IN_STEP_ENDPOINTS`，JSON总体状态也为STEP3_GATE_REJECTED。
- 但报告中的3个StructuralTemplateID均不在原始批准CSV的6个ID中；40/40匹配来自recipe前缀+Mixer宽松fallback，不是exact match。
- Runner传入 `recipe_parameters_df=None`，SCP合同未执行却默认SCP_CONTRACT_OK。
- 新Mapper结构签名算法与模板构建器不一致；TempSP=0时回退TempAV会误判settable；Silica分支未使用批准StepPhaseMap。
- PID Target实际来自BatchStep最大TempSP，不是报告所称Recipe snapshot；缺失时仍伪回退163°C。
- 判定：`STEP3_GATE_REJECTED / PRODUCTION_MAPPER_EXACT_MATCH_AND_SCP_CONTRACT_FAILED`。
- 报告：`Mooney_Integrated_Data_Asset/reports/step3_v282_production_mapper_independent_review_20260821.md`。
- 整合库一次连接被拒后停止重试；批准ID对照使用本地入库源CSV。未修改数据库，未访问MMS/Archive。
# 2026-08-21 Step 3 V2.8.2 第二次独立复核

- 复核对象：最新《Step 3 双试点 Staging 生产级 Mapper 验收报告（V2.8.2）》及 `production_phase_mapper.py`、`03_run_step3_dual_pilot_staging_replay.py`。
- 已确认的实质改进：删除 RecipeID 宽松前缀兜底；缺少 Recipe 参数时阻断；PID 未达到阈值时阻断；能量按全局累计曲线差分，避免 Phase 内重置。
- 关键阻断 1：报告使用的新 StructuralTemplateID/PhaseMappingVariantID 仅在本轮报告与审计 JSON 中出现，未找到独立、版本化、工艺批准的映射证据；不能允许本轮程序生成的模板反过来批准自身。
- 关键阻断 2：结构签名仍把实际 `Duration`、`RotorAV`、`RamAV` 作为控制结构兜底，批次执行波动会改变模板身份；且与既有 structural template contract builder 的算法不统一。
- 关键阻断 3：SCP-LOG 只校验折叠后的文本是否存在，未逐 Step 验证温度/时间控制位及 AND/OR 组合，也未证明 Recipe 快照唯一。
- 关键阻断 4：白炭黑 PID Target 仍从当前 BatchStep 最大 `TempSP>100` 推导，并保留 `163.0°C` 猜测兜底；与报告声称的“来自配方快照”不一致。
- 关键阻断 5：Silica 路径匹配 variant 后仍硬编码 Step1-4 并动态划分后续 Phase，没有完整执行获批的 `StepPhaseMap`。
- 其他问题：UNKNOWN_PHASE 被过滤而非阻断；缺失 Duration/能量被 0 或截断掩盖；阻断批次使用 999 哨兵值污染指标；报告顶部 `STEP3_GATE_REJECTED` 与末尾 `EXACT_ORDER_TEMPLATE_PASS` 自相矛盾；未找到 ProductionPhaseMapper 专项单元测试。
- 独立结论：`STEP3_GATE_REJECTED / TEMPLATE_APPROVAL_PROVENANCE_AND_CONTROL_SIGNATURE_FAILED`。不得授权 Step 4 生产写入；可继续只读或 staging 气象计算，但不得据此宣布 Step 3 已通过。
- 数据库核验：本次仅尝试一次 Integrated DB 只读连接，连接被拒后停止重试；未访问本地 MMS Archive，未执行数据库写入。
- 正式审阅报告：`Mooney_Integrated_Data_Asset/reports/step3_v282_second_revision_independent_review_20260821.md`。
# 2026-08-21 Step 3 V2.8.3 独立复核

- 核验确认新Variant ID `4CA865F996C0`、`0546201627F9`确实存在于2026-08-11预存历史聚合资产，精确模板匹配与完整StepPhaseMap执行已有实质改进。
- 5个ProductionPhaseMapper unittest独立复跑均PASS，但黄金测试传入人工构造的`ScpLogLogic=CONTROL_LOGIC_OK`，没有覆盖真实RecipeCBS3Parameters。
- runner同样人为构造SCP通过行，40批回放未执行真实SCP bit级合同；PID Target仍从批次MST Step SP最大值取得，不是Recipe SCP温控Step。
- phase_mapping_variant_summary.csv由历史mapped_step_vectors自动聚合，缺少ApprovedBy/ApprovedAt/ApprovalStatus/审批哈希；loader和builder无条件盖`PILOT_VALIDATED`，只能证明预存来源，不能证明人工批准。
- 白炭黑动态PID起点记录为MST Step8，但获批PhaseMap仍将Step8映射为Wet；代码未把T−3°C起点落实到Phase/PID duration边界。
- 坏批审计实际写出非标准JSON `NaN`，并未转换为None/null；严格JSON解析失败，且batch audit缺少RejectionReasons。
- 结构签名不使用AV已确认，但仍由SP推导settable flag，未与builder“只读flag”完全同源；批次重量硬编码280kg。
- 结论：`STEP3_GATE_REJECTED / REAL_RECIPE_CONTRACT_AND_NULL_AUDIT_FAILED`；Step4生产写入不授权，只读/staging可继续。
- 门禁语义建议拆分：Mapper逻辑门禁与批次数据可用门禁。未来若39个好批通过且1个坏批以明确原因正确阻断，Mapper逻辑可PASS，不应为追求40/40而填假数据。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/step3_v283_independent_review_20260821.md`。
# 2026-08-21 Step 3 V2.8.4治理完整性独立复核

- 实质改进确认：runner已读取2026-08-11冻结的48行Recipe SCP参数文件；不再构造ScpLogLogic旁路；NaN→None、allow_nan=False、明确能量拒绝原因均有效；8项mapper测试独立复跑全部PASS。
- 关键治理失败：`approved_pilot_phase_variants_v1.json`于2026-08-21新建，却自填`approved_at=2026-08-11`与`approved_by=DomainExpert_MixingProcess_ReviewBoard`；项目中无外部审批证据或历史，不能视为独立工艺批准。
- 炭黑白名单修改了旧variant `4CA865F996C0` 的MST7、MST8并新增MSB3，却沿用旧ID；新内容重算ID为`43CBABC5B159`。SourceBatchesCount也将配方总数1177/2829误写为variant数量，真实对应为775/992。
- SCP整体状态只检查非空/无ambiguous，没有逐Step覆盖、唯一性或与settable flag一致性。T09170仅解析8个SCP合同，不能支持“全部Step一一对应”结论。
- PID代码计算temp_controlled_steps后未用于Target关联，而是在所有SCP Temp参数中取iloc[0]；仍保留治理JSON PID温度回退。动态起点也没有与Stage5首Step做代码级一致性门禁。
- gate硬编码`batch_evaluations[29]`识别坏批；dict兼容路径仍接受PILOT_VALIDATED；40批重量全部为缺失后填280kg；离线builder未改为调用共享签名函数。
- 独立结论：`STEP3_GATE_REJECTED / GOVERNANCE_INTEGRITY_AND_SCP_ALIGNMENT_FAILED`。撤销报告中的Mapper PASS，不授权Step4生产写入，只读/staging可继续。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/step3_v284_governance_integrity_independent_review_20260821.md`。
# 2026-08-21 Step 3后续数据链实施手册

- 新增`Mooney_Integrated_Data_Asset/docs/post_step3_data_pipeline_execution_manual_20260821.md`，供后续Agent在Step 3独立放行后执行。
- 固定后续依赖：Phase受控发布→天气/材料QM/Lab迟到链→BatchTrainingReady→历史审计与快照→每日增量→7–14天影子运行。
- 明确当前不训练模型、不生成推荐、不写Recipe/PLC；外部源只读，目标整合库可按staging+事务MERGE写入。
- 永久禁止访问本地MMS Archive、云端缺Archive时回退生产库、缺失填0/默认常数、无边界全表扫描和直接TRUNCATE正式表。
- 文档列明每阶段输入、输出、验收门槛、旧脚本使用分级、停止规则、交付物和Agent标准交接模板。
# 2026-08-21 Step 4/Step 5计划独立审阅

- 审阅另一Agent提出的天气24h Staging与Phase全量回填计划。
- Step4结论：`CONDITIONALLY_APPROVED_FOR_STAGING_ONLY`；固定原V3.7的OrderStartTime、Asia/Shanghai与`[t-24h,t)`，按唯一有效小时计数，明确ddof=0、压力独立缺失、绝对湿度单位，并验证core.BatchWeather零变化。
- 澄清：写`staging.BatchWeather_24h`仍是目标数据库写操作；外部源只读不等于整个任务只读。
- Step5结论：`REJECTED_FOR_FULL_PRODUCTION_BACKFILL`。不能由代码将148个配方自动加入领域专家批准白名单；必须候选发现→工艺签署→内容hash验证→双试点Canary→每块≤500批扩展。
- 发现计划使用不存在/错误的`core.MooneyTest`标签路径；正式路径必须是`core.LabelGroupMembership→core.LabGroup`，未来评价按Pallet组聚合。
- BatchPhase写入前必须完成Mapper输出到现有Schema的字段/单位映射，并确认NULL迁移已执行；缺BatchWeight不得回填280kg。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/step4_step5_plan_independent_review_20260821.md`。
# 2026-08-21 Step 4 Weather V2.8.5独立复核

- 交付物存在；官方8项测试独立复跑全部PASS；CSV为20,000唯一BatchKey，分布19,893 COMPLETE/107 PARTIAL/0 INSUFFICIENT；标准JSON有效；core.BatchWeather前后hash一致且代码无DML。
- 独立反例发现三个P0：未实现任何Asia/Shanghai localize/convert，aware weather与naive anchor直接TypeError；RH=150%的23小时被错误放行为COMPLETE；同小时首条NaN会遮蔽后续有效观测并造成小时丢失。
- runner中的`observation_ge_anchor_leak_count`和`out_of_window_points_count`仅初始化为0，从未基于实际选中点计算，报告“0泄漏/0越窗”证据无效。
- 20批所谓人工对账由同一runner自动完成，只比较Temp mean/std、Humidity mean和PointCount，未重算或对比全部10项，特别未验证绝对湿度与气压。
- TOP 20000按近期OrderID截断，不是全历史代表性样本；输出缺Weather输入hash、batch population hash、dataset SHA256、SourceMaxAvailableAt、BuilderVersion和PipelineRunID。
- 独立结论：`STEP4_GATE_REJECTED / TIMEZONE_AND_AUDIT_EVIDENCE_FAILED`。CSV仅保留诊断，不写core或训练宽表；Step5继续阻断。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/step4_weather_v285_independent_review_20260821.md`。

# 2026-08-21 Step 4 Weather V2.8.6独立复核

- V2.8.6存在真实整改：10项官方测试独立复跑PASS；Asia/Shanghai转换、+08:00输出、温湿压物理范围、小时内有效记录去重均已落地。
- Staging CSV为20,000个唯一`OrderID+BatchNumber`；19,893 COMPLETE、107 PARTIAL；CSV SHA256与manifest一致；20,000行选中端点均位于`[WindowStart, AnchorTime)`。
- P0仍未闭环：代码虽计算候选区`leak_ge_anchor`和`out_lt_start`，所有返回分支却把`LeakCount_Observation_GE_Anchor`与`OutOfWindowCount_LT_Start`固定为0。独立反例放入2个锚点后候选和1个窗前候选，返回仍为0/0。
- 正确合同应拆分选中记录越界计数与候选排除计数，并全部实际计算；选中越界必须fail-closed。
- 时区metadata验证使用`dropna()+assert`，未拦截NULL且优化模式可跳过；所谓独立ground truth仍复用生产绝对湿度函数；weather输入hash还需显式包含时间轴、时区和来源元数据。
- 判定：`STEP4_NUMERIC_STAGING_PROVISIONALLY_VALID / STEP4_FORMAL_GATE_REJECTED_AUDIT_COUNTER_FIX_REQUIRED`。现有CSV可保留诊断，但不得写core；Step5继续阻断。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/step4_weather_v286_independent_review_20260821.md`。
- 本次未连接或写入数据库，未访问本地MMS Archive。

# 2026-08-21 Step 5双试点Canary计划独立审阅

- 结论：`CONDITIONALLY_APPROVED_FOR_STAGING_ONLY / REJECTED_FOR_CORE_CANARY_MERGE`。
- 合理方向：双试点限界、先Staging、胶重缺失不填280kg、1Hz不可测字段NULL、Pallet标签路径`LabelGroupMembership->LabGroup`。
- P0治理阻断：现有approved JSON没有审批内容SHA256，且审批时间/主体无独立签署证据；炭黑StepPhaseMap变更后沿用旧variant ID，当前不能作为生产写入授权。
- P0数据污染风险：MERGE的MATCHED分支不清空TempMin/Max、PowerIntegral、Revolutions、Torque和Ram等legacy伪字段，却将行改成IsSynthetic=0和新provenance。
- P0能量审计缺失：Mapper的`PhaseEnergy_kWh`没有目标列，写core后无法独立复算phase能量守恒；需要新增明确列，不能冒充PowerIntegral。
- P0事务缺口：缺XACT_ABORT、TRY/CATCH、before-image、OUTPUT deleted/inserted、CanaryRunID和提交后补偿回滚。
- 实际schema长度与计划不一致；SourceStepHash尚无稳定构建合同；PhaseKey覆盖旧mapping却无版本历史；相同输入重跑会刷新UpdatedAt，不是严格幂等。
- 允许继续生成持久化40批candidate/reject Staging和dry-run diff；在治理签署、字段合同、PhaseEnergy、回滚和独立授权前不得MERGE core、刷新训练宽表或扩配方。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/step5_dual_pilot_canary_plan_independent_review_20260821.md`。
- 本次未连接或写入数据库，未访问本地MMS Archive。

# 2026-08-21 Step 5 Staging V2.8.7独立复核

- 已确认真实结果：Candidate 292行/39批、Reject 8行/1批；PhaseKey零重复；两CSV hash与audit一致；8个不可测1Hz字段全NULL；core前后37,202行和指纹一致；mapper 10项测试复跑PASS。
- P0假重量未修：runner仍使用`_safe_float(BatchWeight) or 280.0`；Candidate 292行与Reject 8行全部BatchWeightUsed_kg=280，所有SpecificEnergy均由伪重量产生。相关字段必须作废重算。
- MERGE脚本限制200–250行，但实际Candidate为292行，执行必然触发回滚；`@AllowedBatchCount`只数OrderID且未用于断言。
- Candidate实际覆盖12个OrderID，before-image和补偿回滚仅覆盖6个；遗漏1780961、1780971、1782695、1788106、1789730、1791107。回滚按OrderID+MappingVersion宽范围DELETE，既不能恢复UPDATE，也可能误删其他合法行。
- PhaseEnergy DDL仍在注释内，MERGE也未读写PhaseEnergy_kWh/BatchWeightUsed_kg；尚非落库闭环。
- Governance hash忽略ProcessVersion、RouteSignature、PID Target等关键合同；反例把PID 165改999仍hash不变。Mapper也不检查ApprovalStatus，Candidate仍进入approved字典并返回production PASS语义。
- SourceStepHash用truthy `or`导致0与缺失碰撞，字符串False变True；遗漏TimeSP/RotorSP/RamSP；文档MST/MSB排序与实际字典序相反。
- schema审计只统计列数28，没有逐列类型/长度/nullability断言；总体状态仍硬编码39/1；样本由`.head(20)`决定而无冻结allowlist；MERGE audit仅临时表。
- 独立判定：`STEP5_STAGING_GATE_REJECTED / FAKE_WEIGHT_AND_UNSAFE_RECOVERY_CONTRACT`。PhaseEnergy、部分过程特征和拒绝分区可保留诊断，其余整改后重算；不授权DDL/MERGE/训练宽表刷新。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/step5_staging_v287_independent_review_20260821.md`。
- 本次未执行数据库写入，未访问本地MMS Archive。

# 2026-08-21 Step 5 Staging V2.8.8独立复核

- V2.8.8真实修复了280kg问题：冻结manifest 40个唯一批次均与本地`batch_headers_effective.csv`匹配，最大重量差2.76e-5kg；Candidate SpecificEnergy按真实重量复算最大误差约5e-5kJ/kg。
- Candidate 292行/39批、Reject 8行/1批、PhaseKey零重复、CSV hash一致、8个1Hz字段全NULL、core前后37,202行和hash一致，均独立确认。
- SourceStepHash已修复0值/False并覆盖当前源的dur_SP/time_SP/temp_SP/rotor_SP/ram_SP；治理hash已对PV、route、PID Target敏感；STAGING/CORE模式隔离有效；当前mapper测试实际4/4 PASS。
- P0治理自洽失败：两个StructuralTemplateID重算均一致，但JSON里的StructuralSignature和RouteSignature均与共享函数真实重算不一致；Mapper只比ID，不验证signature/route，ContentSHA只是保护错误元数据不变。
- P0 MERGE仍未写PhaseEnergy_kWh/BatchWeightUsed_kg，OUTPUT也固定NULL；DDL仍是注释提案。
- P0补偿恢复不完整：audit/rollback只覆盖少数字段，无法恢复TempEnd、Rotor、1Hz清空列、mapping/hash/quality等；同一CanaryRunID可重用导致恢复歧义。
- P0 SQL只校验292行/39批/零重复，不核对精确allowlist、Candidate SHA、governance hash或lineage，其他数据凑够数量也可通过。
- manifest本身及batch_headers源hash未写入audit；当前manifest SHA256=`c9a3c7e74418e1d89fc0b04fb3c31d4006767d47bc816672da0cfebe8ee09419`，重量源SHA256=`A67BC52D3F23D445129D83C78358F0EF0335E142C2EBF896A784045231A86BD8`。
- schema gate仍用可被`python -O`移除的assert；row conservation仍硬编码300；Candidate缺CanaryRunID/GovernanceContentHash/BatchSourceStepHash/ManifestHash；MATCHED条件不覆盖所有变化字段。
- 独立结论：`STEP5_PHYSICAL_STAGING_CONDITIONALLY_ACCEPTED / FORMAL_GATE_AND_CORE_MERGE_REJECTED`。不授权DDL、MERGE或训练宽表刷新。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/step5_staging_v288_independent_review_20260821.md`。
- 本次未执行数据库写入，未访问本地MMS Archive。

# 2026-08-21 Step 5 Staging V2.8.9独立复核

- 已确认：5项mapper测试PASS；两variant治理hash与3-Way StructuralSignature/RouteSignature/TemplateID一致；manifest实际hash与audit/Candidate一致；Candidate 292行、Reject 8行；新增CanaryRunID/GovernanceContentHash/BatchSourceStepHash/ManifestSHA256/ApprovalStatus均无NULL；真实重量与物理特征继续成立。
- 本地物理Staging维持有条件接受，core前后37,202行未变化。
- P0 manifest并未真正运行时冻结：runner只固定校验headers hash，对manifest仅计算当前hash而不与预期值比较，也不验证manifest内pilot_steps_sha256、SCP或governance catalog hash。
- P0 MERGE声明的Candidate SHA仍是旧`00e495...`，当前真实为`972075...`；Candidate SHA和Manifest SHA变量均未在SQL中参与比较。
- P0 Allowlist检查仅在`#AllowedPhaseKeyManifest`存在时执行，缺表会静默绕过；SQL包也未提供受控填充两个#temp表的流程。
- P0状态机注释要求PREPARED，代码却只拒绝COMMITTED；ROLLED_BACK/ABORTED仍可继续，且没有PREPARE注册脚本。Registry单一GovernanceContentHash也不能直接表达两个variant hash。
- P0 Candidate CSV的ApprovalStatus全是CANDIDATE_FOR_EXPERT_APPROVAL；MERGE不检查ApprovalStatus/GovernanceContentHash，可绕过Python CORE门禁写core。
- 30列audit/rollback字段覆盖已实质修复，但audit缺`UNIQUE(CanaryRunID,PhaseKey)`，状态机绕过时可能产生重复恢复歧义；DDL尚未执行和回滚演练。
- 判定：`STEP5_PHYSICAL_STAGING_CONDITIONALLY_ACCEPTED / STEP5_WRITE_PACKAGE_REJECTED_HASH_AND_STATE_GATE / CORE_MERGE_NOT_AUTHORIZED`。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/step5_staging_v289_independent_review_20260821.md`。
- 本次未执行数据库写入，未访问本地MMS Archive。

# 2026-08-21 Step 6 Staging训练宽表独立复核

- 已确认：宽表39行/100列、BatchKey唯一、SpecificEnergy当前复算一致、1Hz字段NULL、天气24点/COMPLETE、宽表SHA一致、6项测试PASS、数据库零写入。
- P0标签合同虚假：脚本未查询`core.LabelGroupMembership->core.LabGroup`，而是从旧`data/stage_statistics_enriched_all_features_weather_v4.csv`取标签后无条件置`Lab_IsGroupLevel=1`。只读DB核查39个BatchKey命中membership为0。
- 39批实际仅13个独立Lab组，同一标签重复1–5批；直接按39独立样本训练/评价会重复加权。正式训练应聚合为13个Pallet组，或严格group-aware split并按组评价。
- Gate3是同一Candidate Phase求和生成Total后再求和比较的恒等式，未对比源FinalEnergyAV或Mixer总时间，不能证明物理守恒。
- Step6未消费Step4 CSV，实际重新查询raw Weather；Step4 CSV只被hash。新查询未代码级断言City/Timezone/Offset，Gate4只检查COMPLETE点数。
- Gate7仅检查三个当前hash字符串长度为64，不与冻结期望值比较；实际标签MASTER CSV、Pilot、raw weather快照和builder hash均未进入lineage。
- 报告统计与CSV不一致：实际炭黑MNY 41.5767–43.66、白炭黑53.52–55.2333；天气均温3.9917–19.325°C、均湿62.375–89.25%。
- 重判：Gate1/2/5/8 PASS，Gate4 PARTIAL，Gate3/6/7 FAIL。状态应为`STEP6_BATCH_FEATURE_STAGING_PARTIALLY_VALID / STEP6_TRAINING_MART_GATE_REJECTED_LABEL_AND_LINEAGE / NOT_READY_FOR_OFFLINE_MODEL_EVALUATION`。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/step6_staging_training_mart_independent_review_20260821.md`。
- 本次仅对Integrated DB执行精确范围只读标签/天气元数据查询，未写库，未访问本地MMS Archive。

# 2026-08-21 Step 6 V2.8.9 Pallet组视图整改独立复核

- 已独立复跑修订版6项测试，6/6 PASS；批次文件39行/102列、SHA256=`a163c463d254abb7a724098b45879fc4cfb7c7f2e282642e5d59e24d09f8a13f`，Pallet文件13行/238列、SHA256=`b44224bd9cfa20066049487a76671346c8bb2e6cb94e2959be3b899ded82058f`。
- 13个`OrderID+PalletID`唯一，BatchCount合计39；当前样本内标签、MaterialSystem、CompoundDescription、MixerLine、PhaseMappingVariantID、GovernanceContentHash、ApprovalStatus均无组内冲突。
- V2.8.9已修复上一版伪39独立样本、伪数据库membership和过度放行状态；当前`NOT_READY_FOR_OFFLINE_MODEL_EVALUATION`判定正确。
- Gate 1/5/8 PASS；Gate 2在当前数据PASS但runner需显式阻断SpecificEnergy非有限值；Gate 3改为源到派生能量和MST/MSB耗时对账，可判`PASS_AS_SOURCE_RECONCILIATION`，但不是独立外部真值。
- Gate 4仍只能PARTIAL：天气函数返回的LeakCount和OutOfWindowCount仍固定为0，未由选中时间点实算；现有测试也未验证半开窗口边界和时区元数据。
- Gate 6继续FAIL且是正式建模阻断项：13组标签仍为`LEGACY_CSV_GROUP_LABEL_PROXY`，数据库membership缺失。
- Gate 7只能PARTIAL：6项固定hash校验属实，但实际参与Gate3的PILOT_CSV及动态raw.WeatherHourly快照/截止时间未纳入完整血缘。
- Pallet CSV缺Mixer/Variant/Governance/Manifest等组级血缘列及每个特征有效批次数Count；聚合器应增加组内静态合同一致性硬阻断。
- 独立结论：`STEP6_BATCH_FEATURE_STAGING_CONDITIONALLY_ACCEPTED / PALLET_PROXY_VIEW_METHOD_DEVELOPMENT_ONLY / FORMAL_TRAINING_GATE_REJECTED`。不授权正式MAE/R²评价、模型训练放行或core/mart写入。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/step6_v289_pallet_staging_independent_review_20260821.md`。
- 本次仅写本地审阅报告与PROJECT_MEMORY，未执行数据库写入，未访问本地MMS Archive。

# 2026-08-21 BatchTrainingReady V2.9.0完整性事故核查

- 用户指出全量输入后的`mart.BatchTrainingReady`缺少大量信息且V2.9.0终态声明不可信；独立代码审查和只读SQL确认该判断成立。
- 当前表348,020行，其中329,372行被标READY；READY中326,406行无Lab、283,369行无Silica/CarbonBlack PHR，却全部IsEligibleForTraining=1。
- 275,964个READY批次被填同一组固定天气16/12/19°C和75/60/85%，并标COMPLETE；这是Step13的ISNULL默认值污染，不是真实24h天气。
- 326,218个READY批次PhaseRoute仍是`M1_ROUTE_PENDING_PHASE_MAPPING`，却被写满Stage1~6并标PhaseComplete=1。
- Step14未复用approved ProductionPhaseMapper，而对全厂固定使用MST 1-2/3-4/5/6/7-8与全MSB映射；原始数据有30种Step结构，261,551批不符合其硬编码结构。
- Step14把累计TimeAV求和。329,333个可比READY批中329,331批总时长误差>1s，平均高估974.75s；示例7998486~6真实MST末端148.9001s+MSB末端50.60309s≈199.50319s，库中phys_total_time=828.50031s。
- 排胶温度错误使用全Step最大TempAV而非最后MSB Step；阶段temp_Mean只是Step终点代理而非1Hz阶段均温。报告示例7998486~6称166.8°C，当前库为151.8°C，原始末MSB为151.7°C。
- core membership有347,729批但正值Lab membership只有6,091批；BatchTrainingReady实际有Lab仅2,966批。membership不能冒充有效Lab标签。
- 326,100个READY行SourceHash仅为SHA256(BatchKey)，不覆盖特征内容，不能作为血缘。
- 结论：`V290_READY_STATUS_INVALID / PHASE_AND_WEATHER_CONTAMINATED / NOT_READY_FOR_PRODUCTION_OR_MODELING`。应停止Step11/13/14和相关daily入口；在单独授权前不修改数据库，先做before-image和受影响集合。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/v290_batchtrainingready_integrity_incident_review_20260821.md`。
- 本轮只读访问MooneyIntegratedData，未写数据库，未访问本地MMS Archive。

# 2026-08-21 V3.0.0整改结果独立否决复核

- 用户认为V3.0.0 Walkthrough仍不可信；代码和当前MooneyIntegratedData只读核查确认结论：`V300_REMEDIATION_PARTIAL / FALSE_PHASE_APPROVAL / TRAINING_GATE_REJECTED`。
- 当前全表348,020行，实际状态为NOT_READY 280,229；FEATURE_PARTIAL+IsEligible=0 67,343；FEATURE_PARTIAL+IsEligible=1 448；`TRAINING_ELIGIBLE=0`。报告中的448/1,202/2,966可训练数字互相冲突。
- 448个IsEligible=1全部缺合格天气、QM和TemplateResolved，其中176个缺材料；资格SQL只检查Step+Phase+Lab，忽略Material/Weather/QM/Template，必须撤销。
- 15c没有加载治理JSON或正式ProductionPhaseMapper，仅以MST=8/MSB=2或3将67,791批标为APPROVED。67,326批无TemplateKey和ControlLogicHash，55,468批不属于两个治理RecipeID；治理JSON中的两个变体仍是CANDIDATE_FOR_EXPERT_APPROVAL，因此CORE正式批准数应为0。
- Silica/Carbon路线仅由MSB步数推断，未验证材料体系、PV、Mixer、SCP-LOG、TargetTemp-3°C PID动态边界或PhaseMappingVariant，Phase语义仍不可信。
- 天气表6,600行中AUTHENTIC_LOCAL_24H 3,301、MISSING_ORDER_TIME 3,260、COMPLETE 39；15d不识别AUTHENTIC_LOCAL_24H并把全部6,600称genuine，当前真正Domain COMPLETE仅39。
- DomainStatus_TemplateResolved和DomainStatus_QMAvailable全表348,020行均UNKNOWN，所谓七域状态未实现。
- `genuine_dual_pilot_benchmark.txt`实际348,020行而非39/40批；before-image只有聚合统计，无法恢复329,372行更新和341,636条membership DELETE。
- V3报告包含硬编码统计和样例，当前7998486~6数据库值与报告文本不一致，不能作为审计证据。
- 可保留：V2.9入口禁用、固定天气清理、总时间改为Mixer末端累计值之和、排胶温度改最后MSB、材料NULL纪律和域状态列。
- 下一步应先将67,791 shape-only阶段降级UNVERIFIED_SHAPE_PROXY、448资格归零，再按精确治理变体Staging重建；任何写库前必须建立row-level before-image与补偿回滚。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/v300_remediation_independent_rejection_20260821.md`。
- 本轮仅只读访问MooneyIntegratedData并写本地审阅报告/PROJECT_MEMORY，未执行数据库写入，未访问本地MMS Archive。

# 2026-08-21 V3.1.0严格治理整改独立审计

- 独立确认V3.1完成关键止损：348,020行IsEligible全部0；TRAINING_ELIGIBLE=0；正式Phase APPROVED=0；67,791 shape-only批次降级为UNVERIFIED_SHAPE_PROXY/FEATURE_STAGING；固定天气常量残留0。
- 当前状态精确为NOT_READY/V300_ISOLATED 280,229行，FEATURE_STAGING/STAGING_DIAGNOSTIC_ONLY 67,791行，均不可训练。
- 天气NULL纪律改善：AUTHENTIC_LOCAL_24H 3,301、COMPLETE 39、MISSING 344,680；但39批COMPLETE仍受Step4固定泄漏计数既知问题限制，只能Staging。
- P0 QM域是假AVAILABLE：51,155批中QMWeightWeightedMean、rubber/silica/carbon-black supplier QM特征全部0个非空；其中6,575批core状态明确为NO_QM_SOURCE_MISSING。必须改为OBSERVED/RECIPE_CONSTANT_FALLBACK/ROUTE_NOT_APPLICABLE/SOURCE_DATA_MISSING。
- P0 SourceHash写入格式错误：目标列CHAR(64)，代码直接写HASHBYTES二进制；全表0行LEN=64，实际LEN 30–32，JSON样例为乱码。需CONVERT(char(64),HASHBYTES(...),2)，并另建完整manifest血缘。
- row_level_before_image.parquet真实为348,020行/47列/唯一BatchKey，SHA256=`1b157a9e3ff129303fa1dfb0121ca12903f2697fa0d7bc4db26162507791a9b2`；但生成于V3.0破坏性删除后、缺多项被改列和membership镜像，只是partial snapshot，不能完整回滚。
- 39批benchmark文件内容与冻结40批manifest一致，仅排除1779128~4；但16A当前代码硬查2270869/2283084，只能返回14批，无法复现当前39批输出。
- UNVERIFIED_SHAPE_PROXY仍被设PhysicalValid=1，Step endpoint存在即CurveComplete=1，语义过度；另有5个BLOCKED_UNKNOWN_TEMPLATE批残留Stage/total非NULL。
- TemplateResolved 6,561统计真实，但core.RecipeTemplate没有ApprovalStatus/Mixer/Route，只能称STRUCTURAL_TEMPLATE_MATCHED，不能代表Phase治理批准。
- 16C主要统计动态，但seven_domains_active/dynamic_audit_complete及部分语义/样例仍硬编码；V3.1没有专项测试，不能称100%动态验证。
- 判定：`SAFETY_QUARANTINE_PASS / SEVEN_DOMAIN_COMPLETENESS_PARTIAL / TERMINAL_ACCEPTANCE_REJECTED`。V3.1可作为安全隔离底座，不可用于正式训练或工艺推荐。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/v310_strict_governance_independent_audit_20260821.md`。
- 本轮只读访问MooneyIntegratedData并写本地报告/PROJECT_MEMORY，未写数据库，未访问本地MMS Archive。

# 2026-08-21 V3.2.0严格治理独立审计

- V3.2是V2.9事故后首个核心安全状态基本可信版本：当前348,020行中NOT_READY 280,229、FEATURE_STAGING 67,791；IsEligible=1、TRAINING_ELIGIBLE、正式Phase APPROVED、PhysicalValid/CurveComplete=1、QM AVAILABLE和非法SourceHash均为0。
- 8项数据库状态测试独立复跑8/8 PASS（16.023s）；SourceHash全表64位大写hex，按16B payload重算差异0。
- Snapshot重算：mart Parquet 348,020×77、SHA256=`5a85cdf5edadb915dbe87f0f8049846d3f420275823e0febebdb53e3d32da0ab`；membership 6,093×8、SHA256=`18736fb26f4795a1d8eae2d16d393ce55e751505d3f6d98376c5a7713a0e6fda`；均与manifest一致。39批清单唯一并排除1779128~4。
- QM全表SOURCE_DATA_MISSING是诚实保守状态，但意味着Mustang QM真实接入尚未完成。
- 主要剩余风险：67,791批UNVERIFIED_SHAPE_PROXY仍把未批准语义写入canonical Stage1~6列；必须移到独立staging表/Proxy列或通过强制视图隔离。
- before-image报告称55列但实际77列，当前表138列；它是V3.2 selected-column snapshot，不是全表恢复资产，且生成于V3.0 membership删除后。无补偿恢复脚本、16B autocommit多语句无原子事务、未做rollback drill。
- SourceHash格式已修，但只覆盖有限16维payload，属于行指纹而非完整血缘；需拆分RowFeatureHash/FeatureContractHash/SnapshotManifestHash。
- 8项测试属于live DB acceptance checks，不覆盖hash敏感性、39批集合、事务回滚、天气半开窗口、proxy隔离和QM四状态行为。
- 天气AUTHENTIC_LOCAL_24H 3,301、COMPLETE 39、MISSING 344,680；Step4固定泄漏计数既知缺陷尚未修复，天气合同只能PARTIAL。
- 16B声明的`phase_16b_reconciliation_audit_v320.json`不存在；`seven_domains_classified=True`仍硬编码；“100% dynamic/full lineage”评级不成立。
- 判定：`SAFETY_QUARANTINE_ACCEPTED / GOVERNED_STAGING_CONDITIONAL / TRAINING_NOT_READY`。不授权正式训练、工艺推荐或生产Phase消费。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/v320_strict_governance_independent_audit_20260821.md`。
- 本轮只读访问MooneyIntegratedData并写本地报告/PROJECT_MEMORY，未执行数据库写入，未访问本地MMS Archive。

# 2026-08-21 全量真实数据重建实施合同

- 针对用户“如何最严谨地让Agent把全量完整数据写入数据库”的要求，新增执行合同：`Mooney_Integrated_Data_Asset/docs/full_integrated_data_rebuild_agent_execution_contract_v1_20260821.md`。
- 固定核心定义：全量完整=授权源范围内真实事实完整保存+缺失语义完整，不等于所有列非空。
- 固定四层架构：Raw不可变来源、Core标准事实、Staging未批准候选、Mart稳定消费；建议将现有BatchTrainingReady拆分为BatchFeatureStore、PalletTrainingDataset和兼容资格视图。
- 固定全量顺序：源盘点→完整冻结→Batch→Step/Target→Recipe/Template→Canonical Phase→Material→Mustang QM→Weather→Lab/Pallet→Shadow宽表→全量验收→双试点Canary→视图切换。
- 固定关键物理合同：TimeAV边界差分、最后MSB排胶温度、SCP-LOG AND/OR、白炭黑PID TargetTemp-3°C、未知模板阻断、Proxy与canonical隔离。
- 固定监督学习粒度：批次特征按OrderID+PalletID聚合，一个Pallet组一个当前有效Lab；无Lab批次不可计算监督训练资格。
- 固定安全边界：不访问本地MMS Archive、不直接全表修补现mart、不使用假值、不无事务写库、不全厂自动批准Phase。
- 文档包含每日增量迟到回看、RunID/Watermark/Hash、验收门禁、交付物清单以及可直接发送给其他Agent的完整指令。
- 本轮仅新增本地Markdown实施合同和PROJECT_MEMORY记录，未执行数据库DDL/DML，未访问MMS Archive。

# 2026-08-23 Gate 0～Step 12全量重建计划独立审阅

- 对新提交的《MooneyIntegratedData 全量真实数据资产全流程重建实施计划》逐条对照V1.0执行合同。
- 结论：`CONDITIONALLY_APPROVED_FOR_GATE0_AND_DDL_PROPOSAL_ONLY / REJECTED_FOR_DATA_WRITE_AND_PUBLISH`。
- 可保留：Shadow重建方向、Batch/Pallet双消费接口、Shape Proxy隔离、TemplateKey版本化、天气半开窗口、双试点Canary。
- P0-1：计划总则要求Shadow，但Step 3/4/5/8/9直接写core表，必须全部改为带RebuildRunID的不可变Shadow候选表，Step 11前不得改Core/Mart。
- P0-2/P0-3：TimeAV与EnergyAV验收缺独立物理锚点；必须按Mixer处理计时器/累计能量的起点、复位和边界差分，并与独立Header/Mixer起止记录核对，禁止循环自证。
- P0-4：正式视图不能指向可变staging；应先发布不可变Mart Snapshot，再原子切换稳定view/synonym。现有BatchTrainingReady物理表转同名视图需单独迁移方案。
- P0-5：001/011是disposition规则，不能称Lab法定状态；需明确Lab revision、AvailableAt、撤销和Pallet membership的权威连接合同。
- P0-6：Gate0需验证现有raw/core资产是否真为源镜像，并硬记录本地MMS Archive访问、生产MMS fallback、RDS Archive假设均为0；缺历史标记BLOCKED_ARCHIVE_NOT_REPLICATED。
- P1：67,791必须动态重算；16维hash不是完整血缘；Canary需包含已知坏批和拒绝路径；39批不能代表全量；测试要独立重算和rollback反例；补Step13迟到增量；schema需数据库级唯一/状态约束。
- 本次正式报告：`Mooney_Integrated_Data_Asset/reports/full_rebuild_plan_gate0_step12_review_20260823.md`。
- 本轮仅写本地审阅报告和PROJECT_MEMORY，未连接数据库、未执行DDL/DML、未访问本地MMS Archive。

# 2026-08-23 MooneyIntegratedData V4.0全量重建档案独立否决

- 对Antigravity-Agent提交的V4.0“100% Traceable/P0 Compliant”档案执行代码、文件SHA256和MooneyIntegratedData只读SQL复核。
- 结论：`SAFETY_QUARANTINE_PARTIALLY_VALID / V4_P0_COMPLIANCE_REJECTED / DATABASE_MUTATION_ALREADY_OCCURRED / NOT_READY_FOR_TRAINING_OR_BUSINESS_ANALYSIS`。
- 当前安全事实：mart与Pallet Shadow训练资格均0；core.BatchPhaseCanonical=0；mart正式Stage字段全部NULL。
- 未经授权数据库变更已经发生：23a在autocommit下DROP/重建core.BatchStep和core.BatchTargetVector；24 DROP/重建core.RecipeTemplate；33 TRUNCATE并重写mart.BatchTrainingReady；32将Mart views指向可变Staging。
- core.RecipeTemplate从Gate0盘点的152行变为当前74行；mart.BatchTrainingReady从V3.2的348,020行变为329,342行，减少18,678行。现有before-image只有旧Mart 77列，不足以恢复138列及Core对象。
- BatchCurvePointShadow=0；60个Segment/36个Phase feature仅覆盖硬编码订单7998486的6批，全部PointCount=0、STEP_ENDPOINT_FALLBACK；39黄金Canary实际未匹配，PID动态crossing未实现。
- Phase脚本硬编码StepNo映射和伪VariantID；Template脚本默认PV=1/Mixer=MB02、按RecipeID字符串猜材料、用Step count/max伪造StepPatternHash、未解析真实SCP-LOG。当前74模板全部Other/STANDARD_STAGE。
- Step门禁仅检查负duration标志：实际负duration 7行/6批、负能量3行、DeltaEnergy缺失231,935行、排胶温度缺716批、总时间最大21,586秒；没有独立时长/能量守恒。
- Weather脚本只审计旧core表并打印未来泄漏为0，未执行窗口重算/泄漏查询；Lab脚本只读旧core并硬断言6093，未接CE revision；QM脚本直接全体标missing，未接Mustang。
- PalletShadow 968组全部TemplateKey NULL且LabIsCurrentValid硬编码1；core有6,457 Lab组、6,093 membership覆盖1,988组、4,469组无membership。
- 档案SHA256清单多数为占位符，与实测文件hash不符；档案自身SHA256=`76BB0682E5189375BB20241F4AACA9A7FA1B0F4034AA3021DF8C89569B293EBD`。
- Run1001遗留RUNNING孤儿状态；rollback drill只在临时K1/K2表执行，不证明真实Core/Mart可恢复。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/v400_full_rebuild_independent_rejection_20260823.md`。
- 建议立即冻结20～33脚本与所有DB写操作，先做V3.2 snapshot/当前库差集、RecipeTemplate损失、schema/index/permission变更和SQL Server原生备份恢复点的只读影响评估。
- 本轮数据库连接使用ApplicationIntent=ReadOnly，仅执行SELECT；本地只写报告和PROJECT_MEMORY；未访问本地MMS Archive。

# 2026-08-23 BatchCurve→Step→Phase特征合同补充

- 用户指出Gate 0～Step 12计划遗漏BatchCurve处理；确认这是P0级遗漏，原计划不能仅靠Step端点和Phase Mapper构建正式Phase特征。
- 新增合同：`Mooney_Integrated_Data_Asset/docs/batch_curve_step_phase_feature_contract_v1_20260823.md`。
- 固定正式链路：本批BatchCurve→按Mixer标准化时间轴→对齐本批Step半开区间→按批准治理Variant构建Phase Segment→聚合CURVE_OBSERVED特征。
- 禁止跨批/未来曲线填充；曲线非空不等于完整；未知模板、时间基准不一致、PID阈值未观察到均fail-closed。
- 关键语义：白炭黑PID从获批MST候选区间内首次达到TargetTemp-3°C的曲线时刻开始，可能在Step内部切分；Stage5为该时刻至MST结束，Stage6为MSB至排胶；top+bottom PID延续至最后MSB排胶。
- FeatureBasis严格区分CURVE_OBSERVED、STEP_ENDPOINT_DERIVED、SHAPE_PROXY_DIAGNOSTIC_ONLY，禁止COALESCE静默混用。
- 不规则采样按真实delta_t积分，不默认1Hz；累计EnergyAV需识别基线/复位并做边界差分，必要时与Power积分独立核对。
- 主计划应增加Step3B曲线镜像与质量、Step5A Step/Curve/Phase分段、Step5B Phase曲线特征；三者未通过不得构建正式训练宽表。
- 本轮仅写本地合同和PROJECT_MEMORY，未连接数据库、未执行DDL/DML、未访问本地MMS Archive。

# 2026-08-23 Step14全量Stage回写与训练激活方案否决

- 用户提交基于3,974,749行Step数据向mart.BatchTrainingReady全量回写Stage并将有Lab批次全部激活READY的方案；独立审阅结论：`REJECTED / DO_NOT_AUTHORIZE_SCRIPT_EXECUTION / DO_NOT_WRITE_MART`。
- Step端点数据不等于BatchCurve；duration加权Step终点温度不是真实Phase温度均值，无法定位白炭黑PID在Step内部达到TargetTemp-3°C的动态时刻。
- 固定“Step1～2或1～3”等映射无确定选择合同，属于按步数猜Phase；未知Recipe/PV/Mixer/SCP-LOG/TargetVector模板必须BLOCKED_UNKNOWN_PHASE_MAPPING。
- 方案将炭黑MSB全部归Stage5并令Stage6 NULL，违反B00458已治理的Stage5 MST PID + Stage6 MSB PID结构。
- 比能公式若目标单位kJ/kg必须为PhaseEnergy_kWh*3600/BatchWeight_kg；EnergyAV还需处理baseline/reset/MST-MSB计数语义。
- 同源Step累计时长与由其生成的phys_total_time比较是循环自证，必须使用独立Mixer/Header时间锚点。
- 有Lab不等于批次训练就绪；MNY监督资格属于OrderID+PalletID组，必须满足有效revision、membership、模板、曲线、Phase、Material、Weather、QM和物理门禁。
- 禁止直接UPDATE当前mart.BatchTrainingReady或设置READY/IsEligible/PhaseComplete/PhysicalValid；LightGBM/XGBoost指标不能证明数据语义正确。
- 可将拟议脚本改成只读/Shadow诊断版本，仅生成Step/endpoint/proxy候选与reject，不写Core/Mart、不激活资格。
- 正式报告：`Mooney_Integrated_Data_Asset/reports/step14_full_stage_population_plan_rejection_20260823.md`。
- 本轮仅写本地审阅报告和PROJECT_MEMORY，未连接数据库、未执行DDL/DML、未访问本地MMS Archive。

## 2026-08-23 17:28 V3.7.1 M1训练资产写入
- 复用快照：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_retrain_quality_pid_snapshot_v1\v371_retrain_quality_pid_group_feature_snapshot.csv`，SHA256 `d3f7582edc4e8c3bc727898f97f78a2ad111e68309abe0bb94e985433824f2ad`；未访问本地MMS/Archive。
- M1范围：1197个独立Pallet/Lab组、4100个批次；正式训练单元保持一组一行。
- 新建（仅新增、事务提交）`mart.V371TrainingGroupMatrix`、`mart.V371TrainingBatchFeatureSnapshot`、`mart.V371TrainingBuildAudit`。
- 现有`mart.BatchTrainingReady`未修改；Stage3校准关闭，自动控制关闭。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\Mooney_Integrated_Data_Asset\reports\write_v371_m1_training_assets_V371_M1_20260823T092807Z_82f53420.json`。
## 2026-08-23 MooneyIntegratedData表清理审计
- 只读盘点数据库 60 个表/视图；未执行任何删除。
- 确认 V3.7.1 当前训练资产为 `mart.V371TrainingGroupMatrix`、`mart.V371TrainingBatchFeatureSnapshot`、`mart.V371TrainingBuildAudit`。
- 识别 7 张被否决 V4 staging 遗留表；其中 `BatchFeatureShadow` 和 `PalletTrainingShadow` 被现有 mart 视图依赖，不能直接 DROP。
- 审计报告：`Mooney_Integrated_Data_Asset/reports/database_table_inventory_and_cleanup_review_20260823.md`。

## 2026-08-23 V4遗留表归档与项目目录整理
- 已将 7 张被否决 V4 staging 表及 2 个依赖视图导出归档并通过行数核验，然后事务删除；归档：`Mooney_Integrated_Data_Asset/database_archives/v4_staging_cleanup_20260823/manifest.json`。
- 核心/raw/etl/audit/V3.7.1训练表保留；`mart.BatchTrainingReady`保留但不作为当前训练源。
- 新建干净模型目录 `V371_M1_Training_Ready`，包含模型训练代码、测试、冻结模型包、训练快照、数据库资产说明。
- 明确根目录实验脚本和`scratch`移至`legacy/20260823_project_reorganization`，未删除原始内容。
- 另将`V4_1_Energy_Optimization`、`V4_2_Step_Phase_Architecture`、`Relative_Mooney_Guidance_Pipeline`完整移入同一legacy目录；当前干净目录只保留V3.7.1 M1训练链路资产。

## 2026-08-23 V3.7.1 M1组级训练首轮
- 使用M1质量快照，1197个独立Lab/Pallet组，时间组级切分；Stage3关闭。
- OOT全局MAE 4.5961 MU、RMSE 7.1562、R2 0.7252；配方均值基线MAE 3.9363，模型暂不放行。
- 训练产物与Actual-vs-Prediction图：`reports/v371_challenger/m1_v371_r1_group_training_run_20260823/`。
- 训练最后的控制台中文打印因Windows cp1252编码失败，但模型、指标、摘要均已写出；未产生数据库写入。

## 2026-08-23 17:57 M1-T15760黄金车次集合首轮
- 精确配方 `M1-T15760---- 06 002`，组级样本 445 个；按 TemplateResearchKey 分层。
- 排除 Batch1 和冻结曲线质量风险，以模板内 MNY 中位数-0.5*MAD、单位阶段比能中位数筛选；MNY未用于曲线异常识别。
- 初步黄金集合 45 个Lab组，来自 39 个订单、6 个模板；仅研究用途。
- 交付目录：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\t15760_golden_curve_research_20260823`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\t15760_golden_curve_research_20260823\t15760_golden_selection_audit.json`。

## 2026-08-23 17:59 M1-T15760黄金集合混杂审计
- 对黄金组与同模板合格对照组进行了原料、环境、温度、配比和ProcessVersion分布检查。
- 黄金MNY中位数 54.820，合格对照中位数 60.330；高失衡特征-模板组合 65 个。
- 结论：只允许继续做匹配/调整分析，暂不输出方向性Offset推荐。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\t15760_golden_curve_research_20260823\t15760_golden_confounding_audit.json`。

## 2026-08-23 17:59 M1-T15760黄金集合混杂审计
- 对黄金组与同模板合格对照组进行了原料、环境、温度、配比和ProcessVersion分布检查。
- 黄金MNY中位数 54.820，合格对照中位数 60.200；高失衡特征-模板组合 65 个。
- 结论：只允许继续做匹配/调整分析，暂不输出方向性Offset推荐。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\t15760_golden_curve_research_20260823\t15760_golden_confounding_audit.json`。

## 2026-08-23 T15760同模板黄金组—对照组Step Offset匹配分析
- 只读使用 `MooneyIntegratedData`，未访问本地MMS/Archive，未写数据库。
- 实际字段使用 `core.BatchStep.DurationSec/TempAV/RotorAV`，目标使用 `raw.RecipeStepTargetSnapshot`；正式单位仍为LabelGroupID。
- 同TemplateResearchKey控制RecipeID/ProcessVersion/Mixer/Route/Step模板，再用QM、天气、配方物料和初温标准化最近邻匹配。
- 目标快照无直接ProcessVersion绑定，已标记为限制；观察性Offset不等于因果动作效果，当前不生成推荐。
- 产物：`reports/t15760_golden_curve_research_20260823/t15760_matched_offset_audit.json`及同目录CSV/PNG/MD。

## 2026-08-23 T15760版本统一与Step Offset最终重跑
- 未访问本地MMS/Archive；只读使用云端 `MooneyIntegratedData` 的 `raw.MmsOrderHistory`、`core.Batch`、`core.BatchStep`、`core.LabGroup`、`core.BatchMaterialQM`、`raw.WeatherHourly` 和 `raw.RecipeStepTargetSnapshot`。
- 旧黄金快照的ProcessVersion与云端实际版本不一致；没有覆盖旧事实，按同一云端快照重建 `t15760_unified_candidate_batches_pv27.csv` 与 `t15760_unified_golden_batch_set_pv27.csv`。
- 统一结果：445个LabelGroup、1432条批次；实际版本全部PV27；重新筛选黄金组26个、23个订单；天气用OrderStartTime重算，445/445获得真实24h窗口；QM可用396组，49组源缺失。
- 最终Step实际值-Target匹配：25对有效匹配，17226条Step，ProcessVersion错配0；使用DurationSec/TempAV/RotorAV与Recipe Target，并控制QM覆盖率、天气和初温。
- 加入工程最小效应阈值：duration 1s、temperature 0.5°C、rotor 1rpm；过滤浮点噪声后无稳定Step Offset，当前仍为观察性分析，不生成推荐。
- 统一快照审计：`reports/t15760_golden_curve_research_20260823/t15760_unified_snapshot_audit.json`；最终Offset审计：`reports/t15760_golden_curve_research_20260823/t15760_matched_offset_audit.json`。

## 2026-08-23 T15760同模板黄金组—对照组Step Offset匹配分析
- 只读使用 `MooneyIntegratedData`，未访问本地MMS/Archive，未写数据库。
- 实际字段使用 `core.BatchStep.DurationSec/TempAV/RotorAV`，目标使用 `raw.RecipeStepTargetSnapshot`；正式单位仍为LabelGroupID。
- 同TemplateResearchKey控制RecipeID/ProcessVersion/Mixer/Route/Step模板，再用QM、天气、配方物料和初温标准化最近邻匹配。
- 目标快照无直接ProcessVersion绑定，已标记为限制；观察性Offset不等于因果动作效果，当前不生成推荐。
- 产物：`reports/t15760_golden_curve_research_20260823/t15760_matched_offset_audit.json`及同目录CSV/PNG/MD。

## 2026-08-23 T15760同模板黄金组—对照组Step Offset匹配分析
- 只读使用 `MooneyIntegratedData`，未访问本地MMS/Archive，未写数据库。
- 实际字段使用 `core.BatchStep.DurationSec/TempAV/RotorAV`，目标使用 `raw.RecipeStepTargetSnapshot`；正式单位仍为LabelGroupID。
- 同TemplateResearchKey控制RecipeID/ProcessVersion/Mixer/Route/Step模板，再用QM、天气、配方物料和初温标准化最近邻匹配。
- 目标快照无直接ProcessVersion绑定，已标记为限制；观察性Offset不等于因果动作效果，当前不生成推荐。
- 产物：`reports/t15760_golden_curve_research_20260823/t15760_matched_offset_audit.json`及同目录CSV/PNG/MD。

## 2026-08-23 T15760同模板黄金组—对照组Step Offset匹配分析
- 只读使用 `MooneyIntegratedData`，未访问本地MMS/Archive，未写数据库。
- 实际字段使用 `core.BatchStep.DurationSec/TempAV/RotorAV`，目标使用 `raw.RecipeStepTargetSnapshot`；正式单位仍为LabelGroupID。
- 同TemplateResearchKey控制RecipeID/ProcessVersion/Mixer/Route/Step模板，再用QM、天气、配方物料和初温标准化最近邻匹配。
- 目标快照无直接ProcessVersion绑定，已标记为限制；观察性Offset不等于因果动作效果，当前不生成推荐。
- 产物：`reports/t15760_golden_curve_research_20260823/t15760_matched_offset_audit.json`及同目录CSV/PNG/MD。

## 2026-08-23 T15760同模板黄金组—对照组Step Offset匹配分析
- 只读使用 `MooneyIntegratedData`，未访问本地MMS/Archive，未写数据库。
- 实际字段使用 `core.BatchStep.DurationSec/TempAV/RotorAV`，目标使用 `raw.RecipeStepTargetSnapshot`；正式单位仍为LabelGroupID。
- 同TemplateResearchKey控制RecipeID/ProcessVersion/Mixer/Route/Step模板，再用QM、天气、配方物料和初温标准化最近邻匹配。
- 目标快照无直接ProcessVersion绑定，已标记为限制；观察性Offset不等于因果动作效果，当前不生成推荐。
- 产物：`reports/t15760_golden_curve_research_20260823/t15760_matched_offset_audit.json`及同目录CSV/PNG/MD。

## 2026-08-23 T15760同模板黄金组—对照组Step Offset匹配分析
- 只读使用 `MooneyIntegratedData`，未访问本地MMS/Archive，未写数据库。
- 实际字段使用 `core.BatchStep.DurationSec/TempAV/RotorAV`，目标使用 `raw.RecipeStepTargetSnapshot`；正式单位仍为LabelGroupID。
- 同TemplateResearchKey控制RecipeID/ProcessVersion/Mixer/Route/Step模板，再用QM、天气、配方物料和初温标准化最近邻匹配。
- 目标快照无直接ProcessVersion绑定，已标记为限制；观察性Offset不等于因果动作效果，当前不生成推荐。
- 产物：`reports/t15760_golden_curve_research_20260823/t15760_matched_offset_audit.json`及同目录CSV/PNG/MD。

## 2026-08-23 T15760同模板黄金组—对照组Step Offset匹配分析
- 只读使用 `MooneyIntegratedData`，未访问本地MMS/Archive，未写数据库。
- 实际字段使用 `core.BatchStep.DurationSec/TempAV/RotorAV`，目标使用 `raw.RecipeStepTargetSnapshot`；正式单位仍为LabelGroupID。
- 同TemplateResearchKey控制RecipeID/ProcessVersion/Mixer/Route/Step模板，再用QM、天气、配方物料和初温标准化最近邻匹配。
- 目标快照无直接ProcessVersion绑定，已标记为限制；观察性Offset不等于因果动作效果，当前不生成推荐。
- 产物：`reports/t15760_golden_curve_research_20260823/t15760_matched_offset_audit.json`及同目录CSV/PNG/MD。

## 2026-08-23 T15760同模板黄金组—对照组Step Offset匹配分析
- 只读使用 `MooneyIntegratedData`，未访问本地MMS/Archive，未写数据库。
- 实际字段使用 `core.BatchStep.DurationSec/TempAV/RotorAV`，目标使用 `raw.RecipeStepTargetSnapshot`；正式单位仍为LabelGroupID。
- 同TemplateResearchKey控制RecipeID/ProcessVersion/Mixer/Route/Step模板，再用QM、天气、配方物料和初温标准化最近邻匹配。
- 目标快照无直接ProcessVersion绑定，已标记为限制；观察性Offset不等于因果动作效果，当前不生成推荐。
- 产物：`reports/t15760_golden_curve_research_20260823/t15760_matched_offset_audit.json`及同目录CSV/PNG/MD。

## 2026-08-23 T15760 BatchWeather源值回填
- 仅写入 `MooneyIntegratedData.core.BatchWeather`，源为同库云端同步 `raw.MmsOrderHistory`+`raw.WeatherHourly`；未访问本地MMS/Archive。
- 事务回填批次：21095；更新：831；新增：20264；逐行验证：21095。
- 严格窗口 `[OrderStartTime-24h, OrderStartTime)`，Asia/Shanghai；Before-Image：`reports/t15760_golden_curve_research_20260823/t15760_batchweather_before_image.json`；审计：`reports/t15760_golden_curve_research_20260823/t15760_weather_repair_audit.json`。

## 2026-08-23 T15760动态Step/Phase特征分析
- 使用统一PV27云端快照，只读，未访问本地MMS/Archive，未写数据库。
- 构造温升、升温速率、累计时序、能量增量、PID Target-3°C动态切入和组内车次波动；匹配单位仍为LabelGroupID。
- 结果审计：`reports/t15760_golden_curve_research_20260823/t15760_dynamic_feature_audit.json`；当前仅观察性，禁止直接推荐。

## 2026-08-23 T15760动态差异治理复核
- 对动态特征结果增加未知Phase模板硬门禁：`M1_ROUTE_PENDING_PHASE_MAPPING`/`TARGETVECTOR_*`统一标记 `BLOCKED_UNKNOWN_PHASE_MAPPING`。
- 最终57个动态特征、25对匹配组中，0个可放行的稳定动态差异；38个统计上表面稳定的组合被未知Phase映射阻断。
- 状态保持 `BLOCKED_UNKNOWN_PHASE_MAPPING_AND_NO_CAUSAL_RECOMMENDATION`，不生成Step推荐。

## 2026-08-23 T15760动态Step/Phase特征分析
- 使用统一PV27云端快照，只读，未访问本地MMS/Archive，未写数据库。
- 构造温升、升温速率、累计时序、能量增量、PID Target-3°C动态切入和组内车次波动；匹配单位仍为LabelGroupID。
- 结果审计：`reports/t15760_golden_curve_research_20260823/t15760_dynamic_feature_audit.json`；当前仅观察性，禁止直接推荐。

## 2026-08-23 T15760 PV27 Phase映射覆盖审计
- 只读使用 MooneyIntegratedData；未访问本地MMS/Archive，未写数据库。
- 产物：`reports/t15760_golden_curve_research_20260823/t15760_pv27_phase_mapping_coverage.json`、CSV、MD。
- 实际模板逐一与 RecipeTemplate、PhaseMappingVariant、治理白名单对账；未知/未批准模板统一 BLOCKED_UNKNOWN_PHASE_MAPPING；不创建猜测映射、不生成推荐。

## 2026-08-23 T15760 PV27 Phase映射人工确认包
- 仅基于云端统一快照和实际Step/Target证据生成；未访问本地MMS/Archive，未写数据库。
- 为48个实际模板生成Step证据清单、代表批次和确认矩阵；PhaseName全部留空，未猜测映射。
- 当前状态 BLOCKED_UNKNOWN_PHASE_MAPPING，必须工艺确认后才能进行3-way签名和物理回放。

## 2026-08-23 T15760 PV27 Phase映射人工确认包
- 仅基于云端统一快照和实际Step/Target证据生成；未访问本地MMS/Archive，未写数据库。
- 为48个实际模板生成Step证据清单、代表批次和确认矩阵；PhaseName全部留空，未猜测映射。
- 当前状态 BLOCKED_UNKNOWN_PHASE_MAPPING，必须工艺确认后才能进行3-way签名和物理回放。

## 2026-08-23 T15760 PV27 Phase映射人工确认包
- 仅基于云端统一快照和实际Step/Target证据生成；未访问本地MMS/Archive，未写数据库。
- 为61个实际模板生成Step证据清单、代表批次和确认矩阵；PhaseName全部留空，未猜测映射。
- 当前状态 BLOCKED_UNKNOWN_PHASE_MAPPING，必须工艺确认后才能进行3-way签名和物理回放。

## 2026-08-23 T15760工艺语义补充确认
- 工艺确认：白炭黑PID从 `TargetTemp-3°C` 在Wet阶段首次达到时动态切入，并持续到MSB排胶；任何曲线中间缺失或关键事件缺失均须标记阻断。
- Dry终点定义为Wet开始；Wet终点定义为PID开始。MSB最后一个排胶控制Step的SP为设定值、AV为实际值；排胶开始即该车终止，不再等待排胶完成。
- B1起点为上密炼温度首次下降点，但必须达到至少10°C下降幅度以排除单点异常；上密炼排胶后约15秒对所有配方统一排除，且不跨后续Step继续计入B1。
- SCP-LOG必须从Recipe字段表逐Step解析。AND表示启用条件全部满足才切换，OR表示任一启用条件满足即切换；禁止从曲线反推控制逻辑。
- TargetVector/StepPattern变化只有在实际Step数量、GroupName/StepNo结构和控制逻辑一致时才可候选复用；新增或缺少Step必须独立模板并保持 `BLOCKED_UNKNOWN_PHASE_MAPPING`，不得猜测。
- 本轮规则仍仅作为待实现/待回放合同，未执行数据库写入、未访问本地MMS/Archive、未生成推荐。

## 2026-08-23 T15760异常车次复核：7717440-5粘门事件
- 用户指出 7717440 Batch 5 为明显粘门异常。复核统一云端快照后确认：该批仍进入候选数据，`curve_quality_risk_flag=False`、`curve_physical_invalid_rate=0`、`curve_model_include_group=True`，因此没有被当前曲线质量门禁排除。
- 真实Step证据显示：MST Step10结束约193.70s、TempAV=165.9°C，Step11时间突然延长至457.12s（+263.42s），TempAV反而降至160.7°C；MSB三个Step的TimeAV均为0，且一个Target缺失。这是明显的终端控制/粘门异常信号，但现有质量规则未检查“末Step异常延长+温度反降”和“MSB实际时间全为0”。
- 根因：当前统一快照继承旧曲线质量标志，且云端MooneyIntegratedData当前可见的是Step端点/摘要与`raw.MmsCurveManifest`，没有被当前流水线使用的逐点曲线形态判定；因此没有实现粘门事件检测。
- 结论：7717440-5必须标记为`EXCLUDED_STICKY_DOOR_EVENT`/`BLOCKED_CURVE_EVENT_ANOMALY`，不得进入黄金组、对照匹配或动态Step差异分析。后续需先实现可审计的末Step时间/温度反常、MSB AV缺失/全零、排胶事件一致性门禁并重建T15760候选集。

## 2026-08-23 T15760粘门异常门禁初审
- 只读检查云端MooneyIntegratedData，未访问本地MMS/Archive，未写数据库。
- 新增`47_audit_t15760_sticky_door_events.py`，对末Step异常延长、末Step温降≥10°C、MSB TimeAV全零/缺失进行批次级审计；结果文件为`t15760_sticky_door_event_audit.json`。
- 7717440-5已确认不应进入黄金/对照研究；异常门禁尚为AUDIT_ONLY，下一步需接回统一快照后重跑黄金筛选。

## 2026-08-23 AlarmInfo权威异常源连接审计
- 用户提供 `EFTMMS055.tiremes.contiwan.com / MixingCurveData.dbo.AlarmInfo` 作为严格异常车次来源；规则为 `AlarmType IS NOT NULL` 即异常。
- 只读连接尝试到达SQL Server，但SQL登录返回 `18456 Login failed for user sa`；按用户提供密码的两种转义形式均失败。
- 未读取AlarmInfo、未写入任何数据库、未修改T15760研究快照；待DBA确认SQL登录/密码或提供只读账号后继续。

## 2026-08-23 MixingInfo JudgeReason与T15760黄金组对账
- 使用更正后的只读账号访问 `EFTMMS055.tiremes.contiwan.com / MixingCurveData.dbo.MixingInfo`，并与MooneyIntegratedData.core.Batch按OrderID+BatchID逐车对齐；未访问本地MMS/Archive，未写数据库。
- T15760相关MixingInfo 20,489行，云端车次映射20,489/20,489；当前26个黄金LabelGroup中有15条非空JudgeReason，涉及11个订单，说明黄金组筛选漏用了独立异常判定源。
- 非空JudgeReason包含BottomTemper硬规则、Static/CoolingLike疑似粘门、Experience-TemperCooling粘门、Module-Mark模型异常及First-Batch标记；正式重建时至少应将有JudgeReason的组阻断，First-Batch继续按既有首车规则排除。
- 7717440-5在MixingInfo中为JudgeReason=None/Normal，但Step物理审计发现MSB实际TimeAV全零、温度缺失及MST末Step异常延长，属于 `SOURCE_JUDGE_NORMAL_BUT_PHYSICAL_ANOMALY`，不能因数据库Normal而放行。
- 产物：`reports/t15760_golden_curve_research_20260823/t15760_mixinginfo_judgereason_audit.json`、`t15760_golden_mixinginfo_judgereason.csv`及全量CSV/MD报告。

## 2026-08-23 MixingInfo粘门标记车次偏移核对
- 对全量T15760 MixingInfo中包含“粘门/TemperCooling/CoolingLike”的150条记录检查同订单前后相邻车次。
- 140条为当前车次独立标记，前后车均无同类粘门标记；5条前一车也有同类标记；5条后一车也有同类标记。当前证据不支持“数据库普遍把粘门标记整体错位一车”的假设。
- 相邻重复更像连续异常段或同一异常被相邻车次共同触发，不能简单把异常标签向前/向后平移；需要结合JudgeTime、ProductTime和曲线事件时间进一步确认。
- 7717440-5仍为特殊不一致：MixingInfo本车Normal，但Step物理异常；不能用相邻标签替代本车判定。

## 2026-08-23 粘门影响车次与曲线二次门禁规则确认
- 工艺确认：报警车次N本身的曲线可能仍正常，粘门会影响下一实际车次N+1；因此JudgeReason不能直接删除报警车N，必须映射到同订单按BatchNumber排序的下一实际车次，再对N+1曲线做独立形态复核。
- 下一车复核顺序：同订单下一实际Batch -> MixingInfo.JudgeReason/Remark -> `TemperDataAligned`（必要时原始`TemperData`）曲线解码与完整性 -> 同模板曲线形态基线；下一车缺曲线或形态异常则阻断，不能用报警车N的标签替代。
- MixingInfo温度字段已确认是可用曲线源：每4个十六进制字符对应一个温度值，样例`009B`对应155°C；`TemperDataAligned`长度约600点，需保留原始/对齐来源和缺失审计，不可填补。
- SCP-LOG权威源固定为云端/冻结Recipe `SFEPlant.dbo.RecipeCBS3Parameters`字段表的`RecipeID, StepNumber, ParameterID, ParameterName, ParameterValue, UOM`；`ParameterID`中的`SCP-Log-AND/OR-LBit0`代表时间控制位，`LBit2`代表温度控制位，AND/OR按ParameterID逐Step解析。`raw.RecipeStepTargetSnapshot`中的ScpLogLogic/ControlLogicJson只能作为同源快照结果校验，不能替代Recipe字段源。

## 2026-08-23 T15760粘门异常门禁初审
- 只读检查云端MooneyIntegratedData，未访问本地MMS/Archive，未写数据库。
- 新增`47_audit_t15760_sticky_door_events.py`，对末Step异常延长、末Step温降≥10°C、MSB TimeAV全零/缺失进行批次级审计；结果文件为`t15760_sticky_door_event_audit.json`。
- 7717440-5已确认不应进入黄金/对照研究；异常门禁尚为AUDIT_ONLY，下一步需接回统一快照后重跑黄金筛选。

## 2026-08-23 T15760粘门异常门禁初审
- 只读检查云端MooneyIntegratedData，未访问本地MMS/Archive，未写数据库。
- 新增`47_audit_t15760_sticky_door_events.py`，对末Step异常延长、末Step温降≥10°C、MSB TimeAV全零/缺失进行批次级审计；结果文件为`t15760_sticky_door_event_audit.json`。
- 7717440-5已确认不应进入黄金/对照研究；异常门禁尚为AUDIT_ONLY，下一步需接回统一快照后重跑黄金筛选。

## 2026-08-23 T15760 MixingInfo JudgeReason权威异常源审计
- 只读访问MixingCurveData.dbo.MixingInfo并与MooneyIntegratedData.core.Batch按OrderID+BatchID对齐；未访问本地MMS/Archive，未写数据库。
- 产物：`t15760_mixinginfo_judgereason_audit.json`、全量CSV、黄金组异常CSV。JudgeReason/AbnormalCause非空按权威异常处理。

## 2026-08-23 T15760 JudgeReason黄金组隔离快照
- 基于MixingInfo非空JudgeReason对LabelGroup执行保守隔离；保留旧黄金集，不写数据库、不覆盖原始CSV。
- 新产物：`t15760_golden_judgereason_quarantine_audit.json`、过滤后候选/黄金CSV及报告。
- 数据库Normal但Step物理异常的车次不做相邻车次标签平移，单独保留物理复核状态。

## 2026-08-23 T15760 JudgeReason黄金组隔离快照
- 基于MixingInfo非空JudgeReason对LabelGroup执行保守隔离；保留旧黄金集，不写数据库、不覆盖原始CSV。
- 新产物：`t15760_golden_judgereason_quarantine_audit.json`、过滤后候选/黄金CSV及报告。
- 数据库Normal但Step物理异常的车次不做相邻车次标签平移，单独保留物理复核状态。

## 2026-08-23 粘门报警到下一车曲线影响审计
- 新增`50_audit_sticky_alarm_to_next_batch_curve.py`；按同订单下一实际车次映射报警，不直接删除报警车。
- 使用MixingInfo TemperDataAligned（4位十六进制温度点）审计下一车完整性和≥10°C单点下降；未写数据库、未访问本地MMS/Archive。
- 结果为audit-only，待同模板曲线基线和Phase治理确认后再接入黄金筛选。

## 2026-08-23 粘门影响审计计数语义修正
- 修正`50_audit_sticky_alarm_to_next_batch_curve.py`：严格区分下一车键、下一车MixingInfo记录和有效温度曲线；重复MixingInfo按批次聚合，不丢失非空JudgeReason。
- 重跑事实：150条来源报警，149条找到下一实际车；66条有MixingInfo记录且曲线可解码，84条缺少下一车MixingInfo记录，11条下一车自身有JudgeReason，54条仅触发≥10°C下降复核信号，1条无该单点下降信号。
- 7916071-24仅作为报警源，映射到7916071-25；下一车没有MixingInfo/曲线，因此阻断的是影响分析，不删除7916071-24所在组。

## 2026-08-23 T15760黄金候选事件门禁重建
- 新增`51_audit_t15760_golden_event_gate.py`，将冻结26个黄金候选组按当前车物理异常、报警源车N、下一车N+1影响拆分审计。
- 结果：18组无事件门禁命中；6组当前批次物理异常阻断；1组下一车自身JudgeReason阻断；7916071|7916071006保留为`RETAIN_SOURCE_ALARM_PENDING_SOURCE_CURVE_CHECK`，不因24车报警直接删除。
- 仍为audit-only，未覆盖冻结黄金快照、未批准Phase映射、未写数据库、未访问本地MMS/Archive。

## 2026-08-23 粘门报警到下一车曲线影响审计
- 新增`50_audit_sticky_alarm_to_next_batch_curve.py`；按同订单下一实际车次映射报警，不直接删除报警车。
- 使用MixingInfo TemperDataAligned（4位十六进制温度点）审计下一车完整性和≥10°C单点下降；未写数据库、未访问本地MMS/Archive。
- 结果为audit-only，待同模板曲线基线和Phase治理确认后再接入黄金筛选。

## 2026-08-23 T15760黄金候选事件门禁审计
- 新增`51_audit_t15760_golden_event_gate.py`，将当前批次物理异常、MixingInfo报警源车N、下一车N+1影响拆为独立门禁。
- 报警源车N不直接删除；下一车曲线缺失/自带JudgeReason阻断，≥10°C下降只做同模板形态复核信号。
- 只写本地审计CSV/JSON/报告，未写数据库、未覆盖冻结黄金快照、未访问本地MMS/Archive。

## 2026-08-23 T15760报警源车曲线审计
- 新增`52_audit_t15760_alarm_source_curves.py`，检查报警源车N自身曲线完整性；JudgeReason不直接淘汰源车。
- 输出报警源曲线CSV/JSON/报告，仍需同模板形态基线后才能完成最终黄金判定；只读、未访问本地MMS/Archive。

## 2026-08-23 T15760纯净黄金候选Step Offset研究
- 新增`53_analyze_t15760_pure_golden_step_offsets.py`，基于事件门禁无命中的18组做MNY/比能和实际值-Target偏差描述统计。
- 输出纯净组摘要、Step偏差明细/汇总、两张PNG图和审计报告；仅研究性分析，不做因果或推荐，不覆盖黄金快照。

## 2026-08-23 T15760 Step实际时长口径纠正
- 用户确认：TimeAV/时间AV是Step结束累计读数，单Step实际时长必须为当前结束读数减同一GroupName上一Step结束读数；不能把AV直接当duration。
- 修正`39_match_t15760_golden_step_offsets.py`：按BatchKey+GroupName+数值StepNo排序，重算`DurationSecFromCumulative`；目标时长优先TimeSP组内差分，上一TimeSP缺失时仅使用同Recipe已有DurationSP，并记录`TargetDurationBasis`。
- 云端Recipe目标语义审计：13行/1个PV27 Recipe；11行DurationSP与TimeSP差分一致，2行因上一TimeSP缺失使用DurationSP fallback，0行冲突。
- 重跑Step offset和纯净黄金分析；旧的直接使用累计AV/累计目标的分析结论作废，当前结果仍仅研究性、不做因果或推荐。

## 2026-08-23 T15760纯净黄金候选Step Offset研究
- 新增`53_analyze_t15760_pure_golden_step_offsets.py`，基于事件门禁无命中的18组做MNY/比能和实际值-Target偏差描述统计。
- 输出纯净组摘要、Step偏差明细/汇总、两张PNG图和审计报告；仅研究性分析，不做因果或推荐，不覆盖黄金快照。

## 2026-08-23 T15760同模板黄金组—对照组Step Offset匹配分析
- 只读使用 `MooneyIntegratedData`，未访问本地MMS/Archive，未写数据库。
- 实际字段使用 `core.BatchStep.DurationSec/TempAV/RotorAV`，目标使用 `raw.RecipeStepTargetSnapshot`；正式单位仍为LabelGroupID。
- 同TemplateResearchKey控制RecipeID/ProcessVersion/Mixer/Route/Step模板，再用QM、天气、配方物料和初温标准化最近邻匹配。
- 目标快照无直接ProcessVersion绑定，已标记为限制；观察性Offset不等于因果动作效果，当前不生成推荐。
- 产物：`reports/t15760_golden_curve_research_20260823/t15760_matched_offset_audit.json`及同目录CSV/PNG/MD。

## 2026-08-23 T15760同模板黄金组—对照组Step Offset匹配分析
- 只读使用 `MooneyIntegratedData`，未访问本地MMS/Archive，未写数据库。
- 实际字段使用 `core.BatchStep.DurationSec/TempAV/RotorAV`，目标使用 `raw.RecipeStepTargetSnapshot`；正式单位仍为LabelGroupID。
- 同TemplateResearchKey控制RecipeID/ProcessVersion/Mixer/Route/Step模板，再用QM、天气、配方物料和初温标准化最近邻匹配。
- 目标快照无直接ProcessVersion绑定，已标记为限制；观察性Offset不等于因果动作效果，当前不生成推荐。
- 产物：`reports/t15760_golden_curve_research_20260823/t15760_matched_offset_audit.json`及同目录CSV/PNG/MD。

## 2026-08-23 T15760 Recipe Target Snapshot时间字段语义审计
- 新增`54_audit_t15760_recipe_target_snapshot_semantics.py`，比较Recipe目标快照的累计TimeSP差分与DurationSP，明确DurationSP fallback和冲突状态。
- 该审计用于修正Step实际时长-目标时长口径；只读云端MooneyIntegratedData，未访问本地MMS/Archive、未写数据库。

## 2026-08-23 T15760纯净黄金候选Step Offset研究
- 新增`53_analyze_t15760_pure_golden_step_offsets.py`，基于事件门禁无命中的18组做MNY/比能和实际值-Target偏差描述统计。
- 输出纯净组摘要、Step偏差明细/汇总、两张PNG图和审计报告；仅研究性分析，不做因果或推荐，不覆盖黄金快照。

## 2026-08-23 T15760纯净黄金候选Step Offset研究
- 新增`53_analyze_t15760_pure_golden_step_offsets.py`，基于事件门禁无命中的18组做MNY/比能和实际值-Target偏差描述统计。
- 输出纯净组摘要、Step偏差明细/汇总、两张PNG图和审计报告；仅研究性分析，不做因果或推荐，不覆盖黄金快照。

## 2026-08-23 T15760纯净黄金候选Step Offset研究
- 新增`53_analyze_t15760_pure_golden_step_offsets.py`，基于事件门禁无命中的18组做MNY/比能和实际值-Target偏差描述统计。
- 输出纯净组摘要、Step偏差明细/汇总、两张PNG图和审计报告；仅研究性分析，不做因果或推荐，不覆盖黄金快照。

## 2026-08-23 T15760纯净黄金组目标向量差异分析
- 新增`55_analyze_t15760_pure_golden_target_vector_differences.py`，仅比较Recipe/TargetVector的duration、temperature、rotor、ram设定值，排除实际AV执行偏差。
- 按Recipe、ProcessVersion、Mixer、路线、StepPattern和ControlLogic基础模板分层，固定目标不进入候选；只输出观察性目标差异，不做因果推荐。

## 2026-08-23 T15760车次级Target来源体检
- 新增并运行`56_audit_t15760_batch_bound_target_source.py`，只读检查云端MooneyIntegratedData的BatchStep/BatchTargetVector/Batch/BatchTrainingReady等表结构与T15760样本。
- 目的：不能用配方级RecipeStepTargetSnapshot推断车次间没有Target差异；下一步必须使用真实车次绑定的SP/Target字段。
- 实际AV不作为门尼变化解释变量；本轮未写数据库、未访问本地MMS/Archive。

## 2026-08-23 T15760车次绑定Step Target差异分析
- 新增并运行`57_analyze_t15760_batch_bound_target_differences.py`，从云端`raw.CloudM1StepWide`按车次提取TimeSP/TempSP/RotorSP/RamSP。
- TimeSP按BatchKey+GroupName累计终点读数差分为Step目标时长；实际AV不作为门尼变化解释变量。
- 按纯净黄金组与同基础模板对照组比较目标设定值差异，输出CSV/JSON/Markdown/PNG；结论仅为观察性研究，不生成推荐。
- 未写数据库，未访问本地MMS/Archive；Phase mapping继续保持BLOCKED_UNKNOWN_PHASE_MAPPING。

## 2026-08-23 T15760纯净黄金组实际Step条件差异分析
- 新增并运行`58_analyze_t15760_pure_golden_actual_step_differences.py`，将实际Step时长/温度/转速作为主分析对象。
- TimeAV按车次+GroupName累计终点差分；TempAV/RotorAV使用Step结束值；Actual-Target偏差不作为门尼解释变量。
- 在同模板并控制原料/QM、天气、配方比例等生产前条件后，比较黄金组与对照组实际Step条件；生成CSV/JSON/Markdown/PNG。
- 未写数据库，未访问本地MMS/Archive，Phase mapping仍未批准。

## 2026-08-23 T15760主分析区重排与车次级TimeAV差分纠正
- 新增并运行`59_audit_t15760_phase_priority_actuals.py`：MST Step 1-2只保留为Loading上下文；MSB Step 1-3全部保留为PID主分析候选，明确MSB Step 1不是Loading；MST Step>=3保留为主分析候选但不强行命名Dry/Wet/PID。
- 首轮发现并修正了一个关键口径风险：一个Pallet/LabelGroup包含多个Batch，不能按LabelGroupID+GroupName直接做TimeAV差分；现按物理`BatchKey+GroupName`重算，并对样例重复车次核验Step1=75.29942s、Step2=15.19976s、Step3=15.19982s。
- 真实曲线源状态仍为`CURRENTLY_UNAVAILABLE_SOURCE_DATA_MISSING`，T15760 PV27精确Phase mapping仍为`BLOCKED_UNKNOWN_PHASE_MAPPING`；因此只输出实际Step研究证据，不输出Dry/Wet/PID归因或推荐。
- 只读、未访问本地MMS/Archive、未写数据库；生成59号CSV/JSON/Markdown/PNG产物。

## 2026-08-23 T15760纯净黄金组实际Step条件差异分析
- 新增并运行`58_analyze_t15760_pure_golden_actual_step_differences.py`，将实际Step时长/温度/转速作为主分析对象。
- TimeAV按车次+GroupName累计终点差分；TempAV/RotorAV使用Step结束值；Actual-Target偏差不作为门尼解释变量。
- 在同模板并控制原料/QM、天气、配方比例等生产前条件后，比较黄金组与对照组实际Step条件；生成CSV/JSON/Markdown/PNG。
- 未写数据库，未访问本地MMS/Archive，Phase mapping仍未批准。

## 2026-08-23 T15760纯净黄金组实际Step条件差异分析
- 新增并运行`58_analyze_t15760_pure_golden_actual_step_differences.py`，将实际Step时长/温度/转速作为主分析对象。
- TimeAV按车次+GroupName累计终点差分；TempAV/RotorAV使用Step结束值；Actual-Target偏差不作为门尼解释变量。
- 在同模板并控制原料/QM、天气、配方比例等生产前条件后，比较黄金组与对照组实际Step条件；生成CSV/JSON/Markdown/PNG。
- 未写数据库，未访问本地MMS/Archive，Phase mapping仍未批准。

## 2026-08-23 T15760主分析区重排与MSB PID语义固化
- 新增并运行`59_audit_t15760_phase_priority_actuals.py`。
- MST Step 1-2 保留为Loading上下文，不作为主要门尼分析；MSB Step 1-3 明确保留为PID主分析候选，MSB Step 1不再被误判为Loading。
- 实际时长按同车次/同GroupName的TimeAV累计差分；TempAV/RotorAV为Step结束读数。
- T15760 PV27精确Phase mapping与云端真实1Hz曲线缺失时，禁止命名Dry/Wet/PID，状态保持BLOCKED_UNKNOWN_PHASE_MAPPING / BLOCKED_REAL_BATCHCURVE_UNAVAILABLE。
- 只读、未访问本地MMS/Archive、未写数据库、无推荐输出。

## 2026-08-23 T15760主分析区重排与MSB PID语义固化
- 新增并运行`59_audit_t15760_phase_priority_actuals.py`。
- MST Step 1-2 保留为Loading上下文，不作为主要门尼分析；MSB Step 1-3 明确保留为PID主分析候选，MSB Step 1不再被误判为Loading。
- 实际时长按同车次/同GroupName的TimeAV累计差分；TempAV/RotorAV为Step结束读数。
- T15760 PV27精确Phase mapping与云端真实1Hz曲线缺失时，禁止命名Dry/Wet/PID，状态保持BLOCKED_UNKNOWN_PHASE_MAPPING / BLOCKED_REAL_BATCHCURVE_UNAVAILABLE。
- 只读、未访问本地MMS/Archive、未写数据库、无推荐输出。

## 2026-08-23 T15760 MixingInfo JudgeReason权威异常源审计
- 只读访问MixingCurveData.dbo.MixingInfo并与MooneyIntegratedData.core.Batch按OrderID+BatchID对齐；未访问本地MMS/Archive，未写数据库。
- 产物：`t15760_mixinginfo_judgereason_audit.json`、全量CSV、黄金组异常CSV。JudgeReason/AbnormalCause非空按权威异常处理。

## 2026-08-23 粘门报警到下一车曲线影响审计
- 新增`50_audit_sticky_alarm_to_next_batch_curve.py`；按同订单下一实际车次映射报警，不直接删除报警车。
- 使用MixingInfo TemperDataAligned（4位十六进制温度点）审计下一车完整性和≥10°C单点下降；未写数据库、未访问本地MMS/Archive。
- 结果为audit-only，待同模板曲线基线和Phase治理确认后再接入黄金筛选。

## 2026-08-23 粘门报警到下一车曲线影响审计
- 新增`50_audit_sticky_alarm_to_next_batch_curve.py`；按同订单下一实际车次映射报警，不直接删除报警车。
- 使用MixingInfo TemperDataAligned（4位十六进制温度点）审计下一车完整性和≥10°C单点下降；未写数据库、未访问本地MMS/Archive。
- 结果为audit-only，待同模板曲线基线和Phase治理确认后再接入黄金筛选。

## 2026-08-23 T15760真实温度曲线源提取
- 新增并运行`60_extract_t15760_temperature_curve_source.py`，按T15760实际订单分块只读查询MixingCurveData.dbo.MixingInfo，并按OrderID+BatchNumber去重。
- 产物为` t15760_temperature_curve_source.csv`、审计JSON和报告；曲线范围明确为TEMPERATURE_ONLY，不冒充完整1Hz Power/Torque/Rotor BatchCurve。
- 未访问本地MMS/Archive，未写数据库。

## 2026-08-23 T15760 PV27 Step到Phase候选治理包
- 新增并运行`61_build_t15760_phase_mapping_candidate_pack.py`，基于真实PV27 StepPattern、云端RecipeStepTargetSnapshot（SourceTable指向HESFESFEPLANT.dbo.RecipeCBS3Parameters）、用户确认的Loading/Wet/PID/MSB语义生成候选映射。
- 主模板MST1-11/MSB1-3进入候选；额外MSB4和M1_ROUTE_PENDING模板硬阻断；候选状态不是生产批准，不写approved白名单。
- 真实MixingInfo温度曲线用于TargetTemp-3阈值证据；由于仅温度通道且未完成时间对齐/物理回放，不生成生产Phase特征或推荐。

## 2026-08-23 T15760温度曲线源与映射候选复核补充
- 先后尝试订单级LIKE查询与批量OR查询，因远端响应不可控均主动中止；未进行整表扫描。
- 最终改为按已知T15760物理`OrderID+BatchNumber`逐车精确只读查询MixingInfo，返回1,102/1,230车有效温度曲线：`TemperDataAligned` 1,055车，原始`TemperData`回退47车。
- 生成`60_extract_t15760_temperature_curve_source.py`及温度源CSV/审计JSON/报告；明确仅TEMPERATURE_ONLY，不能替代完整Power/Torque/Rotor 1Hz曲线。
- `61_build_t15760_phase_mapping_candidate_pack.py`复核：主模板646车，初步曲线时间对齐候选620车；额外MSB4 6车和pending route 578车阻断。
- 发现并记录配方快照结构差异：Recipe定义了AVR_MST:12但主实际模板未执行；AVR_MST:2、AVR_MSB:3无SCP行，因此MSB3只能作为排胶阶段容器，不能据此生成duration调节建议。

## 2026-08-23 T15760 PV27 Step到Phase候选治理包
- 新增并运行`61_build_t15760_phase_mapping_candidate_pack.py`，基于真实PV27 StepPattern、云端RecipeStepTargetSnapshot（SourceTable指向HESFESFEPLANT.dbo.RecipeCBS3Parameters）、用户确认的Loading/Wet/PID/MSB语义生成候选映射。
- 主模板MST1-11/MSB1-3进入候选；额外MSB4和M1_ROUTE_PENDING模板硬阻断；候选状态不是生产批准，不写approved白名单。
- 真实MixingInfo温度曲线用于TargetTemp-3阈值证据；由于仅温度通道且未完成时间对齐/物理回放，不生成生产Phase特征或推荐。

## 2026-08-23 T15760 PV27 Step到Phase候选治理包
- 新增并运行`61_build_t15760_phase_mapping_candidate_pack.py`，基于真实PV27 StepPattern、云端RecipeStepTargetSnapshot（SourceTable指向HESFESFEPLANT.dbo.RecipeCBS3Parameters）、用户确认的Loading/Wet/PID/MSB语义生成候选映射。
- 主模板MST1-11/MSB1-3进入候选；额外MSB4和M1_ROUTE_PENDING模板硬阻断；候选状态不是生产批准，不写approved白名单。
- 真实MixingInfo温度曲线用于TargetTemp-3阈值证据；由于仅温度通道且未完成时间对齐/物理回放，不生成生产Phase特征或推荐。

## 2026-08-24 T15760人工参考车次包
- 固定人工对照车次 `2375670~31`（OrderID 2375670, Batch 31）。依据独立事件门禁 `NO_EVENT_GATE_HIT`、非Batch1、PV27主模板、有效温度曲线和非极端门尼筛选。
- 新增逐Step表、标注曲线图、JSON审计与中文核对报告；明确MST1-2为Loading上下文、MST3 Dry、MST4 Oil、MST5-8 Wet、MST9-11 MST PID、MSB1-3 MSB PID。
- 记录PID阈值首次达到约为曲线样本186（Step9附近），首个下降≥10°C约为样本206；曲线时间对齐仍为待批准状态。
- 无数据库写入、未访问本地MMS/Archive、未生成推荐；映射仍为候选待工艺专家审核。

## 2026-08-24 T15760人工参考车次标签与SCP-LOG纠正
- 工艺复核确认 `MNY=53.57` 是 `LabelGroupID=2375670|2375670009` 的组级检测结果，对应车次为 `2375670~31` 至 `2375670~35`；Batch31仅作为人工对照车次，不能作为独立Lab标签。
- 工艺复核确认实际执行车次的SCP-LOG条件为：MST Step3、Step4=`OR`；其余MST1-11及MSB1-3均为`AND`。
- 云端`raw.RecipeStepTargetSnapshot`与UI合同存在3处冲突：快照把MST3/MST4写成AND、MST8写成OR；MSB3没有快照行。已在候选包中保留`SnapshotScpLogLogic`并增加`EffectiveScpLogLogic`和冲突状态，禁止静默覆盖或进入生产白名单。
- 人工参考报告已更新：PID阈值曲线样本约186，现场判断为Step10开始数秒，记录为时间对齐误差；B1温度下降点经工艺确认有效。
- 无数据库写入，未访问本地MMS/Archive，未生成推荐。

## 2026-08-23 T15760 PV27 Step到Phase候选治理包
- 新增并运行`61_build_t15760_phase_mapping_candidate_pack.py`，基于真实PV27 StepPattern、云端RecipeStepTargetSnapshot（SourceTable指向HESFESFEPLANT.dbo.RecipeCBS3Parameters）、用户确认的Loading/Wet/PID/MSB语义生成候选映射。
- 主模板MST1-11/MSB1-3进入候选；额外MSB4和M1_ROUTE_PENDING模板硬阻断；候选状态不是生产批准，不写approved白名单。
- 真实MixingInfo温度曲线用于TargetTemp-3阈值证据；由于仅温度通道且未完成时间对齐/物理回放，不生成生产Phase特征或推荐。

## 2026-08-24 RecipeData SCP-LOG权威源接入准备
- 根据工艺专家确认，正式SCP-LOG来源改为`SFEPlant.dbo.RecipeData`，按VariablePath解析MST/MSB步号及Logic-OR/Logic-AND的Value末位。
- 新增`63_extract_t15760_recipe_data_scp_log_contract.py`；仅允许RDS_APAC云端只读连接，当前凭据缺少RDS_APAC段，真实RecipeData提取状态为`BLOCKED_RECIPE_DATA_SOURCE_UNAVAILABLE`。
- 不使用本地MMS/HF_MMS作为替代，不把截图或RecipeStepTargetSnapshot伪装成RecipeData真值。

## 2026-08-24 SFEAPAC RecipeData全表读取尝试
- 按用户授权，使用项目已确认的云端端点`mustang-apac-rds-01-dbinstance-01.ch6kpnfcmztf.eu-central-1.rds.amazonaws.com`、数据库`HESFESFEPLANT`和SFEAPAC凭据，准备只读读取`dbo.RecipeData`全表并保存本地快照。
- 连接在建连阶段持续无响应，已主动终止；本次没有读到RecipeData，也没有任何数据库写入。当前状态为`BLOCKED_SFEAPAC_CLOUD_CONNECTION_TIMEOUT`，不能把超时解释为表不存在或逻辑解析失败。
- 未访问本地MMS/HF_MMS；解析器仍按RecipeData的VariablePath与Value末位规则保留，待云端连接恢复后重新执行。

## 2026-08-23 T15760 PV27 Step到Phase候选治理包
- 新增并运行`61_build_t15760_phase_mapping_candidate_pack.py`，基于真实PV27 StepPattern、云端RecipeStepTargetSnapshot（SourceTable指向HESFESFEPLANT.dbo.RecipeCBS3Parameters）、用户确认的Loading/Wet/PID/MSB语义生成候选映射。
- 主模板MST1-11/MSB1-3进入候选；额外MSB4和M1_ROUTE_PENDING模板硬阻断；候选状态不是生产批准，不写approved白名单。
- 真实MixingInfo温度曲线用于TargetTemp-3阈值证据；由于仅温度通道且未完成时间对齐/物理回放，不生成生产Phase特征或推荐。

## 2026-08-23 T15760 PV27 Step到Phase候选治理包
- 新增并运行`61_build_t15760_phase_mapping_candidate_pack.py`，基于真实PV27 StepPattern、云端RecipeStepTargetSnapshot（SourceTable指向HESFESFEPLANT.dbo.RecipeCBS3Parameters）、用户确认的Loading/Wet/PID/MSB语义生成候选映射。
- 主模板MST1-11/MSB1-3进入候选；额外MSB4和M1_ROUTE_PENDING模板硬阻断；候选状态不是生产批准，不写approved白名单。
- 真实MixingInfo温度曲线用于TargetTemp-3阈值证据；由于仅温度通道且未完成时间对齐/物理回放，不生成生产Phase特征或推荐。

## 2026-08-24 RecipeData SCP-LOG权威源接入准备
- 根据工艺专家确认，正式SCP-LOG来源改为`SFEPlant.dbo.RecipeData`，按VariablePath解析MST/MSB步号及Logic-OR/Logic-AND的Value末位。
- 新增`63_extract_t15760_recipe_data_scp_log_contract.py`；仅允许SFEAPAC云端只读连接，当前真实RecipeData提取状态为`BLOCKED_RECIPE_DATA_SOURCE_UNAVAILABLE`。
- 不使用本地MMS/HF_MMS作为替代，不把截图或RecipeStepTargetSnapshot伪装成RecipeData真值。

## 2026-08-24 V3.7.1训练资产组级/车次级Step血缘审计
- 新增并运行`64_audit_v371_training_group_matrix_contract.py`，只读检查`mart.V371TrainingGroupMatrix`、`mart.V371TrainingBatchFeatureSnapshot`和审计表；未写数据库、未访问本地MMS/Archive。
- 重点核验：组级Lab标签不能等同单车标签；车次级资产必须保留OrderID+BatchNumber，并且必须有GroupName/StepNo或不可变Step外键。
- 审计产物位于`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\training_asset_contract_audit_20260824`。

## 2026-08-24 V371精确Phase覆盖审计与T15760候选映射重建
- 只读核验`MooneyIntegratedData.core.RecipeTemplate`与`core.PhaseMappingVariant`：当前74个模板/15个Phase变体不能覆盖全部4,100个M1车次，未匹配模板不得猜测Phase。
- 修改`66_build_v371_step_group_staging.py`：加载新的云端RecipeData合同，并仅对现有T15760候选StepPhaseMap做逐车次、逐Step的SCP-LOG一致性校验；逻辑不一致即阻断，候选仍保持`CANDIDATE_MAPPING_REQUIRES_EXPERT_APPROVAL`。
- 重建结果：4,100车、54,116 Step；T15760 646车/9,044 Step进入候选映射；45,072 Step继续`BLOCKED_UNKNOWN_PHASE_MAPPING`；1条负累计时间Step阻断；未写数据库、未访问本地MMS/Archive。
- 新增4项单元测试，全部通过；审计产物位于`reports/v371_challenger/global_rds_active_v1/v371_step_group_staging_v1/v371_step_group_staging_audit.json`。

## 2026-08-24 SFEAPAC RecipeData连接与目标配方合同提取
- 核验云端SFEAPAC端口可达、`master`与`HESFESFEPLANT`登录成功，且`HESFESFEPLANT.dbo.RecipeData`存在；未访问本地MMS/HF_MMS，未执行数据库写入。
- 发现`RecipeData`实际约7,279,875行，不执行全表读取；将`63_extract_t15760_recipe_data_scp_log_contract.py`改为只读取目标`RecipeID=9922A1576030_PV27_002`，并移除不适用于该RDS端点的`MultiSubnetFailover=yes`。
- 成功读取447条目标RecipeData记录，解析32条逻辑证据、16个MST/MSB Step合同，全部`RECIPEDATA_CONTRACT_OK`；MST Step 3、4为`RECIPE_OR`，其余已解析Step为`RECIPE_AND`。
- 产物：`reports/t15760_golden_curve_research_20260823/t15760_recipe_data_target_snapshot.csv`、`t15760_recipe_data_scp_log_evidence.csv`、`t15760_recipe_data_scp_log_contract.csv`及审计JSON；状态`RECIPEDATA_CONTRACT_EXTRACTED`。

## 2026-08-24 V3.7.1训练资产组级/车次级Step血缘审计
- 新增并运行`64_audit_v371_training_group_matrix_contract.py`，只读检查`mart.V371TrainingGroupMatrix`、`mart.V371TrainingBatchFeatureSnapshot`和审计表；未写数据库、未访问本地MMS/Archive。
- 重点核验：组级Lab标签不能等同单车标签；车次级资产必须保留OrderID+BatchNumber，并且必须有GroupName/StepNo或不可变Step外键。
- 审计产物位于`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\training_asset_contract_audit_20260824`。

## 2026-08-24 10:42 V371 RecipeData SCP-LOG合同快照
- 仅从云端`HESFESFEPLANT.dbo.RecipeData`按129个M1 RecipeID分块只读提取；未全表扫描、未访问本地MMS/Archive、未写数据库。
- 读取47066条RecipeData记录，生成1919条Step逻辑合同；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_recipe_data_scp_log_contract_v1\v371_recipe_data_scp_log_contract_audit.json`。

## 2026-08-24 10:44 V371 RecipeData SCP-LOG合同快照
- 仅从云端`HESFESFEPLANT.dbo.RecipeData`按129个M1 RecipeID分块只读提取；未全表扫描、未访问本地MMS/Archive、未写数据库。
- 读取47066条RecipeData记录，生成1919条Step逻辑合同；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_recipe_data_scp_log_contract_v1\v371_recipe_data_scp_log_contract_audit.json`。

## 2026-08-24 10:48 V371组成员与逐Step Staging重构
- 新增组成员桥接与逐车次逐Step特征Staging脚本`66_build_v371_step_group_staging.py`；Step实际值按累计AV差分，SCP-LOG只读RecipeData合同。
- 当前生成1197组、4100车、54116条Step；Phase未有精确治理映射的全部标记`BLOCKED_UNKNOWN_PHASE_MAPPING`，未猜测、未写数据库。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_step_group_staging_v1\v371_step_group_staging_audit.json`。

## 2026-08-24 10:53 V371组成员与逐Step Staging重构
- 新增组成员桥接与逐车次逐Step特征Staging脚本`66_build_v371_step_group_staging.py`；Step实际值按累计AV差分，SCP-LOG只读RecipeData合同。
- 当前生成1197组、4100车、54116条Step；Phase未有精确治理映射的全部标记`BLOCKED_UNKNOWN_PHASE_MAPPING`，未猜测、未写数据库。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_step_group_staging_v1\v371_step_group_staging_audit.json`。

## 2026-08-24 10:58 V371组成员与逐Step Staging重构
- 新增组成员桥接与逐车次逐Step特征Staging脚本`66_build_v371_step_group_staging.py`；Step实际值按累计AV差分，SCP-LOG只读RecipeData合同。
- 当前生成1197组、4100车、54116条Step；Phase未有精确治理映射的全部标记`BLOCKED_UNKNOWN_PHASE_MAPPING`，未猜测、未写数据库。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_step_group_staging_v2\v371_step_group_staging_audit.json`。

## 2026-08-24 11:00 B00458结构等价候选Phase映射
- 针对当前`9922A00458BT_PV77_005`按Settable控制结构签名匹配历史治理变体，不按RecipeID直接复用；匹配结果170车、候选Step 0条。
- SCP-LOG继续以RecipeData校验；状态保持`B00458_CANDIDATE_MAPPING_REQUIRES_EXPERT_APPROVAL`，未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_step_group_staging_v3\b00458_phase_mapping_candidate_audit.json`。

## 2026-08-24 11:01 B00458结构等价候选Phase映射
- 针对当前`9922A00458BT_PV77_005`按Settable控制结构签名匹配历史治理变体，不按RecipeID直接复用；匹配结果170车、候选Step 1700条。
- SCP-LOG继续以RecipeData校验；状态保持`B00458_CANDIDATE_MAPPING_REQUIRES_EXPERT_APPROVAL`，未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_step_group_staging_v3\b00458_phase_mapping_candidate_audit.json`。

## 2026-08-24 11:04 B00458映射解析修正与回归验证
- 发现并修正`67_build_b00458_phase_mapping_candidate.py`的签名解析缺陷：历史签名使用`GroupName:S<StepNo>:PhaseName`，原解析器未将`S<StepNo>`归一化为逐Step Staging使用的数值键，造成候选数虚假为0；未改变映射规则或治理资产。
- 修正后，`9922A00458BT_PV77_005`的170车、1700条Step全部精确命中`StructuralTemplateID=6E0CEA262D17`与`PhaseMappingVariantID=4CA865F996C0`；SCP-LOG RecipeData合同不匹配数为0。逐Step结果为：MST1-2→Stage1_Loading，MST3→Stage2_DryMixing，MST4→Stage3_OilLoading，MST5-6→Stage4_WetMixing_Ramp，MST7-8→Stage5_MST_PID_Silanization，MSB1-2→Stage6_MSB_PID_Step1-2。
- 新增`test_b00458_phase_mapping_candidate.py`，覆盖`S<StepNo>`归一化与畸形签名Fail-Closed；原4项V371测试与新增2项测试均通过。仍为`B00458_CANDIDATE_MAPPING_REQUIRES_EXPERT_APPROVAL`，未写数据库、未访问本地MMS/Archive。

## 2026-08-24 11:05 双试点Phase映射人工审批包
- 新增`68_build_dual_pilot_mapping_approval_packet.py`，仅汇总已生成的候选Staging，不推断新映射、不批准、不写数据库。
- T15760当前候选范围：646车、9044条Step、14条去重Step映射；同Recipe范围另有11010条`BLOCKED_UNKNOWN_PHASE_MAPPING`，已明确列入阻断，不得当作全配方已通过。
- B00458当前PV77_005候选范围：170车、1700条Step、10条去重Step映射，阻断0；RecipeData解析全部`RECIPE_AND`，结构模板/变体为`6E0CEA262D17`/`4CA865F996C0`。
- 审批包：`reports/v371_challenger/global_rds_active_v1/dual_pilot_phase_mapping_approval_packet_v1/`；状态`WAITING_FOR_INDEPENDENT_DOMAIN_EXPERT_APPROVAL`，生产批准仍为False。

## 2026-08-24 11:18 双试点ProcessVersion/TargetVector验证与全局拼装写入
- 新增并运行`69_validate_dual_pilot_version_target_scope.py`：T15760候选范围18个ProcessVersion/14个TargetVector、B00458候选范围6个ProcessVersion/4个TargetVector，均映射跨范围稳定、SCP-LOG跨范围稳定、范围内失败0，审计状态`PASS`。
- 新增并运行`70_write_v371_global_step_phase_assembly.py`，在单一事务中新增`mart.V371TrainingBatchStepSnapshot`（54,116行）、`mart.V371TrainingGroupMemberSnapshot`（4,100行）和`mart.V371TrainingPhaseAssemblyAudit`（1行）。
- 写入结果：候选Phase Step 10,744（T15760 9,044 + B00458 1,700），`BLOCKED_UNKNOWN_PHASE_MAPPING` 43,372；阻断行保留但不具备训练资格。候选行SCP-LOG不合格0、Step时间合同不合格0、Step/成员双向覆盖0、Step主键重复0。
- 旧`mart.V371TrainingGroupMatrix`仍1197行、旧`V371TrainingBatchFeatureSnapshot`仍4100行、`BatchTrainingReady`仍329342行，均未修改；未访问本地MMS/Archive。写入审计：`reports/v371_challenger/global_rds_active_v1/v371_global_step_phase_assembly_v1/write_audit.json`。

## 2026-08-24 11:29 V3.7.1组级Phase训练消费视图
- 新增并运行`71_build_v371_group_phase_training_view.py`，从新Step快照、组成员快照和`mart.V371TrainingGroupMatrix`组级标签拼装，一组一行新增`mart.V371TrainingGroupPhaseFeatureSnapshot`（1197组）与`mart.V371TrainingGroupPhaseBuildAudit`（1行）。
- 其中222组满足所有成员车次均有候选Phase、SCP-LOG和Step时间合同，状态`ELIGIBLE_CANDIDATE_SCOPE`；975组因至少一个成员车次阻断，状态`BLOCKED_GROUP_HAS_UNUSABLE_BATCH`。B00458候选47组，T15760候选175组。
- Phase统计只使用候选Step：按物理车次/Phase先聚合实际Step时长、Step终点TempAV、实际能耗和Target，再按LabelGroupID计算均值/标准差/最小/最大；阻断组Phase特征保持NULL。旧`V371TrainingGroupMatrix`未修改，未访问本地MMS/Archive。

## 2026-08-24 V3.7.1训练资产组级/车次级Step血缘审计
- 新增并运行`64_audit_v371_training_group_matrix_contract.py`，只读检查`mart.V371TrainingGroupMatrix`、`mart.V371TrainingBatchFeatureSnapshot`和审计表；未写数据库、未访问本地MMS/Archive。
- 重点核验：组级Lab标签不能等同单车标签；车次级资产必须保留OrderID+BatchNumber，并且必须有GroupName/StepNo或不可变Step外键。
- 审计产物位于`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\training_asset_contract_audit_20260824`。

## 2026-08-24 11:18 V3.7.1全局Step/Phase拼装写入
- ProcessVersion/TargetVector验证已通过：T15760为18个ProcessVersion/14个TargetVector候选范围，B00458为6个ProcessVersion/4个TargetVector候选范围；各范围映射与SCP-LOG稳定。
- 仅新增并事务写入`mart.V371TrainingBatchStepSnapshot`（54116车次Step行）、`mart.V371TrainingGroupMemberSnapshot`（4100物理车次成员）和`mart.V371TrainingPhaseAssemblyAudit`；保留候选与`BLOCKED_UNKNOWN_PHASE_MAPPING`，未修改旧V371表或`BatchTrainingReady`。
- 候选Step 10744，阻断/不可训练Step 43372；Archive状态仍为`BLOCKED_ARCHIVE_NOT_REPLICATED`，未访问本地MMS/Archive。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_global_step_phase_assembly_v1\write_audit.json`。

## 2026-08-24 11:29 V3.7.1组级Phase训练消费表
- 从新增Step快照、组成员快照和现有组级标签表拼装，一组一行写入`mart.V371TrainingGroupPhaseFeatureSnapshot`（1197组），审计写入`mart.V371TrainingGroupPhaseBuildAudit`。
- Phase特征仅由候选映射Step生成；含任一阻断车次的组保持阻断，不用其它车次替补。训练候选组222，阻断组975。
- 旧`V371TrainingGroupMatrix`未修改，未访问本地MMS/Archive。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_group_phase_training_view_v1\write_audit.json`。

## 2026-08-24 11:34 V3.7.1组级训练输入一致性审计
- 只读审计`V371TrainingGroupMatrix`、`V371TrainingGroupPhaseFeatureSnapshot`和冻结模型元数据；未写数据库、未访问本地MMS/Archive。
- 组级标签粒度、标签对账和候选Phase资格通过；新消费表有222个候选组、975个阻断组。
- 冻结模型要求210列；纠正后的Phase聚合列尚未与旧Stage特征逐列证明同源，状态`BLOCKED_MODEL_INPUT_CONTRACT`，禁止直接训练。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_group_phase_model_input_audit_v1\v371_group_phase_model_input_audit.json`。

## 2026-08-24 11:35 V3.7.1特征血缘缺口
- 只读逐列对账冻结模型210个特征与旧组表/新纠正Phase组表；未写数据库、未访问本地MMS/Archive。
- 旧组表同名特征210列，新纠正Phase表同名特征0列；端点AV/Phase聚合不自动等价为曲线均值/积分，状态`BLOCKED_UNTIL_CURVE_FEATURE_REBUILD_AND_CROSSWALK`。
- 逐列清单：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_feature_lineage_gap_v1\v371_feature_lineage_gap.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_feature_lineage_gap_v1\v371_feature_lineage_gap_audit.json`。

## 2026-08-24 11:42 V3.7.1曲线缺失补全Staging重建
- 新增并运行`74_rebuild_v371_curve_phase_features.py`，只使用冻结云端Active BatchCurve快照与当前候选Step Staging；候选车次816，命中曲线816，未访问本地MMS/Archive。
- 按本车累计Step AV边界构造曲线Segment，白炭黑按`TargetTemp-3°C`本车首次温度crossing切分PID；未通过曲线完整性、时间边界或PID阈值的车次只记录阻断，不填0、不借用其他车次。
- 产物：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_curve_phase_rebuild_v1\v371_curve_rebuild_audit.json`、`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_curve_phase_rebuild_v1\v371_curve_rebuild_report.md`；当前`database_writes=false`、`model_ready=false`。

## 2026-08-24 11:43 V3.7.1曲线缺失补全Staging重建
- 新增并运行`74_rebuild_v371_curve_phase_features.py`，只使用冻结云端Active BatchCurve快照与当前候选Step Staging；候选车次816，命中曲线816，未访问本地MMS/Archive。
- 按本车累计Step AV边界构造曲线Segment，白炭黑按`TargetTemp-3°C`本车首次温度crossing切分PID；未通过曲线完整性、时间边界或PID阈值的车次只记录阻断，不填0、不借用其他车次。
- 产物：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_curve_phase_rebuild_v1\v371_curve_rebuild_audit.json`、`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_curve_phase_rebuild_v1\v371_curve_rebuild_report.md`；当前`database_writes=false`、`model_ready=false`。

## 2026-08-24 11:45 V3.7.1曲线缺失补全Staging重建
- 新增并运行`74_rebuild_v371_curve_phase_features.py`，只使用冻结云端Active BatchCurve快照与当前候选Step Staging；候选车次816，命中曲线816，未访问本地MMS/Archive。
- 按本车累计Step AV边界构造曲线Segment，白炭黑按`TargetTemp-3°C`本车首次温度crossing切分PID；未通过曲线完整性、时间边界或PID阈值的车次只记录阻断，不填0、不借用其他车次。
- 产物：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_curve_phase_rebuild_v1\v371_curve_rebuild_audit.json`、`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_curve_phase_rebuild_v1\v371_curve_rebuild_report.md`；当前`database_writes=false`、`model_ready=false`。

## 2026-08-24 11:47 V3.7.1曲线缺失补全Staging重建
- 新增并运行`74_rebuild_v371_curve_phase_features.py`，只使用冻结云端Active BatchCurve快照与当前候选Step Staging；候选车次816，命中曲线816，未访问本地MMS/Archive。
- 按本车累计Step AV边界构造曲线Segment，白炭黑按`TargetTemp-3°C`本车首次温度crossing切分PID；未通过曲线完整性、时间边界或PID阈值的车次只记录阻断，不填0、不借用其他车次。
- 产物：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_curve_phase_rebuild_v1\v371_curve_rebuild_audit.json`、`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_curve_phase_rebuild_v1\v371_curve_rebuild_report.md`；当前`database_writes=false`、`model_ready=false`。

## 2026-08-24 11:50 V3.7.1曲线解码与Ram路由语义修正
- 修正`r10_curve_shape_features.py`：MMS曲线按有符号16位补码解析，`FFFF`保留为缺失哨兵；`FFFA`等真实负值不再被误判为空。新增测试后，R10解码与Step对齐测试4/4通过。
- 明确Ram仅在MST有物理意义；MSB阶段所有Ram特征置NULL并标记`ROUTE_NOT_APPLICABLE_MSB`，不再把下密炼Ram当作缺失。
- 重跑`74_rebuild_v371_curve_phase_features.py`：816车全部命中曲线，787车正式曲线资格，10车信号部分覆盖，10车PID阈值未观察到，9车本车MSB累计时间为0导致Step边界阻断；不跨车填补、不填0、不写数据库。
- 每个segment补充`GroupName/StepNo/SegmentNo/SourceCurveHash/PhaseMappingVariantID`；独立校验`75_validate_v371_curve_rebuild.py`通过：重复segment键0、曲线重叠0、MSB Ram误用0。
- 生成审计图`76_plot_v371_curve_rebuild_status.py`及`v371_curve_rebuild_status.png`；当前状态仍为`STAGING_ONLY_REVIEW_REQUIRED`，模型输入和数据库写入均未放行。

## 2026-08-24 11:54 V3.7.1曲线范围审计
- 新增并运行`77_audit_v371_curve_scope_counts.py`，只读对比冻结曲线、M1组成员和候选Phase映射范围。
- 冻结云端Active曲线快照为6814个物理车次，时间范围2025-12-01至2026-08-11；当前不是完整12个月快照。
- M1组成员为4100车/1197个Lab组，4100/4100均有冻结曲线；当前候选Phase映射仅816车（T15760 646、B00458 170），其余3284车因未知或未治理Phase映射阻断。
- 因此正式787车是`816候选映射车次`中通过曲线、Step边界和动态PID门禁的数量，不代表全年车次数；全年度扩展必须先补其余M1模板的治理映射，并补取2025-08至2025-11及2026-08-12之后的云端Active快照，不能访问本地Archive。
- 审计：`reports/v371_challenger/global_rds_active_v1/v371_curve_scope_audit_v1/v371_curve_scope_audit.json`；未写数据库。

## 2026-08-24 11:48 V3.7.1曲线缺失补全Staging重建
- 新增并运行`74_rebuild_v371_curve_phase_features.py`，只使用冻结云端Active BatchCurve快照与当前候选Step Staging；候选车次816，命中曲线816，未访问本地MMS/Archive。
- 按本车累计Step AV边界构造曲线Segment，白炭黑按`TargetTemp-3°C`本车首次温度crossing切分PID；未通过曲线完整性、时间边界或PID阈值的车次只记录阻断，不填0、不借用其他车次。
- 产物：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_curve_phase_rebuild_v1\v371_curve_rebuild_audit.json`、`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_curve_phase_rebuild_v1\v371_curve_rebuild_report.md`；当前`database_writes=false`、`model_ready=false`。

## 2026-08-24 12:00 其他M1配方Phase映射候选分层
- 基于当前3249个Phase阻断车次，按材料体系、设备路线、Settable结构签名、Step覆盖和RecipeData SCP-LOG分层；未按RecipeID相似直接复用。
- 精确候选476车/6048条Step，覆盖7个配方；其余状态保持Fail-Closed。候选仍为`CANDIDATE_NOT_APPROVED`，未写数据库。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_all_recipe_mapping_candidates_v1\v371_m1_all_recipe_phase_mapping_candidate_audit.json`；逐Step资产：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_all_recipe_mapping_candidates_v1\v371_m1_other_recipe_phase_mapping_candidate_step_map.csv`。

## 2026-08-24 12:01 其他M1配方Phase映射候选分层
- 基于当前3284个Phase阻断车次，按材料体系、设备路线、Settable结构签名、Step覆盖和RecipeData SCP-LOG分层；未按RecipeID相似直接复用。
- 精确候选476车/6048条Step，覆盖7个配方；其余状态保持Fail-Closed。候选仍为`CANDIDATE_NOT_APPROVED`，未写数据库。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_all_recipe_mapping_candidates_v1\v371_m1_all_recipe_phase_mapping_candidate_audit.json`；逐Step资产：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_all_recipe_mapping_candidates_v1\v371_m1_other_recipe_phase_mapping_candidate_step_map.csv`。

## 2026-08-24 12:01 其他M1配方Phase映射候选分层
- 基于当前3284个Phase阻断车次，按材料体系、设备路线、Settable结构签名、Step覆盖和RecipeData SCP-LOG分层；未按RecipeID相似直接复用。
- 精确候选476车/6048条Step，覆盖7个配方；其余状态保持Fail-Closed。候选仍为`CANDIDATE_NOT_APPROVED`，未写数据库。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_all_recipe_mapping_candidates_v1\v371_m1_all_recipe_phase_mapping_candidate_audit.json`；逐Step资产：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_all_recipe_mapping_candidates_v1\v371_m1_other_recipe_phase_mapping_candidate_step_map.csv`。

## 2026-08-24 12:03 其他M1配方Phase映射候选分层
- 基于当前3284个Phase阻断车次，按材料体系、设备路线、Settable结构签名、Step覆盖和RecipeData SCP-LOG分层；未按RecipeID相似直接复用。
- 精确候选476车/6048条Step，覆盖7个配方；其余状态保持Fail-Closed。候选仍为`CANDIDATE_NOT_APPROVED`，未写数据库。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_all_recipe_mapping_candidates_v1\v371_m1_all_recipe_phase_mapping_candidate_audit.json`；逐Step资产：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_all_recipe_mapping_candidates_v1\v371_m1_other_recipe_phase_mapping_candidate_step_map.csv`。

## 2026-08-24 12:04 其他M1配方候选曲线重建
- 对精确结构候选476车执行BatchCurve→Step→Phase staging重建，曲线命中476车，Segment 6430条，正式曲线资格候选463车。
- 仍遵守同车曲线、累计TimeAV边界、白炭黑`TargetTemp-3°C`动态PID、MSB Ram不适用和Fail-Closed规则；未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_other_recipe_curve_phase_candidates_v1\v371_other_recipe_curve_phase_candidate_audit.json`。

## 2026-08-24 12:04 其他M1配方候选曲线重建
- 对精确结构候选476车执行BatchCurve→Step→Phase staging重建，曲线命中476车，Segment 6430条，正式曲线资格候选463车。
- 仍遵守同车曲线、累计TimeAV边界、白炭黑`TargetTemp-3°C`动态PID、MSB Ram不适用和Fail-Closed规则；未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_other_recipe_curve_phase_candidates_v1\v371_other_recipe_curve_phase_candidate_audit.json`。
## 2026-08-24 12:06 曲线边界Fail-Closed修复
- 在`74_rebuild_v371_curve_phase_features.py`补充物理边界阻断门禁：当`BoundaryStatus`为`BLOCKED_*`时只保留审计，不再输出可能重叠的裁剪Segment；避免将MST/MSB时间重叠误当作正式曲线特征。
- 该修复用于其他配方候选曲线重建前的回归，未写数据库。

## 2026-08-24 12:06 其他M1配方候选曲线重建
- 对精确结构候选476车执行BatchCurve→Step→Phase staging重建，曲线命中476车，Segment 6401条，正式曲线资格候选463车。
- 仍遵守同车曲线、累计TimeAV边界、白炭黑`TargetTemp-3°C`动态PID、MSB Ram不适用和Fail-Closed规则；未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_other_recipe_curve_phase_candidates_v1\v371_other_recipe_curve_phase_candidate_audit.json`。

## 2026-08-24 12:07 其他M1配方版本与Target范围稳定性核验
- 对23个RecipeID+ProcessVersion+TargetVector范围逐一检查Step数量、映射键、SCP-LOG和Phase完整性；稳定23个、阻断0个。
- 仅生成专家批准队列，未更新canonical staging、未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_all_recipe_mapping_candidates_v1\v371_m1_other_recipe_mapping_scope_contract_audit.json`。
## 2026-08-24 12:08 其他M1配方候选曲线与范围验收
- 对476个精确结构候选车次重建同车BatchCurve→Step→Phase：曲线命中476/476，生成6401条Segment；463车通过`NORMAL_COMPLETE_ROUTE + CURVE_COMPLETE`正式曲线资格，13车因信号部分缺失或物理边界阻断保留审计。
- 独立验证通过：Segment重叠0、重复键0、未知Phase写入0、MSB Ram非空0；生成状态图`v371_other_recipe_curve_candidate_status.png`。
- 23个RecipeID+ProcessVersion+TargetVector范围全部通过映射稳定性检查，已生成专家批准队列；仍未写数据库、未进入canonical模型训练表。

## 2026-08-24 12:52 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：DRY_RUN_STAGING。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T045139Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 12:52 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：DRY_RUN_STAGING。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T045232Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 12:53 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：DRY_RUN_STAGING。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T045324Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 12:55 云端M1增量刷新器V1累计AV差分修正
- `step_snapshot.parquet`新增每Step实际`Duration=当前TimeAV−上一Step TimeAV`；保留TimeAV累计终点，不使用固定比例切分、不把设备执行偏差改写为Target。
- 复核结果：专项测试7/7，连同既有曲线测试共10/10通过；云端1订单回放仍为113车、6471行Step、113条曲线、1701条物料，TrainingReady提升数=0。
- 最新审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T045418Z\refresh_audit.json`；Step快照：`step_snapshot.parquet`。

## 2026-08-24 13:02 云端M1每日刷新运行手册与调度入口
- 新增`Mooney_Integrated_Data_Asset/scripts/run_cloud_m1_daily_refresh.ps1`和对应runbook；默认DRY-RUN，未注册旧任务替换，Execute只允许原始内容哈希幂等写入，不提升Phase或TrainingReady。
- PowerShell参数边界校验通过；当前仍要求先完成其他M1结构治理批准与后续宽表门禁。

## 2026-08-24 13:18 云端M1多源增量链路扩展
- 刷新器新增云端 Compound Excellence Lab、Mustang SAP QM、Pallet组级标签、core.Batch/core.BatchStep/core.BatchMaterial、core.LabCurrent/core.LabGroup 和 etl.DependencyState 的事务内处理；天气只从已有 raw.WeatherHourly 按24小时合同重算，缺失不填常数。
- 001/011 才能成为训练标签；034/209及其他状态保留原始记录但不得进入训练资格。Phase尚未治理批准时`IsReadyForMart=0`且TrainingReady提升=0。
- 真实只读回放：1订单/113车/6471 Step行/113曲线/1701物料/337 RecipeData/0 CE Lab/34 QM；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T050059Z\refresh_audit.json`。
- 专项测试9/9通过；未执行`--execute`，未写目标数据库。

## 2026-08-24 13:10 其他配方候选曲线血缘修复
- 修复`79_rebuild_other_recipe_curve_phase_candidates.py`在左右表均存在`PhaseMappingVariantID`时的列冲突；现在每个Segment和车次审计都显式携带候选VariantID与`CANDIDATE_MAPPING_REQUIRES_EXPERT_APPROVAL`。
- 重跑结果：6401 Segment全部有VariantID，476车审计全部有VariantID；独立校验仍为PASS（重叠0、重复键0、MSB Ram误填0、正式曲线资格463车），未写数据库。
- `80_validate_other_recipe_curve_candidates.py`新增VariantID血缘硬门禁，复跑PASS。

## 2026-08-24 13:16 云端M1多订单只读回放
- 3个订单真实回放成功：186车、11120行相关Step、186条曲线、2801条物料、1104条RecipeData、3条CE Lab（3条001/011候选）、40条SAP QM；天气24h特征186车可计算。
- 发现并修复OrderID同时存在数字与`H`前缀格式的安全字面量校验；仅允许`[A-Za-z0-9_-]+`后进入精确订单键SQL，不放宽为任意SQL文本。
- 最新审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T050523Z\refresh_audit.json`；模式DRY_RUN_STAGING，未写目标数据库。

## 2026-08-24 12:54 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：DRY_RUN_STAGING。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T045418Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 13:01 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：DRY_RUN_STAGING。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T050059Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 13:03 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多3个订单；模式：DRY_RUN_STAGING。
- 读取订单3、车次186、Step行11120、曲线186、物料2801、RecipeData1104、CE Lab3、QM40；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T050326Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 13:05 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多3个订单；模式：DRY_RUN_STAGING。
- 读取订单3、车次186、Step行11120、曲线186、物料2801、RecipeData1104、CE Lab3、QM40；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T050423Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 13:05 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多3个订单；模式：DRY_RUN_STAGING。
- 读取订单3、车次186、Step行11120、曲线186、物料2801、RecipeData1104、CE Lab3、QM40；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T050523Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 13:07 其他M1配方候选曲线重建
- 对精确结构候选476车执行BatchCurve→Step→Phase staging重建，曲线命中476车，Segment 6401条，正式曲线资格候选463车。
- 仍遵守同车曲线、累计TimeAV边界、白炭黑`TargetTemp-3°C`动态PID、MSB Ram不适用和Fail-Closed规则；未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_other_recipe_curve_phase_candidates_v1\v371_other_recipe_curve_phase_candidate_audit.json`。

## 2026-08-24 13:24 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337、CE Lab0、QM34；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T052334Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 13:27 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337、CE Lab0、QM34；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T052708Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 13:30 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337、CE Lab0、QM34；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T052914Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 13:34 云端M1增量链路单实例、真实Schema与幂等修复
- 性能：raw ContentHash 检查由逐行查询改为按业务键分块预取；增加 `cloud_m1_daily_refresh.lock` 单实例锁和阶段日志，防止重复执行树并发争用目标库。
- 真实Schema对齐：`core.BatchStep` 实际只有 BatchKey/StepNo/TimeAV/DurationSec/TempAV/RotorAV/EnergyAV/DeltaEnergy_kWh/PressureAV/RebuildRunID 等字段；不再假设不存在的 StepKey、Target 或 SCP-LOG 列。Target 与 SCP-LOG 继续保存在 raw.MmsBatchDataHistory/raw RecipeData，供后续映射合同使用。
- 数据纪律：累计 TimeAV 和 EnergyAV 均按同组 Step 差分；所有核心物理数值经有限值检查，非有限值写 NULL，不填假值。修复 LabelGroupMembership 与 DependencyState 更新参数错误。
- 源字段修正：云端 Orders 的工程版本使用 `ProdVersion`（如 PV80），内部数字 `ProcessVersion` 不再误当工程版本；RecipeID 缺失时使用 CompoundDescription，CompoundCode 作为 RecipeCode。
- 实证执行：订单7994929单实例成功；113车、6471 raw Step、1701 raw物料、113天气状态、113 DependencyState；首次执行审计 `20260824T052914Z`。第二次同订单执行 raw 六类插入均为0，PipelineRun=108 SUCCESS，core.Batch=113、core.BatchStep=1469、重复(BatchKey,GroupName,StepNo)=0、core.LabelGroupMembership=113，证明幂等更新无重复。
- 安全边界：本轮无本地MMS/Archive访问；Phase映射仍为 `BLOCKED_UNKNOWN_PHASE_MAPPING_UNTIL_GOVERNANCE_APPROVAL`；TrainingReady提升=0。旧 raw 历史中存在一版较早来源哈希，保留为历史审计，不直接删除。

## 2026-08-24 13:38 配方身份合同修正
- 工艺确认：`9922A00218BQ_PV80_006` 是完整配方身份；末尾 `_006` 变化即视为配方变化，不能按 `9922A00218BQ` 或仅 `PV80` 合并。
- 云端 Orders 只读核验：RecipeID 为空时，使用完整 `CompoundDescription` 作为 RecipeID；`CompoundCode` 仅作基础物料/RecipeCode；工程版本使用 `ProdVersion`（PV80），不使用内部数字 `ProcessVersion=5`。
- 代码与测试：`bounded_orders_sql()` 已选择 CompoundCode、完整 RecipeID 和 ProdVersion；专项测试由9项增至10项。核心订单7994929已更新为 RecipeID=`9922A00218BQ_PV80_006`、ProcessVersion=`PV80`。
- 修正后的同订单二次执行审计：`20260824T053206Z`，raw 六类插入均为0、核心域更新成功、TrainingReady提升0；说明完整配方身份已稳定且增量幂等。

## 2026-08-24 13:34 其他M1配方治理范围按完整RecipeID重建
- 重新运行`82_validate_other_recipe_mapping_scope_contract.py`，严格使用完整RecipeID作为治理主键；末尾公式修订号（如`_006`）变化不合并，单独进入审批范围。
- 结果：7个其他M1 RecipeID、23个 RecipeID+ProcessVersion+TargetVector 范围、476个候选车次；23/23范围Step数稳定、映射键稳定、SCP-LOG合同完整，阻断范围0。
- 审批队列：`reports/v371_challenger/global_rds_active_v1/v371_all_recipe_mapping_candidates_v1/v371_m1_other_recipe_expert_approval_queue.csv`；审计：`v371_m1_other_recipe_mapping_scope_contract_audit.json`，仅文件生成，数据库写入0。
- 当前仍不可宣称生产映射已批准；下一步是按完整RecipeID逐范围补齐独立Phase映射治理证据，再进入canonical Phase staging/backfill。

## 2026-08-24 13:37 其他配方曲线Phase名称断链修复
- 发现缺陷：候选 Step 映射中的 `Stage6_MSB_PID_Step1/2` 在旧曲线重建中被通用 `canonical_phase()` 改写为 `Stage6_BottomMixing`，造成 Step→Phase 名称断链。
- 修复：`74_rebuild_v371_curve_phase_features.py` 增加 `preserve_phase_names` 受控参数；其他配方候选以 `True` 调用，严格保留该完整RecipeID/Variant的候选Phase名称，仅保留动态PID切分逻辑，不再使用通用别名覆盖。
- `80_validate_other_recipe_curve_candidates.py` 增加候选Phase集合血缘门禁；每个Segment的PhaseName必须存在于同车精确候选Step映射中。
- 重跑结果：476车、6401 Segment、973组Phase特征；正式曲线资格463车；曲线命中476/476；Phase血缘失败0、重叠0、重复0、MSB Ram误填0；独立验证PASS；数据库写入0。
- 代表车 `2370888~1` 已验证输出 `Stage6_MSB_PID_Step1/2`，不再输出 `Stage6_BottomMixing`。审计：`reports/v371_challenger/global_rds_active_v1/v371_other_recipe_curve_phase_candidates_v1/v371_other_recipe_curve_phase_candidate_validation.json`。

## 2026-08-24 13:32 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337、CE Lab0、QM34；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T053206Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 13:33 其他M1配方版本与Target范围稳定性核验
- 对23个RecipeID+ProcessVersion+TargetVector范围逐一检查Step数量、映射键、SCP-LOG和Phase完整性；稳定23个、阻断0个。
- 仅生成专家批准队列，未更新canonical staging、未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_all_recipe_mapping_candidates_v1\v371_m1_other_recipe_mapping_scope_contract_audit.json`。

## 2026-08-24 13:33 其他M1配方版本与Target范围稳定性核验
- 对23个RecipeID+ProcessVersion+TargetVector范围逐一检查Step数量、映射键、SCP-LOG和Phase完整性；稳定23个、阻断0个。
- 仅生成专家批准队列，未更新canonical staging、未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_all_recipe_mapping_candidates_v1\v371_m1_other_recipe_mapping_scope_contract_audit.json`。

## 2026-08-24 13:36 其他M1配方候选曲线重建
- 对精确结构候选476车执行BatchCurve→Step→Phase staging重建，曲线命中476车，Segment 6401条，正式曲线资格候选463车。
- 仍遵守同车曲线、累计TimeAV边界、白炭黑`TargetTemp-3°C`动态PID、MSB Ram不适用和Fail-Closed规则；未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_other_recipe_curve_phase_candidates_v1\v371_other_recipe_curve_phase_candidate_audit.json`。

## 2026-08-24 13:39 其他M1 canonical Phase staging生成
- 使用完整RecipeID、精确候选Step→Phase、同车曲线Segment生成3732行Phase staging，覆盖471车/7个配方。
- 完整曲线车次463，部分曲线车次8；未知Phase行0；未写数据库，TrainingReady=0。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_other_recipe_canonical_phase_staging_v1\v371_other_recipe_canonical_phase_staging_audit.json`；源功率仅保留原始积分并标记未确认单位，不换算kWh。

## 2026-08-24 13:41 M1 Phase全量重建缺口清单
- 当前快照共4100车/129个完整RecipeID；canonical staging覆盖471车，未映射或边界阻断3629车。
- 生成按RecipeID和RebuildStatus分层的车次清单及汇总；未写数据库，TrainingReady全部保持阻断。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_m1_phase_rebuild_gap_inventory_v1\m1_phase_rebuild_gap_inventory_audit.json`。

## 2026-08-24 13:42 M1 Phase全量重建缺口清单
- 当前快照共4100车/129个完整RecipeID；canonical staging覆盖471车，未映射或边界阻断3629车。
- 生成按RecipeID和RebuildStatus分层的车次清单及汇总；未写数据库，TrainingReady全部保持阻断。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_m1_phase_rebuild_gap_inventory_v1\m1_phase_rebuild_gap_inventory_audit.json`。

## 2026-08-24 13:49 云端M1有界水位分页
- `83_cloud_m1_daily_refresh_v1.py` 增加 `(OrderStartTime, OrderID)` 游标；游标时间使用字符串精度，避免pyodbc旧datetime精度导致上一订单重放。
- `87_run_cloud_m1_watermark_refresh.py` 增加分页与本地水位；仅`--execute`成功页推进水位，dry-run不写水位、不写目标库。PowerShell每日入口已切换到水位分页器。
- 实测：两页page-size=1连续回放，第一页7994929，第二页H7994579，未重复；H前缀订单精确子表无数据时不替换为数字兄弟订单，保留依赖阻断。
- 测试：云端刷新专项11/11 PASS；watermark dry-run两页完成，`state_written=false`，未生成水位状态。Phase/TrainingReady提升仍为0。

## 2026-08-24 13:42 M1 Phase全量重建缺口清单
- 当前快照共4100车/129个完整RecipeID；canonical staging覆盖471车，未映射或边界阻断3629车。
- 生成按RecipeID和RebuildStatus分层的车次清单及汇总；未写数据库，TrainingReady全部保持阻断。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_m1_phase_rebuild_gap_inventory_v1\m1_phase_rebuild_gap_inventory_audit.json`。

## 2026-08-24 13:45 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：DRY_RUN_STAGING。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337、CE Lab0、QM34；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054532Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 13:46 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：DRY_RUN_STAGING。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337、CE Lab0、QM34；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054610Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 13:48 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：DRY_RUN_STAGING。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337、CE Lab0、QM34；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054820Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 13:48 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：DRY_RUN_STAGING。
- 读取订单1、车次0、Step行0、曲线0、物料0、RecipeData320、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054841Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 13:52 M1 Phase映射专家审批队列
- 将当前4100车/129个完整RecipeID拆为311个精确治理范围；按未知结构、路线缺口、已有候选和canonical候选分类。
- 所有范围NOT_APPROVED，未猜测Phase、未写数据库、TrainingReady提升0。审批队列：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_m1_phase_mapping_expert_review_queue_v1\m1_phase_mapping_expert_review_queue.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_m1_phase_mapping_expert_review_queue_v1\m1_phase_mapping_expert_review_queue_audit.json`。

## 2026-08-24 13:54 M1 Phase映射专家审批队列
- 将当前4100车/129个完整RecipeID拆为311个精确治理范围；按未知结构、路线缺口、已有候选和canonical候选分类。
- 所有范围NOT_APPROVED，未猜测Phase、未写数据库、TrainingReady提升0。审批队列：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_m1_phase_mapping_expert_review_queue_v1\m1_phase_mapping_expert_review_queue.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_m1_phase_mapping_expert_review_queue_v1\m1_phase_mapping_expert_review_queue_audit.json`。

## 2026-08-24 13:54 M1 Phase映射专家审批队列
- 将当前4100车/129个完整RecipeID拆为311个精确治理范围；按未知结构、路线缺口、已有候选和canonical候选分类。
- 所有范围NOT_APPROVED，未猜测Phase、未写数据库、TrainingReady提升0。审批队列：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_m1_phase_mapping_expert_review_queue_v1\m1_phase_mapping_expert_review_queue.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_m1_phase_mapping_expert_review_queue_v1\m1_phase_mapping_expert_review_queue_audit.json`。

## 2026-08-24 13:59 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337、CE Lab0、QM34；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T055859Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 14:03 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337、CE Lab0、QM34；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T060236Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 14:04 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337、CE Lab0、QM34；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T060352Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 14:07 云端M1增量刷新器V1
- 范围：2026-08-19T00:00:00 至 2026-08-20T00:00:00（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次113、Step行6471、曲线113、物料1701、RecipeData337、CE Lab0、QM34；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T060629Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-24 14:09 每日增量M1 Phase治理桥接
- 从MooneyIntegratedData只读生成113车/1个完整RecipeID/1个精确Phase审核范围；数据库写入0、TrainingReady提升0。
- 审核队列：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_incremental_phase_mapping_review_queue_v1\incremental_m1_phase_mapping_review_queue.csv`；RecipeData Step目标：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_incremental_phase_mapping_review_queue_v1\incremental_m1_recipe_step_targets.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_incremental_phase_mapping_review_queue_v1\incremental_m1_phase_mapping_review_queue_audit.json`。

## 2026-08-24 14:11 每日增量M1 Phase治理桥接
- 从MooneyIntegratedData只读生成113车/1个完整RecipeID/1个精确Phase审核范围；数据库写入0、TrainingReady提升0。
- 审核队列：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_incremental_phase_mapping_review_queue_v1\incremental_m1_phase_mapping_review_queue.csv`；RecipeData Step目标：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_incremental_phase_mapping_review_queue_v1\incremental_m1_recipe_step_targets.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_incremental_phase_mapping_review_queue_v1\incremental_m1_phase_mapping_review_queue_audit.json`。
## 2026-08-24 14:12 云端M1每日链路P0血缘与配方合同修复
- 修复 `83_cloud_m1_daily_refresh_v1.py`：云端真实 `dbo.BatchData`/`dbo.BatchMaterials` 现在显式写 `IsSynthetic=0` 与 RAW provenance；仅对 `etl.PipelineRun.RunType='DAILY_CLOUD_M1_REFRESH'` 的既有行执行限定修复，禁止误改旧资产。
- 订单7994929受控重放：6471条BatchData与1701条BatchMaterial血缘已纠正；行数未增加，坏血缘计数均为0；PipelineRun 111成功，Phase写入0、TrainingReady提升0。
- 将权威 `HESFESFEPLANT.dbo.RecipeData` 接入 `raw.RecipeStepTargetSnapshot`：完整RecipeID=`9922A00218BQ_PV80_006`，写入16个Recipe Step（MSB 5 + MST 11），MST.03=`RECIPE_OR`，其余当前行按源表为`RECIPE_AND`。PipelineRun 112写入16行；PipelineRun 113重放为0插入/0更新/16 unchanged。
- 每车从同车实际Step键、Target向量及完整RecipeID的SCP-LOG重算 `StepPatternHash`、`TargetVectorHash`、`ControlLogicHash`。7994929的113车均为1个结构/1个Target/1个逻辑哈希，缺失0；TemplateKey继续保持NULL，未知Phase映射不提前批准。
- 专项合同测试14/14 PASS；PowerShell每日入口语法通过，并在execute成功后自动生成增量Phase治理审核队列。

## 2026-08-24 14:13 9922A00218BQ_PV80_006增量Phase治理事实
- MooneyIntegratedData已包含该完整RecipeID的113车、26个Pallet组、113车曲线、0个当前Lab组；PhaseRow=0、TrainingReady提升=0。
- RecipeData定义结构为MST 11 + MSB 5（16 Step），订单7994929实际执行结构为MST 8 + MSB 5（13 Step）；缺失于实际执行的Recipe Step为MST.09/10/11。
- 按“Step数量或Pattern改变不得沿用映射”的工艺合同，状态修正为 `BLOCKED_RECIPE_EXECUTION_STEP_PATTERN_MISMATCH`。审核资产：`reports/v371_challenger/global_rds_active_v1/v371_incremental_phase_mapping_review_queue_v1/incremental_m1_phase_mapping_review_queue_audit.json`。
## 2026-08-24 14:18 M1数据专用每日任务发布包
- 新增跨运行watermark V2：未完成的旧窗口必须续跑，只有 `window_complete=true` 后才接受新的滚动窗口，防止MaxPages截断后漏掉窗口后段订单；17项合同测试全部PASS。
- 每日入口在云端源/核心域事务成功后自动生成增量Phase治理队列；仍不执行模型训练、Stage3校准、Phase猜测、`core.BatchPhase`写入或TrainingReady提升。
- 已发布独立包到 `\\10.246.97.159\d\MNY master batch prediction model pipline\releases\M1DataAssetDaily_20260824_v1`，共11个显式文件，服务器包Python编译通过，计划任务安装脚本PLAN_ONLY验证通过。
- 新任务名固定为 `MooneyM1DataAssetDaily`、计划账号`uif45510`、默认每日02:15、S4U；旧 `MooneyDailyPipeline`（会训练/校准模型）不复用。远程Task Scheduler查询返回Access denied，尚未在服务器注册任务，需服务器管理员执行发布包内安装脚本的 `-Register`。
## 2026-08-24 14:27 463车Core精确合同修复执行
- 云端按100个精确OrderID核验：463/463车的完整RecipeID、实际StepKey、全部BatchData Target与RecipeData SCP-LOG均与候选快照一致；Target不一致0、Logic不一致0。
- PipelineRunID=116，事务更新463车：完整RecipeID、工程ProcessVersion、StepPatternHash、TargetVectorHash、ControlLogicHash；数据库复核463/463通过。
- TemplateKey仍NULL、Status=`BLOCKED_MAPPING`；Phase写入0、TrainingReady提升0。Before-image SHA256=`f43de6b5ab10c46be5e52d4cad64e6b58fbfee0c251bb8ec621abedd7800a9fc`；审计：`reports/v371_challenger/global_rds_active_v1/v371_approved_scope_core_contract_reconciliation_v1/approved_scope_core_contract_repair_execution_audit.json`。

## 2026-08-24 完整RecipeID末尾修订号治理确认
- 工艺再次确认：`9922A00218BQ_PV80_006`是完整配方号；末尾`_006`变化表示配方变化，必须作为独立RecipeID、独立治理范围和独立模型/Phase模板身份处理。
- 物理`StepPatternHash`和`TargetVectorHash`在不同配方修订间允许偶然相同，但不得据此合并配方；`ControlLogicHash`、治理Scope和最终`TemplateKey`必须绑定完整RecipeID。
- 新增回归测试，明确`_005`与`_006`即使Step Target相同也不得共享控制合同身份。

## 2026-08-24 MSB B1温降边界合同V2
- 按工艺确认修正：B1起点使用MST终点后首次瞬时温降`>=10°C`；不再仅用“曲线末端减MSB累计时长”倒推。候选映射明确包含的MST末Step不再被构建器强制改为PostPID tail。
- 新增正反例测试：首次12°C温降精确定位B1；提供温度曲线但未观察到10°C温降时返回`BLOCKED_MSB_DISCHARGE_DROP_NOT_FOUND`。相关5项边界测试全部通过。
- 816车冻结曲线重建中807车识别到真实温降；新B1相对旧倒推起点平均提前约79秒。其他配方456车共同Phase中，MSB Phase均温/终温最大变化68°C；Dry/Wet/MST PID边界不变。
- 7个已写车次因未观察到10°C温降撤销候选资格，必须通过V2事务删除旧Phase并清空训练宽表旧阶段值。

## 2026-08-24 14:35 463车精确Phase CORE写前审计
- 完整RecipeID治理：7个配方、23个已明确批准范围、463车、5879条真实Step、3668条精确Phase；PhaseKey重复0。
- 发现并冻结旧合成Phase 2436行；只允许事务内精确替换，能量单位未确认字段保持NULL，TrainingReady提升0。
- 写入集：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_phase_core_merge_v1\approved_m1_phase_core_write_set.csv`；before-image：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_phase_core_merge_v1\core_batch_phase_before_image.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_phase_core_merge_v1\approved_m1_phase_core_merge_preflight_audit.json`。

## 2026-08-24 14:37 463车精确Phase CORE写前审计
- 完整RecipeID治理：7个配方、23个已明确批准范围、463车、5879条真实Step、3668条精确Phase；PhaseKey重复0。
- 发现并冻结旧合成Phase 2436行；只允许事务内精确替换，能量单位未确认字段保持NULL，TrainingReady提升0。
- Phase写入集：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_phase_core_merge_v1\approved_m1_phase_core_write_set.csv`；Batch映射写入集：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_phase_core_merge_v1\approved_m1_core_batch_mapping_write_set.csv`；before-image：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_phase_core_merge_v1\core_batch_phase_before_image.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_phase_core_merge_v1\approved_m1_phase_core_merge_preflight_audit.json`。

## 2026-08-24 14:41 463车精确Phase CORE事务写入
- CanaryRunID=`M1PHASE_20260824_463_V1`；旧合成Phase 2436行已审计替换为3668行同车曲线精确Phase，最终463车/3668行/PhaseKey重复0。
- 新Phase全部IsSynthetic=0；能量单位未确认字段和RequiredFieldComplete保持NULL/0；TrainingReady提升0。
- 完整RecipeID、工程ProcessVersion、TemplateKey及PhaseMappingVariant已绑定；执行审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_phase_core_merge_v1\approved_m1_phase_core_merge_execution_audit.json`。

## 2026-08-24 14:46 463车BatchTrainingReady可信过程特征预检
- 完整RecipeID 7个/463车；Step与Target JSON、Stage1-6时长/温度/转速及物理总时长完整，必需过程字段NULL=0。
- 能量单位未确认字段保持NULL；IsEligibleForTraining提升0。写入集：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_training_process_backfill_v1\approved_m1_batch_training_process_feature_write_set.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_training_process_backfill_v1\approved_m1_batch_training_process_feature_audit.json`。

## 2026-08-24 14:47 463车BatchTrainingReady可信过程特征写入
- 完整RecipeID 7个/463车；Step与Target JSON、Stage1-6时长/温度/转速及物理总时长完整，必需过程字段NULL=0。
- 能量单位未确认字段保持NULL；IsEligibleForTraining提升0。写入集：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_training_process_backfill_v1\approved_m1_batch_training_process_feature_write_set.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_training_process_backfill_v1\approved_m1_batch_training_process_feature_audit.json`。

## 2026-08-24 14:51 V3.7.1可信过程Pallet组训练矩阵预检
- 406个有标签车次对应116组；104个成员完整且模板一致组/363车进入矩阵；6个成员不完整组和6个组内Template不一致组硬阻断；每组只贡献1个MNY标签。
- 数值过程特征108列、NULL=0；未对能量/物料/天气/QM做伪填。矩阵：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_canonical_process_group_matrix_v1\v371_canonical_process_group_matrix.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_canonical_process_group_matrix_v1\v371_canonical_process_group_matrix_audit.json`。

## 2026-08-24 14:51 V3.7.1可信过程Pallet组训练矩阵写入
- 406个有标签车次对应116组；104个成员完整且模板一致组/363车进入矩阵；6个成员不完整组和6个组内Template不一致组硬阻断；每组只贡献1个MNY标签。
- 数值过程特征108列、NULL=0；未对能量/物料/天气/QM做伪填。矩阵：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_canonical_process_group_matrix_v1\v371_canonical_process_group_matrix.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_canonical_process_group_matrix_v1\v371_canonical_process_group_matrix_audit.json`。

## 2026-08-24 14:57 V3.7.1曲线缺失补全Staging重建
- 新增并运行`74_rebuild_v371_curve_phase_features.py`，只使用冻结云端Active BatchCurve快照与当前候选Step Staging；候选车次816，命中曲线816，未访问本地MMS/Archive。
- 按本车累计Step AV边界构造曲线Segment，白炭黑按`TargetTemp-3°C`本车首次温度crossing切分PID；未通过曲线完整性、时间边界或PID阈值的车次只记录阻断，不填0、不借用其他车次。
- 产物：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_curve_phase_rebuild_v1\v371_curve_rebuild_audit.json`、`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_curve_phase_rebuild_v1\v371_curve_rebuild_report.md`；当前`database_writes=false`、`model_ready=false`。

## 2026-08-24 14:58 其他M1配方候选曲线重建
- 对精确结构候选476车执行BatchCurve→Step→Phase staging重建，曲线命中476车，Segment 6303条，正式曲线资格候选456车。
- 仍遵守同车曲线、累计TimeAV边界、白炭黑`TargetTemp-3°C`动态PID、MSB Ram不适用和Fail-Closed规则；未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_other_recipe_curve_phase_candidates_v1\v371_other_recipe_curve_phase_candidate_audit.json`。

## 2026-08-24 14:59 其他M1 canonical Phase staging生成
- 使用完整RecipeID、精确候选Step→Phase、同车曲线Segment生成3676行Phase staging，覆盖464车/7个配方。
- 完整曲线车次456，部分曲线车次8；未知Phase行0；未写数据库，TrainingReady=0。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_other_recipe_canonical_phase_staging_v1\v371_other_recipe_canonical_phase_staging_audit.json`；源功率仅保留原始积分并标记未确认单位，不换算kWh。

## 2026-08-24 14:59 其他M1配方版本与Target范围稳定性核验
- 对23个RecipeID+ProcessVersion+TargetVector范围逐一检查Step数量、映射键、SCP-LOG和Phase完整性；稳定23个、阻断0个。
- 仅生成专家批准队列，未更新canonical staging、未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_all_recipe_mapping_candidates_v1\v371_m1_other_recipe_mapping_scope_contract_audit.json`。

## 2026-08-24 15:07 温降边界V2 Phase CORE写前审计
- 完整RecipeID治理：7个配方、23个批准范围中456车通过温降门禁、5788条真实Step、3612条精确Phase；7车因未观察到≥10°C排胶温降撤销。
- 冻结当前V1 Phase 3668行；事务内替换为V2并撤销7车，能量单位未确认字段保持NULL，TrainingReady提升0。
- Phase写入集：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_phase_core_merge_v2_tempdrop\approved_m1_phase_core_write_set.csv`；Batch映射写入集：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_phase_core_merge_v2_tempdrop\approved_m1_core_batch_mapping_write_set.csv`；before-image：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_phase_core_merge_v2_tempdrop\core_batch_phase_before_image.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_phase_core_merge_v2_tempdrop\approved_m1_phase_core_merge_preflight_audit.json`。

## 2026-08-24 15:08 温降边界V2 Phase CORE事务写入
- CanaryRunID=`M1PHASE_20260824_456_V2_TEMPDROP`；V1 Phase 3668行已审计替换为V2 3612行，456车通过；7车因未观察到≥10°C排胶温降撤销。
- 新Phase全部IsSynthetic=0；能量单位未确认字段和RequiredFieldComplete保持NULL/0；TrainingReady提升0。
- 完整RecipeID、工程ProcessVersion、TemplateKey及PhaseMappingVariant已绑定；执行审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_phase_core_merge_v2_tempdrop\approved_m1_phase_core_merge_execution_audit.json`。

## 2026-08-24 15:13 463车BatchTrainingReady可信过程特征预检
- 完整RecipeID 7个/463车；Step与Target JSON、Stage1-6时长/温度/转速及物理总时长完整，必需过程字段NULL=0。
- 能量单位未确认字段保持NULL；IsEligibleForTraining提升0。写入集：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_training_process_backfill_v2_tempdrop\approved_m1_batch_training_process_feature_write_set.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_training_process_backfill_v2_tempdrop\approved_m1_batch_training_process_feature_audit.json`。

## 2026-08-24 15:14 463车BatchTrainingReady可信过程特征写入
- 完整RecipeID 7个/463车；Step与Target JSON、Stage1-6时长/温度/转速及物理总时长完整，必需过程字段NULL=0。
- 能量单位未确认字段保持NULL；IsEligibleForTraining提升0。写入集：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_training_process_backfill_v2_tempdrop\approved_m1_batch_training_process_feature_write_set.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_approved_scope_training_process_backfill_v2_tempdrop\approved_m1_batch_training_process_feature_audit.json`。

## 2026-08-24 15:14 V3.7.1可信过程Pallet组训练矩阵预检
- 406个有标签车次对应116组；104个成员完整且模板一致组/363车进入矩阵；6个成员不完整组和6个组内Template不一致组硬阻断；每组只贡献1个MNY标签。
- 数值过程特征108列、NULL=0；未对能量/物料/天气/QM做伪填。矩阵：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_canonical_process_group_matrix_v2_tempdrop\v371_canonical_process_group_matrix.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_canonical_process_group_matrix_v2_tempdrop\v371_canonical_process_group_matrix_audit.json`。

## 2026-08-24 15:14 V3.7.1可信过程Pallet组训练矩阵写入
- 406个有标签车次对应116组；104个成员完整且模板一致组/363车进入矩阵；6个成员不完整组和6个组内Template不一致组硬阻断；每组只贡献1个MNY标签。
- 数值过程特征108列、NULL=0；未对能量/物料/天气/QM做伪填。矩阵：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_canonical_process_group_matrix_v2_tempdrop\v371_canonical_process_group_matrix.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_canonical_process_group_matrix_v2_tempdrop\v371_canonical_process_group_matrix_audit.json`。

## 2026-08-24 15:15 发布V3.7.1当前过程组训练视图
- `mart.V371CanonicalProcessGroupTrainingCurrent` 仅选择合同 `V371-CANONICAL-PROCESS-GROUP-V2-TEMPDROP` 的最新构建 `V371PROCGRP-F16D489BAF83F7A7`；104个Pallet/Lab组、363个成员批次，禁止混入已淘汰边界版本。
- RecipeID保留完整修订尾码；该视图仅为过程域可训练接口，不宣称能量、物料、天气或QM域已完备。

## 2026-08-24 15:21 T15760精确配方Phase Staging
- 完整RecipeID `9922A1576030_PV27_002`；候选646车，正式曲线合格617车；其余按物理/PID门禁隔离。
- 使用本车累计AV差分、MST排胶后首个>=10C温降作为B1起点、TargetTemp-3C动态PID并持续至MSB排胶；不跨车填充、不访问本地Archive、不写数据库。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_t15760_exact_canonical_phase_staging_v1\t15760_exact_canonical_phase_staging_audit.json`。

## 2026-08-24 15:22 T15760精确配方Phase Staging
- 完整RecipeID `9922A1576030_PV27_002`；候选646车，正式曲线合格617车；其余按物理/PID门禁隔离。
- 使用本车累计AV差分、MST排胶后首个>=10C温降作为B1起点、TargetTemp-3C动态PID并持续至MSB排胶；不跨车填充、不访问本地Archive、不写数据库。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_t15760_exact_canonical_phase_staging_v1\t15760_exact_canonical_phase_staging_audit.json`。

## 2026-08-24 15:52 已批准候选范围Core合同对账
- 云端精确OrderID核验154订单/617车/1个完整RecipeID；预检PASS=617，BLOCKED=0，执行=False。
- Phase写入0、TrainingReady提升0；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_t15760_exact_core_contract_reconciliation_v1\approved_scope_core_contract_reconciliation_audit.json`。

## 2026-08-24 15:53 V3.7.1 V2训练接口与T15760扩展
- 温降边界V2批次过程特征已事务回填至 `mart.BatchTrainingReady`：456车，PipelineRunID=124；406车有标签、50车无标签，必需过程字段NULL=0，未伪造能耗且未提升 `IsEligibleForTraining`。
- Pallet/Lab组过程矩阵已重建并提交：BuildRunID=`V371PROCGRP-F16D489BAF83F7A7`，PipelineRunID=125；104个独立组/363成员车/108过程特征，标签及数值NULL=0。发布当前视图 `mart.V371CanonicalProcessGroupTrainingCurrent`，只选择 `V371-CANONICAL-PROCESS-GROUP-V2-TEMPDROP` 最新构建。
- RecipeID身份合同保持完整尾码，`9922A00218BQ_PV80_006` 与任何 `_005`/其它尾码均视为不同配方，禁止归一化合并。
- T15760完整RecipeID `9922A1576030_PV27_002`：646候选车逐车重算，617车通过本车曲线完整性、MST后首个>=10°C温降B1起点、TargetTemp-3°C PID及完整8 Phase；29车阻断。云端APAC RDS精确对账617/617 PASS，Target差异0、RecipeData SCP-LOG差异0；当前仅Staging，Core写入0。
- 云端精确对账改为10订单小分片并逐片缓存到 `reports/_cloud_exact_order_chunk_cache`，中断不重复读取成功分片；未访问本地MMS/Archive。

## 2026-08-24 16:01 T15760 Phase事务写前预检
- `9922A1576030_PV27_002`精确617车/4936 Phase，云端对账617/617 PASS；写集与Before-Image已生成，当前未写Core。

## 2026-08-24 16:04 T15760 Phase Core事务提交
- Canary=`M1PHASE_20260824_T15760_617_V1`，617车删除旧Phase 3636行，写入4936行；能耗字段全NULL，IsSynthetic=0，事务后校验通过。

## 2026-08-24 16:09 T15760 BatchTrainingReady过程特征回填
- 617车阶段时长/温度/转速/Step JSON已写入；能耗全NULL、IsEligibleForTraining=0；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\t15760_training_process_backfill_v1\t15760_training_process_execution_audit.json`。

## 2026-08-24 16:11 M1全映射过程组矩阵V3
- 合并旧批准范围与T15760精确617车，Build=`V371PROCGRP-8AD264F8F894E580`，训练组261、成员车919；能耗/物料/天气/QM不做填补。

## 2026-08-24 16:11 发布V3.7.1当前过程组训练视图
- `mart.V371CanonicalProcessGroupTrainingCurrent` 仅选择合同 `V371-CANONICAL-PROCESS-GROUP-V3-M1-ALL-MAPPED` 的最新构建 `V371PROCGRP-8AD264F8F894E580`；104个Pallet/Lab组、363个成员批次，禁止混入已淘汰边界版本。
- RecipeID保留完整修订尾码；该视图仅为过程域可训练接口，不宣称能量、物料、天气或QM域已完备。

## 2026-08-24 16:14 B00458精确Phase Staging
- 完整RecipeID `9922A00458BT_PV77_005`，170车逐车曲线重建，正式曲线合格169车；只读Staging，未访问本地Archive。

## 2026-08-24 16:16 已批准候选范围Core合同对账
- 云端精确OrderID核验44订单/169车/1个完整RecipeID；预检PASS=169，BLOCKED=0，执行=False。
- Phase写入0、TrainingReady提升0；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_b00458_exact_core_contract_reconciliation_v1\approved_scope_core_contract_reconciliation_audit.json`。

## 2026-08-24 16:19 B00458 Phase事务写前预检
- `9922A00458BT_PV77_005`精确169车/1183 Phase，云端对账169/169 PASS；写集和Before-Image已生成，未写Core。

## 2026-08-24 16:24 B00458 Phase Core事务提交
- 更正记录：Recipe route=`M1_B00458_EXACT`，Canary=`M1PHASE_20260824_B00458_169_V1`，169车删除旧Phase 1002行，写入1183行；能耗字段全NULL，IsSynthetic=0，事务后校验通过。

## 2026-08-24 16:27 B00458 BatchTrainingReady过程特征回填
- 169车阶段时长/温度/转速/Step JSON已写入；能耗全NULL、IsEligibleForTraining=0；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\b00458_train_backfill_v1\b00458_training_process_execution_audit.json`。

## 2026-08-24 16:27 M1全映射过程组矩阵V3
- 合并旧批准范围、T15760精确617车与B00458精确169车，Build=`V371PROCGRP-08C741A90B98F752`，训练组307、成员车1084；能耗/物料/天气/QM不做填补。

## 2026-08-24 16:27 发布V3.7.1当前过程组训练视图
- `mart.V371CanonicalProcessGroupTrainingCurrent` 仅选择合同 `V371-CANONICAL-PROCESS-GROUP-V3-M1-ALL-MAPPED` 的最新构建 `V371PROCGRP-08C741A90B98F752`；当前期望307个Pallet/Lab组、1084个成员批次，禁止混入已淘汰边界版本。
- RecipeID保留完整修订尾码；该视图仅为过程域可训练接口，不宣称能量、物料、天气或QM域已完备。

## 2026-08-24 16:30 完整配方身份合同复核
- 完整配方身份必须以源 RecipeID 全字符串（例如 `9922A00218BQ_PV80_006`）作为治理键；任何尾码变化视为新配方版本，禁止合并。
- 云端 `core.Batch` 与 `mart.BatchTrainingReady` 当前物理存储为拆分字段（主体 `RecipeID` + 工艺字段 `ProcessVersion`），不能将拆分字段直接等同于完整 RecipeID；完整身份继续由候选快照/RecipeData 合同保存并参与范围过滤。
- B00458 已按完整源身份 `9922A00458BT_PV77_005` 建立 169 车精确候选；数据库写入仅使用其物理批次键和治理哈希，未用其它尾码代替。

## 2026-08-24 16:33 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共347981车，解析347492车，阻断489车，完整配方471个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-08-24 16:34 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填329342行；M1训练车329342，身份解析329342，阻断0，未改变训练资格。

## 2026-08-24 16:52 M1每日数据资产发布包V2
- 发现服务器旧发布包 `M1DataAssetDaily_20260824_v1` 未包含完整RecipeID身份刷新脚本。
- 新建服务器发布包 `releases/M1DataAssetDaily_20260824_v2`，加入更新后的云端刷新入口、`109_materialize_m1_full_recipe_identity.py`、`110_attach_full_recipe_identity_to_training_ready.py` 和管理员安装脚本；未覆盖V1，未启动旧 `MooneyDailyPipeline`。

## 2026-08-24 16:39 M1完整配方重建库存
- 生成471个完整RecipeID的车次、Step、模板、Phase状态、标签覆盖和阻断库存；数据库只读，CSV/JSON审计位于 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_m1_recipe_rebuild_inventory_v1`。

## 2026-08-24 16:45 V371组级身份安全视图
- 发布 `mart.V371TrainingGroupMatrixCurrent`，在旧组快照上增加完整RecipeID、身份状态、当前Phase合同状态和ModelUseStatus；1197组中身份解析754组，仍明确标记为旧特征桥接，不宣称最终训练宽表完成。
## 2026-08-24 继续固定 M1 每日刷新 V2 发布清单
- 核对服务器发布包 `\\10.246.97.159\d\MNY master batch prediction model pipline\releases\M1DataAssetDaily_20260824_v2`，确认新版安装器、云端刷新入口、完整 RecipeID 身份脚本 109/110 均存在。
- 发现 V2 目录内原有 manifest 仍标记为 V1，新增 `M1DataAssetDaily_20260824_v2_manifest.json`，记录实际 V2 文件 SHA256；V1 manifest 未覆盖。
- 任务运行边界保持：仅云端 RDS、M1 only、无本地 MMS Archive、无模型训练、无 Stage3 校准、无 Phase 猜测、无 TrainingReady 自动晋级。
## 2026-08-24 M1 每日任务服务器管理员 Runbook V2
- 新增 `Mooney_Integrated_Data_Asset/deployment/M1_DAILY_SERVER_ADMIN_RUNBOOK_V2.md`。
- 明确推荐将 `releases/M1DataAssetDaily_20260824_v2` 复制到服务器本地 `C:\ProgramData\Continental\Mooney\M1DataAssetDaily_v2` 后注册 S4U 任务，避免计划任务运行时访问 UNC 失败。
- 明确凭据单独存放并限制 ACL；任务仅云端 RDS、M1 only、无 Archive 回退、无模型训练和无自动晋级。
## 2026-08-24 M1 剩余配方精确映射进度审计
- 新增 `reports/v371_challenger/global_rds_active_v1/m1_recipe_mapping_progress_v1.md`。
- 明确 471 个完整 RecipeID、347,981 个 M1 批次、476 个稳定结构候选批次、2,794 个未知结构批次和 489 个身份缺失批次的当前状态。
- 明确候选配方 `9922A00458BT_PV77_008` 内含 10/11 Step 混合，必须按 Scope 分流；禁止按 FullRecipeID 粗粒度直接合并。
- 明确 `9922A00218BQ_PV80_006` 为 16 Step 未知映射，不能复用 T15760 的 13 Step 映射，继续 Fail-Closed。

## 2026-08-24 16:54 其他M1 canonical Phase staging生成
- 使用完整RecipeID、精确候选Step→Phase、同车曲线Segment生成3676行Phase staging，覆盖464车/7个配方。
- 完整曲线车次456，部分曲线车次8；未知Phase行0；未写数据库，TrainingReady=0。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_other_recipe_canonical_phase_staging_v1\v371_other_recipe_canonical_phase_staging_audit.json`；源功率仅保留原始积分并标记未确认单位，不换算kWh。

## 2026-08-24 16:54 M1 Phase全量重建缺口清单
- 当前快照共4100车/129个完整RecipeID；canonical staging覆盖464车，未映射或边界阻断3636车。
- 生成按RecipeID和RebuildStatus分层的车次清单及汇总；未写数据库，TrainingReady全部保持阻断。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_m1_phase_rebuild_gap_inventory_v1\m1_phase_rebuild_gap_inventory_audit.json`。

## 2026-08-24 16:56 M1 Phase全量重建缺口清单
- 当前快照共4100车/129个完整RecipeID；canonical staging覆盖464车，未映射或边界阻断2839车。
- 生成按RecipeID和RebuildStatus分层的车次清单及汇总；未写数据库，TrainingReady全部保持阻断。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_m1_phase_rebuild_gap_inventory_v1\m1_phase_rebuild_gap_inventory_audit.json`。

## 2026-08-24 16:56 M1 Phase全量重建缺口清单
- 当前快照共4100车/129个完整RecipeID；canonical staging覆盖464车，未映射或边界阻断2839车。
- 生成按RecipeID和RebuildStatus分层的车次清单及汇总；未写数据库，TrainingReady全部保持阻断。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_m1_phase_rebuild_gap_inventory_v1\m1_phase_rebuild_gap_inventory_audit.json`。
## 2026-08-24 M1 Phase缺口库存V2精确试点感知修正
- 重跑 84/85/86：其他配方 canonical staging 3676行、464车，独立验证 PASS；未写数据库。
- 修正 `86_build_m1_phase_rebuild_gap_inventory.py`，接入 T15760/B00458 独立精确 Staging 清单，避免把已有精确试点误报为 `CANDIDATE_MAPPING_NOT_IN_CURRENT_QUEUE`。
- 新口径：786车 `EXISTING_EXACT_PHASE_STAGING_READY`，11车 `EXISTING_EXACT_PHASE_STAGING_BLOCKED`，456车其他配方 canonical ready，8车 partial；未知结构仍为2794车，未写入 TrainingReady。
- 新审计：`reports/v371_challenger/global_rds_active_v1/v371_m1_phase_rebuild_gap_inventory_v1/m1_phase_rebuild_gap_inventory_audit.json`，版本 `V2_EXACT_STAGING_AWARE`，SHA256 `f89726299c8614b89965e5c9b08b294d96980a9e69d3f6a6bd91644102f66216`。

## 2026-08-24 16:58 M1未知结构审核队列
- 对2794个未知结构车次按StructuralSignature、RouteSignature、Step数、材料体系和路线分组为85个Scope。
- 已知签名仅标记为SCOPE复核，未知签名标记为NEW_STRUCTURAL_MAPPING_REQUIRED；全部未批准、未写库。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_unknown_m1_structural_review_queue_v1\unknown_m1_structural_review_queue_audit.json`。

## 2026-08-24 16:59 M1未知结构审核队列
- 对2794个未知结构车次按StructuralSignature、RouteSignature、Step数、材料体系和路线分组为85个Scope。
- 已知签名仅标记为SCOPE复核，未知签名标记为NEW_STRUCTURAL_MAPPING_REQUIRED；全部未批准、未写库。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_unknown_m1_structural_review_queue_v1\unknown_m1_structural_review_queue_audit.json`。
## 2026-08-24 M1未知结构审核队列与材料体系防串模板
- 新增并运行 `113_build_unknown_m1_structural_review_queue.py`，将2794个未知车次聚合为85个结构Scope。
- 初版发现仅按结构签名+路线会把白炭黑与炭黑的同设备结构误匹配；已修正为 `StructuralSignature + RouteSignature + MaterialSystem` 三元键。
- 修正后85/85 Scope 均为 `NEW_STRUCTURAL_MAPPING_REQUIRED`，无任何自动复用、批准或数据库写入。
- 审计队列：`reports/v371_challenger/global_rds_active_v1/v371_unknown_m1_structural_review_queue_v1/unknown_m1_structural_review_queue_audit.json`，队列SHA256 `4ec9b2da13d509d3b435e678f3e6e032aa276c5ef308fa8bd2ba522c148e0534`。
## 2026-08-24 M1每日任务V2服务器路径计划验证
- 使用服务器 UNC 发布包和云端凭据路径运行安装器的 PLAN_ONLY 验证，Runner、用户 `uif45510`、02:15、S4U 和 RuntimeRoot 均正确解析。
- 验证未注册计划任务、未启动刷新、未写数据库；服务器管理员仍需在服务器本机复制到本地目录后执行 `-Register` 并首次手动运行。

## 2026-08-24 17:00 未知M1 Scope代表Step审核包
- 为85个未知结构Scope选取85个真实代表车次，共1044行Step；保留Target、AV、SCP-LOG和AND/OR字段。
- ProposedPhaseName全部为NULL，状态为BLOCKED_UNKNOWN_PHASE_MAPPING，未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_unknown_m1_structural_review_queue_v1\unknown_m1_scope_representative_step_pack_audit.json`。
## 2026-08-24 未知M1 Scope代表Step审核包
- 新增并运行 `114_build_unknown_m1_scope_representative_step_pack.py`，为85个未知结构Scope选取85个真实代表车次，共1044行Step。
- 保留实际值、Target、SCP-LOG来源、AND/OR位和Step边界；ProposedPhaseName 1044/1044 为 NULL，全部 `BLOCKED_UNKNOWN_PHASE_MAPPING`，未写数据库。
- 审计队列 SHA256 `980085d73efcb87ad80a8dcca81ebb1e5e570e1dcdc3e4f38d5f3d484ae2a18b`。
## 2026-08-24 M1审核脚本回归验证
- `86_build_m1_phase_rebuild_gap_inventory.py`、`113_build_unknown_m1_structural_review_queue.py`、`114_build_unknown_m1_scope_representative_step_pack.py` 均通过 `py_compile`。
- 三份审计 JSON 均保持 `database_writes=false`、`local_mms_archive_access=false`；代表Step包所有 ProposedPhaseName 为空。

## 2026-08-24 17:02 M1未知结构审核队列
- 对2794个未知结构车次按StructuralSignature、RouteSignature、Step数、材料体系和路线分组为85个Scope。
- 已知签名仅标记为SCOPE复核，未知签名标记为NEW_STRUCTURAL_MAPPING_REQUIRED；全部未批准、未写库。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_unknown_m1_structural_review_queue_v1\unknown_m1_structural_review_queue_audit.json`。

## 2026-08-24 17:02 未知M1 Scope代表Step审核包
- 为85个未知结构Scope选取85个真实代表车次，共1044行Step；保留Target、AV、SCP-LOG和AND/OR字段。
- ProposedPhaseName全部为NULL，状态为BLOCKED_UNKNOWN_PHASE_MAPPING，未写数据库。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_unknown_m1_structural_review_queue_v1\unknown_m1_scope_representative_step_pack_audit.json`。
## 2026-08-24 M1未知结构Scope优先级分层
- 远程读取服务器计划任务被拒绝：`schtasks /Query` 返回 `Access is denied`，未进行远程写入或启动；服务器管理员仍可在本机查看LastTaskResult。
- 为85个未知结构Scope增加真实覆盖优先级：P0 7个Scope/1624车，P1 13个Scope/641车，P2 65个Scope/529车。
- 重跑代表Step审核包，85个代表车次、1044行Step、ProposedPhaseName仍100% NULL。
## 2026-08-24 最终训练表与每日流程目标纠偏
- 审计确认：`mart.V371TrainingGroupMatrixCurrent` 是桥接视图，不是物理最终训练表；每日 v2 流程当前实际写入 raw/core，并由 109/110 更新 `core.BatchRecipeIdentity` 和 `mart.BatchTrainingReady` 身份字段，未刷新 V371 训练资产。
- 新增合同文档 `Mooney_Integrated_Data_Asset/docs/v371_final_training_dataset_daily_contract_v1.md`，指定未来正式物理训练表为 `mart.V371TrainingDataset`，一行一个 LabelGroupID，BatchTrainingReady 仅作为兼容层。
- 更新每日 runner 警告和服务器 Runbook，明确当前尚未刷新 `mart.V371TrainingDataset`，避免把旧 BatchTrainingReady 误当最终训练源。
- 已将更新后的 runner、Runbook 和 V2 manifest 同步到服务器发布包 `releases/M1DataAssetDaily_20260824_v2`；未注册、未启动任务。

## 2026-08-24 17:09 V371最终训练表DDL只读前置核验
- 只读核对 `mart.V371TrainingGroupMatrix`（1197行/229列）和 `mart.V371TrainingGroupPhaseFeatureSnapshot`（1197行/250列）。
- 确认 `mart.V371TrainingDataset` 尚不存在；生成一行一个LabelGroupID的DDL提案，未执行DDL/DML、未提升训练资格。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_training_dataset_preflight_v1\v371_training_dataset_preflight_audit.json`。

## 2026-08-24 17:10 V371最终训练表DDL只读前置核验
- 只读核对 `mart.V371TrainingGroupMatrix`（1197行/229列）和 `mart.V371TrainingGroupPhaseFeatureSnapshot`（1197行/250列）。
- 确认 `mart.V371TrainingDataset` 尚不存在；生成一行一个LabelGroupID的DDL提案，未执行DDL/DML、未提升训练资格。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_training_dataset_preflight_v1\v371_training_dataset_preflight_audit.json`。
## 2026-08-24 V371TrainingDataset物理表只读前置核验
- 只读连接 MooneyIntegratedData 确认：`mart.V371TrainingGroupMatrix` 1197行/229列，`mart.V371TrainingGroupPhaseFeatureSnapshot` 1197行/250列，`mart.V371TrainingDataset` 不存在。
- 新增并运行 `115_build_v371_training_dataset_preflight.py`，生成471列一行一个LabelGroupID的DDL提案；LabelGroupID/TrainingUnit强制NOT NULL，唯一索引和BuildRunID字段已纳入。
- DDL未执行、TrainingReady未提升、未访问本地MMS/Archive；DDL SHA256 `161c408b3543994a45a96e0be60076dfc7d9845c6a603610f3357f4d33453723`。

## 2026-08-24 17:12 V371最终训练表Staging联合快照
- 只读合并GroupMatrix、GroupPhaseFeatureSnapshot和最新CanonicalProcessGroupMatrix，生成1197个LabelGroupID、612列联合特征。
- 资格分布：{'BLOCKED_PHASE_OR_LABEL_OR_PROCESS': 994, 'ELIGIBLE_FOR_V371_TRAINING_DATASET': 203}；未执行DDL/DML，未提升训练资格。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_training_dataset_staging_v1\v371_training_dataset_staging_audit.json`。

## 2026-08-24 17:12 V371最终训练表Staging联合快照
- 只读合并GroupMatrix、GroupPhaseFeatureSnapshot和最新CanonicalProcessGroupMatrix，生成1197个LabelGroupID、611列联合特征。
- 资格分布：{'BLOCKED_PHASE_OR_LABEL_OR_PROCESS': 994, 'ELIGIBLE_FOR_V371_TRAINING_DATASET': 203}；未执行DDL/DML，未提升训练资格。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_training_dataset_staging_v1\v371_training_dataset_staging_audit.json`。

## 2026-08-24 17:13 V371最终训练表DDL只读前置核验
- 只读核对 `mart.V371TrainingGroupMatrix`（1197行/229列）和 `mart.V371TrainingGroupPhaseFeatureSnapshot`（1197行/250列）。
- 确认 `mart.V371TrainingDataset` 尚不存在；生成一行一个LabelGroupID的DDL提案，未执行DDL/DML、未提升训练资格。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_training_dataset_preflight_v1\v371_training_dataset_preflight_audit.json`。

## 2026-08-24 17:13 V371最终训练表DDL只读前置核验
- 只读核对 `mart.V371TrainingGroupMatrix`（1197行/229列）和 `mart.V371TrainingGroupPhaseFeatureSnapshot`（1197行/250列）。
- 确认 `mart.V371TrainingDataset` 尚不存在；生成一行一个LabelGroupID的DDL提案，未执行DDL/DML、未提升训练资格。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_training_dataset_preflight_v1\v371_training_dataset_preflight_audit.json`。

## 2026-08-25 08:44 修正V371最终训练资产为车次粒度
- 明确废止将1,197个LabelGroupID作为最终训练表的做法；只读使用V371TrainingBatchFeatureSnapshot、V371TrainingGroupMemberSnapshot、V371TrainingGroupMatrix和V371TrainingBatchStepSnapshot，构建一车一行的4100车Staging资产。
- LabelGroupID保留为组级Lab标签和评价键，不把同一Lab值当作独立车次标签；训练评价需先预测车次，再按LabelGroupID聚合均值。
- 当前未执行DDL/DML；未知Phase映射和待专家批准映射均阻断训练资格，不猜测Phase。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_batch_training_dataset_staging_v1\v371_batch_training_dataset_staging_audit.json`。

## 2026-08-25 08:45 批次粒度最终训练表DDL前置核验
- 生成车次粒度 `mart.V371TrainingDataset` DDL提案：唯一键为PhysicalBatchKey，LabelGroupID仅建普通索引，禁止错误的组级唯一约束。
- 仅生成提案，未执行DDL/DML。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_batch_training_dataset_staging_v1\v371_batch_training_dataset_ddl_preflight_audit.json`。

## 2026-08-25 08:53 M1按阶段语义与duration包络生成Phase映射候选
- 不再将T15760结构硬套全部配方；对4100车/54116个Step按胶料体系、路线、阶段顺序、历史实际duration包络和Silica TargetTemp-3°C动态PID阈值生成候选。
- 候选570车，阻断3530车；全部未写库、未提升训练资格。无油路线因缺少已批准的独立阶段合同保持阻断。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_duration_phase_mapping_candidates_v1\m1_duration_phase_mapping_candidate_audit.json`。

## 2026-08-25 08:55 M1按阶段语义与duration包络生成Phase映射候选
- 不再将T15760结构硬套全部配方；对4100车/54116个Step按胶料体系、路线、阶段顺序、历史实际duration包络和Silica TargetTemp-3°C动态PID阈值生成候选。
- 候选488车，阻断3612车；全部未写库、未提升训练资格。无油路线因缺少已批准的独立阶段合同保持阻断。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_duration_phase_mapping_candidates_v1\m1_duration_phase_mapping_candidate_audit.json`。

## 2026-08-25 只读审计：旧Phase分段中的投油字段与压栓下降逻辑
- 检查 `Mooney_Prediction_Pipeline/data_processing/curve_segmenter_all_compounds.py`、`preprocess_raw_curves.py`、`pipeline_orchestrator.py` 与 `model_analysis/predict_order_v3.py`，未执行数据库读取或写入。
- 旧代码投油Step不是来自 `RecipeData`：它从 `BatchData.VariablePath='SCP-1-Step-Time-rel-s'` 中取值，并用 `RecipeMaterials.StepNumber=bd.StepNo AND MaterialCode LIKE '%CS4%'`筛选；输出字段为 `oil.StepNo`、`CurrentValue=bd.Value`、`PrevStepValue=bd_prev.Value`。这只是旧逻辑证据，不是最终RecipeData合同。
- 旧 `find_ram_first_drop()` 只取首个单点 `diff < -5`，找不到时硬回退约30秒；不能满足连续下降与严格异常判定要求，必须改为连续/去抖判定并对不确定结果Fail-Closed。
- 旧PID阈值使用 `TargetTemperature-20°C`，缺失时默认 `140°C`；新合同必须改为两类胶料统一 `TargetTemperature-3°C`，缺失不得默认。
- 旧代码存在以 `t1+45s`、`oil_start+40s`估算投油边界和以`N-1/N/N+1`排除粘门相邻车次的历史旁路；这些不能进入新正式映射，除非作为显式诊断字段保留。
- 结论：旧代码可复用曲线解码、特征聚合框架和Isolation Forest思路，但投油Step、Ram连续下降、PID阈值、MSB统一BottomMixing边界必须按新合同重构；当前未写库、未提升训练资格。

## 2026-08-25 V3分段契约重建与T15760旧逻辑审计
- 新增纯函数模块 `Mooney_Prediction_Pipeline/data_processing/phase_segmentation_contract_v3.py`：CS4*唯一投油证据、TimeAV差分、连续Ram下降、Target-3°C PID、MSB统一BottomMixing；缺失或歧义均Fail-Closed，无默认值。
- 新增测试 `Mooney_Prediction_Pipeline/tests/test_phase_segmentation_contract_v3.py`。当前环境未安装pytest，因此未宣称pytest通过；已完成py_compile和核心纯函数smoke test，结果 `PURE_CONTRACT_SMOKE_PASS`。
- 审计结论：现有T15760候选脚本的PID Target-3逻辑部分正确，但仍存在固定MST步序、MSB拆分为多个PID Step名、旧duration候选合同等问题，不视为最终准确映射；T15760必须用V3契约重放后再恢复训练资格。
- 本轮只改本地纯函数与测试，未访问本地MMS Archive，未写任何数据库，未提升任何训练资格。

## 2026-08-25 08:56 M1按阶段语义与duration包络生成Phase映射候选
- 不再将T15760结构硬套全部配方；对4100车/54116个Step按胶料体系、路线、阶段顺序、历史实际duration包络和Silica TargetTemp-3°C动态PID阈值生成候选。
- 候选488车，阻断3612车；全部未写库、未提升训练资格。无油路线因缺少已批准的独立阶段合同保持阻断。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_duration_phase_mapping_candidates_v1\m1_duration_phase_mapping_candidate_audit.json`。

## 2026-08-25 09:48 V3双胶料Phase分段只读回放
- 试点：T15760 PV27 与 B00458 PV77；曲线来自RDS Active BatchCurve快照，RecipeMaterials仅按两个RecipeID读取CS4证据。
- 结果：0通过/10阻断；未写数据库、未访问本地MMS Archive、未批准全量映射。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v3_phase_contract_dual_pilot_replay_v1\v3_dual_pilot_replay_audit.json`。

## 2026-08-25 09:48 V3双胶料Phase分段只读回放
- 试点：T15760 PV27 与 B00458 PV77；曲线来自RDS Active BatchCurve快照，RecipeMaterials仅按两个RecipeID读取CS4证据。
- 结果：0通过/10阻断；未写数据库、未访问本地MMS Archive、未批准全量映射。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v3_phase_contract_dual_pilot_replay_v1\v3_dual_pilot_replay_audit.json`。

## 2026-08-25 09:49 V3双胶料Phase分段只读回放
- 试点：T15760 PV27 与 B00458 PV77；曲线来自RDS Active BatchCurve快照，RecipeMaterials仅按两个RecipeID读取CS4证据。
- 结果：10通过/0阻断；未写数据库、未访问本地MMS Archive、未批准全量映射。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v3_phase_contract_dual_pilot_replay_v1\v3_dual_pilot_replay_audit.json`。

## 2026-08-25 09:51 M1全量Phase V3只读回放
- 范围：4100车/129个RecipeID；曲线来自RDS Active快照，RecipeMaterials按RecipeID分块只读读取CS4证据。
- 结果：4062通过/38阻断；未写库、未访问本地MMS Archive、未提升训练资格。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v3_phase_contract_all_m1_replay_v1\v3_all_m1_replay_audit.json`。

## 2026-08-25 09:57 M1全量Phase V3只读回放
- 范围：4100车/129个RecipeID；曲线来自RDS Active快照，RecipeMaterials按RecipeID分块只读读取CS4证据。
- 结果：4062通过/34阻断；未写库、未访问本地MMS Archive、未提升训练资格。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v3_phase_contract_all_m1_replay_v2\v3_all_m1_replay_audit.json`。

## 2026-08-25 All-M1 MixingInfo第一异常门禁
- 新增 `123_gate_mixinginfo_before_phase_mapping_all_m1.py`。只按4,100批次精确OrderID+BatchID受限读取 MixingCurveData.dbo.MixingInfo，不扫描本地MMS/Archive、不写数据库。
- 先处理MixingInfo异常；粘门源车N映射到同订单下一实际车N+1，下一车需继续曲线形态门禁；通过者才允许Phase映射。
- 运行状态：BLOCKED。

## 2026-08-25 All-M1 MixingInfo第一异常门禁
- 新增 `123_gate_mixinginfo_before_phase_mapping_all_m1.py`。只按4,100批次精确OrderID+BatchID受限读取 MixingCurveData.dbo.MixingInfo，不扫描本地MMS/Archive、不写数据库。
- 先处理MixingInfo异常；粘门源车N映射到同订单下一实际车N+1，下一车需继续曲线形态门禁；通过者才允许Phase映射。
- 运行状态：BLOCKED。

## 2026-08-25 All-M1 MixingInfo第一异常门禁
- 新增 `123_gate_mixinginfo_before_phase_mapping_all_m1.py`。只按4,100批次精确OrderID+BatchID受限读取 MixingCurveData.dbo.MixingInfo，不扫描本地MMS/Archive、不写数据库。
- 先处理MixingInfo异常；粘门源车N映射到同订单下一实际车N+1，下一车需继续曲线形态门禁；通过者才允许Phase映射。
- 运行状态：MIXINGINFO_GATE_COMPLETE_MAPPING_MUST_USE_ELIGIBLE_ONLY。

## 2026-08-25 All-M1 MixingInfo第一异常门禁
- 新增 `123_gate_mixinginfo_before_phase_mapping_all_m1.py`。只按4,100批次精确OrderID+BatchID受限读取 MixingCurveData.dbo.MixingInfo，不扫描本地MMS/Archive、不写数据库。
- 先处理MixingInfo异常；粘门源车N映射到同订单下一实际车N+1，下一车需继续曲线形态门禁；通过者才允许Phase映射。
- 运行状态：MIXINGINFO_GATE_COMPLETE_MAPPING_MUST_USE_ELIGIBLE_ONLY。

## 2026-08-25 10:07 M1全量Phase V3只读回放
- 范围：3033车/104个RecipeID；曲线来自RDS Active快照，RecipeMaterials按RecipeID分块只读读取CS4证据。
- 结果：3019通过/13阻断；未写库、未访问本地MMS Archive、未提升训练资格。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v3_phase_contract_all_m1_replay_v3_mixinginfo_clean\v3_all_m1_replay_audit.json`。

## 2026-08-25 10:32 M1 V3.7 MixingInfo硬门槛训练数据物化
- 以MixingInfo门禁与V3 Phase PASS交集作为准入，计划批次4100，最终可训练3019批；分表保存批次223列、Step、曲线原文、物料、天气、Lab组及成员关系。
- 写入状态：MATERIALIZED_SUCCESS；未访问本地MMS/Archive。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\m1_training_dataset_v371_materialization_v1\m1_training_materialization_audit.json`。

## 2026-08-25 10:38 M1 V3.7 MixingInfo硬门槛训练数据物化
- 空值审计：只读检查云端目标库 `mart.M1TrainingBatchV371`（3019行、234列）及来源 `mart.V371TrainingBatchFeatureSnapshot`（4100行、223列），未访问本地MMS/Archive，未写数据库。
- 事实：来源快照本身已有一批全空列，物化表是原样带入，并非本次INSERT把有效值清空。
- 全空列主要包括 `__group_std` 组内统计列（批次级快照没有组内统计来源）、1Hz功率/扭矩/效率相关字段及其缺失指示列。当前 `mart.M1TrainingStepV371`只有TimeAV/TempAV/RotorAV/EnergyAV等字段，没有Power/Torque/效率原始列，不能凭空补算。
- `GateReason`、`Reasons` 在通过批次中为NULL属于空原因字段，不代表门禁失败；Stage3/Stage4功率、扭矩、效率相关字段有594/3019批次为空，其余主要阶段过程字段并非全空。
- 结论：当前表不能宣称“所有字段完整”或直接作为不加筛选的训练矩阵。后续应建立字段合同，区分 `ROUTE_NOT_APPLICABLE`、`SOURCE_DATA_MISSING`、`NOT_COMPUTABLE_FROM_SOURCE`，补齐 `__missing` 标志；物理字段保持NULL，不填0。
- 用户指出功率/扭矩/效率应从BatchCurve拆分。本次复核确认：本地冻结曲线快照含 `Curve1/Curve2/Curve5/Curve6/Curve7`，旧代码明确映射为温度/功率/扭矩/转子/压栓；`step_aligned_curve_features.py`也已有解码与按Step边界聚合函数。
- 关键缺口：`124_materialize_m1_training_dataset_v371.py`当前只把原始曲线写入 `M1TrainingCurveV371`，没有调用Step-aligned curve feature builder，也没有把Curve2/Curve5聚合回`M1TrainingBatchV371`。因此功率/扭矩/效率字段的空值主要是“曲线特征尚未重算回填”，不能判定为源数据不存在。
- 新增 `125_build_real_curve_phase_features_v371.py`，只读读取`mart.M1TrainingStepV371`与`mart.M1TrainingCurveV371`，真实解码Curve1/2/5/6/7，按本批累计Step AV边界和首次温降>=10°C确定MSB起点，输出`reports/real_curve_phase_features_v371/real_curve_phase_features_staging.csv`。
- 本次实测：3019批输入，698批成功生成`CURVE_OBSERVED`阶段功率/扭矩/转速/压栓/温度/eta特征，2321批因Step源的PhaseName为空而严格阻断`BLOCKED_UNKNOWN_PHASE_MAPPING`；未写数据库、未访问本地MMS/Archive、未使用合成功率/扭矩。
- 以MixingInfo门禁与V3 Phase PASS交集作为准入，计划批次4100，最终可训练3019批；分表保存批次223列、Step、曲线原文、物料、天气、Lab组及成员关系。
- 写入状态：MATERIALIZED_SUCCESS；未访问本地MMS/Archive。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\m1_training_dataset_v371_materialization_v1\m1_training_materialization_audit.json`。

## 2026-08-25 11:04 M1真实曲线Step-Phase映射重建
- 使用`phase_segmentation_contract_v3.py`、同批Curve1/Curve7和云端RecipeMaterials CS4证据；dry-run映射候选2974批、39417行，数据库写入=False。
- 未访问本地MMS/Archive；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\real_curve_phase_features_v371\m1_step_phase_mapping_rebuild_audit.json`。

## 2026-08-25 11:06 M1真实曲线Step-Phase映射重建
- 使用`phase_segmentation_contract_v3.py`、同批Curve1/Curve7和云端RecipeMaterials CS4证据；dry-run映射候选2974批、39417行，数据库写入=False。
- 未访问本地MMS/Archive；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\real_curve_phase_features_v371\m1_step_phase_mapping_rebuild_audit.json`。

## 2026-08-25 11:08 M1真实曲线Step-Phase映射重建
- 使用`phase_segmentation_contract_v3.py`、同批Curve1/Curve7和云端RecipeMaterials CS4证据；dry-run映射候选3019批、39976行，数据库写入=False。
- 未访问本地MMS/Archive；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\real_curve_phase_features_v371\m1_step_phase_mapping_rebuild_audit.json`。

## 2026-08-25 11:09 M1真实曲线Step-Phase映射重建
- 使用`phase_segmentation_contract_v3.py`、同批Curve1/Curve7和云端RecipeMaterials CS4证据；dry-run映射候选3019批、39976行，数据库写入=True。
- 未访问本地MMS/Archive；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\real_curve_phase_features_v371\m1_step_phase_mapping_rebuild_audit.json`。

## 2026-08-25 11:16 真实曲线Phase特征回写
- 基于同批Curve1/2/5/6/7解码和Step边界，候选3010批，目标真实曲线特征列35列；数据库写入=False。不覆盖Step/Phase时长，不使用合成功率扭矩。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\real_curve_phase_features_v371\real_curve_phase_features_apply_audit.json`。

## 2026-08-25 11:16 真实曲线Phase特征回写
- 基于同批Curve1/2/5/6/7解码和Step边界，候选3010批，目标真实曲线特征列35列；数据库写入=True。不覆盖Step/Phase时长，不使用合成功率扭矩。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\real_curve_phase_features_v371\real_curve_phase_features_apply_audit.json`。

## 2026-08-25 11:17 真实曲线Phase特征回写
- 修正：Curve7仅用于MST区间的Loading边界；MSB无压栓动作，不再把MSB尾段当作Curve7缺失。9个当前曲线边界失败批次的旧曲线派生值已清为NULL，并将对应缺失标志设为1。

## 2026-08-25 M1训练表空值与每日链路审计
- 修正`phase_segmentation_contract_v3.py`：油路线中Loading后、CS4投油Step之前必须归为`Stage2_DryMixing`，CS4 Step才是`Stage3_OilLoading`，投油后至PID为`Stage4_WetMixing`。
- 修正后3019批Step→Phase全部通过，重新生成并回写真实Curve特征；当前有油路线Stage2功率特征基本完整，无油路线Stage3/Stage4为空属于阶段不适用。
- 仍有部分Stage5功率/扭矩/效率字段按批次为空，需做信号级缺失审计；不能用0填充。
- 当前每日计划任务实际调用`run_cloud_m1_daily_refresh.ps1`，链路为`87_run_cloud_m1_watermark_refresh.py`→云端raw/core源刷新→RecipeIdentity→Phase review queue。尚未调用MixingInfo硬门禁、`126_rebuild_m1_step_phase_mapping_v371.py`、`125_build_real_curve_phase_features_v371.py`或`127_apply_real_curve_phase_features_v371.py`，因此每日增量尚未完成最终训练表晋级。

## 2026-08-25 工艺专家Step调整动作空间合同补充
- Step 1–2属于投料，永久不可推荐、不可修改Target；任何候选必须保持其Target不变。
- 压栓/ram默认不进入动作空间；仅保留SP-AV跟踪、异常审计和物理诊断，不生成ram调整建议。
- Dry mixing允许duration、rotor、temperature动作；候选必须按“达到目标温度且/或满足时间”的联合完成条件评价，不能只看单一参数。
- Wet mixing的降RPM必须与允许/要求duration增加及温度终点保持组成耦合动作；禁止把RPM下降作为独立动作。
- Carbon Black PID开放上密炼和下密炼duration，单次duration变化以5–10秒为基本动作粒度。
- Silica PID仅开放下密炼duration，10秒一档；不预设duration与门尼的单调关系，必须由历史回放/模型确认方向并检查过调风险。
- 以上规则属于推荐系统硬门禁和动作空间合同，适用于历史黄金差异分析、Step向量搜索、MANUAL_REVIEW_ONLY推荐及后续每日增量服务。
- 白炭黑PID总时长新增硬门禁：PID总时间从真实`TargetTemperature-3°C`首次温度crossing开始，持续到MSB排胶开始；由上密炼PID实际duration与下密炼实际duration组成，不计排胶开始后的额外尾部时间。
- PID TargetTemperature=165°C时，总PID duration必须在80–140 s；TargetTemperature=155°C时必须在130–230 s；TargetTemperature=142°C时必须在270–330 s。候选超过对应上限、低于对应下限或无法唯一匹配规则时直接阻断，不得进入推荐排序。
- 该规则适用于白炭黑PID duration调整、Step向量搜索、历史反事实回放和MANUAL_REVIEW_ONLY输出；拒绝码建议统一为`REJECTED_SILICA_PID_TOTAL_DURATION_LIMIT`或`BLOCKED_UNSUPPORTED_PID_TARGET_DURATION_RULE`。
- 基于同批Curve1/2/5/6/7解码和Step边界，候选3010批，目标真实曲线特征列35列；数据库写入=True。不覆盖Step/Phase时长，不使用合成功率扭矩。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\real_curve_phase_features_v371\real_curve_phase_features_apply_audit.json`。

## 2026-08-25 11:24 M1真实曲线Step-Phase映射重建
- 使用`phase_segmentation_contract_v3.py`、同批Curve1/Curve7和云端RecipeMaterials CS4证据；dry-run映射候选3019批、39976行，数据库写入=False。
- 未访问本地MMS/Archive；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\real_curve_phase_features_v371\m1_step_phase_mapping_rebuild_audit.json`。

## 2026-08-25 11:26 M1真实曲线Step-Phase映射重建
- 使用`phase_segmentation_contract_v3.py`、同批Curve1/Curve7和云端RecipeMaterials CS4证据；dry-run映射候选3019批、39976行，数据库写入=True。
- 未访问本地MMS/Archive；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\real_curve_phase_features_v371\m1_step_phase_mapping_rebuild_audit.json`。

## 2026-08-25 11:28 真实曲线Phase特征回写
- 基于同批Curve1/2/5/6/7解码和Step边界，候选3010批，目标真实曲线特征列35列；数据库写入=True。不覆盖Step/Phase时长，不使用合成功率扭矩。
- 审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\real_curve_phase_features_v371\real_curve_phase_features_apply_audit.json`。

## 2026-08-25 11:42 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054820Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断113批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260825T034222Z_audit.json`。

## 2026-08-25 11:42 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054820Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断113批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260825T034252Z_audit.json`。

## 2026-08-25 11:43 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054820Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断113批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260825T034346Z_audit.json`。

## 2026-08-25 11:44 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054820Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过107批，阻断6批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260825T034432Z_audit.json`。

## 2026-08-25 11:45 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054841Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260825T034545Z_audit.json`。

## 2026-08-25 11:48 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054820Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过107批，阻断6批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260825T034748Z_audit.json`。

## 2026-08-25 11:49 云端M1增量刷新器V1
- 范围：2026-08-25T10:49:05 至 2026-08-25T11:49:05（半开区间），M1，最多1个订单；模式：DRY_RUN_STAGING。
- 读取订单0、车次0、Step行0、曲线0、物料0、RecipeData0、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_daily_test\reports\cloud_m1_daily_refresh\20260825T034907Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-25 11:49 云端M1增量刷新器V1
- 范围：2026-08-25T10:49:27 至 2026-08-25T11:49:27（半开区间），M1，最多1个订单；模式：DRY_RUN_STAGING。
- 读取订单0、车次0、Step行0、曲线0、物料0、RecipeData0、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_daily_test2\reports\cloud_m1_daily_refresh\20260825T034929Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-25 11:49 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_daily_test2\reports\cloud_m1_daily_refresh\20260825T034929Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260825T034943Z_audit.json`。
## 2026-08-25 每日增量四道门禁链路接入
- 新增 `Mooney_Integrated_Data_Asset/pipeline/128_daily_m1_phase_gate_finalize_v371.py`：只消费 `83_cloud_m1_daily_refresh_v1.py` 当前 run 目录的精确快照，按车次执行 MixingInfo 异常门禁、粘门源车到下一车影响门禁、V3 Step→Phase、同批 Curve1/2/5/6/7 真实曲线特征、M1Training* 精确幂等 upsert；不访问本地 MMS/Archive，不用合成功率/扭矩，不把异常车次所在 Lab 组整体删除。
- 修改 `87_run_cloud_m1_watermark_refresh.py` 输出 `latest_run_directories.json`，修改 `run_cloud_m1_daily_refresh.ps1` 在 raw/core refresh 后消费每个精确 run 目录；RuntimeRoot 与项目目录均支持。
- 受控回放：113 车次输入，109 通过 MixingInfo，107 完成 Phase 与真实曲线特征，6 阻断；提交写入 113 Batch、1469 Step、113 Curve。审计：`reports/cloud_m1_daily_refresh/daily_phase_gate_finalize/DAILY_20260825T034748Z_audit.json`。
- 编排器 dry-run 已验证：无订单窗口正常完成，不触发错误；数据库写入仍由 `-Execute` 明确控制。
## 2026-08-25 数据库三重清理核查
- 对 `MooneyIntegratedData` 执行对象/行数、代码与 SQL 引用、外键/视图/模块依赖三轮检查。
- 结论：没有发现可在不破坏当前链路、治理验收、失败审计、回滚或历史复现的前提下安全删除的表；本轮未执行 DROP。
- 审计：`reports/database_cleanup_three_check_audit_20260825.md`、`reports/db_cleanup_inventory.json`。

## 2026-08-25 Silane 配比字段全空诊断
- 只读核查确认 `mart.M1TrainingBatchV371.weight_pct_silian` 3,132/3,132 全 NULL，不是实际配方无 Silane。
- `mart.M1TrainingMaterialV371` 已存在 1,304 行 `MaterialCategory='SILANE'` 的真实物料；当前物料快照 PHR 也 21,871/21,871 全空，说明装配阶段未做真实胶重归一化。
- 诊断报告：`reports/silane_feature_missing_diagnosis_20260825.md`。尚未写回，避免未经 staging 对账直接修改训练表。

## 2026-08-25 PID 派生字段全空诊断
- `mart.M1TrainingBatchV371` 的 9 个 PID 字段（目标/实际 top、bottom、total duration，start step、found rate、target temperature）均为 3,132/3,132 NULL。
- 只读确认 `M1TrainingStepV371` 已有真实 Step Target/AV，RecipeData 已有 PID Temperature 与 SCP Step Time；根因是当前物化流程未执行 PID 动态起点和 Step AV 差分聚合。
- 不得使用 MST/MSB 总时长、固定步号或 0 替代；诊断报告：`reports/pid_feature_missing_diagnosis_20260825.md`。

## 2026-08-25 12:03 Silane/PID 特征重算
- staging 批次3132，Silane有效1304，Silica PID通过2667，阻断7；写入=False。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-25 12:04 Silane/PID 特征重算
- staging 批次3132，Silane有效1304，Silica PID通过2667，阻断7；写入=False。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-25 12:13 Silane/PID 特征重算
- staging 批次3132，Silane有效1304，Silica PID通过2667，阻断7；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-25 12:17 Silane/PID 特征重算
- staging 批次3132，Silane有效1304，Silica PID通过2495，阻断179；写入=False。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-25 12:21 Silane/PID 特征重算
- staging 批次3132，Silane有效1304，Silica PID通过2495，阻断179；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-25 12:28 Silane/PID 特征重算
- staging 批次3132，Silane有效1304，Silica PID通过2495，阻断179；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-25 12:29 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段；单车组保持 NULL；写入=False，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-25 12:30 V371 组级标准差重建
- 服务器任务 `MooneyM1DataAssetDaily` 已触发但 `Last Result=1`；同一 release 只读复现确认 `PythonPath=python.exe` 在 S4U 非交互环境不可解析，报 `PROJECT_PYTHON_NOT_FOUND:python.exe`，昨日没有成功写入。
- 已把修正版 runner、128/129/130 脚本同步到服务器 release，并保留旧 runner 备份；重注册前必须改为服务器实际 Python 绝对路径。
- 每日链路已同步：`run_cloud_m1_daily_refresh.ps1` 在增量 finalize 后调用 `129_rebuild_silane_pid_features_v371.py --keys-file`，并调用 `130_rebuild_group_std_features_v371.py`；不访问本地 MMS/Archive。
- 本机可见最近调度日志为 DRY-RUN、0 新订单、0 数据库写入；本机未找到 `MooneyM1DataAssetDaily` 任务，服务器昨日是否成功需在服务器本机查看任务历史和 scheduler log。
- LabelGroupID 回聚合 40 个 __group_std 字段；单车组保持 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-25 13:25 Silane/PID 特征重算
- staging 批次3132，Silane有效1304，Silica PID通过2628，阻断46；写入=False。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-25 13:31 Silane/PID 特征重算
- staging 批次3132，Silane有效1304，Silica PID通过2628，阻断46；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-25 13:31 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段；单车组保持 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-25 13:33 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-25 13:34 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-25 每日增量PipelineRun 138诊断
- 服务器release已同步最新129动态PID、131组级PID同步和daily runner；文件时间分别约13:22、13:34、13:35。
- 过去24小时任务日志`refresh_20260825_131259.log`最后停在`lab domains complete: 5 groups`，未进入Batch/BatchStep/Material、Weather、DependencyState及commit；`etl.PipelineRun`中未发现138，说明该事务未提交/已回滚，不能视为成功。
- 该次运行开始于最新脚本同步前，不能用来验证最新129/131；当前已无证据表明仍在运行。数据库另有历史PipelineRun 137处于RUNNING，需在服务器检查是否为遗留任务并确认无锁竞争。

## 2026-08-25 13:35 M1TrainingBatchV371全表NULL审计
- 只读审计3132批、234列：身份主键字段无NULL；8个曲线质量/组风险字段全表NULL（curve_batch_count、curve_statistical_outlier_batch_count、curve_if_score_mean/min、curve_statistical_outlier_rate、curve_valid_batch_rate、curve_quality_risk_flag、Reasons），不能作为训练特征使用。
- material_system为空113批，且phase_route及阶段物理字段同步为空，标记为未知路线阻断；CarbonBlack 345批PID字段为空符合不适用语义；Silica 2674批中PID通过2628、阻断46。
- 发现旧流程遗留pid_data_valid_flag错误：CarbonBlack数值为空但曾标1，已修正为路由/实际值联合有效标志；Silica有效2628、阻断46，CarbonBlack和未知路线均为0。
- Silica的weight_pct_silian有1304批真实值，1370批无值；其中1356批在core.BatchMaterial及M1TrainingMaterial均无物料行，不能填0，需标记SOURCE_DATA_MISSING并阻断依赖该特征的训练切片。

## 2026-08-25 M1黄金曲线/连续升高趋势可行性体检
- 只读使用 `reports/t15760_golden_curve_research_20260823/t15760_candidate_eligible_after_judgereason_filter.csv`：同一 RecipeID=9922A1576030、ProcessVersion=PV27、MixerLine=ML09，共314个Lab组、258个有标签订单。
- 按订单内Lab组MNY均值/中位数聚合，并按OrderID排序后，连续三个有标签订单每一步至少上升2 MU的序列为19个，至少上升3 MU的序列为10个；这只是数值触发候选，不代表工艺因果已成立。
- 代表序列：7940154→7947677→7951751，中位MNY 58.745→63.990→68.440（+5.245、+4.450）；7955506→7959223→7959284，55.540→58.440→63.260（+2.900、+4.820）。
- 触发实现必须使用LabelGroup级有效Lab结果、结果可用时间、001/011状态、曲线/粘门/下一车门禁及精确模板；不可把一个Lab结果复制成多个车次标签，也不能仅凭OrderID连续性宣称物理连续。
- 结论：过程偏离预警链路可先实现为MANUAL_REVIEW_ONLY诊断；Target→actual Phase B3当前仍未通过时，禁止把黄金曲线差异转换为Target推荐。推荐链需等待B3控制传递、动作支持和混杂门禁通过。
## 2026-08-25 每日云端M1增量链路索引与自动恢复修复
- 修复 `87_run_cloud_m1_watermark_refresh.py`：运行摘要和 `latest_run_directories.json` 改为原子写入，避免 wrapper 读取半成品或遗漏索引。
- 修复 `run_cloud_m1_daily_refresh.ps1`：若旧版 87 已生成 `pages[].run_directory` 但缺少索引，wrapper 从最近的 watermark summary 自动恢复索引；恢复只写 RuntimeRoot JSON，不修改数据库。
- 增加空索引、无运行目录和不存在运行目录的硬阻断，防止误消费旧运行目录。
- 修复每日 Recipe 身份门禁：CompoundDescription fallback 只能用于诊断，不得通过 RecipeData/Phase 映射；统一记录 `BLOCKED_RECIPE_ID_FALLBACK`。
- 运行语义保持：raw/core 入库可成功，Lab/天气延迟自动保留待补状态；未通过精确 Recipe、Phase 治理或数据完整性门禁时不提升 TrainingReady。
- 未访问本地 MMS/Archive，未修改生产 Recipe/PLC。
- 增加有界延迟依赖重试：每日任务从 `etl.DependencyState` 仅读取最多 100 个 `PENDING_LAB_GROUP/PENDING_WEATHER/PENDING_QM` 或 Recipe 合同阻断订单，再按精确 OrderID 从云端重取；不做全表 Orders 扫描，不把未知 Phase 映射自动猜测为通过。
- 发现并修复每日后处理缺口：`128_daily_m1_phase_gate_finalize_v371.py` 现在按精确车次增量 upsert `mart.M1TrainingMaterialV371`，并同步 upsert `mart.V371TrainingGroupMemberSnapshot`；新车次不再只存在于 `core.BatchMaterial` 而缺失于训练物料/组成员粒度。写入结果加入 `MaterialUpsert` 与 `GroupMemberUpsert` 审计字段。
- 继续补齐每日 V371 子表：按当前 run 的精确 BatchKey/LabelGroupID 从 `core.BatchWeather`、`core.LabGroup`、`core.LabelGroupMembership` 读取并增量 upsert 到 `mart.M1TrainingWeatherV371`、`mart.M1TrainingLabGroupV371`、`mart.M1TrainingLabMembershipV371`；缺失 Lab/天气保持零行/阻断，不填伪值。新增 `WeatherUpsert`、`LabGroupUpsert`、`LabMembershipUpsert` 审计。
- 针对单日数据量和服务器内存风险，将每日 wrapper 默认改为 `LookbackHours=24`、`MaxOrders=5`、`MaxPages=100`；每页独立提取/事务/快照/finalizer，页数增加不扩大单页内存峰值；延迟依赖重试上限改为10个订单。
- 增加每日启动资源门禁：启动云端取数前读取 Windows 总内存/可用内存、CPU 使用率及其他 Python/PowerShell 进程工作集；默认可用内存至少4GB、CPU不超过85%、其他脚本合计不超过8GB、单个脚本不超过4GB。资源不足每30秒重试，最长60分钟，超时安全退出且不启动取数；资源快照写入 scheduler log。
- 修复每日 V371 终结器的关键回写缺口：`128_daily_m1_phase_gate_finalize_v371.py` 现在在同一事务内按精确车次从 `core.BatchMaterial`、`core.BatchMaterialQM`、`core.BatchMaterialQMFeature` 聚合并投影物料比例、PHR、QM 状态和供应商特征到 `mart.M1TrainingBatchV371`；源数据缺失仍保留 NULL，禁止默认值填充。
- 修复真实曲线阶段字段映射：`*_CurveDuration_s` 正确映射到 `*_Duration`，不再因字符串替换错误导致所有阶段时长无法回写；同步修正 `127_apply_real_curve_phase_features_v371.py`。
- 诊断确认 `DAILY_20260825T034748Z` 的113行全部缺失物料/QM/PID，是旧终结器只写身份字段且826增量尚未经过 V371 finalizer 的结果；`etl.PipelineRun` 1134-1137 均为 raw/core SUCCESS 但 `training_ready_promotions=0`，未将其误判为训练可用。

## 2026-08-26 09:59 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054820Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过107批，阻断6批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T015911Z_audit.json`。

## 2026-08-26 10:00 Silane/PID 特征重算
- staging 批次113，Silane有效0，Silica PID通过0，阻断0；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-26 10:03 延迟Lab组每日同步
- 从 `core.LabGroup` 增量幂等投影到 `mart.M1TrainingLabGroupV371`；本次候选6466组，写入=True。不访问本地MMS/Archive，不填充Lab值。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\lab_sync_20260826T020330Z.json`。

## 2026-08-26 10:06 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054820Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过107批，阻断6批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T020544Z_audit.json`。

## 2026-08-26 10:08 Silane/PID 特征重算
- staging 批次113，Silane有效0，Silica PID通过0，阻断0；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。
- 2026-08-26 Phase合同复核：确认昨日已建立完整的RecipeData/SCP-LOG、RecipeID+ProcessVersion+TargetVector+StepPattern、批准PhaseMappingVariant、曲线边界及PID规则合同；但当前日增量128终结器仅调用`phase_segmentation_contract_v3.py`的边界分段函数，未消费83已生成的`recipe_scp_log_contract.parquet`，也未执行批准Variant精确匹配。因此当前路径属于“部分复用合同”，不能宣称与昨日完整Phase mapper等价。特别是CarbonBlack批次若在MST PID阈值后被标为Stage5_PID，与既定“炭黑无PID、下密炼统一BottomMixing”规则冲突，须在继续全量写入前修复并回放。
- 2026-08-26 训练主表粒度整改：确认`mart.M1TrainingBatchV371`只能保存存在真实Lab结果的`LabelGroupID`对应车次；无Lab车次只保留在core/过程子表，不得因订单被抓取而晋级训练主表。128终结器新增`derive_lab_labelled_batch_keys()`，以`core.LabGroup`有效数值结果+`IsEligibleForTraining=1`与`core.LabelGroupMembership`交集作为精确写入键；无标签车次不再写入。当前历史误写的7994929共113车次无LabGroup，待完成备份后清理。测试23项通过。

## 2026-08-26 10:22 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054820Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过107批，阻断6批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T022200Z_audit.json`。

## 2026-08-26 10:23 M1训练主表标签范围审计
- `mart.M1TrainingBatchV371`总行3132，有效Lab车次4810，待清理无Lab行1218；Before-image `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\label_scope_reconcile\before_image_20260826T022317Z.parquet` SHA256=f09ecdc9b2c4dbab7c1327913d26cbf28962bcb9fdfde2805b303ca12c6c2fc9；写入=False。不访问本地MMS/Archive。

## 2026-08-26 10:24 M1训练主表标签范围审计
- `mart.M1TrainingBatchV371`总行3132，有效Lab车次4810，待清理无Lab行1218；Before-image `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\label_scope_reconcile\before_image_20260826T022409Z.parquet` SHA256=f09ecdc9b2c4dbab7c1327913d26cbf28962bcb9fdfde2805b303ca12c6c2fc9；写入=True。不访问本地MMS/Archive。
- 2026-08-26 10:25 标签范围修正完成：`mart.M1TrainingBatchV371`无Lab行清理已提交，删除1218行（348订单），Before-image `reports/cloud_m1_daily_refresh/label_scope_reconcile/before_image_20260826T022409Z.parquet`，SHA256=`f09ecdc9b2c4dbab7c1327913d26cbf28962bcb9fdfde2805b303ca12c6c2fc9`；数据库复核总行1914、有效Lab车次1914、无效Lab车次0，订单7994929行数0。128终结器及134标签范围脚本、daily wrapper已同步服务器release `M1DataAssetDaily_20260824_v2`，三文件本地/服务器SHA256一致。不动core及Step/Curve/Material子表。

## 2026-08-26 10:37 延迟Lab车次晋级
- pending候选0，有效Lab过滤后待晋级0，写入=False；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260826T023754Z.json`。

## 2026-08-26 10:38 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054820Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过107批，阻断6批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T023812Z_audit.json`。

## 2026-08-26 10:38 延迟Lab车次晋级
- pending候选113，有效Lab过滤后待晋级0，写入=False；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260826T023842Z.json`。
- 2026-08-26 延迟Lab链路完成：128在无Lab时不写`M1TrainingBatchV371`，同时保存完整过程候选到`daily_phase_gate_finalize/pending_labelled_batches/*.parquet`；新增135仅当`core.LabGroup`有效结果+`LabelGroupMembership`精确命中且MixingInfo/Phase/Curve通过时晋级主表，并输出精确keys供PID/组特征后处理。7994929回放产生pending 113、Lab晋级候选0，主表仍0行。daily wrapper顺序为LabSync→135 delayed promotion→134 label scope reconciliation→PID/组后处理。25项回归测试通过，脚本已同步服务器release，SHA256一致。

## 2026-08-26 10:41 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260824T054820Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过107批，阻断6批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T024053Z_audit.json`。
- 2026-08-26 SCP-LOG结构化落库：128现在把RecipeData解析出的`EffectiveScpLogLogic/ContractStatus/SourceAuthority/OR_LastBinaryBit/AND_LastBinaryBit`逐Step写入`mart.M1TrainingStepV371`，不再只拼接到MappingBasis。7994929精确回放后Step总数1469，其中1391条通过`MAPPED_V371_RECIPEDATA_CONTRACT`并具备结构化SCP字段；主训练表仍为0行（无Lab）。
- 2026-08-26 标签范围最终复核：`134_reconcile_m1_training_batch_label_scope_v371.py` 只读核验 `mart.M1TrainingBatchV371` 为1914行，ValidLabBatchKeys为4810，RowsToRemove=0；`135_promote_delayed_lab_batches_v371.py` 只读核验pending为113车次、PromotionCandidateCount=0。服务器release与本地128/134/135及daily wrapper逐文件SHA256一致，25项合同测试通过；无本地MMS/Archive访问。
- 2026-08-26 指定订单Lab缺失审计：云端CE精确查询确认8001854/8001894/8005185/8009114/8009116/8009357/8012700共有真实MS1+3检测记录，但目标库当前仅有8012700的raw/core Lab记录；其余7单因早于24h增量窗口未被回补。8019747仅有一条空占位记录（test_variable、pallet_id、sample_id、test_result均为空），不能作为训练标签。尝试8单精确回放时目标库血缘修复阶段发生SQL死锁，事务未提交；随后终止本次两个回放进程释放锁，目标库无新增半写入。未访问本地MMS/Archive。
- 2026-08-26 指定订单补回：对真实CE检测订单执行精确订单键回放，未扫描全库。8001854进入4车、8001894进入10车、8005185进入14车、8009116进入17车、8009357进入4车、8012700进入27车；8009114有实验室记录但资格状态为0，因此0车进入训练主表；8019747为空占位不进入。目标表只保留有效Lab绑定车次，过程Step/Curve/Material仍按过程车次保存。为降低补回耗时，83脚本的raw BatchData/Material批量写入、BatchStep集合式upsert及COLLATE修复已通过真实回放验证；本地与服务器release SHA256均为F59968C39211CB76351B090A222D253412E3CE76E3CA1F7AA38949B57E2BACFA，服务器旧版本已备份。最终`mart.M1TrainingBatchV371`为1990行，134标签范围复核RowsToRemove=0，135延迟Lab晋级候选0。未访问本地MMS/Archive。

## 2026-08-26 10:44 延迟Lab车次晋级
- pending候选113，有效Lab过滤后待晋级0，写入=False；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260826T024445Z.json`。

## 2026-08-26 10:44 M1训练主表标签范围审计
- `mart.M1TrainingBatchV371`总行1914，有效Lab车次4810，待清理无Lab行0；Before-image `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\label_scope_reconcile\before_image_20260826T024446Z.parquet` SHA256=c461bba43526b44355c6fe02b25727188090fae290efbf6c8e5c61976f4f10a8；写入=False。不访问本地MMS/Archive。

## 2026-08-26 11:26 云端M1增量刷新器V1
- 范围：2026-08-19T11:18:09 至 2026-08-26T11:18:09（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次73、Step行4654、曲线73、物料1101、RecipeData447、CE Lab2、QM6；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T031809Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-26 11:27 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T031809Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过66批，阻断7批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T032704Z_audit.json`。

## 2026-08-26 11:56 云端M1增量刷新器V1
- 范围：2026-08-19T11:55:20 至 2026-08-26T11:55:20（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次148、Step行9418、曲线148、物料2229、RecipeData447、CE Lab5、QM6；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T035520Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-26 11:58 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T035520Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过130批，阻断18批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T035652Z_audit.json`。

## 2026-08-26 11:58 云端M1增量刷新器V1
- 范围：2026-08-19T11:58:18 至 2026-08-26T11:58:18（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次73、Step行4651、曲线73、物料1097、RecipeData447、CE Lab5、QM12；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T035818Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-26 11:59 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T035818Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过60批，阻断13批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T035903Z_audit.json`。

## 2026-08-26 11:59 云端M1增量刷新器V1
- 范围：2026-08-19T11:58:18 至 2026-08-26T11:58:18（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次73、Step行4643、曲线73、物料1097、RecipeData447、CE Lab2、QM6；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T035926Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-26 12:00 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T035926Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过65批，阻断8批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T040001Z_audit.json`。

## 2026-08-26 12:01 云端M1增量刷新器V1
- 范围：2026-08-19T11:58:18 至 2026-08-26T11:58:18（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次124、Step行7893、曲线124、物料1870、RecipeData447、CE Lab6、QM6；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T040019Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-26 12:01 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T040019Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过107批，阻断17批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T040117Z_audit.json`。

## 2026-08-26 12:04 云端M1增量刷新器V1
- 范围：2026-08-19T11:58:18 至 2026-08-26T11:58:18（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次22、Step行1400、曲线22、物料331、RecipeData447、CE Lab2、QM6；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T040151Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-26 12:04 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T040151Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过18批，阻断4批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T040425Z_audit.json`。

## 2026-08-26 12:05 云端M1增量刷新器V1
- 范围：2026-08-19T11:58:18 至 2026-08-26T11:58:18（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次146、Step行9299、曲线146、物料2198、RecipeData447、CE Lab8、QM6；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T040433Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-26 12:07 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T040433Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过130批，阻断16批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T040549Z_audit.json`。

## 2026-08-26 12:07 延迟Lab车次晋级
- pending候选772，有效Lab过滤后待晋级0，写入=False；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260826T040726Z.json`。

## 2026-08-26 12:07 M1训练主表标签范围审计
- `mart.M1TrainingBatchV371`总行1990，有效Lab车次4859，待清理无Lab行0；Before-image `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\label_scope_reconcile\before_image_20260826T040728Z.parquet` SHA256=c461bba43526b44355c6fe02b25727188090fae290efbf6c8e5c61976f4f10a8；写入=False。不访问本地MMS/Archive。

## 2026-08-26 13:47 Silane/PID 特征重算
- staging 批次1990，Silane有效874，Silica PID通过1683，阻断40；写入=False。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-26 13:48 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T031809Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过66批，阻断7批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T054809Z_audit.json`。

## 2026-08-26 13:50 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T035520Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过130批，阻断18批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T054923Z_audit.json`。

## 2026-08-26 13:50 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T035818Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过60批，阻断13批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T055020Z_audit.json`。

## 2026-08-26 13:51 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T035926Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过65批，阻断8批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T055100Z_audit.json`。

## 2026-08-26 13:52 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T040019Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过107批，阻断17批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T055135Z_audit.json`。

## 2026-08-26 13:52 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T040151Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过18批，阻断4批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T055221Z_audit.json`。

## 2026-08-26 13:53 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T040433Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过130批，阻断16批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T055245Z_audit.json`。

## 2026-08-26 13:56 Silane/PID 特征重算
- staging 批次1977，Silane有效861，Silica PID通过1674，阻断36；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-26 14:01 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=False，物理无效=0，统计异常=76，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 14:02 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=0，统计异常=76，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 14:11 Silane/PID 特征重算
- staging 批次1977，Silane有效861，Silica PID通过1674，阻断36；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-26 14:11 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段；单车组保持 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-26 14:11 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-26 14:19 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=False，通过=0，阻断=1977，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 14:22 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=False，通过=0，阻断=1977，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 14:28 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=False，通过=1953，阻断=24，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 14:33 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=1953，阻断=24，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 14:37 Silane/PID 特征重算
- staging 批次1977，Silane有效861，Silica PID通过1674，阻断36；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-26 14:37 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段；单车组保持 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-26 14:37 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-26 14:37 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=0，统计异常=77，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 14:34 V371真实曲线阶段特征最终回填与每日链路同步
- 修复`137_rebuild_real_curve_phase_features_v371.py`的阶段区间遍历错误：由误遍历DataFrame列名改为逐阶段记录，避免将所有批次错误阻断；Python编译通过。
- 以真实`M1TrainingStepV371`累计AV/SP、`M1TrainingCurveV371` Curve1/2/5/6/7和`core.Batch`真实胶重重建并写回97个曲线派生字段：1,953批通过、24批因`BLOCKED_PID_TOP_OR_BOTTOM_PHASE_NOT_OBSERVED`保持阻断，未填充默认值；数据库写入1,953行。
- 顺序重算129 PID、130组统计、131组PID、136曲线质量；结果分别为PID 1,674/1,710通过、组统计645组、质量重建1,977批。当前`mart.M1TrainingBatchV371`为1,977行、键无重复、状态均为`ELIGIBLE_M1_TRAINING`。
- 已同步至服务器发布目录`\\10.246.97.159\d\MNY master batch prediction model pipline\releases\M1DataAssetDaily_20260824_v2`，128/129/136/137及每日wrapper本地与服务器SHA256逐项一致；本轮不访问本地MMS/Archive。

## 2026-08-26 14:58 V371体系差异与曲线信号硬门禁修正
- 炭黑按最新工艺合同不适用PID：清除`Stage5_PID_*`及`pid_*`派生残留，保留`Stage6_BottomMixing`；白炭黑保留动态PID。
- 质量脚本按物料体系分别要求：炭黑Dry+Bottom，白炭黑Dry+PID+Bottom；避免把炭黑结构性无PID误判为物理异常。
- `mart.M1TrainingBatchV371`最终核验：1,977行、TrainingBatchKey无重复；1,953行`ELIGIBLE_M1_TRAINING`且必需阶段温度/功率空值为0；24行明确`BLOCKED_REAL_CURVE_FEATURES`，均为真实PID边界未观测。可训练集合不含这24行；曲线质量、PID、组统计已重算。
- 136/137/每日wrapper已同步服务器并逐项SHA256一致；不访问本地MMS/Archive。

## 2026-08-26 14:47 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=False，通过=1953，阻断=24，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 14:52 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=1953，阻断=24，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 14:55 Silane/PID 特征重算
- staging 批次1977，Silane有效861，Silica PID通过1674，阻断36；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-26 14:55 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段；单车组保持 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-26 14:55 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-26 14:55 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=267，统计异常=60，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 14:57 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=0，统计异常=83，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 15:28 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T152829_audit.json`。
- 实际批次数=812，待修复=812，QM源返回=133。

## 2026-08-26 15:35 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T153518_audit.json`。
- 实际批次数=1977，待修复=1977，QM源返回=18。

## 2026-08-26 15:39 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T153912_audit.json`。
- 实际批次数=1977，待修复=1977，QM源返回=18。

## 2026-08-26 15:45 Silane/PID 特征重算
- staging 批次1977，Silane有效861，Silica PID通过1820，阻断36；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-26 15:58 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T155854_audit.json`。
- 实际批次数=1977，待修复=1977，QM源返回=424。

## 2026-08-26 15:59 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T155955_audit.json`。
- 实际批次数=1977，待修复=1977，QM源返回=0。

## 2026-08-26 16:03 云端M1增量刷新器V1
- 范围：2026-03-01T00:00:00 至 2026-03-02T00:00:00（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次45、Step行2299、曲线45、物料316、RecipeData393、CE Lab1、QM45；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T080206Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-26 16:05 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T160553_audit.json`。
- 实际批次数=1977，待修复=1977，QM源返回=45。

## 2026-08-26 16:06 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T080206Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过39批，阻断6批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T080612Z_audit.json`。

## 2026-08-26 16:12 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T161226_audit.json`。
- 实际批次数=1974，待修复=1974，QM源返回=0。

## 2026-08-26 16:13 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T080206Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过39批，阻断6批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T081251Z_audit.json`。

## 2026-08-26 16:13 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T161337_audit.json`。
- 实际批次数=1974，待修复=1974，QM源返回=0。

## 2026-08-26 16:23 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T162340_audit.json`。
- 实际批次数=1978，待修复=1978，QM源返回=45。

## 2026-08-26 16:24 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T080206Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过39批，阻断6批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T082359Z_audit.json`。

## 2026-08-26 16:25 Silane/PID 特征重算
- staging 批次4，Silane有效0，Silica PID通过0，阻断0；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-26 16:25 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=4，阻断=0，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 16:26 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段；单车组保持 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-26 16:26 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-26 16:26 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=146，统计异常=84，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 16:30 延迟链路缺失根因与修复
- 对订单`2397476`逐车核查确认：云端真实返回45个BatchHeader、2,299行BatchData、45条BatchCurve、315行BatchMaterials、1条Compound Excellence Lab、45条QM及45条天气；原先V371缺少的是后续投影链路，不是源数据不存在。
- 根因1：物料代码`CC0326`（名称为`N 326`）、`CE2501`、`CR41...`被旧分类器写成`UNKNOWN`，导致真实QM虽按ChargeID进入`raw.SupplierQMHistory`，却无法按物料类别聚合为橡胶/炭黑特征。修复`83_cloud_m1_daily_refresh_v1.py`与`138_refresh_pending_qm_weather_v371.py`，以受治理代码前缀优先分类：`CC→CARBON_BLACK`、`CE(非CE19)→POLYMER`、`CR41→POLYMER`、`CN→NATURAL_RUBBER`、`CS10/CS12→SILICA`、`CS4→OIL`、`CA551→SILANE`；不生成任何默认值。
- 根因2：Phase门禁删除的有Lab标签车次在下一轮延迟修复中不可见。修复`138`全量修复范围：除现有V371行外，加入精确`LabelGroupMembership + LabGroup.LabCurrentResult`关联且缺少V371行的车次；保留按目标库限定，不访问云端全表。
- 根因3：全量QM车次参数一次性可能超过SQL Server 2100参数限制。修复`138`按每500个BatchKey分块查询，并增加`--order-id`精确修复参数；避免大窗口失败及无界远程查询。
- 针对`2397476`使用`--full-reconcile --order-id 2397476`：QM重建4个车次、天气45个车次均为`COMPLETE`；再执行Phase/真实曲线/PID/组级统计/曲线质量链路。最终该Lab组对应的4个车次全部为`ELIGIBLE_M1_TRAINING`，物料QM字段、天气字段、Stage字段、真实曲线特征均有值；同订单其余41车无Lab标签，未进入训练表。证据：`reports/cloud_m1_daily_refresh/daily_phase_gate_finalize/DAILY_20260826T082359Z_audit.csv`及后续V371审计。
- 例外语义：CarbonBlack的PID字段是`NOT_APPLICABLE`，因此`pid_actual_*`为NULL不属于缺链；Silica才要求真实PID字段通过。所有阻断仍Fail-Closed。

## 2026-08-26 16:33 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=False，通过=2，阻断=0，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 16:40 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=1942，阻断=36，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 16:56 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=1942，阻断=36，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 17:02 曲线派生字段断链修复与全量复核
- 根因确认：`137_rebuild_real_curve_phase_features_v371.py` 原先使用 `COALESCE` 写回；真实曲线不存在某阶段时，旧的Step派生值被保留，形成“看似有值、实际血缘错误”。
- 修复：新增 `curve_feature_columns()`，对137拥有的Stage/phys/cb/PID派生字段使用直接赋值；成功重算的非适用阶段写NULL，阻断批次清空旧曲线派生字段后保持 `BLOCKED_REAL_CURVE_FEATURES`。不填默认值、不改变原始Step/Curve表。
- 重新处理全库当前1978行：真实曲线通过1942、PID合同阻断36；CarbonBlack的Stage5/PID字段清零为不适用；全量复核显示：可训练行无“正时长Stage4但温度为空”，无CarbonBlack残留PID值，无阻断行残留旧曲线特征。Silica中393行、CarbonBlack中50行的Stage4为空均对应真实`Stage4_Duration=0`/非适用路径，不是漏导入。
- 服务器发布目录 `\\10.246.97.159\d\MNY master batch prediction model pipline\releases\M1DataAssetDaily_20260824_v2\Mooney_Integrated_Data_Asset` 已同步83/128/129/137/138及每日Wrapper，6/6 SHA256一致。每日顺序保持：有界云端raw/core→138 QM/天气回补→128 MixingInfo/Phase/Label门禁→129 PID→137真实曲线直接覆盖→组级与质量字段；无本地MMS/Archive、无自动模型晋级。

## 2026-08-26 17:08 每日物料链路硬门禁加强
- `83_cloud_m1_daily_refresh_v1.py` 的DependencyState新增真实物料存在性判断；无`BatchMaterials`时写入`PENDING_MATERIAL`，禁止仅凭旧宽表值晋级。
- `87_run_cloud_m1_watermark_refresh.py` 将目标库中既有V371行的物料/QM/天气缺口加入有界TOP-N精确订单重试；仍不扫描云端全表。
- `128_daily_m1_phase_gate_finalize_v371.py` 的最终静态源门禁增加精确`core.BatchMaterial`、认证`core.BatchMaterialQM`及COMPLETE/PARTIAL天气存在性；从真实物料计算`is_oil_loading_present`，不再保留旧值。`135`延迟Lab晋级也复用同一源域门禁。
- 语法与重试SQL测试通过；重试查询当前返回`2296902,2298539,2298569,2298755,2298768`五个有界订单。服务器8个相关脚本已同步，SHA256全部一致。尚未执行历史缺口的云端补数，待按小批量重试，补不到的保持阻断，不填假值。

## 2026-08-26 17:25 历史物料链路首轮补数与精确范围修复
- 首轮从云端按5个明确订单ID读取并写入真实raw/core：343个BatchHeader、20,314行BatchData、343条BatchCurve、5,869条BatchMaterials、1,129条RecipeData、17条CE Lab、350条QM；未访问本地MMS/Archive。
- 修复`138.all_mart_batches()`中`OR`与订单过滤的优先级漏洞；`--order-id`现已实际限定为单订单。新增显式订单修复时强制重新读取该订单QM，防止旧`ELIGIBLE`状态遮蔽新到源数据。
- 订单`2296902`精确回补4车，随后四个订单按精确范围完成QM/天气/物料投影；当前可训练行共975行，`ELIGIBLE_SOURCE_CHAIN_GAPS=0`。967行旧记录因源域未齐被`BLOCKED_SOURCE_DOMAIN_PENDING`，36行因真实Silica PID合同失败被`BLOCKED_REAL_CURVE_FEATURES`，均不可训练。
- 每日流程已加入物料缺口检测与TOP-N精确重试；`128`/`135`晋级前必须存在精确BatchMaterial、认证BatchMaterialQM和COMPLETE/PARTIAL BatchWeather。相关服务器脚本已重新校验，`138`修复版SHA256与服务器一致。

## 2026-08-26 17:10 云端M1增量刷新器V1
- 范围：2025-01-01T00:00:00 至 2027-01-01T00:00:00（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次343、Step行20314、曲线343、物料5869、RecipeData1129、CE Lab17、QM350；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T090911Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-26 17:11 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T171121_audit.json`。
- 实际批次数=1978，待修复=1978，QM源返回=0。

## 2026-08-26 17:17 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T171725_audit.json`。
- 实际批次数=4，待修复=4，QM源返回=77。

## 2026-08-26 17:18 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T171854_audit.json`。
- 实际批次数=10，待修复=10，QM源返回=78。

## 2026-08-26 17:19 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T171906_audit.json`。
- 实际批次数=3，待修复=3，QM源返回=58。

## 2026-08-26 17:19 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T171922_audit.json`。
- 实际批次数=4，待修复=4，QM源返回=77。

## 2026-08-26 17:19 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T171937_audit.json`。
- 实际批次数=4，待修复=4，QM源返回=78。

## 2026-08-26 17:23 云端M1增量刷新器V1
- 范围：2025-01-01T00:00:00 至 2027-01-01T00:00:00（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次4、Step行244、曲线4、物料68、RecipeData444、CE Lab2、QM102；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T092258Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-26 17:23 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T172352_audit.json`。
- 实际批次数=4，待修复=4，QM源返回=77。

## 2026-08-26 17:24 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T092258Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断4批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T092408Z_audit.json`。

## 2026-08-26 17:28 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T092258Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过4批，阻断0批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T092807Z_audit.json`。

## 2026-08-26 17:29 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T092258Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过4批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T092903Z_audit.json`。

## 2026-08-26 17:48 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T174831_audit.json`。
- 实际批次数=781，待修复=755，QM源返回=0。

## 2026-08-26 17:49 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T092258Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过4批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T094906Z_audit.json`。

## 2026-08-26 17:49 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260826T174957_audit.json`。
- 实际批次数=4，待修复=4，QM源返回=0。

## 2026-08-26 17:54 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T092258Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过4批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260826T095356Z_audit.json`。

## 2026-08-26 17:54 Silane/PID 特征重算
- staging 批次4，Silane有效4，Silica PID通过3，阻断1；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-26 17:54 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=3，阻断=1，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 17:56 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段；单车组保持 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-26 17:56 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-26 17:56 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=37，统计异常=81，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-26 18:00 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-26 18:02 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查4车，完整3车，从训练表隔离0车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260826T100252Z.json`。

## 2026-08-26 18:03 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查4车，完整3车，从训练表隔离1车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260826T100303Z.json`。

## 2026-08-26 18:03 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1981车，完整31车，从训练表隔离0车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260826T100307Z.json`。
## 2026-08-27 09:40 每日 M1 链路停滞根因、全配方证据与服务器修复
- 审计服务器 `MooneyM1DataAssetDaily` 2026-08-27 02:15 日志：原始层 PipelineRun 1197/1198/1199 均 SUCCESS；主分页读取 31 个 M1 订单、513 批，依赖重试 10 个订单/795 批。训练晋级为 0，日志停止在 `DAILY_FOUR_GATE_FINALIZER_PENDING`，说明失败/停滞发生在原始写入之后、Phase/QM/Lab/训练表终态之前。
- 数据库与冻结 Parquet 证明本次并非只读取 T15760：主分页含 21 种 M1 CompoundName；含 T15760、T09170、B00460B、B00163、R00218、T33025 等。源码入口仍为 `CompoundName LIKE 'M1-%'`，没有 T15760 硬过滤。
- 根因：服务器 release 的 128/130/138/wrapper 落后于本地修正版，且缺少 139 完整性门禁；旧 138 在 raw/core 完成后再次发起远端 QM 查询，任务未进入后续 finalizer。今日源页同时返回 CE Lab=0、QM=0，因此即使 raw/core 已写入，也不能把无 Lab 标签批次伪晋级到 `mart.M1TrainingBatchV371`。
- 修复 87 watermark：完成旧 backlog 后从上一窗口 `end_time` 连续追赶至本次 requested end，消除旧窗口完成后直接跳到当前 24h 而漏掉中间时间段的风险；仍按页限制内存。
- 修复 139 exact-key 空范围语义：每日窗口尚无 eligible Lab member 时返回 `NO_ELIGIBLE_TRAINING_ROWS_IN_SCOPE` 成功 no-op，不再误报管线失败；全表空表仍 fail-closed。
- wrapper 已接入 139 exact-key 终态完整性硬门禁，并使用 138 `--cached-qm-only`，避免对同一日已缓存的真实 QM 再做重复远程重查。仅完整、真实、Lab-member 行可保留在训练表；raw/core/Step/Curve 证据不删除。
- 已部署并 SHA256 对齐服务器 release：87、128、130、138、139、`run_cloud_m1_daily_refresh.ps1`。部署前文件保存在服务器 `deployment_backups/before_daily_fix_20260827_093538`。Python compile、PowerShell parse、watermark gap 单元反例均通过。严格不访问本地 MMS/Archive。

## 2026-08-27 09:59 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近24小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260827T095949_audit.json`。
- 实际批次数=1，待修复=1，QM源返回=0。

## 2026-08-27 10:06 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260827T100606_audit.json`。
- 实际批次数=3026，待修复=2997，QM源返回=0。

## 2026-08-27 10:09 每日增量四道门禁终结器
- 精确消费 `\\10.246.97.159\d\MNY master batch prediction model pipline\data_asset_runtime\reports\cloud_m1_daily_refresh\20260827T014158Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过399批，阻断46批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260827T020709Z_audit.json`。

## 2026-08-27 10:12 每日增量四道门禁终结器
- 精确消费 `\\10.246.97.159\d\MNY master batch prediction model pipline\data_asset_runtime\reports\cloud_m1_daily_refresh\20260827T014237Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过513批，阻断24批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260827T021013Z_audit.json`。

## 2026-08-27 10:13 每日增量四道门禁终结器
- 精确消费 `\\10.246.97.159\d\MNY master batch prediction model pipline\data_asset_runtime\reports\cloud_m1_daily_refresh\20260827T014320Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过172批，阻断12批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260827T021242Z_audit.json`。

## 2026-08-27 10:15 每日增量四道门禁终结器
- 精确消费 `\\10.246.97.159\d\MNY master batch prediction model pipline\data_asset_runtime\reports\cloud_m1_daily_refresh\20260827T014343Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过402批，阻断49批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260827T021353Z_audit.json`。

## 2026-08-27 10:16 每日增量四道门禁终结器
- 精确消费 `\\10.246.97.159\d\MNY master batch prediction model pipline\data_asset_runtime\reports\cloud_m1_daily_refresh\20260827T014424Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过175批，阻断23批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260827T021552Z_audit.json`。

## 2026-08-27 10:20 每日增量四道门禁终结器
- 精确消费 `\\10.246.97.159\d\MNY master batch prediction model pipline\data_asset_runtime\reports\cloud_m1_daily_refresh\20260827T014448Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过719批，阻断76批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260827T021719Z_audit.json`。

## 2026-08-31 09:17 云端M1增量刷新器V1
- 范围：2026-08-28T00:00:00 至 2026-08-31T01:00:00（半开区间），M1，最多5个订单；模式：DRY_RUN_STAGING。
- 读取订单5、车次203、Step行12965、曲线203、物料3324、RecipeData1760、CE Lab2、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260831T011702Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:24 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次159、Step行7871、曲线159、物料2032、RecipeData1656、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T012318Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:24 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次4、Step行204、曲线4、物料30、RecipeData1089、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T012411Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:29 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次44、Step行2915、曲线44、物料705、RecipeData2005、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T012927Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:30 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次173、Step行10627、曲线173、物料2560、RecipeData1397、CE Lab7、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T012954Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:31 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次95、Step行4196、曲线95、物料1525、RecipeData1201、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013058Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:32 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次99、Step行5893、曲线99、物料1671、RecipeData1510、CE Lab1、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013132Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:32 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次45、Step行2393、曲线45、物料765、RecipeData1926、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013206Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:33 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次108、Step行4291、曲线108、物料1734、RecipeData1659、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013232Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:33 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次96、Step行5976、曲线96、物料1446、RecipeData1580、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013308Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:34 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次170、Step行10862、曲线170、物料2504、RecipeData1932、CE Lab1、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013339Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:34 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次2、Step行100、曲线2、物料18、RecipeData1579、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013440Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:36 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次241、Step行12490、曲线241、物料3404、RecipeData1067、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013453Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:36 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次0、Step行0、曲线0、物料0、RecipeData1506、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013601Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:36 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次73、Step行3626、曲线73、物料1174、RecipeData1924、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013613Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:37 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次75、Step行4097、曲线75、物料1103、RecipeData1743、CE Lab2、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013639Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:37 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次73、Step行4650、曲线73、物料1173、RecipeData1408、CE Lab2、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013707Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:38 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次115、Step行6345、曲线115、物料1548、RecipeData1827、CE Lab2、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013738Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:39 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次201、Step行11024、曲线201、物料2976、RecipeData1596、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013819Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:39 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次32、Step行1462、曲线32、物料448、RecipeData1957、CE Lab1、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013909Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:40 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次102、Step行6564、曲线102、物料1722、RecipeData1861、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013924Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:40 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次29、Step行2017、曲线29、物料551、RecipeData1595、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014006Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:40 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次44、Step行1986、曲线44、物料616、RecipeData1370、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014024Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:41 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次199、Step行9197、曲线199、物料3332、RecipeData1563、CE Lab5、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014045Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:42 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次41、Step行1981、曲线41、物料697、RecipeData2070、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014142Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:43 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次203、Step行12965、曲线203、物料3324、RecipeData1760、CE Lab2、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014203Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:43 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次0、Step行0、曲线0、物料0、RecipeData1766、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014304Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:43 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次20、Step行1168、曲线20、物料362、RecipeData1784、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014313Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:44 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次112、Step行7171、曲线112、物料1979、RecipeData2120、CE Lab5、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014329Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:44 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次115、Step行5123、曲线115、物料1845、RecipeData1548、CE Lab2、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014414Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:45 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次73、Step行4655、曲线73、物料1100、RecipeData1867、CE Lab4、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014453Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:45 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次84、Step行5224、曲线84、物料1268、RecipeData1313、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014522Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:46 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次77、Step行4340、曲线77、物料830、RecipeData1907、CE Lab3、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014553Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:47 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次164、Step行7109、曲线164、物料2362、RecipeData1520、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014623Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:47 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次125、Step行6011、曲线125、物料1698、RecipeData1389、CE Lab3、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014709Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:48 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次71、Step行5296、曲线71、物料1142、RecipeData1460、CE Lab1、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014750Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:49 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次232、Step行12498、曲线232、物料3436、RecipeData1364、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014829Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:50 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次73、Step行3532、曲线73、物料806、RecipeData919、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014934Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:50 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次5、Step行210、曲线5、物料70、RecipeData1974、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015004Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:51 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次119、Step行5980、曲线119、物料1899、RecipeData981、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015021Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:51 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次91、Step行5698、曲线91、物料1554、RecipeData1824、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015105Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:52 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次111、Step行4931、曲线111、物料1217、RecipeData1927、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015140Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:53 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次281、Step行17556、曲线281、物料4529、RecipeData1691、CE Lab1、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015218Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:53 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次0、Step行0、曲线0、物料0、RecipeData2036、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015338Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:53 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次0、Step行0、曲线0、物料0、RecipeData1514、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015350Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:54 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次42、Step行3123、曲线42、物料673、RecipeData1516、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015359Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:54 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次62、Step行3453、曲线62、物料889、RecipeData1667、CE Lab2、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015421Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:55 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次0、Step行0、曲线0、物料0、RecipeData868、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015452Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:55 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次150、Step行8152、曲线150、物料2285、RecipeData1937、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015500Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:56 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次56、Step行3250、曲线56、物料928、RecipeData1908、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015548Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:56 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次50、Step行2032、曲线50、物料594、RecipeData1657、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015622Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:57 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次120、Step行4717、曲线120、物料1782、RecipeData1264、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015649Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:58 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次73、Step行4649、曲线73、物料1100、RecipeData1477、CE Lab1、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015730Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:58 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次75、Step行3075、曲线75、物料677、RecipeData2002、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015802Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 09:59 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次69、Step行3794、曲线69、物料692、RecipeData1525、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015830Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:00 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次279、Step行16432、曲线279、物料4198、RecipeData784、CE Lab4、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015902Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:00 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次13、Step行843、曲线13、物料247、RecipeData1972、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020023Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:01 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次109、Step行5964、曲线109、物料1744、RecipeData2006、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020043Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:02 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次64、Step行3489、曲线64、物料1071、RecipeData1935、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020131Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:02 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次0、Step行0、曲线0、物料0、RecipeData1778、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020204Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:02 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次0、Step行0、曲线0、物料0、RecipeData2053、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020213Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:03 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次172、Step行7799、曲线172、物料2467、RecipeData1666、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020223Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:03 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次69、Step行3542、曲线69、物料1218、RecipeData1938、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020308Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:03 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次16、Step行1019、曲线16、物料224、RecipeData1738、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020338Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:04 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次136、Step行8213、曲线136、物料2465、RecipeData2010、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020354Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:05 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次16、Step行787、曲线16、物料288、RecipeData1534、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020450Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:06 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次180、Step行8121、曲线180、物料2504、RecipeData1298、CE Lab1、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020506Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:07 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次146、Step行9298、曲线146、物料2197、RecipeData1595、CE Lab6、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020604Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:08 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次169、Step行10984、曲线169、物料2839、RecipeData1560、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020706Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:08 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次49、Step行3662、曲线49、物料790、RecipeData1986、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020812Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:08 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次16、Step行1132、曲线16、物料272、RecipeData1439、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020842Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:09 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次40、Step行1990、曲线40、物料523、RecipeData2010、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020859Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:09 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次0、Step行0、曲线0、物料0、RecipeData2085、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020923Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:09 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次14、Step行878、曲线14、物料230、RecipeData2013、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020932Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:09 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单5、车次9、Step行413、曲线9、物料134、RecipeData1576、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020948Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:10 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多5个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次0、Step行0、曲线0、物料0、RecipeData335、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020959Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:13 云端M1增量刷新器V1
- 范围：2026-08-26T09:23:11 至 2026-08-31T09:23:11（半开区间），M1，最多10个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单10、车次795、Step行47807、曲线795、物料12313、RecipeData2306、CE Lab30、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T021008Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-08-31 10:16 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近120小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260831T101622_audit.json`。
- 实际批次数=5000，待修复=5000，QM源返回=0。

## 2026-08-31 10:16 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T012927Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过39批，阻断5批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T021633Z_audit.json`。

## 2026-08-31 10:17 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T012954Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过153批，阻断20批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T021704Z_audit.json`。

## 2026-08-31 10:18 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013058Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过88批，阻断7批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T021813Z_audit.json`。

## 2026-08-31 10:19 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013132Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过91批，阻断8批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T021851Z_audit.json`。

## 2026-08-31 10:19 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013206Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过43批，阻断2批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T021929Z_audit.json`。

## 2026-08-31 10:20 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013232Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过107批，阻断1批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T021954Z_audit.json`。

## 2026-08-31 10:21 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013308Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过93批，阻断3批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022037Z_audit.json`。

## 2026-08-31 10:22 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013339Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过156批，阻断14批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022121Z_audit.json`。

## 2026-08-31 10:22 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013440Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过2批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022219Z_audit.json`。

## 2026-08-31 10:23 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013453Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过233批，阻断8批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022236Z_audit.json`。

## 2026-08-31 10:23 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013601Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022334Z_audit.json`。

## 2026-08-31 10:24 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013613Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过68批，阻断5批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022347Z_audit.json`。

## 2026-08-31 10:24 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013639Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过72批，阻断3批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022426Z_audit.json`。

## 2026-08-31 10:25 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013707Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过66批，阻断7批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022518Z_audit.json`。

## 2026-08-31 10:26 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013738Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过107批，阻断8批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022559Z_audit.json`。

## 2026-08-31 10:27 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013819Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过194批，阻断7批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022647Z_audit.json`。

## 2026-08-31 10:27 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013909Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过30批，阻断2批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022740Z_audit.json`。

## 2026-08-31 10:28 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T013924Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过95批，阻断7批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022804Z_audit.json`。

## 2026-08-31 10:28 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014006Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过28批，阻断1批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022839Z_audit.json`。

## 2026-08-31 10:29 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014024Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过42批，阻断2批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022901Z_audit.json`。

## 2026-08-31 10:30 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014045Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过188批，阻断11批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T022934Z_audit.json`。

## 2026-08-31 10:30 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014142Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过40批，阻断1批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T023044Z_audit.json`。

## 2026-08-31 10:32 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014203Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过188批，阻断15批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T023118Z_audit.json`。

## 2026-08-31 10:32 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014304Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T023219Z_audit.json`。

## 2026-08-31 10:32 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014313Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过20批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T023230Z_audit.json`。

## 2026-08-31 10:33 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014329Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过96批，阻断16批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T023249Z_audit.json`。

## 2026-08-31 10:34 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014414Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过110批，阻断5批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T023341Z_audit.json`。

## 2026-08-31 10:34 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014453Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过69批，阻断4批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T023432Z_audit.json`。

## 2026-08-31 10:35 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014522Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过74批，阻断10批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T023519Z_audit.json`。

## 2026-08-31 10:36 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014553Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过74批，阻断3批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T023602Z_audit.json`。

## 2026-08-31 10:37 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014623Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过119批，阻断45批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T023648Z_audit.json`。

## 2026-08-31 10:38 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014709Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过116批，阻断9批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T023752Z_audit.json`。

## 2026-08-31 10:39 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014750Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过58批，阻断13批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T023847Z_audit.json`。

## 2026-08-31 10:40 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014829Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过206批，阻断26批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T023927Z_audit.json`。

## 2026-08-31 10:40 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T014934Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过72批，阻断1批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024040Z_audit.json`。

## 2026-08-31 10:41 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015004Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过5批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024109Z_audit.json`。

## 2026-08-31 10:41 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015021Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过113批，阻断6批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024127Z_audit.json`。

## 2026-08-31 10:42 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015105Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过79批，阻断12批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024218Z_audit.json`。

## 2026-08-31 10:43 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015140Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过108批，阻断3批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024255Z_audit.json`。

## 2026-08-31 10:44 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015218Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过266批，阻断15批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024333Z_audit.json`。

## 2026-08-31 10:44 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015338Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024457Z_audit.json`。

## 2026-08-31 10:45 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015350Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024505Z_audit.json`。

## 2026-08-31 10:45 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015359Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过40批，阻断2批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024518Z_audit.json`。

## 2026-08-31 10:46 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015421Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过57批，阻断5批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024543Z_audit.json`。

## 2026-08-31 10:46 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015452Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024607Z_audit.json`。

## 2026-08-31 10:46 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015500Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过147批，阻断3批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024622Z_audit.json`。

## 2026-08-31 10:47 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015548Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过49批，阻断7批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024715Z_audit.json`。

## 2026-08-31 10:47 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015622Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过49批，阻断1批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024739Z_audit.json`。

## 2026-08-31 10:48 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015649Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过115批，阻断5批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024807Z_audit.json`。

## 2026-08-31 10:49 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015730Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过69批，阻断4批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024851Z_audit.json`。

## 2026-08-31 10:49 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015802Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过73批，阻断2批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T024933Z_audit.json`。

## 2026-08-31 10:50 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015830Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过66批，阻断3批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025003Z_audit.json`。

## 2026-08-31 10:51 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T015902Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过257批，阻断22批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025042Z_audit.json`。

## 2026-08-31 10:52 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020023Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过12批，阻断1批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025157Z_audit.json`。

## 2026-08-31 10:53 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020043Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过103批，阻断6批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025224Z_audit.json`。

## 2026-08-31 10:53 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020131Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过58批，阻断6批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025327Z_audit.json`。

## 2026-08-31 10:53 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020204Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025356Z_audit.json`。

## 2026-08-31 10:54 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020213Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025407Z_audit.json`。

## 2026-08-31 10:55 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020223Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过164批，阻断8批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025427Z_audit.json`。

## 2026-08-31 10:55 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020308Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过66批，阻断3批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025522Z_audit.json`。

## 2026-08-31 10:56 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020338Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过14批，阻断2批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025601Z_audit.json`。

## 2026-08-31 10:57 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020354Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过114批，阻断22批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025631Z_audit.json`。

## 2026-08-31 10:57 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020450Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过15批，阻断1批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025729Z_audit.json`。

## 2026-08-31 10:58 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020506Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过169批，阻断11批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025800Z_audit.json`。

## 2026-08-31 10:59 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020604Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过130批，阻断16批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T025901Z_audit.json`。

## 2026-08-31 11:00 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020706Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过159批，阻断10批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T030008Z_audit.json`。

## 2026-08-31 11:01 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020812Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过43批，阻断6批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T030105Z_audit.json`。

## 2026-08-31 11:01 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020842Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过10批，阻断6批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T030134Z_audit.json`。

## 2026-08-31 11:01 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020859Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过38批，阻断2批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T030149Z_audit.json`。

## 2026-08-31 11:02 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020923Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T030203Z_audit.json`。

## 2026-08-31 11:02 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020932Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过11批，阻断3批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T030217Z_audit.json`。

## 2026-08-31 11:02 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020948Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过7批，阻断2批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T030231Z_audit.json`。

## 2026-08-31 11:02 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020959Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T030239Z_audit.json`。

## 2026-08-31 11:07 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T021008Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过719批，阻断76批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T030324Z_audit.json`。

## 2026-08-31 11:08 延迟Lab组每日同步
- 从 `core.LabGroup` 增量幂等投影到 `mart.M1TrainingLabGroupV371`；本次候选6526组，写入=True。不访问本地MMS/Archive，不填充Lab值。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\lab_sync_20260831T030814Z.json`。

## 2026-08-31 11:08 延迟Lab车次晋级
- pending候选8700，有效Lab过滤后待晋级5，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260831T030823Z.json`。

## 2026-08-31 11:08 M1训练主表标签范围审计
- `mart.M1TrainingBatchV371`总行2030，有效Lab车次4984，待清理无Lab行0；Before-image `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\label_scope_reconcile\before_image_20260831T030830Z.parquet` SHA256=5eae89442b89454fd8fff8968613a93898c91f59632dc905d7b6eccd3edf8ffd；写入=False。不访问本地MMS/Archive。

## 2026-08-31 11:10 Silane/PID 特征重算
- staging 批次28，Silane有效23，Silica PID通过24，阻断1；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-31 11:11 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=27，阻断=1，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 11:11 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-31 11:11 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-31 11:13 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=37，统计异常=82，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 11:13 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共357219车，解析356730车，阻断489车，完整配方501个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-08-31 11:13 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填0行；M1训练车329342，身份解析329342，阻断0，未改变训练资格。

## 2026-08-31 11:13 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查28车，完整18车，从训练表隔离10车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T031347Z.json`。

## 2026-08-31 11:15 每日增量M1 Phase治理桥接
- 从MooneyIntegratedData只读生成7165车/74个完整RecipeID/95个精确Phase审核范围；数据库写入0、TrainingReady提升0。
- 审核队列：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_incremental_phase_mapping_review_queue_v1\incremental_m1_phase_mapping_review_queue.csv`；RecipeData Step目标：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_incremental_phase_mapping_review_queue_v1\incremental_m1_recipe_step_targets.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_incremental_phase_mapping_review_queue_v1\incremental_m1_phase_mapping_review_queue_audit.json`。

## 2026-08-31 11:16 每日增量M1 Phase治理桥接
- 从MooneyIntegratedData只读生成7165车/74个完整RecipeID/95个精确Phase审核范围；数据库写入0、TrainingReady提升0。
- 审核队列：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_incremental_phase_mapping_review_queue_v1\incremental_m1_phase_mapping_review_queue.csv`；RecipeData Step目标：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_incremental_phase_mapping_review_queue_v1\incremental_m1_recipe_step_targets.csv`；审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_incremental_phase_mapping_review_queue_v1\incremental_m1_phase_mapping_review_queue_audit.json`。

## 2026-08-31 11:23 Silane/PID 特征重算
- staging 批次83，Silane有效79，Silica PID通过83，阻断0；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-31 11:24 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=83，阻断=0，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 11:25 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查83车，完整18车，从训练表隔离65车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T032500Z.json`。

## 2026-08-31 11:27 延迟Lab车次晋级
- pending候选8700，有效Lab过滤后待晋级75，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260831T032746Z.json`。

## 2026-08-31 11:29 Silane/PID 特征重算
- staging 批次75，Silane有效66，Silica PID通过71，阻断1；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-31 11:31 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=74，阻断=1，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 11:31 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-31 11:31 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-31 11:32 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=37，统计异常=85，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 11:32 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查75车，完整0车，从训练表隔离75车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T033252Z.json`。

## 2026-08-31 11:34 延迟Lab车次晋级
- pending候选8700，有效Lab过滤后待晋级75，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260831T033412Z.json`。

## 2026-08-31 11:37 Silane/PID 特征重算
- staging 批次75，Silane有效66，Silica PID通过71，阻断1；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-31 11:38 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=74，阻断=1，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 11:38 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-31 11:39 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-31 11:39 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=37，统计异常=85，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 11:39 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查75车，完整57车，从训练表隔离18车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T033946Z.json`。

## 2026-08-31 11:45 Mustang Lab恢复、五日回填与每日链路修复终态
- 权限实测：`mustangmaster` 的 Compound Excellence Lab 已恢复，近7日查询返回23,540行；供应商QM schema仍无权，固定降级为 `PENDING_SUPPLIER_QM_ACCESS`，不填默认值、不丢弃同次已取得的Lab。
- 修复 `83_cloud_m1_daily_refresh_v1.py`：Lab与QM改为独立异常边界，QM失败不再清空真实Lab；120小时回填读取361个订单并对10个延迟Lab订单精确重试。`raw.CELabHistory`现有10,412行，最新源修改时间2026-08-31 00:13:56；`core.LabGroup` 6,526组，最新TestTime 2026-08-30 22:04:32。
- 最近生产日Core覆盖复核：2026-08-26至08-30分别为982/1493/1354/1578/1195车；`core.BatchStep`、`core.BatchMaterial`、`mart.M1TrainingCurveV371`对这些车均100%命中。Phase映射按MixingInfo异常和治理合同保守放行，非100%属于真实阻断，不猜测映射。
- 为4,018,008行 `core.BatchStep` 新增 `IX_BatchStep_BatchKey_GroupName_StepNo`，消除每日MERGE业务键全表扫描；迁移资产 `Mooney_Integrated_Data_Asset/sql/09_add_batchstep_daily_merge_index.sql` 已同步服务器release。
- 修复每日wrapper两项缺陷：Windows PowerShell stderr假失败改为按Python真实退出码判断；后处理键从“最后一个分页审计”改为聚合本次全部分页审计，避免早期分页漏做PID/真实曲线特征。
- 修复 `137_rebuild_real_curve_phase_features_v371.py` 状态机：仅当精确键PID与真实曲线均通过时，将 `BLOCKED_PID_FEATURES_PENDING` 或 `LAB_LABELLED_READY` 恢复为 `ELIGIBLE_M1_TRAINING`；异常、未知映射及其他阻断状态不得恢复。
- 本轮最终：`mart.M1TrainingBatchV371` 中 `RunID LIKE DAILY_20260831%` 共75车，75/75为 `ELIGIBLE_M1_TRAINING`；延迟候选另有18车按完整性门禁隔离（1真实曲线阻断、9真实Silane物料缺失、4 QM+Phase缺失、4 Phase合同不完整），证据保留在raw/core/Step/Curve/Pending与审计表。
- 服务器release三项最终文件（83、137、daily wrapper）及回归测试已同步且SHA256一致；PowerShell解析、Python编译及新增2项“全分页键聚合/可逆状态恢复”回归断言通过。全程未访问本地MMS/Archive，未训练模型、未自动晋升模型、未写Recipe/PLC。

## 2026-08-31 11:53 V371配方静态与真实材料特征重建
- 批次2012，配方89，Top/Bottom Fill解析0/0；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-08-31 11:55 V371配方静态与真实材料特征重建
- 批次2012，配方89，Top/Bottom Fill解析2012/2012；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-08-31 11:59 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查2012车，完整103车，从训练表隔离0车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T035902Z.json`。

## 2026-08-31 15:28 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查2012车，完整1055车，从训练表隔离0车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T072840Z.json`。

## 2026-08-31 15:28 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查2012车，完整1055车，从训练表隔离957车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T072856Z.json`。

## 2026-08-31 15:30 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1055车，完整1055车，从训练表隔离0车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T073039Z.json`。

## 2026-08-31 15:37 延迟Lab车次晋级
- pending候选8700，有效Lab过滤后待晋级83，写入=False；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260831T073758Z.json`。

## 2026-08-31 15:39 延迟Lab车次晋级
- pending候选8700，有效Lab过滤后待晋级83，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260831T073908Z.json`。

## 2026-08-31 15:39 V371配方静态与真实材料特征重建
- 批次83，配方11，Top/Bottom Fill解析83/83；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-08-31 15:40 Silane/PID 特征重算
- staging 批次83，Silane有效67，Silica PID通过65，阻断8；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-31 15:41 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=75，阻断=8，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 15:41 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-31 15:41 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-31 15:42 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=8，统计异常=30，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 15:42 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查83车，完整51车，从训练表隔离32车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T074227Z.json`。

## 2026-08-31 15:55 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020842Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过10批，阻断6批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T075550Z_audit.json`。

## 2026-08-31 15:56 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020932Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过11批，阻断3批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T075632Z_audit.json`。

## 2026-08-31 15:56 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020948Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过7批，阻断2批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T075651Z_audit.json`。

## 2026-08-31 15:57 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_backfill_20260831\reports\cloud_m1_daily_refresh\20260831T020948Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过7批，阻断2批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260831T075728Z_audit.json`。

## 2026-08-31 15:59 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=51，阻断=0，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 15:59 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-31 15:59 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-31 16:00 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=0，统计异常=29，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 16:00 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查51车，完整51车，从训练表隔离0车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T080019Z.json`。

## 2026-08-31 16:02 延迟Lab车次晋级
- pending候选8700，有效Lab过滤后待晋级32，写入=False；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260831T080244Z.json`。

## 2026-08-31 16:03 延迟Lab车次晋级
- pending候选8700，有效Lab过滤后待晋级32，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260831T080312Z.json`。

## 2026-08-31 16:03 V371配方静态与真实材料特征重建
- 批次32，配方8，Top/Bottom Fill解析32/32；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-08-31 16:04 Silane/PID 特征重算
- staging 批次32，Silane有效16，Silica PID通过14，阻断8；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-31 16:05 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=24，阻断=8，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 16:05 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-31 16:05 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-31 16:06 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=8，统计异常=30，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 16:06 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查32车，完整0车，从训练表隔离32车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T080627Z.json`。

## 2026-08-31 16:08 V371配方静态与真实材料特征重建
- 批次1106，配方30，Top/Bottom Fill解析1106/1106；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-08-31 16:08 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1106车，完整1106车，从训练表隔离0车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T080833Z.json`。

## 2026-08-31 16:14 每日全M1配方统一映射链路验收与服务器发布
- 每日源查询合同固定为 `CompoundName LIKE 'M1-%'`，无 T15760/B00458 配方白名单；按 `OrderStartTime + OrderID` 游标分页并在前一日窗口未完成时续跑，禁止静默跳过剩余订单。
- 每个读取到的精确 RecipeID（包含 ProcessVersion 与末尾配方修订号）均从云端 Active `RecipeData` 和 `RecipeCBS3Parameters` 构建 Step Target、SCP-LOG、填充系数和 Target Temperature；所有批次写入 core Batch/BatchStep/BatchMaterial/BatchWeather/BatchTargetVector。
- Phase 映射按真实材料体系分为 SILICA 与 CARBONBLACK；非试点配方同样运行 MixingInfo 异常门禁、RecipeData 合同、Phase 映射和真实 Curve 特征。成功批次更新 `core.Batch.PhaseRoute/Status/IntegrationStatus`，并写入 `mart.M1TrainingStepV371` 与 `mart.M1TrainingCurveV371`；未知材料体系、缺 RecipeData 或未知映射 Fail-Closed，不猜测。
- `mart.M1TrainingBatchV371` 保持真实 Lab 标签粒度：只有有有效 LabGroupMembership 的对应车次且通过所有硬门禁才进入；无 Lab 的正常生产批次保留在 core 和 V371 Step/Curve 子表，等待 Lab 延迟到达后由 delayed promotion 精确晋级。
- 非试点真实回放已验证：`9922A00011CU_PVAJ_007` 映射为 SILICA_6_PHASE，`9922A1043026_PV19_005` 映射为 CARBON_BLACK_5_PHASE；7 个有效批次通过，2 个异常批次按车次隔离。
- 当前训练主表事实：1106 个完整批次、30 个精确 RecipeID；T15760 为776批（70.2%）。历史偏斜来自既有 Lab/完整链路覆盖，不代表每日脚本只抓试点配方。
- 合同测试 34/34 PASS；Python 编译和 PowerShell 解析通过。最新8个文件已哈希一致同步到服务器每日任务 release `M1DataAssetDaily_20260824_v2`；回滚备份位于 `deployment_backup/all_m1_mapping_20260831_161443`。
- 供应商QM暂不可用时保持真实 NULL 并仅作为质量审计，不阻断非QM配方/Phase链路；绝不以常数或其他配方值填充。全链路不访问本地MMS/Archive。

## 2026-08-31 16:21 延迟Lab车次晋级
- pending候选8700，有效Lab过滤后待晋级32，写入=False；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260831T082134Z.json`。

## 2026-08-31 16:22 延迟Lab车次晋级
- pending候选8700，有效Lab过滤后待晋级32，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260831T082229Z.json`。

## 2026-08-31 16:22 V371配方静态与真实材料特征重建
- 批次32，配方8，Top/Bottom Fill解析32/32；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-08-31 16:23 Silane/PID 特征重算
- staging 批次32，Silane有效16，Silica PID通过14，阻断8；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-31 16:24 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=24，阻断=8，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 16:24 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查32车，完整0车，从训练表隔离32车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T082436Z.json`。

## 2026-08-31 16:26 延迟Lab车次晋级
- pending候选8700，有效Lab过滤后待晋级32，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260831T082612Z.json`。

## 2026-08-31 16:26 V371配方静态与真实材料特征重建
- 批次32，配方8，Top/Bottom Fill解析32/32；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-08-31 16:27 Silane/PID 特征重算
- staging 批次32，Silane有效16，Silica PID通过14，阻断8；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-31 16:28 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=24，阻断=8，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 16:28 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-31 16:28 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-31 16:29 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=8，统计异常=30，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 16:29 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查32车，完整0车，从训练表隔离0车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T082933Z.json`。

## 2026-08-31 16:32 Silane/PID 特征重算
- staging 批次32，Silane有效32，Silica PID通过14，阻断8；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-08-31 16:32 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查32车，完整16车，从训练表隔离0车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T083204Z.json`。

## 2026-08-31 16:32 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查32车，完整16车，从训练表隔离16车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T083223Z.json`。

## 2026-08-31 16:32 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-08-31 16:32 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-08-31 16:33 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=0，统计异常=30，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-08-31 16:33 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1122车，完整1122车，从训练表隔离0车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260831T083307Z.json`。

## 2026-08-31 16:36 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260831T163602_audit.json`。
- 实际批次数=4945，待修复=4945，QM源返回=0。

## 2026-09-01 09:27 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260901T092739_audit.json`。
- 实际批次数=500，待修复=500，QM源返回=1682。

## 2026-09-01 09:33 云端M1增量刷新器V1
- 范围：2026-08-31T02:15:03 至 2026-09-01T02:15:03（半开区间），M1，最多25个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单25、车次353、Step行19685、曲线353、物料4817、RecipeData6015、CE Lab9、QM517；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_nightly_recovery_20260901_v2\reports\cloud_m1_daily_refresh\20260901T013054Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-01 09:37 云端M1增量刷新器V1
- 范围：2026-08-31T02:15:03 至 2026-09-01T02:15:03（半开区间），M1，最多10个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单10、车次795、Step行47807、曲线795、物料12313、RecipeData2324、CE Lab30、QM866；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_nightly_recovery_20260901_v2\reports\cloud_m1_daily_refresh\20260901T013311Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-01 09:39 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_nightly_recovery_20260901_v2\reports\cloud_m1_daily_refresh\20260901T013054Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过330批，阻断23批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260901T013816Z_audit.json`。

## 2026-09-01 09:44 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\runtime_nightly_recovery_20260901_v2\reports\cloud_m1_daily_refresh\20260901T013311Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过721批，阻断74批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260901T014056Z_audit.json`。

## 2026-09-01 09:46 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260901T094650_audit.json`。
- 实际批次数=4961，待修复=4961，QM源返回=0。

## 2026-09-01 09:47 延迟Lab组每日同步
- 从 `core.LabGroup` 增量幂等投影到 `mart.M1TrainingLabGroupV371`；本次候选6531组，写入=True。不访问本地MMS/Archive，不填充Lab值。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\lab_sync_20260901T014737Z.json`。

## 2026-09-01 09:47 延迟Lab车次晋级
- pending候选9053，有效Lab过滤后待晋级30，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260901T014749Z.json`。

## 2026-09-01 09:48 V371配方静态与真实材料特征重建
- 批次93，配方11，Top/Bottom Fill解析93/93；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-09-01 09:49 Silane/PID 特征重算
- staging 批次93，Silane有效93，Silica PID通过81，阻断8；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-09-01 09:52 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=85，阻断=8，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-01 09:52 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-09-01 09:52 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-09-01 09:52 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=8，统计异常=30，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-01 09:52 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查93车，完整77车，从训练表隔离16车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260901T015256Z.json`。

## 2026-09-01 09:58 实验胶X硬排除
- 规则：CompoundName 任意位置含 X 即实验胶；core 精确BatchKey=1493，训练资产写入清理=False。原始/core证据保留并标记，训练mart完全排除。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\experimental_compound_exclusion\experimental_x_exclusion_20260901T015832Z.json`，键清单 SHA256=6a9371eb3f02b843abeb13f89d9f66172ee1b3abb8ea3f4ba6964baa6b32ca85。

## 2026-09-01 09:58 实验胶X硬排除
- 规则：CompoundName 任意位置含 X 即实验胶；core 精确BatchKey=1493，训练资产写入清理=True。原始/core证据保留并标记，训练mart完全排除。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\experimental_compound_exclusion\experimental_x_exclusion_20260901T015853Z.json`，键清单 SHA256=6a9371eb3f02b843abeb13f89d9f66172ee1b3abb8ea3f4ba6964baa6b32ca85。

## 2026-09-01 10:01 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260901T100107_audit.json`。
- 实际批次数=17，待修复=17，QM源返回=156。

## 2026-09-01 10:01 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260901T100129_audit.json`。
- 实际批次数=13，待修复=13，QM源返回=143。

## 2026-09-01 10:01 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260901T100153_audit.json`。
- 实际批次数=14，待修复=14，QM源返回=194。

## 2026-09-01 10:02 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260901T100210_audit.json`。
- 实际批次数=5，待修复=5，QM源返回=109。

## 2026-09-01 10:02 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260901T100226_audit.json`。
- 实际批次数=6，待修复=6，QM源返回=126。

## 2026-09-01 10:02 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260901T100244_audit.json`。
- 实际批次数=4，待修复=4，QM源返回=45。

## 2026-09-01 10:03 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260901T100308_audit.json`。
- 实际批次数=27，待修复=27，QM源返回=151。

## 2026-09-01 10:03 V371配方静态与真实材料特征重建
- 批次1128，配方30，Top/Bottom Fill解析1128/1128；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-09-01 10:09 Silane/PID 特征重算
- staging 批次1128，Silane有效1128，Silica PID通过1110，阻断0；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-09-01 10:13 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=1128，阻断=0，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-01 10:14 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=0，统计异常=27，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-01 10:14 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-09-01 10:14 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-09-01 10:15 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1128车，完整1128车，从训练表隔离0车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260901T021501Z.json`。

## 2026-09-02 每日链失败窗口恢复与终态复核
- 事实纠正：02:15 计划任务的 raw/core 部分成功（128订单、7个运行目录，PipelineRun 1338-1344 全部 SUCCESS/RowsFailed=0），但旧 wrapper 在首个 Phase finalizer 处因 Windows PowerShell 将 Python stderr 当作终止错误而退出，不能标记为端到端成功。
- 使用修复后的服务器 release 对7个 exact-key run directory 完成 MixingInfo→Phase→真实Curve→V371 重放，并依次完成 Lab同步、标签范围、Recipe/Material、Silica PID、Curve Phase、Group std、Group PID、Curve quality、完整性门禁。
- 数据库终态：`mart.M1TrainingBatchV371` 1128行、30个RecipeID、30个Compound，X实验胶0行；`DAILY_20260902T030102Z` 新增/重建48行；完整性1128/1128，缺失隔离0行。
- 关键域：formulation/weather/dry/bottom/Silica-PID 必需字段缺失均为0；`core.BatchMaterialQM` 精确BatchKey命中1128/1128。单项QM仅在对应材料/检测项目有真实源值时写入，不使用0或配方常数伪填。
- 部署修复：SQL标量NaN/Inf→NULL；128使用`MOONEY_CREDENTIALS_JSON`；补齐4个Phase/Curve依赖；wrapper按原生exit code判断stderr；X实验胶三层排除。服务器与本地83、128、wrapper哈希一致。
- 独立审计：`reports/daily_pipeline_recovery_audit_20260902.md`。下一次02:15计划任务的端到端返回0仍需实际调度确认，不提前宣称成功。

## 2026-09-02 PID actual 大量重复值根因审计
- `9922A1576030_PV27_002` 的 `pid_actual_bottom_duration_s=105.699` 覆盖556批、114订单、157 Lab组；对应556条Curve1和Curve7哈希均各自唯一，排除跨批曲线复制。
- 直接查询云端 `HESFESFEPLANT.dbo.BatchData`，订单7445527/7450048的MSB Step累计AV源值已稳定为75.29942/90.49918/105.699；重复不是129脚本广播，而是PLC时间控制完成扫描的源记录特性。
- 当前字段语义为MSB最后Step累计TimeAV；对同Target桶几乎等于Target+固定扫描延迟，B3自然支持度不足时必须标记`FIXED_IN_HISTORY`/`CONTROL_TRANSFER_UNINFORMATIVE`，不得宣称学会Target→actual响应。
- 发现PID列当前为字符型并暴露浮点字符串噪声；后续应数值化并新增curve-event elapsed字段，但本次只读审计未修改数据库。报告：`reports/pid_actual_value_repetition_audit_20260902.md`。

## 2026-09-04 T15760 托盘级弱标签与分布优化方向核查
- 当前训练面 T15760：763个物理批次、216个Pallet/Lab组、170个订单；组内批次数为1~6，84组包含5车，正式独立标签样本量是216组而非763车。
- `mart.M1TrainingLabGroupV371` 可匹配215组，其中210组仅1次Lab、4组2次、1组3次；raw CE中7组存在多个不同结果。故多次检测确实存在，但绝大多数托盘无法仅靠自身检测估计完整分布，必须用分层/部分汇聚的弱监督分布模型。
- 当前215组中仅72组的Lab均值落在60~66；组均值总体约59.01、组间标准差约5.28。现有重复检测跨度非常小，不足以单独训练稳定的组内方差模型；需把组内批次过程离散度、订单漂移和跨组残差共同用于尺度头。
- 推荐将问题定义为Pallet bag级多实例分布学习：位置头、尺度头、P(60<=MNY<=66)与能耗多目标；禁止给每车复制同一标签后按763个独立样本评价。Step影响按同RecipeID/ProcessVersion/TemplateKey、原料QM/天气匹配后，用实际增量duration、rotor、temperature与物理交互筛选，结果先解释为关联，Target推荐仍依赖B3和人工微试验。

## 2026-09-03 10:04 每日增量四道门禁终结器
- 精确消费 `\\10.246.97.159\d\MNY master batch prediction model pipline\data_asset_runtime\reports\cloud_m1_daily_refresh\20260902T181507Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过238批，阻断58批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T020409Z_audit.json`。

## 2026-09-03 10:06 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近24小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260903T100639_audit.json`。
- 实际批次数=435，待修复=435，QM源返回=0。

## 2026-09-03 10:09 每日增量四道门禁终结器
- 精确消费 `\\10.246.97.159\d\MNY master batch prediction model pipline\data_asset_runtime\reports\cloud_m1_daily_refresh\20260902T181507Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过238批，阻断58批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T020728Z_audit.json`。

## 2026-09-03 10:14 每日增量四道门禁终结器
- 精确消费 `\\10.246.97.159\d\MNY master batch prediction model pipline\data_asset_runtime\reports\cloud_m1_daily_refresh\20260902T181622Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过311批，阻断19批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T021051Z_audit.json`。

## 2026-09-03 10:19 每日增量四道门禁终结器
- 精确消费 `\\10.246.97.159\d\MNY master batch prediction model pipline\data_asset_runtime\reports\cloud_m1_daily_refresh\20260902T181733Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过721批，阻断74批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T021513Z_audit.json`。

## 2026-09-03 10:20 延迟Lab组每日同步
- 从 `core.LabGroup` 增量幂等投影到 `mart.M1TrainingLabGroupV371`；本次候选6534组，写入=True。不访问本地MMS/Archive，不填充Lab值。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\lab_sync_20260903T022042Z.json`。

## 2026-09-03 10:20 延迟Lab车次晋级
- pending候选9679，有效Lab过滤后待晋级15，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260903T022056Z.json`。

## 2026-09-03 10:21 M1训练主表标签范围审计
- `mart.M1TrainingBatchV371`总行1143，有效Lab车次5000，待清理无Lab行0；Before-image `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\label_scope_reconcile\before_image_20260903T022131Z.parquet` SHA256=eeb00317f1295f2ec34653d7c8951d2df928da3f0cf4b7959b18e4b206869d69；写入=False。不访问本地MMS/Archive。

## 2026-09-03 10:21 V371配方静态与真实材料特征重建
- 批次63，配方7，Top/Bottom Fill解析63/63；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-09-03 10:23 Silane/PID 特征重算
- staging 批次63，Silane有效63，Silica PID通过52，阻断7；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-09-03 10:25 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=56，阻断=7，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-03 10:26 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-09-03 10:27 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-09-03 10:27 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=7，统计异常=28，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-03 10:28 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共359969车，解析359480车，阻断489车，完整配方508个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-09-03 10:28 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填0行；M1训练车327982，身份解析327982，阻断0，未改变训练资格。

## 2026-09-05 T15760 托盘分布控制项目独立审计

- 审阅外部 `M1_T15760_step_analysis`，确认固定Step→PID、同集残差冒称批间sigma、硬编码控制/节能收益等阻断项；未修改原文件或生产库。
- 新建该项目 `independent_review`，实现真实标签/成员/Step审计及托盘内同车联合占比。5项测试通过、连续两次全量快照审计产物哈希一致；没有训练/推荐发布。
- 完整结果及数据哈希入口：`PROJECT_MEMORY/experiments/20260905_t15760_distribution_control_independent_review.md`。下一步复用 `core.BatchRecipeIdentity`、真实曲线和动态Phase，先确认检测取样/复测机制，再做订单时间外验证。

## 2026-09-05 T15760 真实Step接入与托盘取样研究

- 用户确认不同位置取样、非同位置复测，随机性未验证；保存到 `contracts/t15760_pallet_sampling_contract_20260905.md`。
- SELECT接入mart.M1TrainingStepV371等8类现有资产；离线恢复107车1498个实际/1498个目标时长派生值。精确云端核对多批号重复称量，离线重建104车配比，未修改生产库/每日任务。
- 最终127个完整托盘组；订单时间外测试33组，模型改善尚不能可靠支持控制推荐。8项测试通过、连续两次结果哈希一致。
- 报告/数据哈希入口：`PROJECT_MEMORY/experiments/20260905_t15760_governed_step_distribution_research.md`；不要把离线修复误报成生产部署已完成。

## 2026-09-03 10:28 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查63车，完整48车，从训练表隔离15车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260903T022830Z.json`。

## 2026-09-03 10:30 云端M1增量刷新器V1
- 范围：2026-09-02T10:29:25 至 2026-09-03T10:29:25（半开区间），M1，最多5个订单；模式：DRY_RUN_STAGING。
- 读取订单5、车次82、Step行3976、曲线82、物料1353、RecipeData1770、CE Lab0、QM192；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T022942Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-03 10:30 云端M1增量刷新器V1
- 范围：2026-09-02T10:29:25 至 2026-09-03T10:29:25（半开区间），M1，最多5个订单；模式：DRY_RUN_STAGING。
- 读取订单5、车次73、Step行4649、曲线73、物料1096、RecipeData1752、CE Lab5、QM81；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023007Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-03 10:31 云端M1增量刷新器V1
- 范围：2026-09-02T10:29:25 至 2026-09-03T10:29:25（半开区间），M1，最多5个订单；模式：DRY_RUN_STAGING。
- 读取订单5、车次224、Step行9333、曲线224、物料3473、RecipeData1839、CE Lab1、QM187；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023043Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-03 10:31 云端M1增量刷新器V1
- 范围：2026-09-02T10:29:25 至 2026-09-03T10:29:25（半开区间），M1，最多5个订单；模式：DRY_RUN_STAGING。
- 读取订单5、车次26、Step行1652、曲线26、物料365、RecipeData1315、CE Lab0、QM67；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023116Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-03 10:31 云端M1增量刷新器V1
- 范围：2026-09-02T10:29:25 至 2026-09-03T10:29:25（半开区间），M1，最多5个订单；模式：DRY_RUN_STAGING。
- 读取订单5、车次53、Step行3033、曲线53、物料851、RecipeData2083、CE Lab0、QM102；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023133Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-03 10:32 云端M1增量刷新器V1
- 范围：2026-09-02T10:29:25 至 2026-09-03T10:29:25（半开区间），M1，最多5个订单；模式：DRY_RUN_STAGING。
- 读取订单5、车次27、Step行1411、曲线27、物料304、RecipeData1955、CE Lab1、QM166；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023155Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-03 10:32 云端M1增量刷新器V1
- 范围：2026-09-02T10:29:25 至 2026-09-03T10:29:25（半开区间），M1，最多5个订单；模式：DRY_RUN_STAGING。
- 读取订单5、车次119、Step行6997、曲线119、物料1984、RecipeData1732、CE Lab2、QM154；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023216Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-03 10:33 云端M1增量刷新器V1
- 范围：2026-09-02T10:29:25 至 2026-09-03T10:29:25（半开区间），M1，最多5个订单；模式：DRY_RUN_STAGING。
- 读取订单5、车次107、Step行6646、曲线107、物料1702、RecipeData2122、CE Lab0、QM259；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023249Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-03 10:33 云端M1增量刷新器V1
- 范围：2026-09-02T10:29:25 至 2026-09-03T10:29:25（半开区间），M1，最多5个订单；模式：DRY_RUN_STAGING。
- 读取订单5、车次0、Step行0、曲线0、物料0、RecipeData1389、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023323Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-03 10:33 云端M1增量刷新器V1
- 范围：2026-09-02T10:29:25 至 2026-09-03T10:29:25（半开区间），M1，最多5个订单；模式：DRY_RUN_STAGING。
- 读取订单2、车次0、Step行0、曲线0、物料0、RecipeData813、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023333Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-03 10:33 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T022942Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过77批，阻断5批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T023353Z_audit.json`。

## 2026-09-03 10:34 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023007Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过69批，阻断4批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T023403Z_audit.json`。

## 2026-09-03 10:34 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023043Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过210批，阻断14批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T023419Z_audit.json`。

## 2026-09-03 10:34 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023116Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过23批，阻断3批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T023428Z_audit.json`。

## 2026-09-03 10:34 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023133Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过46批，阻断7批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T023436Z_audit.json`。

## 2026-09-03 10:34 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023155Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过23批，阻断4批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T023444Z_audit.json`。

## 2026-09-03 10:34 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023216Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过100批，阻断19批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T023456Z_audit.json`。

## 2026-09-03 10:35 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023249Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过77批，阻断30批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T023511Z_audit.json`。

## 2026-09-03 10:35 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023323Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T023516Z_audit.json`。

## 2026-09-03 10:35 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260903T023333Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T023521Z_audit.json`。

## 2026-09-03 10:57 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T024931Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过660批，阻断68批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T025742Z_audit.json`。

## 2026-09-03 11:02 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260826T024931Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过660批，阻断68批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260903T025828Z_audit.json`。

## 2026-09-03 11:23 全量同步Redshift CE Lab与会员关系补齐
- 从Redshift同步CE Lab数据10587行，最新化验时间推进至2026-09-03 01:48:47；
- `raw.CELabHistory` 净增251行；`core.LabGroup` 聚合6664组（合格3865组）；
- `core.LabelGroupMembership` 增量回填2484批次；写入=True。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\sync_redshift_lab_20260903T032328Z.json`。

## 2026-09-03 11:25 延迟Lab组每日同步
- 从 `core.LabGroup` 增量幂等投影到 `mart.M1TrainingLabGroupV371`；本次候选6709组，写入=True。不访问本地MMS/Archive，不填充Lab值。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\lab_sync_20260903T032504Z.json`。

## 2026-09-03 11:40 延迟Lab车次晋级
- pending候选9679，有效Lab过滤后待晋级27，写入=False；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260903T034032Z.json`。

## 2026-09-03 11:41 延迟Lab车次晋级
- pending候选9679，有效Lab过滤后待晋级27，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260903T034101Z.json`。

## 2026-09-03 11:41 V371配方静态与真实材料特征重建
- 批次27，配方7，Top/Bottom Fill解析27/27；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-09-03 11:42 Silane/PID 特征重算
- staging 批次27，Silane有效27，Silica PID通过24，阻断3；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-09-03 11:45 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=24，阻断=3，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-03 11:45 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-09-03 11:45 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-09-03 11:50 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=3，统计异常=31，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-03 11:59 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共359969车，解析359480车，阻断489车，完整配方508个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-09-03 11:59 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填0行；M1训练车327982，身份解析327982，阻断0，未改变训练资格。

## 2026-09-03 11:59 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1155车，完整1102车，从训练表隔离53车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260903T035934Z.json`。

## 2026-09-04 09:28 每日增量四道门禁终结器
- 精确消费 `\\10.246.97.159\d\MNY master batch prediction model pipline\data_asset_runtime\reports\cloud_m1_daily_refresh\20260903T181507Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断754批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260904T012854Z_audit.json`。

## 2026-09-04 09:46 全量同步Redshift CE Lab与会员关系补齐
- 从Redshift同步CE Lab数据10610行，最新化验时间推进至2026-09-04 01:01:40；
- `raw.CELabHistory` 净增24行；`core.LabGroup` 聚合6680组（合格3875组）；
- `core.LabelGroupMembership` 增量回填0批次；写入=True。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\sync_redshift_lab_20260904T014650Z.json`。

## 2026-09-04 09:47 延迟Lab组每日同步
- 从 `core.LabGroup` 增量幂等投影到 `mart.M1TrainingLabGroupV371`；本次候选6726组，写入=True。不访问本地MMS/Archive，不填充Lab值。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\lab_sync_20260904T014742Z.json`。

## 2026-09-04 09:47 延迟Lab车次晋级
- pending候选9679，有效Lab过滤后待晋级15，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260904T014755Z.json`。

## 2026-09-04 09:48 V371配方静态与真实材料特征重建
- 批次1117，配方35，Top/Bottom Fill解析1117/1117；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-09-04 09:53 Silane/PID 特征重算
- staging 批次1117，Silane有效1117，Silica PID通过1096，阻断3；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-09-04 09:57 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=1114，阻断=3，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-04 09:58 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-09-04 09:58 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-09-04 09:58 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=3，统计异常=28，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-04 10:00 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共361006车，解析360517车，阻断489车，完整配方508个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-09-04 10:01 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填0行；M1训练车327982，身份解析327982，阻断0，未改变训练资格。

## 2026-09-04 10:01 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1117车，完整1102车，从训练表隔离15车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260904T020108Z.json`。

## 2026-09-04 10:07 延迟Lab车次晋级
- pending候选9679，有效Lab过滤后待晋级15，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260904T020734Z.json`。

## 2026-09-04 10:08 V371配方静态与真实材料特征重建
- 批次1117，配方35，Top/Bottom Fill解析1117/1117；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-09-04 10:12 Silane/PID 特征重算
- staging 批次1117，Silane有效1117，Silica PID通过1096，阻断3；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-09-04 10:15 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=1114，阻断=3，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-04 10:15 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-09-04 10:15 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-09-04 10:16 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=3，统计异常=28，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-04 10:16 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共361006车，解析360517车，阻断489车，完整配方508个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-09-04 10:16 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填0行；M1训练车327982，身份解析327982，阻断0，未改变训练资格。

## 2026-09-04 10:16 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1117车，完整1107车，从训练表隔离10车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260904T021621Z.json`。

## 2026-09-04 10:41 云端M1增量刷新器V1
- 范围：2026-09-04T08:41:24 至 2026-09-04T10:41:24（半开区间），M1，最多2个订单；模式：DRY_RUN_STAGING。
- 读取订单0、车次0、Step行0、曲线0、物料0、RecipeData0、CE Lab0、QM0；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260904T024141Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-04 10:41 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260904T024141Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过0批，阻断0批；写入=False。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260904T024158Z_audit.json`。

## 2026-09-04 10:42 M1训练主表标签范围审计
- `mart.M1TrainingBatchV371`总行1107，有效Lab车次6680，待清理无Lab行0；Before-image `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\label_scope_reconcile\before_image_20260904T024242Z.parquet` SHA256=eeb00317f1295f2ec34653d7c8951d2df928da3f0cf4b7959b18e4b206869d69；写入=False。不访问本地MMS/Archive。

## 2026-09-04 10:57 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共361006车，解析360517车，阻断489车，完整配方508个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-09-04 10:57 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填0行；M1训练车327982，身份解析327982，阻断0，未改变训练资格。

## 2026-09-06 T15760全量Lab、同订单Step联合与真实曲线分布研究
- 以全部LabCurrent为总体：809有效托盘/870真实检测，恢复587完整Step组；不再用127组代表全量。记录级正常完整261组，905车真实曲线参考重放、222完整曲线托盘。
- 单项/联合Step及曲线分布模型均未形成稳健优于基线的调参证据；保留失败，不宣称黄金最优或因果降低MU。三组多位置合格历史参考及逐车SP/AV、曲线图已导出。
- SQL全部只读，未更改生产库/每日任务/PLC。12项测试PASS，重复总体快照SHA256一致。
- 完整记录：`PROJECT_MEMORY/experiments/20260906_t15760_full_lab_within_order_curve_distribution.md`；研究目录：上级`M1_T15760_step_analysis/within_order_full_lab_v2`，主报告`全量研究结果与下一步.md`。目标仍进行中。

## 2026-09-06 09:19 全量同步Redshift CE Lab与会员关系补齐
- 从Redshift同步CE Lab数据10677行，最新化验时间推进至2026-09-05 22:57:11；
- `raw.CELabHistory` 净增69行；`core.LabGroup` 聚合6726组（合格3905组）；
- `core.LabelGroupMembership` 增量回填0批次；写入=True。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\sync_redshift_lab_20260906T011919Z.json`。

## 2026-09-06 09:22 延迟Lab组每日同步
- 从 `core.LabGroup` 增量幂等投影到 `mart.M1TrainingLabGroupV371`；本次候选6774组，写入=True。不访问本地MMS/Archive，不填充Lab值。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\lab_sync_20260906T012244Z.json`。

## 2026-09-06 09:24 M1训练主表标签范围审计
- `mart.M1TrainingBatchV371`总行1111，有效Lab车次6715，待清理无Lab行0；Before-image `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\label_scope_reconcile\before_image_20260906T012421Z.parquet` SHA256=5eae89442b89454fd8fff8968613a93898c91f59632dc905d7b6eccd3edf8ffd；写入=False。不访问本地MMS/Archive。

## 2026-09-06 09:24 延迟Lab车次晋级
- pending候选9891，有效Lab过滤后待晋级10，写入=False；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260906T012438Z.json`。

## 2026-09-06 09:27 延迟Lab车次晋级
- pending候选12820，有效Lab过滤后待晋级10，写入=False；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260906T012731Z.json`。

## 2026-09-06 09:35 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近120小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260906T093521_audit.json`。
- 实际批次数=3555，待修复=3552，QM源返回=0。

## 2026-09-06 09:35 延迟Lab车次晋级
- pending候选12820，有效Lab过滤后待晋级42，写入=False；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260906T013546Z.json`。

## 2026-09-06 09:36 延迟Lab车次晋级
- pending候选12820，有效Lab过滤后待晋级42，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260906T013621Z.json`。

## 2026-09-06 09:37 V371配方静态与真实材料特征重建
- 批次42，配方5，Top/Bottom Fill解析42/42；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-09-06 09:38 Silane/PID 特征重算
- staging 批次42，Silane有效42，Silica PID通过38，阻断4；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-09-06 09:39 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=38，阻断=4，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-06 09:39 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-09-06 09:39 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-09-06 09:40 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=4，统计异常=37，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-06 09:40 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共362376车，解析361887车，阻断489车，完整配方510个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-09-06 09:40 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填0行；M1训练车327982，身份解析327982，阻断0，未改变训练资格。

## 2026-09-06 09:40 V371训练表终态完整性硬门禁
- 范围=KEY_FILE，检查42车，完整29车，从训练表隔离13车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260906T014055Z.json`。

## 2026-09-06 09:56 云端M1增量刷新器V1
- 范围：2026-09-03T00:00:00 至 2026-09-05T00:00:00（半开区间），M1，最多1个订单；模式：DRY_RUN_STAGING。
- 读取订单1、车次73、Step行4615、曲线73、物料1096、RecipeData450、CE Lab9、QM101；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260906T015602Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-06 09:58 云端M1增量刷新器V1
- 范围：2026-09-03T00:00:00 至 2026-09-05T00:00:00（半开区间），M1，最多1个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单1、车次73、Step行4615、曲线73、物料1096、RecipeData450、CE Lab9、QM101；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260906T015715Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-06 09:58 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260906T095833_audit.json`。
- 实际批次数=15，待修复=15，QM源返回=51。

## 2026-09-06 09:59 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260906T015715Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过66批，阻断7批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260906T015859Z_audit.json`。

## 2026-09-06 10:05 Silane/PID 特征重算
- staging 批次1154，Silane有效1154，Silica PID通过1136，阻断0；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-09-06 10:05 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-09-06 10:05 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-09-06 10:05 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=0，统计异常=34，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-06 10:07 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=1154，阻断=0，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-06 10:08 V371配方静态与真实材料特征重建
- 批次1154，配方33，Top/Bottom Fill解析1154/1154；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-09-06 10:08 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共362449车，解析361960车，阻断489车，完整配方510个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-09-06 10:08 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填0行；M1训练车327982，身份解析327982，阻断0，未改变训练资格。

## 2026-09-06 10:08 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1154车，完整1140车，从训练表隔离14车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260906T020837Z.json`。

## 2026-09-06 10:21 延迟Lab车次晋级
- pending候选12893，有效Lab过滤后待晋级27，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260906T022112Z.json`。

## 2026-09-06 10:21 V371配方静态与真实材料特征重建
- 批次1167，配方36，Top/Bottom Fill解析1167/1167；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-09-06 10:25 Silane/PID 特征重算
- staging 批次1167，Silane有效1167，Silica PID通过1145，阻断4；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-09-06 10:27 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=1163，阻断=4，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-06 10:27 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-09-06 10:27 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-09-06 10:28 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=4，统计异常=35，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-06 10:28 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共362449车，解析361960车，阻断489车，完整配方510个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-09-06 10:28 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填0行；M1训练车327982，身份解析327982，阻断0，未改变训练资格。

## 2026-09-06 10:28 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1167车，完整1154车，从训练表隔离13车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260906T022827Z.json`。

## 2026-09-06 10:33 云端M1增量刷新器V1
- 范围：2026-08-20T00:00:00 至 2026-09-06T00:00:00（半开区间），M1，最多19个订单；模式：DRY_RUN_STAGING。
- 读取订单19、车次809、Step行52799、曲线809、物料13129、RecipeData4621、CE Lab13、QM1081；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260906T023044Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-06 10:41 云端M1增量刷新器V1
- 范围：2026-08-20T00:00:00 至 2026-09-06T00:00:00（半开区间），M1，最多19个订单；模式：EXECUTE_RAW_ONLY。
- 读取订单19、车次810、Step行52863、曲线810、物料13144、RecipeData4621、CE Lab13、QM1081；未生成Phase，TrainingReady提升数=0。
- 结果审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260906T023411Z\refresh_audit.json`；本地MMS连接=0；未知映射保持阻断。

## 2026-09-06 10:51 每日增量四道门禁终结器
- 精确消费 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\20260906T023411Z`；MixingInfo→Phase映射→真实曲线特征→回写按车次执行。通过730批，阻断80批；写入=True。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\daily_phase_gate_finalize\DAILY_20260906T024218Z_audit.json`。

## 2026-09-06 10:51 延迟Lab车次晋级
- pending候选13703，有效Lab过滤后待晋级13，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260906T025141Z.json`。

## 2026-09-06 10:52 V371配方静态与真实材料特征重建
- 批次1199，配方37，Top/Bottom Fill解析1199/1199；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-09-06 10:56 Silane/PID 特征重算
- staging 批次1199，Silane有效1199，Silica PID通过1177，阻断4；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-09-06 10:58 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=1195，阻断=4，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-06 10:58 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-09-06 10:58 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-09-06 10:58 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=4，统计异常=46，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-06 10:58 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共363259车，解析362770车，阻断489车，完整配方510个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-09-06 10:58 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填0行；M1训练车327982，身份解析327982，阻断0，未改变训练资格。

## 2026-09-06 10:59 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1199车，完整1186车，从训练表隔离13车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260906T025903Z.json`。

## 2026-09-06 T15760初步试验只读核查与节能主目标更新
- 独立研究读取8054131及两个近期同完整配方同机台对照：确认73车MST Step7 SP35（对照42），实际时间延长；7个位置Lab结果保留，6个低于60、一个64.74，不宣称稳定达标。
- 用户明确仍要求60–66 MU，并将平均每车节能至少2 kWh列为核心目标。订单中途试验应按完整托盘边界设计，不把位置SampleID认成单车标签。
- 已执行18项固定物理假设，64托盘/29订单/43依赖对；17项支持足够、BHq<0.1为0。7916095托盘004/006有2.063kWh差额但各仅一次化验，保留为待核查线索。
- 6项新增契约测试通过；本研究没有SQL写入/每日任务修改/配方PLC操作。证据与后续计划：`PROJECT_MEMORY/experiments/20260906_8054131_trial_and_energy_priority.md`；研究主线尚未完成。

## 2026-09-06 T15760全量源成员恢复与显式首车排除
- 对528有有效Lab订单进行云端只读成员核查，54组缺成员恢复，缺口84→30；另外2组旧/源集合冲突保留。新增1362条精确成员曲线，云端关联车2280。
- 用户要求排除Batch1：109含首车托盘保留检测和成员但标记混合暴露；主完整对照80组/37订单首车0，不把标签重分给剩余正常车。
- 18固定物理项校正后仍无稳定关联；独立能量核对使7916095的2kWh候选不通过（curve2.063 vs MSB末EnergyAV1.375kWh）。尚无可放行节能推荐。
- 8项恢复/排除/计量契约测试通过；详情`PROJECT_MEMORY/experiments/20260906_t15760_coverage_recovery_first_batch.md`。研究只读，未部署每日脚本，目标仍在继续。

## 2026-09-06 T15760单位胶重能耗目标更新与候选称重审计
- 用户明确改为真实kg归一化kWh/kg，正节约不设2kWh/车硬门槛，AV和曲线积分不要求完全相等；MU仍60–66且稳定性不恶化。此前因2kWh双口径不满足而否决的历史记录不作为新门禁。
- 主完整托盘250中71组已测位置全达标；冻结6组候选27车，fresh实际Header胶重全部一致、25车逐事件质量平衡核查支持，2车重复称重解释仍未闭环，未猜填。
- 最低线索7618595|7618595002曲线0.123615kWh/kg（MU64.28一次）；多位置候选2370966|2370966002为0.133606（61.17/63.59）。不能宣称稳定最优或直接改SP。
- 8项测试及端到端研究成功，可视化报告和代码在`../M1_T15760_step_analysis/specific_energy_v4`；详细记忆`PROJECT_MEMORY/experiments/20260906_t15760_specific_energy_mass_candidates.md`。本研究无数据库/每日任务/PLC写入。

## 2026-09-06 T15760候选天气/实称材料/QM上下文匹配
- 冻结60托盘/51订单，读取3505物料行、2733小时气象及2004原始QM；不按Lab/能耗挑选对照。8项测试PASS。
- 7618595在3个天气/实称配比近邻独立订单中保留低单耗现象（curve下降中位8.60%、AV7.90%）；2370966对4近邻不再有一致节能优势，仅保留质量参考。
- 严格3种主原料同lot对照均0，不能宣称原料影响已排除。优先定位7618595/002与7614430/002：同记录SP、Wet实际时间与Bottom做功不同；两组各一次检测，暂不能发布SP推荐。
- 报告/可视化`../M1_T15760_step_analysis/context_matching_v5/候选上下文匹配与实际工况复核.md`；记忆`PROJECT_MEMORY/experiments/20260906_t15760_context_matching.md`。本轮无业务SQL写入，无每日/PLC变更。

## 2026-09-07 T15760逐Step轨迹V6与反例核查V7
- V6离线重放20托盘/17订单/81车：MST7+MSB1占重点对照单耗差额78.3%，±2s窗口敏感性方向不反转；不是因果贡献。相似三维轨迹最近5独立订单仅1组已测全部达标。7项测试通过，详见`PROJECT_MEMORY/experiments/20260907_t15760_step_trajectory_v6.md`。
- V7重新SELECT4订单：源Lab6条、core6条、成员26行全一致。7631249/003的49.38 MU及第10–14车关联确认无错配，但原料、天气、全阶段热暴露与检验时序不相同，不能归因于某一个Step。
- 完整源/成员/阶段/QM/时序对账和可视化保存在`../M1_T15760_step_analysis/discordant_cases_v7`，6项测试通过；记忆`PROJECT_MEMORY/experiments/20260907_t15760_discordant_case_v7.md`。根因未确定，不推荐SP，无业务SQL或PLC改动。

## 2026-09-07 T15760停放确认及真实QM+阶段联合验证V8
- 用户确认24h内取样、无统一停放，影响不大；不再把开测时间差作为主要解释、样本筛除或研究阻断项，已记录独立工艺合同。
- 检出2366549~9曲线汇总含-32768rpm却被旧primary门禁放行；新增研究负转速门禁，保留全部源/标签/正常成员，不将坏车移除后的剩余成员伪装成完整托盘标签。
- 全旧primary1438车本项仅1车异常；清洗后59托盘/51订单/62检测（仅3多位置托盘）。订单留出MAE：基线6.534、QM天气6.572、阶段6.845、交互6.725 MU，未改善，不给SP推荐。
- 6项测试通过，图与报告在`../M1_T15760_step_analysis/joint_qm_phase_v8`；详细记忆`PROJECT_MEMORY/experiments/20260907_t15760_typed_qm_joint_phase_v8.md`。无SQL、每日流水线或PLC写入。

## 2026-09-07 T15760 Wet/MSB实际能耗差额拆解V9
- 重放66达标托盘244车真实曲线。Wet起止温匹配25对，23对节能，中位差0.002606kWh/kg；时间中位短2.02s、均功率低22.83kW，仍非因果。
- MSB时长/总PID/起止温匹配15对，仅6对节能，9对相反；不能拼接各阶段最低值作为最优曲线。
- 已记录Step SP在Wet25/25和MSB15/15对均一致，未提供Target干预证据；白炭黑PID转速动作空间保持锁定。
- 6项测试PASS，可视化已核查，详见`PROJECT_MEMORY/experiments/20260907_t15760_wet_msb_energy_v9.md`及研究目录`wet_msb_energy_v9`。只写研究产物和记忆，无SQL/每日任务/PLC修改。

## 2026-09-07 T15760 Wet进入状态与SP识别边界V10
- 25对中18对前段未多做功，16对同时Wet节能；Dry起温差≤1°C的6对仍全部Wet节能，前段机械投入/起温均不是单一充分解释。
- 现成上下文仅覆盖8/25对，天气+实称组成匹配仅2对，三主料同lot为0；17对待补证据，不把快照缺口当源数据不存在。
- 主清洁942车SP审计：933车可观测、21同版本层无真实记录SP差异。8050512九车缺SP，不是另一个设定方案。历史人工试验是否存在与此冻结样本结论分开。
- 5项测试通过，图已查看，详情`PROJECT_MEMORY/experiments/20260907_t15760_wet_entry_state_v10.md`。无SQL、每日脚本或PLC改动，根因/Target推荐仍未完成。

## 2026-09-07 T15760对照原料/天气/QM缺口补齐V11
- 完成38托盘36订单159车的有界SELECT和旧快照复用，2281物料行、783天气小时、2368原始QM行；所有25对现有双侧来源。无业务SQL写入/每日任务/PLC变更。
- 天气+实称配比匹配2→8对，7/8仍Wet节能，中位差0.001868kWh/kg；三主料源批号全同仍0。27/38托盘独立质量重构通过，其余保留未闭环状态。
- 精确白炭黑QM与天气配比同时可比4对的105°C失重方向与Wet差额一致，但扩展15对有反例，不能宣称根因。未核准S批号别名不强制纳入模型。
- 6项测试通过，可视化已核查；详见`PROJECT_MEMORY/experiments/20260907_t15760_context_completion_v11.md`和研究目录`context_completion_v11`。Target推荐目标尚未完成。

## 2026-09-07 14:41 全量同步Redshift CE Lab与会员关系补齐
- 从Redshift同步CE Lab数据10694行，最新化验时间推进至2026-09-07 01:50:30；
- `raw.CELabHistory` 净增18行；`core.LabGroup` 聚合6738组（合格3916组）；
- `core.LabelGroupMembership` 增量回填5批次；写入=True。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\sync_redshift_lab_20260907T064148Z.json`。

## 2026-09-07 14:53 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近72小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260907T145325_audit.json`。
- 实际批次数=1961，待修复=1938，QM源返回=0。

## 2026-09-07 16:26 延迟Lab组每日同步
- 从 `core.LabGroup` 增量幂等投影到 `mart.M1TrainingLabGroupV371`；本次候选6787组，写入=True。不访问本地MMS/Archive，不填充Lab值。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\lab_sync_20260907T082626Z.json`。

## 2026-09-07 16:26 延迟Lab车次晋级
- pending候选14557，有效Lab过滤后待晋级32，写入=False；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260907T082643Z.json`。

## 2026-09-07 16:27 延迟Lab车次晋级
- pending候选14557，有效Lab过滤后待晋级32，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260907T082701Z.json`。

## 2026-09-07 16:27 M1训练主表标签范围审计
- `mart.M1TrainingBatchV371`总行1209，有效Lab车次6784，待清理无Lab行0；Before-image `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\label_scope_reconcile\before_image_20260907T082725Z.parquet` SHA256=5eae89442b89454fd8fff8968613a93898c91f59632dc905d7b6eccd3edf8ffd；写入=False。不访问本地MMS/Archive。

## 2026-09-07 16:28 V371配方静态与真实材料特征重建
- 批次1209，配方37，Top/Bottom Fill解析1209/1209；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-09-07 16:31 Silane/PID 特征重算
- staging 批次1209，Silane有效1209，Silica PID通过1187，阻断4；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-09-07 16:35 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=1205，阻断=4，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-07 16:35 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-09-07 16:35 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-09-07 16:36 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=4，统计异常=50，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-07 16:36 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共364113车，解析363624车，阻断489车，完整配方512个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-09-07 16:36 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填0行；M1训练车327982，身份解析327982，阻断0，未改变训练资格。

## 2026-09-07 16:36 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1209车，完整1196车，从训练表隔离13车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260907T083616Z.json`。

## 2026-09-08 T15760 V12失重线索与批号来源复核（研究只读）
- 未按MU/能耗筛选的56托盘48订单中，精确批号105°C失重与Wet kWh/kg正向关联保留；共原料批号整组留出能耗预测仅改善约3%、增量区间跨0，门尼预测未改善。不能确认为根因或发布SP推荐。
- 精确核查221个S号：容器桥0命中，KnownMaterialLots全命中且候选SAP物料一致，但LastChargeID全空，尚无直接批号桥接，不晋级别名。无数据库/每日脚本/产线修改。
- 7项测试PASS，图已查看，报告位于同级`M1_T15760_step_analysis/lot_bridge_v12/失重线索复核与原料批号审计.md`；完整记忆`PROJECT_MEMORY/experiments/20260908_t15760_loss105_unselected_v12.md`。下一步改查真实同条件SP变化，不用固定SP下自然差异伪造Target效应。

## 2026-09-08 T15760 V13真实SP切换回放（研究只读）
- 一年645个M1混炼订单，482个有非首车Step；单内变化0，同版本42层无SP变化；另查明23次跨raw版本SP切换。累计端点先差分，九个端点平移不当九个动作。
- 6订单/122曲线的三次事件：B1设定90→75s全车kWh/kg观察低2.07%，缺可比Lab；MST3时长47→35s的Dry节省被Wet额外做功超过，全车高0.42%；MST7 42→35rpm保留负功率待复核两车后仅低约0.05%，且MU未满足60–66。均为混杂的跨版本事件，不下发SP。
- 10项测试通过，图已核查，无生产/每日流程变更。报告同级`M1_T15760_step_analysis/sp_interventions_v13/真实SP变化与三次事件回放.md`，完整记忆`PROJECT_MEMORY/experiments/20260908_t15760_real_sp_events_v13.md`。

## 2026-09-08 10:04 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近168小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260908T100448_audit.json`。
- 实际批次数=5000，待修复=5000，QM源返回=0。

## 2026-09-08 10:05 全量同步Redshift CE Lab与会员关系补齐
- 从Redshift同步CE Lab数据10722行，最新化验时间推进至2026-09-08 01:44:05；
- `raw.CELabHistory` 净增30行；`core.LabGroup` 聚合6757组（合格3944组）；
- `core.LabelGroupMembership` 增量回填23批次；写入=True。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\sync_redshift_lab_20260908T020534Z.json`。

## 2026-09-08 10:07 延迟Lab组每日同步
- 从 `core.LabGroup` 增量幂等投影到 `mart.M1TrainingLabGroupV371`；本次候选6808组，写入=True。不访问本地MMS/Archive，不填充Lab值。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\lab_sync_20260908T020757Z.json`。

## 2026-09-08 10:08 延迟Lab车次晋级
- pending候选14557，有效Lab过滤后待晋级23，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260908T020814Z.json`。

## 2026-09-08 10:08 M1训练主表标签范围审计
- `mart.M1TrainingBatchV371`总行1219，有效Lab车次6841，待清理无Lab行0；Before-image `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\label_scope_reconcile\before_image_20260908T020818Z.parquet` SHA256=5eae89442b89454fd8fff8968613a93898c91f59632dc905d7b6eccd3edf8ffd；写入=False。不访问本地MMS/Archive。

## 2026-09-08 10:08 V371配方静态与真实材料特征重建
- 批次1219，配方38，Top/Bottom Fill解析1219/1219；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-09-08 10:14 Silane/PID 特征重算
- staging 批次1219，Silane有效1219，Silica PID通过1197，阻断4；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-09-08 10:19 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=1215，阻断=4，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-08 10:19 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-09-08 10:19 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-09-08 10:20 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=4，统计异常=60，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-08 10:20 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共364693车，解析364204车，阻断489车，完整配方512个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-09-08 10:20 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填0行；M1训练车327982，身份解析327982，阻断0，未改变训练资格。

## 2026-09-08 10:20 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1219车，完整1206车，从训练表隔离13车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260908T022027Z.json`。

## 2026-09-08 10:20 自动化延迟化验与托盘自愈巡检
- 扫描回填缺失PalletID车次855个，新增会员关系26；
- 自动晋级合格批次 23 个，通过 116 字段硬门禁；耗时 1331.28s。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\auto_reconcile\auto_reconcile_20260908T015816Z.json`。

## 2026-09-08 10:37 延迟QM与天气域有界回补
- 新增 `138_refresh_pending_qm_weather_v371.py`：只按最近168小时M1批次、ChargeID小批量查询Mustang QM，并按缺口日期增量补入真实Asia/Shanghai天气；随后重算BatchWeather及V371链接。无本地MMS/Archive、无默认值、无模型提升。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_domain_repair\20260908T103756_audit.json`。
- 实际批次数=5000，待修复=5000，QM源返回=0。

## 2026-09-08 10:38 全量同步Redshift CE Lab与会员关系补齐
- 从Redshift同步CE Lab数据10722行，最新化验时间推进至2026-09-08 01:44:05；
- `raw.CELabHistory` 净增0行；`core.LabGroup` 聚合6757组（合格3944组）；
- `core.LabelGroupMembership` 增量回填0批次；写入=True。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\sync_redshift_lab_20260908T023821Z.json`。

## 2026-09-08 10:39 延迟Lab组每日同步
- 从 `core.LabGroup` 增量幂等投影到 `mart.M1TrainingLabGroupV371`；本次候选6808组，写入=True。不访问本地MMS/Archive，不填充Lab值。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\lab_sync\lab_sync_20260908T023913Z.json`。

## 2026-09-08 10:39 延迟Lab车次晋级
- pending候选14557，有效Lab过滤后待晋级13，写入=True；不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\delayed_lab_promotion\delayed_lab_promotion_20260908T023923Z.json`。

## 2026-09-08 10:39 M1训练主表标签范围审计
- `mart.M1TrainingBatchV371`总行1219，有效Lab车次6841，待清理无Lab行0；Before-image `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\label_scope_reconcile\before_image_20260908T023926Z.parquet` SHA256=5eae89442b89454fd8fff8968613a93898c91f59632dc905d7b6eccd3edf8ffd；写入=False。不访问本地MMS/Archive。

## 2026-09-08 10:40 V371配方静态与真实材料特征重建
- 批次1219，配方38，Top/Bottom Fill解析1219/1219；写入=True。仅使用云端RecipeData和core.BatchMaterial，不使用供应商QM，不访问本地MMS/Archive。

## 2026-09-08 10:43 Silane/PID 特征重算
- staging 批次1219，Silane有效1219，Silica PID通过1197，阻断4；写入=True。报告 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_silane_pid_feature_rebuild_v371\audit.json`。

## 2026-09-08 10:47 V371真实曲线阶段特征重建
- 使用M1TrainingStepV371累计AV/SP、M1TrainingCurveV371真实Curve1/2/5/6/7和core.Batch真实胶重；白炭黑PID按TargetTemp-3C首次真实 crossing 动态切分，MSB为底部连续阶段；写入=True，通过=1215，阻断=4，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_real_curve_phase_feature_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-08 10:47 V371 组级标准差重建
- LabelGroupID 回聚合 40 个 __group_std 字段，统一 ddof=0；单车组在基础特征有效时观测波动=0，结构不适用仍保留 NULL；写入=True，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_group_std_rebuild_v371\audit.json`。

## 2026-09-08 10:47 V371组级PID同步
- 以M1TrainingBatchV371真实PID和GroupMemberSnapshot重建组级PID；仅全成员完整且目标温度唯一的Silica组写入，CarbonBlack明确不适用，部分组保持NULL。状态=COMMITTED。

## 2026-09-08 10:48 V371真实曲线质量字段重建
- 基于`M1TrainingCurveV371`真实Curve1/Curve2与阶段物理字段，按material_system|phase_route做时间前80% IsolationForest，仅统计异常保留风险标记；重建7个curve质量组字段，写入=True，物理无效=4，统计异常=60，审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_curve_quality_rebuild_v371\audit.json`。不访问本地MMS/Archive。

## 2026-09-08 10:48 M1完整RecipeID身份表
- 新增/幂等更新 `core.BatchRecipeIdentity`，从云端 `raw.MmsOrderHistory.CompoundDescription` 按车次保存完整配方号；尾码不裁剪、不合并。共364693车，解析364204车，阻断489车，完整配方512个。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\v371_challenger\global_rds_active_v1\v371_full_recipe_identity_v1\m1_full_recipe_identity_audit.json`。

## 2026-09-08 10:48 BatchTrainingReady完整RecipeID挂接
- 增加 `RecipeIdentityKey` 与 `RecipeIdentityStatus`，从 `core.BatchRecipeIdentity` 回填0行；M1训练车327982，身份解析327982，阻断0，未改变训练资格。

## 2026-09-08 10:48 V371训练表终态完整性硬门禁
- 范围=FULL_TABLE，检查1219车，完整1206车，从训练表隔离13车；原始/core/Step/Curve及审计证据保留。不访问本地MMS/Archive。审计 `C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\m1_training_batch_completeness_gate_v371\completeness_20260908T024852Z.json`。

## 2026-09-08 10:48 自动化延迟化验与托盘自愈巡检
- 扫描回填缺失PalletID车次0个，新增会员关系0；
- 自动晋级合格批次 13 个，通过 116 字段硬门禁；耗时 1015.58s。审计：`C:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\reports\cloud_m1_daily_refresh\auto_reconcile\auto_reconcile_20260908T023156Z.json`。

## 2026-09-08 T15760 MSB B1十订单只读复核 V14
- 预固定前后5+5单，真实曲线/胶重订单等权能耗约−1.83%；有符号功率与共同完整车位敏感性方向一致。B1前75秒能耗反而+0.000764kWh/kg，尾段减少−0.002334，不能把平均功率上升误读为全时段做功提高。
- 8条Lab/8个singleton托盘，不支持MU60–66稳定性放行；4完整曲线袋观察值均范围外。179/190车称重对账完整，主要lot严格对照0；7445593~19参考165总PID149.7043s超140上限，未晋级。
- 12项测试通过；生产数据库、日更脚本与SP零写入。详见`PROJECT_MEMORY/experiments/20260908_t15760_msb_replication_v14.md`与兄弟研究目录`msb_replication_v14/MSB切换十订单复核.md`。下一步优先完整多取样托盘的前段额外做功与必要保温区间研究，不单纯再缩短B1。

## 2026-09-09 T15760 订单内受控试验设计 V16
- 8067458 PLC实际实验：35–36 Step7转速约40，37–38 B1实减10且转速回35；39–40实测组合待操作确认。独立托盘010/011/012源成员各2车已同步，Lab仍0。局部能耗响应有利但各实验段全车比能未低于前后邻近可用对照；不宣布节能/质量或倒置上机下机因果。详见`PROJECT_MEMORY/experiments/20260909_t15760_8067458_plc35_response.md`。
- V17集中试验清单：一次列出A/W/T/WT两因子候选；W为MST7 35→40rpm新待审试探，T/WT缩B1仍因当前热过程风险暂缓。允许事先工艺质量批准限量试验后统一看Lab，不允许实时异常门禁事后处理；短段混合托盘不能归因单一SP。详见`PROJECT_MEMORY/experiments/20260909_t15760_trial_matrix_v17.md`，无生产变更。
- 追加8067458在产订单只读核查：raw51/MSB累计75-90-105/Step7已35。正常11车B1至68秒才达161，13车粘门且14车边界复算失败；0Lab/14PalletID未到。暂缓B1减10秒，先恢复后A段取样和现场控制确认；详见`PROJECT_MEMORY/experiments/20260909_t15760_8067458_live_baseline.md`。
- 新增首轮草案`../M1_T15760_step_analysis/controlled_trial_v16/T15760首轮订单内试验方案_V16.md`；候选仅有条件B1持续时间−10s，累计SP与相对SP明确分开，等待当前订单/控制截图核准。
- 固定完整托盘边界A1/B/A2、不同位置取样、60–66及全程PID硬门禁、真实kWh/kg主口径；异常与不合格原记录不删，不把一次试验或多个Sample当独立干预。没有试验执行或生产/每日脚本/SP写入。
- 详见`PROJECT_MEMORY/experiments/20260909_t15760_controlled_trial_v16.md`。

## 2026-09-08 T15760 多取样完整袋/B1热过程 V15
- 固定18完整多取样袋/17单/90车，40个SampleID及结果重新向mustangmaster逐条核对一致，成员及源Step可复算；保守17袋10严格层11对照。未拟合伪独立车次标签模型。
- 8009116同SP的007袋相对003总能耗−3.97%，Wet−11.37%，MSB反而+5.63%；007的B1热轨迹更分散，26车未在B1内达到参考TempSP。两袋MU取样跨度15.04/19.96，不能把该车指定为高/低门尼来源，也不宣称已定位因果。
- 44/90车首次达TempSP后有回落，深至TempSP−3仅2车，禁止把目标附近波动全判异常。完整化验/原材料/历史SCP限制如实保留；12tests通过，生产/日更/SP零写入。详见`PROJECT_MEMORY/experiments/20260908_t15760_b1_thermal_bags_v15.md`。

## 2026-09-09 8067458五个截图样本结果关联

- 将用户5张原图复制归档到`controlled_trial_v16/screenshot_lab_20260909/evidence/`，新增原文转录JSON、关联JSON、离线分析脚本和《8067458_五个截图样本检测结果分析.md》；旧报告加入后续证据说明，不更改旧快照Lab0。
- Sample/Charge：005=51.290、014=63.180、034=58.370、036=55.270、052=58.570 MU；保留MS1+3/100°C/Small方法，不将Tested当Passed，不传播给同托盘其他成员。036对034/052差−3.100/−3.300，非因果估计。
- 唯一关联、原图hash、显示数值一致、精确差值、既有过程资格和不传播标签断言通过。005/014不作正常对照；037–040缺结果，不能评价减时/联合质量效果。
- 同步实验记忆与decision_log。数据库/PLC/模型/旧冻结过程快照改动0；本轮没有刷新数据库Lab。

## 2026-09-09 8067458第二次截图增量038/039

- 新归档汇总图与转录/审计JSON、离线分析脚本及《8067458_新增038与039检测结果分析.md》。7个车号中新增038=52.190、039=57.640；5个旧值一致不重复计数，新增Sample Code未知。
- 038/039对034差−6.18/−0.73 MU，039比036/038高2.37/5.45；仅描述、不推协同或因果。旧报告加增量提示，记忆与决策日志同步。
- 唯一车号、旧值匹配、新ID未知、精确差值、不传播标签断言通过；原图/过程hash保留；数据库/PLC/模型/旧快照改动0。

## 2026-09-09 T15760 V18门尼稳定与真实节拍试验设计

- 新增V18完整方案、现场简表、设置/段序/拟议阈值JSON、设计核验脚本与周期证据JSON；用户目标更新为60–66优先，否则维持稳定，正式现场质量标准另行核对。
- A35/75、J5新40/70、J10复验40/65；9段对称回切，条件性18纯车27位置样本36检测，过渡另计。MST/MSB分别按同一Charge生效，保留混合/过渡和所有失败账。
- 旧Header源hash验证，联合BatchCycle238.205s未证明整线少10s；长空档和6处负间隔不删，要求下轮真实完成事件与等待记录。脚本设置累计、B2/B3固定、段序、数量和未授权状态断言通过。
- 同步decision_log及`PROJECT_MEMORY/experiments/20260909_t15760_v18_stability_cadence_trial.md`。无新数据库查询/生产/PLC/模型改动，待新订单/现场控制与正式质量范围核准。

## 2026-09-09 V19能量闭合/历史温度范围与两个较长处理设计

- 新增两个只读分析脚本，12车6非重叠能量块/5完整托盘/6对照差值闭合，联合MST8–9增加定位；真实原源hash与结果独立保存v19_energy_ledger。
- 核验历史精确配方ML09 477订单及14个MST9=160支持订单，明确历史组合不同和当前10步仅时间条件；不将温度SP或范围当安全保证。
- 新增V19完整报告/状态JSON，两个待审候选E1温度、E2温度+RPM，B1均75；A-E1-A-E2-A长托盘，能耗优先。V18文档标记已替代，未执行原合同/hash保留。
- 同步decision_log及V19实验记忆。无新云端查询、无PLC/生产/数据库/模型修改。当前温控兼容性未核准，trial_authorized=false。

## 2026-09-10 V20条件性两档温度备选

- 新增《V20_扩大温度范围后的两档温度备选方案.md》及状态JSON；A161/T160/T159仅温度变化，MST7固定35/B1固定75，长托盘分段和能量/稳定性双约束。
- 159明确超核验历史下限，仅为范围扩展后候选，不把假设询问当执行授权；MST10时间条件限制继续明确，缺审批不执行。
- 同步V20实验记忆和decision_log；无生产/PLC/数据库/模型变更，无新增实验或云端查询。

## 2026-09-10 V21多轮三因素研究与实际曲线报告

- 新增`controlled_trial_v16/v21_research/`，含按参考表排列的联合矩阵、4单覆盖+4单重复路线、每单两处理及完整托盘/位置采样设计、条件性内部点/扩温方向、统计方法与优化特征字典。
- 新分析034–052的19车/7托盘，包含试验后的013/014/015；六段能量账、48项功率—时长/公斤精确分解和±2秒内部边界敏感性，保留未通过筛选的047/049原账，不增补门尼标签。
- 新增3张实际曲线/托盘能量/分解科学图及中文报告；源哈希、资格、测试、设计秩输出可复现。采用现有Python312分析环境，因捆绑运行时缺matplotlib与Parquet引擎；未安装或修改依赖。
- V21为当前完整研究路线，旧版本留存；不宣称门尼因果根因已识别或方案已节能，温控审批和真实新实验仍为执行前条件。无云端数据查询、无PLC/数据库/模型变更。

## 2026-10-19 全厂炭黑 Top 5 与白炭黑 Top 5 能耗潜力排行与分步优化模块落地

- **改动类型**：全厂能耗潜力模型、分步遥测方差归因、自动化优化引擎及工程报告
- **核心产物**：
  1. `compound_energy_optimizer.py`（独立、可复用批处理优化器，支持 `--mode rank/diagnose/all`）；
  2. `reports/compound_energy_optimization_auto/Top5_CarbonBlack_and_Silica_Energy_Optimization_Report.md` 与 `docs/Top5_CarbonBlack_and_Silica_Energy_Optimization_Report.md`（完整分步计算依据工程报告）；
  3. `reports/compound_energy_optimization_auto/plant_wide_energy_potential_ranking.csv`（全厂所有胶料能耗潜力排行表）；
  4. `reports/compound_energy_optimization_auto/top_compounds_step_diagnostics_summary.csv`（核心胶料分步诊断与调整依据表）；
  5. `reports/compound_energy_optimization_auto/top5_cb_and_silica_energy_potential_dashboard.png` 与 `reports/compound_energy_optimization_auto/b00458_b00163_and_top_compounds_step_diagnosis.png`（高清可视化双看板）；
  6. `scratch/top_compounds_steps_raw.csv`（47.1 万行完整分步遥测库）、`scratch/compound_bom_classification.csv`、`scratch/all_compounds_energy_ranking.csv`。
- **业务成果**：
  - 基于全厂 20,985 车真实一混数据与 BOM 填料分类，评出纯炭黑 Top 5（B15563R1、B00460B、B00163、R00218W2、B00458）与白炭黑 Top 5（T15760、T20153W、T09170、T33025W3、T17771），年化节电潜力达 250 MWh/年（全族 295 MWh/年）；
  - 明确标注三大计算依据：分步能耗方差贡献度 $W_i$ 锁定关键调整步（排除 Step 1~4 初期吃粉，锁定 Step $\ge 5$ 高剪切升温步及 MSB 固定时间步）；非牛顿流体剪切稀化幂律公式 $\Delta P / P \approx 1.10 (\Delta N / N)$ 计算 MST 降速 $-5\text{ RPM}$（$-1.32 \sim -2.27\text{ kWh/t}$）；边际功率积分速率 $e_{\text{ton,sec}} = \bar{P}_{\text{MSB}} / (3.6 W_{\text{batch}})$ 计算 MSB 缩短秒数；
  - 重点明确 `M1-B00458` 与 `M1-B00163` 在 `92-1020` 产线上的分步方案（MST Step 7 降 5 RPM + MSB S1 减 4s + MSB S2 减 5s，合计预期节能分别达 $-5.15\text{ kWh/t}$ 与 $-5.33\text{ kWh/t}$）。
- **验证**：全量测试套件通过；图表无文字重叠；CSV 数据完整导出。详见 `PROJECT_MEMORY/experiments/20261019_plantwide_compound_energy_optimization_top5.md`。

## 2026-10-19 12:36 全厂胶料能耗优化接入 PID 总体时长 Hard Rule 刚性门禁与结果重算

- **改动类型**：工艺安全与硅烷化/热暴露红线门禁升级（PID Total Duration Hard Rule）、同订单内分位数去混杂、代码与全套报表同步更新
- **核心规则（Hard Rule）**：
  - 从上密炼（`MST_PID_Duration`：首次达到 `Target - 3°C` 至上密炼结束）到下密炼（`MSB_PID_Duration`：下密炼全步总时长）加总计算的总体时长 $t_{\text{PID,total}} = t_{\text{PID,MST}} + t_{\text{PID,MSB}}$ 必须严格落在对应温度档位区间内：
    - **Target 165°C**：**`80 – 140 s`**
    - **Target 155°C**：**`130 – 230 s`**
    - **Target 142°C**：**`270 – 330 s`**
- **核心产物更新**：
  1. `compound_energy_optimizer.py`（新增 `PID_DURATION_HARD_RULES`、`map_pid_target_tier()`、同订单内 `P25`/`P75` 分位数分步方差归因以消除跨订单 MSB 配方切换混杂、以及总体时长低于下限时的 `MSB = 0s` 刚性截断保护）；
  2. `reports/compound_energy_optimization_auto/Top5_CarbonBlack_and_Silica_Energy_Optimization_Report.md` 与 `docs/Top5_CarbonBlack_and_Silica_Energy_Optimization_Report.md`（新增第 3.4 节总体时长 Hard Rule 公式、第四章 10 大胶料合规性全景审计表、以及 `B00458`/`B00163` 双配方状态精确指导）；
  3. `reports/compound_energy_optimization_auto/top_compounds_step_diagnostics_summary.csv`（新增 `PID_Target_Tier_C`、`PID_Rule_Band_s`、`Curr_PID_Top_MST_s`、`Curr_PID_Bottom_MSB_s`、`Curr_PID_Total_s`、`Post_DOE_PID_Total_s`、`PID_HardRule_Status`、`PID_HardRule_Note` 8 个合规审计字段）；
  4. `reports/compound_energy_optimization_auto/top5_cb_and_silica_energy_potential_dashboard.png` 与 `reports/compound_energy_optimization_auto/b00458_b00163_and_top_compounds_step_diagnosis.png`（同步更新 Hard-Rule 修正后的单吨节电与 MSB 动作标注）。
- **关键重算结论**：
  - **白炭黑 Top 5（全部为 Target 165°C，硬门禁 `80–140 s`）**：5 款胶料调整前总体 PID 时长在 `110.9s ~ 133.1s`，执行 `MST 降 3~5 RPM + MSB 缩时 8~10s` 后总体 PID 时长落在 **`100.9s ~ 123.1s`**，**100% 满足 `[80, 140] s`**；其中 `M1-T17771` 原本高能耗批次达 `141.3s`（超出 140s 上限），缩时 `-10s` 后回到 `123.1s`，同时消除了超时过早焦烧隐患；
  - **炭黑 Top 5（Target 155°C 档，硬门禁 `130–230 s`）**：
    - `M1-B15563R1`（`175.7s -> 170.3s`，严格 post-S5 口径 `141.0s -> 132.0s`）、`M1-B00460B`（`152.0s -> 142.0s`）、`M1-R00218W2`（`167.2s -> 158.9s`）均严格满足 `[130, 230] s`（注：`B15563R1` 二阶段禁止继续减至 `-15s`，否则将跌破 `130s` 下限）；
    - **`M1-B00458` 与 `M1-B00163` 触发 Hard Rule 刚性截断保护**：两款胶料在 `92-1020` 近期已由旧版 3 步 MSB（`167.5s`）切换为新版 2 步 MSB 短配方，当前总体时长分别为 **`83.2s`（最新订单 `80.4s`）** 与 **`99.8s`（最新订单 `92.3s`）**，已处于/逼近下限！根据 Hard Rule，**新版 2 步 MSB 配方下严禁再缩短 MSB 时间（强制截断 `MSB Δt = 0 s`）**，仅执行 **`MST Step 7 降速 -5 RPM`**（`B00458: 45->40 RPM` 节电 **`-1.55 kWh/t`** 并延长 MST `+2.8s` 至 `86.0s`；`B00163: 47->42 RPM` 节电 **`-2.41 kWh/t`** 并延长 MST `+3.2s` 至 `103.0s`）；仅在排产回退至旧版 3 步 MSB（`167.5s`）时方可执行 `MSB -9s`。

## 2026-10-19 12:57 规则适用范围修正：PID 总体时长 Hard Rule 仅对白炭黑（Silica）胶料生效，炭黑（Carbon Black）豁免

- **改动类型**：工艺门禁适用范围修正（Silica-Only PID Hard Rule Gate）、炭黑胶料恢复全额 `MST + MSB` 节能推荐
- **业务规则修正说明**：
  - 用户明确确认：**三种 PID Target Temperature 的总体时长硬规则（`165°C: 80–140s`、`155°C: 130–230s`、`142°C: 270–330s`）仅对白炭黑胶料（Silica，`M1-T*` 系列）生效，炭黑胶料（Carbon Black，`M1-B*`、`M1-R*`、`M1-A*` 系列）无硅烷化反应，不受该硬规则影响**。
- **核心产物更新与重算结果**：
  1. **白炭黑 Top 5（`M1-T15760`、`M1-T20153W`、`M1-T09170`、`M1-T33025W3`、`M1-T17771`）**：严格执行 **Target 165°C (`80–140 s`)** 硬门禁，调整后总体 PID 时长分别为 **`100.9s`、`113.5s`、`111.7s`、`109.6s`、`123.1s`**，100% 满足 `[80, 140] s`，预期单吨节电 **`-5.00 ~ -6.38 kWh/t`**；
  2. **炭黑 Top 5（`M1-B15563R1`、`M1-B00460B`、`M1-B00163`、`M1-R00218W2`、`M1-B00458`）**：标记为 `EXEMPT (CarbonBlack - Not Subject to Silica PID Rule)`，全额执行 **`MST` 高剪切步降速 `-5 RPM` + `MSB` 固定时间步缩减 `-9s / -10s`**：
     - `M1-B15563R1`: `MST S5 -5 RPM` + `MSB S1 -4s + MSB S2 -5s (-9s)` $\to$ 预期节电 **`-4.29 kWh/t`**；
     - `M1-B00460B`: `MST S8 -5 RPM` + `MSB S1 -5s + MSB S2 -5s (-10s)` $\to$ 预期节电 **`-5.36 kWh/t`**；
     - `M1-B00163`: `MST S7 -5 RPM` + `MSB S1 -4s + MSB S2 -5s (-9s)` $\to$ 预期节电 **`-5.83 kWh/t`**；
     - `M1-R00218W2`: `MST S7 -5 RPM` + `MSB S1 -5s + MSB S2 -5s (-10s)` $\to$ 预期节电 **`-3.53 kWh/t`**；
     - `M1-B00458`: `MST S7 -5 RPM` + `MSB S1 -4s + MSB S2 -5s (-9s)` $\to$ 预期节电 **`-5.38 kWh/t`**。
