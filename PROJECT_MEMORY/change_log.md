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

| 2026-08-14 | 版本语义审计 | `stage_statistics_enriched.csv` + `local_history_label_release_v1/` | 确认旧历史源没有订单真实 ProcessVersion；RecipeID 中 `_PVxx_` 仅为配方标识的一部分，不再冒充订单工艺版本 | B00458/T33025/T15760 与当前订单实际 ProcessVersion 逐项对照 | 完成；标记 `LEGACY_ORDER_PROCESS_VERSION_UNAVAILABLE` |
| 2026-08-14 | 模型工程 | `v371_source_weighted_group_atomic_batch_v1.py` + H2 runner/tests | 新增 Lab 组来源可靠度权重；历史准入采用精确 RecipeID+Mixer+Route+静态工艺合同，权重采用 180 天半衰期及逐 RecipeID 25% 上限 | 4 项新增单元测试通过；有效组权重逐组/逐配方守恒 | 完成 |
| 2026-08-14 | 模型对照 | `v371_c1_history_weighted_h2_v1/` | H2 仅放行 13/385 个历史组（均为 B00458），总有效票数 2.630；固定 211 组 OOT 与 H0 对照 | MAE 5.502→5.575；RMSE 9.004→9.200；方向 42.9%→38.1%；连续两次模型和指标 SHA256 一致 | H2 拒绝，不发布 |
| 2026-08-14 | 过程信号诊断 | `diagnose_m1_phase_delta_signal_v1.py` + `m1_phase_delta_signal_v1/` | 对同 Order/Recipe/ProcessVersion/Mixer/Template 的相邻 Lab 组构造 Phase 执行差；训练折拟合固定正则 Phase bundle，验证集选择，OOT 只评估 | 147/36/29 个 train/val/OOT 转移；2 项单元测试通过；已生成 Actual vs Prediction 图 | 完成，所有 bundle 不放行 |
| 2026-08-14 | 模型对照 | `run_v371_c1_thermal_budget_h3_v1.py` + `v371_c1_thermal_budget_h3_v1/` | 在 C1 中增加6个训练片 nominal 驱动的油后温度×Wet热预算低维特征；无油路线为物理不适用零 | Val MAE +0.0139、OOT MAE +0.0124；订单内方向不变、delta幅度比仅0.353；1项新增测试通过；已生成全览图 | H3拒绝，不发布 |
| 2026-08-14 | 方案合同 | `docs/M1大规模无标签订单半监督路线_V3_20260814.md` | 将每日大量无Lab的M1订单用于Phase-aware自监督表示、真实Lab组级监督和主动检测；当前关闭伪MNY与RL | 对照既有Teacher–Student/Representation V2失败结果和当前标签合同 | 设计完成，待无标签语料清单审计 |
