# 永久决策日志

## 2026-08-13：曲线异常处理

- 决策：物理无效曲线剔除；统计异常曲线保留并加风险标记；MNY 波动不用于删除。
- 原因：同订单内 MNY 波动可能是真实工艺差异，不能把目标波动误判为脏数据。
- 实现：`Mooney_Prediction_Pipeline/model_training/curve_quality_contract_v1.py`。
- 结果：6,814 车次中物理无效 0，统计异常 217，风险 Lab 组 133。
- 结论：风险标记本身没有明显降低 OOT MAE，但应作为质量审计字段保留。

## 2026-08-13：数值尺度处理

- 决策：不对全部模型特征做统一 Z-score/Min-Max。
- 原因：主模型和主要专家为 LightGBM，树模型不依赖尺度；保留温度、秒数、RPM 的物理单位有利于解释和推荐边界。
- 已做：Isolation Forest 内部使用训练片 RobustScaler。
- 约束：若启用 Ridge/Huber，只在线性专家内部做训练片 RobustScaler，并随模型包保存。

## 2026-08-13：缺失值语义

- 决策：V3.7.1 过程差值不再用 `fillna(0.0)` 伪造正常值。
- 原因：缺失不等于刚好等于路线 nominal；应保留 NaN 并使用显式 missing flag。
- 范围：只影响 V3.7.1 Challenger 子类，冻结 V3.7 不变。

## 2026-08-13：批次级 PID 与机制特征

- 决策：R1B 保留批次级训练口径，并接入动态 PID 特征；7 个手工机制交互项不进入当前主模型。
- 结果：PID+质量 R1B Global OOT MAE 4.394、T15760 4.830、B00458 4.071；加入机制项后分别为 4.408、4.839、4.276。
- 原因：机制项没有稳定增益，继续堆叠会增加过拟合和推荐解释风险。
- 下一步：ProcessVersion/TemplateKey 时间安全残差、同订单前序状态和路线内执行偏差。

## 2026-08-13：扩展双胶料验收

- 决策：在原有 B00458、T15760 外，各材料体系增加两个有足够覆盖的精确配方版本：CarbonBlack 为 M3B-B00460 和 M2-B00163；Silica 为 T09170 和 T33025。
- 原因：验收不能只看 MAE，必须同时评估同订单内预测变化方向和批次间波动捕捉。
- 数据口径：每个 LabelGroup 内多车次预测先取均值，与单一 MNY 比较一次；不把一个 Lab 值重复计数。
- 结论：六个配方版本均完成组级 OOT 审计，但多个配方预测 Std/实际 Std 明显低于 0.70，不能放行。后续继续修复 ProcessVersion/TemplateKey 偏差和同订单前序状态残差，不打开 Stage3 掩盖基础模型问题。

## 2026-08-13：验收输入文件修正

- 发现：`r1b_group_predictions.csv` 不是可靠的原始车次输入，部分 Lab 组包含 OrderID/PalletID 为空的连接辅助行；将其再次平均会人为压低预测波动。
- 修正：验收改用 `r1b_batch_predictions.csv`，该文件是一车次一行且没有空 OrderID/BatchNumber；之后再按 LabelGroup 求车次预测均值。
- 影响：前一版扩展验收结果作废；当前报告和 JSON/CSV 均已用修正口径重跑。

## 2026-08-13：M1 范围硬约束

- 决策：本次模型训练、验收和后续推荐只看 `M1-` 胶料；M2、M3B、PB、R1 等全部排除。
- 当前状态：R1B artifact 原本来自 Global Active 快照，因此这次 M1 结果只是现有模型的 M1 子集回放，不等同于严格 M1-only 重训模型。
- 结论：正式验收前必须用 M1 数据重新训练 Challenger；新增 M1 炭黑版本各只有 2 个 OOT Lab 组，只能作为探索性证据，不可作为正式统计放行。

## 2026-08-13：严格 M1-only Challenger 重训

- 实施：用 M1-only 批次/组级快照重训 `V3.7.1-Corrected-Challenger-R1B-M1-Only`；M2、M3、PB、R1 不进入 Stage1、Stage1b、Stage2 的任何拟合。
- 数据：4,100 个 M1 批次、1,197 个 Lab 组；train 2,704 批次/791 组，val 680/195，OOT 716/211；物理批次重复为 0。
- 结果：M1-only Global OOT MAE 5.188；T15760 精确版本 OOT MAE 约 5.124，预测 Std/实际 Std 约 0.514；模型仍未达到 2.5 MU 和波动捕捉放行要求。
- 保留：Global R1B artifact 不覆盖，继续作为对照；Stage3 在线校准仍关闭。

## 2026-08-14：前沿方法研究结论

- 决策：优先研究 M1 的因果过程状态、局部/JIT 概率专家和灰盒过程响应；暂不直接切换 Transformer 或在线强化学习。
- 原因：公开橡胶密炼研究已经证明软测量、半监督、JIT-GPR、鲁棒深核模型对延迟/稀疏 MNY 有针对性；密炼控制研究强调扭矩目标和热历史目标。当前项目的主要短板是批次过程状态解释和预测振幅，不是模型规模不足。
- 控制路线：先批次间人工影子推荐，再过程约束 Bayesian Optimization，再影子 MPC，最后才考虑安全/离线 RL。
- 约束：所有研究继续 M1-only、云端 RDS Active-only、Stage3 关闭、能耗仅审计、无 Recipe/PLC 写入。

## 2026-08-14：本地历史 CSV 接入决策

- 决定：`data/stage_statistics_enriched.csv` 作为本地历史原始曲线候选源，允许只读、隔离重建；不访问 MMS/Archive。
- 禁止：`stage_statistics_enriched_all_features_weather_v4.csv` 不得直接并入 V3.7.1 训练，因为它包含跨批均值填补、按 MNY 波动删订单、Silica PHR 重复求和及旧 Stage6 ram/天气合同。
- 标签门禁：历史 CSV 无 disposition code，且旧复测逻辑使用均值；在 001/011 和最新有效 Lab 结果重新确认前，候选标签状态统一为 `BLOCKED_UNVERIFIED_HISTORICAL_LAB`。
- 去重门禁：与当前修正快照重叠时，以当前快照为准；固定 211 个 OOT Lab 组不得进入历史补充训练。
- 推荐边界：历史原始曲线可增强 Phase 门尼模型，但因缺少完整 Step Target、SCP-LOG 和模板执行向量，不能单独用于 Step→Phase 推荐模型。

## 2026-08-14：本地历史隔离与 Phase 小批验证

- 已形成 20,126 个本地历史 Phase 重建候选、6,190 个 Lab/Pallet 组，且与当前修正快照和固定 validation/OOT 无重叠。
- 历史标签继续全部阻断，训练可用数为 0；不因候选数量增加而绕过 disposition 和 Lab 修订合同。
- 六个验收胶料各100车的当前批次独立 Phase 诊断重建全部成功，证明保存的原始曲线具有技术可恢复性。
- 该 Phase 输出仍是 V3.7 t1–t7 启发式诊断，不替代精确 Step Target/SCP-LOG 合同；只有 Lab 和 Step 合同补齐后才可进入正式重训与推荐。

## 2026-08-14：本地历史 Lab 标签释放

- 只信任已有 `lab_group_canonical_001_011.csv` 的规范结果；旧历史 CSV 内嵌 MNY 不作为训练目标。
- 1,128个历史组可由规范Lab重新确认，但旧MNY只有543组完全一致，222组差异超过1MU，最大差异33.23MU。
- 为保护固定验证/OOT，只释放精确RecipeID存在固定holdout且生产时间严格早于其起点的385组、1,385车；其余继续阻断。
- 释放后的过程Parquet不含任何MNY/Lab目标列，标签独立按LabelGroupID保存；下一步必须先实现训练/服务同源Phase特征，再做固定211组OOT的C1历史增强对照。

## 2026-08-14：等权历史增强 C1 被拒绝

- H0 使用当前训练767个Lab组；H1在完全相同特征、模型和固定Val/OOT上增加385个安全历史Lab组。
- 固定211组OOT中，H0 MAE为5.502 MU、H1为6.467 MU；H1 Bias升至2.169 MU，R²由0.529降至0.339。
- 跨配方总体预测Std/实际Std为1.329并不代表批次波动捕捉良好；订单内相邻Lab组的变化幅度比只有0.255，方向准确率52.4%，趋势门禁失败。
- 385个历史组中有200组来自两个T33025 RecipeID，占52.0%；加入后这两个RecipeID的历史占比分别为83.3%和71.4%，等权合并移动了共享Stage1全局表面，并伤害无历史补充的配方。
- 决策：不发布H1，不以“更多历史数据”作为自动增益；下一历史候选必须使用训练折内RecipeID权重上限、时间衰减、ProcessVersion/模板相似性门禁，并与H0固定对照。
- 工程稳定性：子专家特征顺序改为确定性排序，LightGBM启用deterministic/force_col_wise；连续两次模型和指标哈希完全一致。

## 2026-08-14：H2 加权历史增强被拒绝

- 版本真相：旧 `stage_statistics_enriched.csv` 没有订单真实 `ProcessVersion`。RecipeID 中的 `_PVxx_` 不得作为订单版本，也不得宣称完成版本级 TemplateKey 匹配；统一审计状态为 `LEGACY_ORDER_PROCESS_VERSION_UNAVAILABLE`。
- H2 准入只使用目标无关信息：精确 RecipeID、MixerLine、PhaseRoute，以及当前训练期静态温度/填充因子/材料比例范围；MNY 不参与相似性或权重计算。
- 权重合同：每个历史 Lab 组基础权重 0.25，180 天半衰期；每个 RecipeID 的历史有效票数最多为当前训练 Lab 组数的 25%。
- 实际只有 13/385 个历史 Lab 组通过，全部属于 B00458；42 个批次的总有效票数为 2.6295 组。
- 固定 211 组 OOT：H0 MAE/RMSE 为 5.502/9.004，H2 为 5.575/9.200；H2 Bias 1.782 超过 1.5，R² 由 0.529 降至 0.508。订单内 delta Pearson 0.072→0.141，但方向准确率 42.9%→38.1%，delta Std 比 0.349→0.304。
- 决策：H2 不晋级。受控历史没有解决订单内波动学习，且旧源缺少真实版本/Step Target 合同；不继续靠调整历史权重追逐固定 OOT。
- 下一重点：回到当前同源数据，分解“跨配方总体相关高、同订单 delta 弱”的来源，建立只在训练折内学习的订单内过程偏差/前序状态残差候选；严格保留 H0、R1B 和冻结 V3.7 对照。

## 2026-08-14：订单内 Phase Delta 单阶段专家不晋级

- 严格转移单元要求相邻 Lab 组属于同一 OrderID、RecipeID、订单 ProcessVersion、MixerLine、TemplateKey 和同一时间切片；train/val/OOT 分别只有 147/36/29 对。
- 固定 Ridge 正则、仅按验证集 ΔMNY MAE 选择得到 Wet bundle，但验证集 MAE 5.795 略差于 carry-forward 5.782，因此在查看 OOT 前已经不具备晋级资格。
- OOT Wet bundle ΔMNY/level MAE 为 5.174，carry-forward 为 5.357；但方向准确率仅 52.4%，预测 ΔStd/实际 ΔStd 仅 0.124，仍严重压缩真实波动。
- 12 个过程量在可用切片中 Spearman 符号一致，但强度很弱；其中最可信的物理方向为：Stage3 油后温度与 MNY 正向，Stage4 Wet duration/rotor integral 与 MNY 负向。这与“油后过热压缩 Wet 热预算、Wet 时间增加通常降低门尼”的专家机制一致。
- 决策：不发布任何单阶段 Phase-delta 专家，也不把前一 Lab MNY 接入正式在线预测。下一候选只允许使用低维、有物理方向约束的 `oil-post temperature × wet thermal budget` 耦合残差，并按 Silica/CarbonBlack 分层；禁止重新引入 40 维自由残差模型。

## 2026-08-14：H3 热预算特征与大规模无标签 M1 决策

- H3 将油后温度超额、Wet duration/rotor integral 缺口及交互加入原四路线 Stage2；nominal 仅使用当前训练片，且无油路线显式物理零。
- H3 没有稳定增益：Val MAE 比 H0 差0.0139 MU，OOT MAE差0.0124 MU；订单内方向准确率仍42.9%，delta Std比仅由0.349变为0.353。拒绝 H3，不继续手工堆叠热预算特征。
- 每日大量无 Lab 的 M1 订单应接入，但主要用途是自监督过程表示、正常流形/OOD、过程代理多任务和主动检测，不是直接伪造 MNY。
- 既有半监督实验只有15个可用伪标签组，Teacher本身MAE约4.56且delta幅度明显不足，因此当前继续关闭伪 MNY；弱 Teacher 会把均值回归复制给 Student。
- 历史001/011 Lab作为唯一监督真相；一个Pallet组只贡献一次标签。监督目标同时包含组级MNY和同订单真实标签对的delta/ranking，以直接约束方向与波动。
- 当前不做强化学习。历史生产记录是受工程师策略影响的观察数据，缺少可识别反事实动作奖励；先完成自监督表示、真实Lab监督、Step→Phase响应和安全影子验证。
- 数据优先级：先建立云端复制库的每日M1无标签增量Parquet语料与覆盖审计；按小日期/订单窗口读取，禁止本地MMS/Archive和全库扫描。

## 2026-08-17：相对门尼改善 Gate 0 / A / B2 / B3 决策

- B3已提前与B2并行；两个试点都未通过，禁止进入Target搜索。
- Gate 0未发现任何满足Lab组数、三控制水平和独立transition联合门槛的杠杆，DML按计划不开发、不运行。
- B00458旧Step快照与当前Pallet标签没有物理键交集，批次不能代替独立Lab组；B3可使用无Lab过程数据，因果支持必须阻断。
- 方案A只作为去混杂历史包络影子证据；即使统计包络存在，也不能声称工艺动作可降低具体MU。
- B2 OOT方向0.762不能覆盖验证集方向0.414及delta幅度比0.176的失败。
- 双试点均未通过B6前，人工微试验和任何自动控制继续禁止。
