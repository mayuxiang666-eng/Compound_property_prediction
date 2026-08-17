# 当前项目状态

更新时间：2026-08-13

## 数据边界

- 训练与验证使用云端 RDS APAC Active 快照。
- Archive 未复制到云端，状态为 `BLOCKED_ARCHIVE_NOT_REPLICATED`。
- 本地 MMS/Archive 不得作为兜底数据源。
- 数据源构建必须记录 SHA256、截止时间和表/文件范围。

## 模型状态

- 冻结 V3.7：保留，不覆盖。
- V3.7.1 Challenger：四阶段解耦结构保留。
- Stage3 在线校准：关闭。
- 当前开发模型：`V3.7.1-Corrected-Challenger-R1B-Group-Atomic-Batch`，路径为 `reports/v371_challenger/global_rds_active_v1/v371_r1b_pid_quality`。
- M1-only 重训模型：`V3.7.1-Corrected-Challenger-R1B-M1-Only`，路径为 `reports/v371_challenger/global_rds_active_v1/v371_r1b_m1_only`；不覆盖 Global R1B。
- 当前模型不能生产放行，也不能生成推荐值。
- 当前业务范围进一步收紧为 M1-only；M2、M3B、PB、R1 等不进入本模型训练、验收和推荐。
- M1-only 模型已完成重训；M2/M3/PB/R1 不参与任何模型学习。
- 已扩展验收配方：CarbonBlack 为 M1-B00458、M1-B00460B、M1-B15563R1；Silica 为 M1-T15760、M1-T09170、M1-T33025。
- 验收指标已增加同订单方向准确度、显著方向准确度和预测波动/实际波动比；所有 LabelGroup 内车次预测先求均值再与单一 MNY 比较。
- 当前扩展验收仍未放行，主要问题是多个精确配方的预测 Std/实际 Std 小于 0.70，仍然低估车次间/组间波动。

## 已完成修正

- Lab 标签按 `OrderID + PalletID` 组级处理，一个组一个标签。
- 评价按“逐车次预测 → 同 Lab 组预测均值 → 与唯一 MNY 比较一次”。
- PID 时间按每个 Mixer 内累计 Step 时间的一阶差分计算。
- 白炭黑 PID 起点按 MST 首次达到 `Target - 3°C` 动态识别。
- 白炭黑 PID 总时间硬门禁：165°C 为 80–140s，155°C 为 130–230s，142°C 为 270–330s。
- 物理无效曲线剔除，统计异常曲线保留并增加风险标记。
- V3.7.1 过程缺失不再转换为 0，保留 NaN 和 missing flag；冻结 V3.7 未修改。

## 当前已知性能

早期缺失安全版本 OOT（开发诊断，非当前最强 R1B）：

| 范围 | MAE | 预测Std/实际Std |
|---|---:|---:|
| Global | 约 4.60 | 约 0.97 |
| T15760 | 约 4.98 | 约 0.55 |
| B00458 | 约 3.91 | 约 0.45 |

当前 R1B PID+质量版本的组级 OOT 参考：Global MAE 约 4.394，T15760 约 4.830，B00458 约 4.071；仍未达到 2.5 MU 和波动捕捉放行要求。

当前主要问题不是统一归一化，而是模型对车次/订单过程差异的有效解释不足，尤其是 T15760 和 B00458 的预测振幅不足。

## 下一优先级

1. M1 causal state encoder：当前车次已完成 Phase + 同订单历史完成车次状态。
2. M1 local/JIT probabilistic experts：RecipeVersion/设备/路线/StepPattern 分层，输出不确定性和 OOD。
3. M1 gray-box Step Target → 实际过程响应模型。
4. 过程约束 Bayesian Optimization 的离线影子推荐。
5. 仅依据时间 OOT 选择模型，继续保留冻结 V3.7 对照；Stage3/MPC/RL 延后。

## 2026-08-14 最新检查点：历史增强 H0/H1/H2

- 同源 current-batch-only Phase C1 的当前数据基线 H0：固定 211 个 OOT Lab 组，MAE 5.502、RMSE 9.004、Bias 1.221、R² 0.529。
- 等权历史 H1 已拒绝：OOT MAE 6.467。
- 静态合同匹配、时间衰减、逐 RecipeID 限额的 H2 也已拒绝：仅 13 个历史组通过（总有效权重 2.630），OOT MAE 5.575、Bias 1.782。
- H0/H2 订单内趋势都未过门槛：方向准确率分别为 42.9%/38.1%，delta Std 比为 0.349/0.304；总体散点相关高主要来自配方间差异，不能证明捕捉到同订单波动。
- 旧历史源缺少订单真实 ProcessVersion，状态为 `LEGACY_ORDER_PROCESS_VERSION_UNAVAILABLE`；不能用 RecipeID 中 `_PVxx_` 代替订单版本，也不能用该历史源证明版本级模板匹配。
- 下一步停止继续调历史权重，优先在当前同源数据上构建训练折内的订单内过程偏差、同订单前序状态和 Phase 敏感残差候选，并与 H0/R1B/冻结 V3.7 统一比较。
- 最新 Phase-delta 诊断：Wet bundle 在验证集没有优于 carry-forward，OOT 变化幅度比仅 0.124，不晋级。稳定但弱的方向证据支持 Stage3 油后温度升高→MNY 上升、Stage4 Wet duration/rotor integral 增加→MNY 下降；下一模型只研究两者的低维热预算耦合，不再增加高维通用残差专家。
- H3 低维热预算耦合也未改善 H0（Val/OOT MAE 分别恶化0.0139/0.0124 MU），已拒绝。下一主线改为建立真正的每日 M1 无标签云端增量语料，先做 Phase-aware 自监督过程表示和主动检测；Teacher 达标前不启用伪 MNY，强化学习继续延后。

## 2026-08-17 最新检查点：相对门尼改善 Gate 0 / A / B2 / B3

- 已实现概率降低请求与失败关闭推荐接口，但当前不生成推荐。
- Gate 0审计1,127个杠杆，`CAUSAL_ACTION_SUPPORTED=0`；B4为`BLOCKED_NO_CAUSAL_ACTION_SUPPORT`。
- 方案A完成1,173个M1 Lab组的静态/原料/环境交叉拟合，OOT MAE 5.246 MU；历史包络只保留影子关联含义。
- B2验证集方向0.414、降低召回0.357、delta Std比0.176，未通过。
- B3按最大精确结构模板评估：T15760 275车、B00458 186车；两个试点均为`CONTROL_TRANSFER_UNAVAILABLE`。
- B5/B6/B7全部保持阻断；没有DML、Target推荐、人工微试验或控制写入。
- 两次完整运行的A/B2/B3核心指标和预测哈希一致；详见`PROJECT_MEMORY/m1_relative_improvement_gate0_ab_v1_20260817.md`。
