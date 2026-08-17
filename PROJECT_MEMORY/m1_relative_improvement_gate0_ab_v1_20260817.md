# M1 相对门尼改善 Gate 0 / A / B2 / B3 V1

日期：2026-08-17

## 范围与安全边界

- 仅 M1；Stage3关闭；能耗不参与排序；不写Recipe/PLC。
- 只读冻结快照，运行期间数据库访问、本地MMS访问和Archive访问均为0。
- Lab评价单位仍为 `OrderID+PalletID`；没有把同一个Lab标签按批次数重复计权。
- 主报告：`reports/v371_challenger/m1_relative_improvement_gate0_ab_v1/M1相对门尼改善_Gate0_A_B2_B3实验报告.md`。

## Gate 0

- 审计1,127个按配方、版本、设备、结构模板和Step杠杆分层的动作行。
- 状态：424个`LOCKED_EXPERT_RULE`、365个`INSUFFICIENT_LAB_SUPPORT`、338个`FIXED_IN_HISTORY`。
- `CAUSAL_ACTION_SUPPORTED=0`，因此DML没有创建或运行，状态为`BLOCKED_NO_CAUSAL_ACTION_SUPPORT`。
- T15760在同一结构模板内Target基本固定；跨版本变化不能冒充同模板自然实验。
- B00458旧Step快照订单与当前Lab/Pallet快照无物理键交集，仅可用于无Lab的B3过程响应，Gate 0必须阻断。

## 方案 A

- 1,173个M1 Lab组完成训练折交叉拟合静态/原料/环境基线。
- 静态基线OOT MAE 5.246 MU，Bias -1.073 MU。
- 7个模板满足当前历史包络统计条件，但统一保持`HISTORICAL_ENVELOPE_SHADOW_ONLY`；不得解释为因果工艺最优或具体MU收益。

## B2

- train/val/OOT严格同订单同版本同模板转移数为147/36/29。
- 验证集方向准确率0.414、降低方向召回0.357、delta Std比0.176、P10-P90覆盖率0.667。
- OOT方向准确率0.762，但召回0.600、delta Std比0.229、覆盖率0.724；验证集失败不能被OOT单项方向覆盖。
- 结论：`B2_GATE_FAILED`。

## B3

- T15760最大精确结构模板：275车；Dry终点/油后/排胶温度MAE为6.105/3.433/3.786°C，失败。
- B00458主模板`0A497A75CABE`：186车；Dry/油后/Wet终点温度MAE为2.195/2.336/3.427°C，PID duration MAE 18.912s，失败。
- 部分输出通过不代表完整路径通过；当前还没有覆盖受动作影响的全部V3.7 Phase均值、积分和执行特征。
- 两个试点均为`CONTROL_TRANSFER_UNAVAILABLE`。

## 停止决策

- B4：`BLOCKED_NO_CAUSAL_ACTION_SUPPORT`。
- B5：`BLOCKED_PREREQUISITE_GATES`，不生成Target向量。
- B6：未开始。
- B7：`BLOCKED_DUAL_PILOT_GATE`，不允许人工微试验。
- 扩展配方T09170、T33025W3、B00460B、B15563R1缺少同合同Step快照，逐个标记`RECIPE_SPECIFIC_POLICY_REQUIRED`。

## 可重复性

同一快照连续两次运行，以下SHA256完全一致：

- B2 metrics：`C2C071D5837FED52A67F7721884DE7957311E2816E3129D7821CF1D87AC56CD2`
- B2 predictions：`DA4E0434E1EAE9C4E8A8D60704AB4E353054F0BE24CE4B68FD729825503CC207`
- B3 metrics：`FC54060C33DACC98498DFC186EAB7DCC52793BBAB2E75FB6D2A4B94D9D1812B0`
- B3 predictions：`D070DB3A06115FDB5342B74B294D235D9AD5BBCCD5413B8DFF229BCA5DBEEE9E`
- A envelope audit：`DEEC57268C34FB3575906D465F18869939E5E804F8AA23961E57999CEDDBD21C`

## 下一允许动作

1. 补齐B00458同源Pallet标签桥，不读取本地Archive。
2. 将B3扩展到实际Phase均值、积分、duration/rotor执行结果及终点，而非只预测终点。
3. Gate 0仍无因果支持时，继续A+B2关联诊断；不得开发DML或进入推荐搜索。
