# 门尼预测项目长期知识库

这是本项目的本地、可追溯知识库，供后续 Agent 继续工作时读取。

## 使用规则

1. 先读取 `current_state.md`、`decision_log.md` 和 `experiment_index.csv`。
2. 新实验只追加文件，不覆盖既有实验结论。
3. 每个实验必须记录：数据快照、代码版本/路径、输入 SHA256、评价口径、结果、失败原因和下一步。
4. 不保存数据库密码、连接字符串中的密码、Token 或个人凭据。
5. 所有模型结论必须区分：开发诊断、影子验证、生产放行。
6. 未经明确授权，不访问本地 MMS/Archive，不写 Recipe、PLC 或生产数据库。

## 当前知识入口

- [当前状态](current_state.md)
- [永久决策日志](decision_log.md)
- [逐次改动登记](change_log.md)
- [实验索引](experiment_index.csv)
- [数据与模型合同](contracts/data_model_contract.md)

## 当前核心目标

- Lab 组级评价口径正确：逐车次预测后按同一 Lab 组求均值，再与唯一 MNY 比较。
- OOT MAE 目标：`≤2.5 MU`。
- 同时捕捉同订单/同组内车次门尼变化趋势。
- Stage3 在线校准关闭；能耗模型未完成前只审计；不自动控制。
