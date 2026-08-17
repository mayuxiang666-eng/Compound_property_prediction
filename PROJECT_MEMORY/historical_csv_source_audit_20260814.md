# 本地历史 CSV 补充源审计（2026-08-14）

## 审计边界

- 仅只读检查项目 `data/` 目录及现有本地快照。
- 未访问本地 MMS、SFEPLANT_ARCHIVE 或任何生产数据库。
- 当前对照快照：`m1_only_batch_pid_quality_features.csv`，4,100 个 M1 物理批次，时间范围 2025-12-01 至 2026-08-11。
- 物理批次键统一为 `normalize(OrderID) + BatchNumber`；历史 `OrderID` 的 `YYYY~` 前缀只作为来源标记，比较时移除。

## 文件结论

### 1. `data/stage_statistics_enriched.csv`：首选历史候选源，但必须隔离重建

- SHA256：`102EE42ED07E19C77723FE69E0408898ADAD03B1538F598BC626E927ED60F892`
- 23,740 行，规范化后 23,740 个唯一物理批次、5,651 个订单、7,268 个 `OrderID+PalletID` Lab 组。
- 23,737 行属于 M1；时间范围 2024-05-26 至 2026-06-28。
- 与当前 4,100 批次快照重叠 3,539 批次；新增候选 20,201 批次；当前快照起点之前有 19,429 批次。
- 含每批独立的 `temp/power/Torque/RotorSpeed/WayofRam` 原始十六进制曲线，以及 RecipeID、配方静态信息、PalletID 和 MNY。
- 12 行缺少至少一条主要曲线；温度/功率/扭矩/转速长度有 71 行不一致，必须进入物理质量阻断或人工审计。
- 五信号完整曲线没有发现跨批复制；唯一重复的曲线哈希来自 11 行五信号全部缺失记录。
- 文件不含 disposition code，不能直接证明标签均为 001/011。
- 旧流水线对 Lab 复测使用均值聚合，当前合同要求使用最新有效结果并保留修订/撤销审计，因此 CSV 内 MNY 只能视为候选标签，需与可信 Lab 快照重验。
- 文件缺少完整 BatchData Step 表、逐 Step Target/Value、SCP-LOG AND/OR 和执行 Target 向量；可用于重建门尼 Phase/曲线模型，但不能单独支撑 Step→Phase 推荐过程响应模型。

决定：将此文件作为历史原始曲线候选源。禁止把旧派生特征直接拼接；必须逐批独立解码，以当前无泄漏构建器重新生成 Phase 特征，保留 NaN，禁止跨批/未来均值填补，并与当前快照跨源去重。

### 2. `data/stage_statistics_enriched_all_features_weather_v4.csv`：仅作旧模型复现和差异审计

- SHA256：`1536E47E99A6F04CB9682770D74AB8D2EF7A6E1B90A4A2E2B79797E7C6ACBB96`
- 18,783 个唯一批次、5,668 个 Lab 组；时间范围 2024-05-30 至 2026-06-28。
- 与当前快照重叠 2,639 批次；新增候选 16,144 批次。
- T15760 的 `silica_phr` 为约 272.4/274.4，仍包含已确认的重复求和错误，正确合同应约为 136.2（按版本重算，不能硬编码）。
- 旧 `curve_segmenter_all_compounds.py` 会按胶料同时间点均值填补异常曲线，并删除订单内 MNY 极差大于 10 MU 的订单；两者均违反当前“禁止跨批填曲线、MNY 波动不作为删除条件”的合同。
- 与当前修正快照 2,639 个重叠批次比较，Silica PHR、Stage6 ram、天气、比能量等多列存在显著差异。

决定：禁止将该文件直接并入 V3.7.1 训练或半监督表示学习。只允许用于冻结 V3.7 复现、旧新特征差异和历史覆盖索引。

### 3. `data/WO_ratio_s156.csv`：专用辅助特征源

- SHA256：`970112DD5D3CA009B9CDCA7C7A076A64EDBFE0DFB37EE80193E8018C928A0620`
- 99,330 行，535 个 FMF 订单、32,720 个订单批次组合；时间范围 2025-06-22 至 2026-07-23。
- 一批多行是物料/来源 WO 明细，不是重复批次事实。
- 只覆盖两台设备和 S156/回用胶相关物料谱系，不含通用 MNY Lab 标签及完整密炼 Phase 曲线。

决定：不能扩大通用门尼监督样本；仅可在相关 M1 配方中作为回用胶/来源批次比例辅助特征，且必须按实际物料重量聚合并做覆盖率消融。

## 试点历史覆盖

`stage_statistics_enriched.csv` 按胶料代码汇总（含多 RecipeID/版本）：

| 胶料代码 | 物理批次 | Lab 组 |
|---|---:|---:|
| T15760 | 2,928 | 949 |
| B00458 | 3,133 | 975 |
| T09170 | 5,473 | 1,362 |
| T33025 | 3,815 | 1,012 |
| B00460 | 533 | 195 |
| B15563 | 182 | 58 |

这些数字是候选覆盖，不等于可训练样本。正式样本数必须在 disposition、Lab 修订状态、物理曲线质量、模板一致性、M1 范围和固定 OOT 排除后重新计算。

## 尚未完成审计的 OneDrive 云占位文件

同步进程关闭时以下文件返回 `The cloud file provider is not running`，本次未读取内容，不能作出训练放行判断：

- `data/enriched_mny_all.csv`
- `data/M1-B00460B--- 01 003.csv`
- `data/oilload_mapping_master_batch.csv`
- `data/stage_statistics.csv`

其中 `oilload_mapping_master_batch.csv` 在旧部署文档中定义为母胶含油量与炭黑分类映射，理论上是辅助映射，不是独立 Lab 历史；仍需文件落地后按字段和主键实审。

## 下一步数据接入顺序

1. 不修改固定的 211 组当前 OOT；任何重叠批次按当前修正快照优先并去重。
2. 从 `stage_statistics_enriched.csv` 建立隔离的历史候选清单，先排除非 M1、缺失/长度不一致曲线及当前 val/OOT 组。
3. 用当前无泄漏曲线构建器逐批重建 Phase；不得调用旧 `curve_segmenter_all_compounds.py` 的跨批均值填补和 MNY 极差删除逻辑。
4. 对历史标签重新确认 001/011、最新有效复测、修订和撤销；确认前状态为 `BLOCKED_UNVERIFIED_HISTORICAL_LAB`。
5. Silica PHR、天气、QM 和 ProcessVersion/Target 合同全部使用当前构建逻辑重算，禁止复用旧 V4 数值。
6. 先做“历史重建 + 固定 OOT”的 C1 对照；确认 MAE、组内趋势和波动捕捉改善后，再将这些批次用于多实例/半监督模型。
7. Step 推荐仍只使用具备完整 Step/Target/SCP-LOG/模板证据的数据；原始曲线 CSV 不足以替代 Step 历史。
