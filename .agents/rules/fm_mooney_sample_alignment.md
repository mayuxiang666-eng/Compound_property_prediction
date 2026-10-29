# 终胶门尼取样与车次/托盘对齐铁律 (FM Mooney Sample & Batch/Pallet Alignment Rule)

## 核心物理与取样对应规则

1. **常规检测频次**：
   - 一个终胶（FM）订单的门尼检测（`ML1+3` / `MS1+3`）一般有 **3 次**（首托盘、中间托盘、尾托盘），有时只有 **2 次**（或偶发其他次数）。

2. **首尾精确到车次，中间精确到托盘**：
   - **首个样本（`SampleID = '001'`，位于订单第 1 个托盘 `...001`）**：精准对应**该订单的第 1 车（`BatchNumber = 1`）**。
   - **最后一个托盘的门尼样本（仅当该样本所在托盘确实是该订单的最后一个托盘 `Max(PalletID)` 时）**：精准对应**该订单的最后一车（`Max(BatchNumber)`）**。
   - **中间托盘的门尼样本**：无法精确到具体某一车，只能对应到**该样本所属的整个托盘（`PalletID`）所包含的 3~4 车终胶**，需按托盘级（或组内关联）处理。

3. **两样本订单的防错铁律（以订单 `8197867` 为实测反例）**：
   - **严禁盲目假设“第二个样本就是最后一车/最后一个托盘”**。
   - **实测证据（订单 `8197867`）**：
     - SFE `BatchHeader` 与 `PalletSample` 中该订单共有 **18 个托盘**（`8197867001` ~ `8197867018`，共 63 车，样本号 `001` ~ `099`）。
     - Redshift 实验室表 `he_datamarts.compound_excellence_datamart` 中该订单仅有 **2 条** `ML1+3` 门尼记录：
       1. `SampleID = '001'`（位于首托盘 `8197867001`，`ML1+3 = 76.37`）$\rightarrow$ 对应**第 1 车（Batch 1）**。
       2. `SampleID = '017'`（位于第 3 托盘 `8197867003`，`ML1+3 = 77.42`）$\rightarrow$ **仅是中间托盘（包含 Batch 9~11），绝非最后一个托盘（`8197867018`）**！
   - **强制校验流程**：
     1. 先查 SFE 数据库 `PalletSample` 和 `BatchHeader`，获取该订单的总托盘数、首托盘号 `Min(PalletID)`、尾托盘号 `Max(PalletID)`、首车 `Batch 1`、尾车 `Max(BatchNumber)`，以及各 `SampleID` 对应的真实 `PalletID`。
     2. 再将实验室检测结果中的 `(order_id, pallet_id, sample_id)` 与 SFE 的托盘拓扑比对，严格判定它是：
        - **首车样本**（`SampleID == '001'` 且 `PalletID == Min(PalletID)` $\rightarrow$ 1:1 绑定 `BatchNumber = 1`）
        - **尾车样本**（`PalletID == Max(PalletID)` 且为该尾托盘门尼样本 $\rightarrow$ 1:1 绑定 `BatchNumber = Max(BatchNumber)`）
        - **中间托盘样本**（`Min(PalletID) < PalletID < Max(PalletID)` $\rightarrow$ 绑定该 `PalletID` 下的所属车次集合）

4. **分析阶段优先级（先终胶自闭环，后母胶溯源）**：
   - 母胶及更前置输入属于终胶的输入边界条件，其物理影响会直接反映在终胶自身的密炼过程曲线（温度、功率、压栓、转速）中。
   - 第一步必须先单独做**“终胶密炼过程曲线 $\rightarrow$ 终胶门尼检测结果”**的直接相关性与因果/条件依赖分析；验证并定位终胶门尼波动对应终胶曲线的具体变化后，再向母胶延伸。

5. **保留全部有效检测结果并引入公差与目标值（严禁盲目剔除超差/非 Passed 样本）**：
   - 只要 `test_result` 为有效数值，**无论 `test_status` 是什么（Passed / Failed / Warning 等），一律作为有效输入保留**，因为检测结果的好坏本身就是密炼过程曲线异常或波动的直接物理结果。
   - 必须从 Redshift `he_datamarts.compound_excellence_datamart` 中同步提取：
     - `test_target`（目标中心值）
     - `tolerance_lower` / `tolerance_upper`（公差上下限，如 71 ~ 91）
     - `warning_lower` / `warning_upper`（预警上下限，如 73 ~ 89）
   - 基于上述字段构建门尼偏离度（`test_result - test_target`、`|test_result - test_target|`）及质量判定分档（正常区间、预警区间、超差区间）进行联合分析。
