# 当前数据与模型合同

## 评价单位

`LabelGroupID = OrderID + PalletID` 是 Lab 标签单位。一个 Lab 结果对应多个车次时：

```text
每个 Batch 独立预测
→ 按 LabelGroupID 求预测均值
→ 与该组唯一 MNY 比较一次
```

不能把同一个 MNY 复制到每个 Batch 后直接计算车次级 MAE。

## 物理主键

`OrderID + BatchNumber`

## 推荐/反事实路径

```text
Step Target 向量
→ 过程响应模型
→ 预测实际过程变量
→ Phase 特征
→ V3.7.1 raw prediction
```

禁止直接把 Target 当成实际 Phase 特征。

## 缺失语义

- 物理不适用：允许显式物理零，并带 applicability 状态。
- 来源缺失：保留 NaN 和 missing flag。
- 不能把来源缺失统一填成 0。

## 质量语义

- `CURVE_PHYSICAL_INVALID`：不进入主模型总体。
- `CURVE_STATISTICAL_OUTLIER`：进入模型，增加风险字段。
- `CURVE_VALID`：正常过程质量。

## 模型放行

- Stage3 校准关闭。
- 自动控制关闭。
- 能耗不参与资格和排序。
- 当前模型为开发诊断，未满足 2.5 MU 目标前不得生成 MANUAL_REVIEW_ONLY 推荐。
