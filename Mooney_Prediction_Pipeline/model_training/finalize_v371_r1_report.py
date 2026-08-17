"""Finalize the R1 report from completed artifacts without retraining."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1" / "v371_r1_temporal_group"


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows: list[list[str]] = []
    for values in frame.itertuples(index=False, name=None):
        row = []
        for value in values:
            if pd.isna(value):
                row.append("")
            elif isinstance(value, (float, np.floating)):
                row.append(f"{float(value):.4f}")
            else:
                row.append(str(value))
        rows.append(row)
    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in [headers, ["---"] * len(headers), *rows]
    )


def main() -> None:
    summary = json.loads((OUTPUT / "r1_summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(OUTPUT / "r1_metrics.csv")
    selected = summary["selected_candidate"]
    scopes = ["global", "T15760", "T15760_M1_06_002", "T15760_M1_06_002_PV46", "B00458"]
    candidates = [selected, "C3_RANDOM_KFOLD_COMPOUND_BALANCED", "baseline_recipe_train_mean"]
    focus = metrics.loc[
        metrics["scope"].isin(scopes)
        & metrics["candidate"].isin(candidates)
        & metrics["split"].isin(["val", "oot"])
    ].copy()
    focus = focus[[
        "candidate", "split", "scope", "groups", "mae", "rmse", "bias", "r2",
        "prediction_to_actual_std_ratio", "within_5_mny_rate",
    ]]
    val = focus.loc[focus["split"].eq("val")]
    get = lambda candidate, scope, column: float(
        val.loc[val["candidate"].eq(candidate) & val["scope"].eq(scope), column].iloc[0]
    )
    t_c3_mae = get("C3_RANDOM_KFOLD_COMPOUND_BALANCED", "T15760", "mae")
    t_r1_mae = get(selected, "T15760", "mae")
    t_c3_bias = get("C3_RANDOM_KFOLD_COMPOUND_BALANCED", "T15760", "bias")
    t_r1_bias = get(selected, "T15760", "bias")
    g_c3_mae = get("C3_RANDOM_KFOLD_COMPOUND_BALANCED", "global", "mae")
    g_r1_mae = get(selected, "global", "mae")

    report = f"""# V3.7.1 R1 等权 Lab 组 + 时间交叉拟合实验报告

## 本轮结论

R1 已完成，但**不通过模型放行**。选中候选为 `{selected}`：

- T15760 验证 MAE 从 C3 的 `{t_c3_mae:.3f}` 改善到 `{t_r1_mae:.3f}` MNY；Bias 从 `{t_c3_bias:+.3f}` 降到 `{t_r1_bias:+.3f}` MNY。
- 全局验证 MAE 为 `{g_r1_mae:.3f}`，与 C3 的 `{g_c3_mae:.3f}` 基本持平，没有达到至少 10% 改善门槛。
- 已查看 OOT 仅作开发诊断；它在 C1/C2/C3 阶段已被查看，不能再作为最终盲测。
- B00458 当前 OOT 只有 15 个 Lab 组，既未达到 30 组门槛，R1 表现也不稳定。
- Stage3 保持关闭；不生成推荐、不写 Recipe/PLC/生产预测表。

## 已修复的设计缺陷

1. 训练表强制一行一个 `OrderID+PalletID` Lab 组，损失权重严格为 1。
2. 移除 C3 的配方频次平衡，避免 T15760 因历史组多被系统性降权。
3. Stage1/Stage1b 和 Silica/Carbon Black 子专家改为按 `lab_time` 的 expanding-window OOF。
4. Stage1b 只使用更早的独立 Lab 事件；未知专家路线继续硬阻断。
5. 正式指标合同锁定为：先预测属于同一 Lab 结果的批次，再按 `OrderID+PalletID` 取预测均值，与 Lab 比较一次。当前 R1 先验证组聚合训练路径；批次级并行模型列为下一实验。

## 关键指标

{markdown_table(focus)}

## 为什么旧报告约 2.5–2.9 MAE 不能直接比较

旧报告包含按批次行推进的 Stage3 offset，同一个 Pallet 的标签可能在连续批次行中被重复用于校准；`lag=3` 不一定代表三个独立 Lab 反馈。旧切分也不是本轮严格时间切分。因此它代表旧系统整链结果，不等价于本轮 `raw + Stage3 off + 时间验证` 的基础模型结果。

## 下一步决定

1. R2：按 `RecipeID + ProcessVersion + MixerLine + TemplateKey` 建立层级 Phase nominal，并对新版本回退到精确 Recipe/设备/路线，不能把工程师正常调过的不同版本混成一个粗均值。
2. 同时建立 batch→group 版本：批次特征逐批预测，整个 Lab 组原子进入同一时间折，每批损失权重为 `1/组内批次数`，最后按组平均后只计一次指标。
3. R3：仅在通用模型基础上，对 T15760/B00458 增加局部残差专家；不得用试点专家掩盖全局模型失败。
4. 等待新的未查看时间窗作为真正锁定 OOT。当前模型状态保持 `NOT_RELEASED_DEVELOPMENT_DIAGNOSTIC`。

## 运行与数据边界

- 数据：APAC RDS Active-only 的冻结特征快照。
- 本次训练数据库访问：`false`。
- 本地 MMS/Archive 访问：`false`。
- Archive：`BLOCKED_ARCHIVE_NOT_REPLICATED`。
- 重复输入最大预测差：`{summary['same_input_repeat_max_abs_delta']}`。
- 训练/验证/OOT Lab 组：`{summary['train_groups']}/{summary['validation_groups']}/{summary['oot_groups']}`。
- 输入 SHA256：`{summary['source_sha256']}`。
- 模型 SHA256：`{summary['model_sha256']}`。

## 时间交叉拟合审计

```json
{json.dumps(summary['crossfit_audit'], ensure_ascii=False, indent=2)}
```
"""
    destination = OUTPUT / "R1_实验报告.md"
    destination.write_text(report, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
