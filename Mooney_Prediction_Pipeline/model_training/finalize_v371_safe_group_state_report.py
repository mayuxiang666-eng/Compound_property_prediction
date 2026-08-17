"""Finalize the safe state replay report from already-computed CSV artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1" / "v371_safe_group_state_diagnostic"


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = []
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
    metrics = pd.read_csv(OUTPUT / "safe_group_state_metrics.csv")
    availability = pd.read_csv(OUTPUT / "feedback_availability_audit.csv")
    focus = metrics.loc[
        metrics["model"].isin(["R1_GROUP_AGGREGATE_RAW", "R1B_BATCH_TO_GROUP_RAW", "C3_RAW"])
        & metrics["split"].isin(["val", "oot"])
        & metrics["scope"].isin(["global", "T15760", "B00458"])
        & (
            metrics["history_groups"].eq(0)
            | (metrics["history_groups"].eq(5)
               & metrics["support_policy"].eq("ALL_GROUPS_ZERO_OFFSET_UNTIL_READY"))
        )
    ][[
        "model", "history_groups", "split", "scope", "groups", "mae", "rmse", "bias", "r2",
        "prediction_to_actual_std_ratio",
    ]]
    unique = availability.drop_duplicates(["LabelGroupID"]).copy()
    delay = pd.to_numeric(unique["FeedbackDelayHours"], errors="coerce")
    delay_summary = pd.DataFrame([
        {
            "groups": int(unique["LabelGroupID"].nunique()),
            "median_delay_hours": float(delay.median()),
            "p95_delay_hours": float(delay.quantile(0.95)),
            "nonpositive_delay_count": int((delay <= 0).sum()),
        }
    ])
    ready = metrics.loc[
        metrics["history_groups"].eq(5)
        & metrics["support_policy"].eq("READY_GROUPS_ONLY")
        & metrics["split"].isin(["val", "oot"])
        & metrics["scope"].isin(["global", "T15760", "B00458"])
    ][["model", "split", "scope", "groups", "mae", "rmse", "bias"]]
    report = f"""# V3.7.1 安全 Lab 组状态先验诊断

## 结论

本报告只使用当前组生产完成时已经返回的、更早且独立的 Lab 组。一个 Pallet 无论有多少批次都只消费一次反馈；当前组不能校准自身。`history_groups=5` 使用过去 5 个可用组误差的中位数，offset 限制为 ±5 MNY。

Stage3 仍为 `false`，所有结果只用于解释旧 V3.7 与当前 raw 模型的差距。

## Raw 与安全 5 组状态对照

{markdown_table(focus)}

## 已具备 5 个历史反馈组的子集

{markdown_table(ready)}

## Lab 反馈延迟

{markdown_table(delay_summary)}

## 解释规则

- 如果安全状态明显改善 raw，说明旧报告约 2.5–2.9 MAE 的一部分来自动态状态，而非基础四阶段模型本身。
- 如果可用历史组不足或反馈延迟过长，则这种改善不能转化为下一车次能力。
- 动态状态不得替代 raw 模型放行，也不能掩盖 T15760/B00458 的负 R² 或振幅不足。

## 数据边界

- 只读取冻结 Active-only CSV；数据库访问：`false`。
- 本地 MMS/Archive：`false`；Archive：`BLOCKED_ARCHIVE_NOT_REPLICATED`。
- 自动推荐/控制：`false`。
"""
    destination = OUTPUT / "安全组级状态诊断报告.md"
    destination.write_text(report, encoding="utf-8")
    summary = {
        "contract": "PRIOR_INDEPENDENT_LAB_GROUPS_AVAILABLE_BY_PRODUCTION_COMPLETE_TIME",
        "stage3_enabled": False,
        "groups": int(unique["LabelGroupID"].nunique()),
        "median_feedback_delay_hours": float(delay.median()),
        "p95_feedback_delay_hours": float(delay.quantile(0.95)),
        "local_mms_access": False,
        "database_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "report": str(destination),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**summary, "report": str(destination).encode("ascii", "backslashreplace").decode("ascii")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
