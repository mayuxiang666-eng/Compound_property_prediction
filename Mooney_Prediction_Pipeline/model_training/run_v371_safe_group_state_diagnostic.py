"""Replay prior-Lab state with production/feedback-time availability checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training import run_v371_r1_temporal_group as r1
from Mooney_Prediction_Pipeline.model_training.group_metric_contract import metric_block
from Mooney_Prediction_Pipeline.model_training.stage3_group_calibration import (
    replay_group_stage3,
)


OUTPUT = r1.DEFAULT_BASE / "v371_safe_group_state_diagnostic"
R1_OUTPUT = r1.DEFAULT_BASE / "v371_r1_temporal_group"
R1B_OUTPUT = r1.DEFAULT_BASE / "v371_r1b_group_atomic_batch"
BATCH_SOURCE = r1.DEFAULT_BASE / "phase_batch_features_active_v1.csv"


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if pd.isna(value):
                rendered.append("")
            elif isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.4f}")
            else:
                rendered.append(str(value))
        rows.append(rendered)
    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in [headers, ["---"] * len(headers), *rows]
    )


def production_times() -> pd.DataFrame:
    batch = pd.read_csv(
        BATCH_SOURCE,
        usecols=["LabelGroupID", "BatchStartTime", "BatchEndTime"],
        low_memory=False,
    )
    batch["BatchStartTime"] = pd.to_datetime(batch["BatchStartTime"], errors="coerce", utc=True)
    batch["BatchEndTime"] = pd.to_datetime(batch["BatchEndTime"], errors="coerce", utc=True)
    return batch.groupby("LabelGroupID", as_index=False).agg(
        GroupProductionStart=("BatchStartTime", "min"),
        GroupProductionComplete=("BatchEndTime", "max"),
    )


def load_predictions() -> pd.DataFrame:
    outputs = []
    r1_predictions = pd.read_csv(
        R1_OUTPUT / "r1_selected_group_predictions.csv", low_memory=False
    )
    r1_predictions["Model"] = "R1_GROUP_AGGREGATE_RAW"
    outputs.append(r1_predictions)

    r1b = pd.read_csv(R1B_OUTPUT / "r1b_group_predictions.csv", low_memory=False)
    r1b = r1b.loc[r1b["Candidate"].eq("R1B_GROUP_ATOMIC_BATCH")].copy()
    r1b["Model"] = "R1B_BATCH_TO_GROUP_RAW"
    outputs.append(r1b)

    for split, filename in (("val", "c3_val_predictions.csv"), ("oot", "c3_oot_predictions.csv")):
        c3 = pd.read_csv(r1.DEFAULT_BASE / filename, low_memory=False)
        selected = c3[[
            "LabelGroupID", "OrderID", "PalletID", "CompoundName", "RecipeID",
            "ProcessVersion", "MixerLine", "material_system", "phase_route", "TemplateKey",
            "MNY", "lab_time", "split",
        ]].copy()
        selected["RawPrediction"] = c3["C3Prediction"]
        selected["Model"] = "C3_RAW"
        outputs.append(selected)
    common = sorted(set.intersection(*(set(frame.columns) for frame in outputs)))
    required = [
        "LabelGroupID", "CompoundName", "RecipeID", "MixerLine", "phase_route",
        "MNY", "lab_time", "split", "RawPrediction", "Model",
    ]
    if not set(required).issubset(common):
        raise ValueError(f"Prediction artifacts lack state diagnostic columns: {set(required)-set(common)}")
    return pd.concat([frame[common] for frame in outputs], ignore_index=True)


def scope_mask(frame: pd.DataFrame, scope: str) -> pd.Series:
    compound = frame["CompoundName"].astype(str).str.upper()
    if scope == "global":
        return pd.Series(True, index=frame.index)
    if scope == "T15760":
        return compound.str.contains("T15760", regex=False)
    if scope == "B00458":
        return compound.str.contains("B00458", regex=False)
    raise ValueError(scope)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions().merge(
        production_times(), on="LabelGroupID", how="left", validate="many_to_one"
    )
    predictions["FeedbackTime"] = pd.to_datetime(predictions["lab_time"], errors="coerce", utc=True)
    predictions["PredictionTime"] = pd.to_datetime(
        predictions["GroupProductionComplete"], errors="coerce", utc=True
    )
    predictions["ActualMNY"] = pd.to_numeric(predictions["MNY"], errors="coerce")
    predictions["GroupRawPrediction"] = pd.to_numeric(
        predictions["RawPrediction"], errors="coerce"
    )
    predictions["CalibrationScope"] = (
        predictions["RecipeID"].astype(str)
        + "|" + predictions["MixerLine"].astype(str)
        + "|" + predictions["phase_route"].astype(str)
    )
    if predictions[["PredictionTime", "FeedbackTime"]].isna().any().any():
        raise ValueError("BLOCKED_MISSING_PREDICTION_OR_FEEDBACK_TIME")

    replay_outputs = []
    metric_rows = []
    for model_name, model_frame in predictions.groupby("Model", sort=True):
        model_frame = model_frame.sort_values(
            ["PredictionTime", "LabelGroupID"], kind="stable"
        ).reset_index(drop=True)
        raw_metric_frame = model_frame.rename(
            columns={"ActualMNY": "MetricActual", "GroupRawPrediction": "MetricPrediction"}
        )
        for split in ("val", "oot"):
            for scope in ("global", "T15760", "B00458"):
                subset = raw_metric_frame.loc[
                    raw_metric_frame["split"].eq(split) & scope_mask(raw_metric_frame, scope)
                ]
                block = metric_block(
                    subset, actual_col="MetricActual", prediction_col="MetricPrediction"
                )
                metric_rows.append(
                    {"model": model_name, "history_groups": 0, "split": split, "scope": scope,
                     "support_policy": "RAW_NO_STATE", **block}
                )
        for history in (3, 5, 10):
            replay = replay_group_stage3(model_frame, history_groups=history, max_abs_offset=5.0)
            replay["Model"] = model_name
            replay["HistoryGroups"] = history
            replay_outputs.append(replay)
            for split in ("val", "oot"):
                for scope in ("global", "T15760", "B00458"):
                    selected = replay.loc[
                        replay["split"].eq(split) & scope_mask(replay, scope)
                    ].copy()
                    all_block = metric_block(
                        selected,
                        actual_col="ActualMNY",
                        prediction_col="CalibratedPrediction",
                    )
                    metric_rows.append(
                        {"model": model_name, "history_groups": history, "split": split,
                         "scope": scope, "support_policy": "ALL_GROUPS_ZERO_OFFSET_UNTIL_READY",
                         **all_block}
                    )
                    ready = selected.loc[selected["Stage3SupportGroupCount"].ge(history)]
                    ready_block = metric_block(
                        ready,
                        actual_col="ActualMNY",
                        prediction_col="CalibratedPrediction",
                    )
                    metric_rows.append(
                        {"model": model_name, "history_groups": history, "split": split,
                         "scope": scope, "support_policy": "READY_GROUPS_ONLY", **ready_block}
                    )

    replays = pd.concat(replay_outputs, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    availability = predictions.assign(
        FeedbackDelayHours=(predictions["FeedbackTime"] - predictions["PredictionTime"]).dt.total_seconds() / 3600.0
    )[[
        "Model", "LabelGroupID", "PredictionTime", "FeedbackTime", "FeedbackDelayHours",
        "CalibrationScope", "split",
    ]]
    replays.to_csv(OUTPUT / "safe_group_state_replay.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT / "safe_group_state_metrics.csv", index=False, encoding="utf-8-sig")
    availability.to_csv(OUTPUT / "feedback_availability_audit.csv", index=False, encoding="utf-8-sig")

    focus = metrics.loc[
        metrics["model"].isin(["R1_GROUP_AGGREGATE_RAW", "R1B_BATCH_TO_GROUP_RAW", "C3_RAW"])
        & metrics["split"].isin(["val", "oot"])
        & metrics["scope"].isin(["global", "T15760", "B00458"])
        & (
            metrics["history_groups"].eq(0)
            | (metrics["history_groups"].eq(5) & metrics["support_policy"].eq("ALL_GROUPS_ZERO_OFFSET_UNTIL_READY"))
        )
    ][[
        "model", "history_groups", "split", "scope", "groups", "mae", "rmse", "bias", "r2",
        "prediction_to_actual_std_ratio",
    ]]
    delays = availability.drop_duplicates(["LabelGroupID"]).agg(
        groups=("LabelGroupID", "nunique"),
        median_delay_hours=("FeedbackDelayHours", "median"),
        p95_delay_hours=("FeedbackDelayHours", lambda x: x.quantile(0.95)),
        nonpositive_delay_count=("FeedbackDelayHours", lambda x: int((x <= 0).sum())),
    ).to_frame().T
    report = f"""# V3.7.1 安全 Lab 组状态先验诊断

## 目的

检查旧 V3.7 低 MAE 中动态校准的真实贡献，同时消除同一 Pallet 多批次重复消费 Lab 的泄漏。每个 `OrderID+PalletID` 只是一条反馈事件；预测当前组时，只允许使用 `FeedbackTime <= 当前组生产完成时间` 的更早独立 Lab 组。

## 关键指标

`history_groups=0` 是 raw；`history_groups=5` 是过去 5 个已到达独立 Lab 组误差中位数，单个 offset 限制在 ±5 MNY。所有候选仍为诊断，`Stage3Enabled=false`。

{markdown_table(focus)}

## 反馈延迟审计

{markdown_table(delays)}

## 判定

- 若安全 5 组状态显著改善 raw，旧 2.5 左右 MAE 的一部分确实来自动态状态，而不是基础模型本身。
- 若 ready group 很少，说明 Lab 延迟使实时状态不可部署，不能以回放改善宣称上线能力。
- 该状态层不参与 R1/R1B 基础模型选择，Stage3 继续关闭。

## 边界

- 仅使用已冻结 Active-only CSV；数据库访问：`false`。
- 本地 MMS/Archive：`false`；自动控制：`false`。
"""
    (OUTPUT / "安全组级状态诊断报告.md").write_text(report, encoding="utf-8")
    summary = {
        "contract": "PRIOR_INDEPENDENT_LAB_GROUPS_AVAILABLE_BY_PRODUCTION_COMPLETE_TIME",
        "stage3_enabled": False,
        "rows": int(len(replays)),
        "groups": int(predictions["LabelGroupID"].nunique()),
        "models": sorted(predictions["Model"].unique()),
        "local_mms_access": False,
        "database_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        **summary,
        "report": str(OUTPUT / "安全组级状态诊断报告.md").encode("ascii", "backslashreplace").decode("ascii"),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
