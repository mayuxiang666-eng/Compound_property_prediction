"""Train the batch-predict-then-Lab-group-mean V3.7.1 challenger."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training import run_v371_r1_temporal_group as r1
from Mooney_Prediction_Pipeline.model_training.batch_phase_training_contract import (
    build_batch_phase_training_frame,
)
from Mooney_Prediction_Pipeline.model_training.group_metric_contract import (
    GROUP_METRIC_CONTRACT,
    aggregate_batch_predictions,
)
from Mooney_Prediction_Pipeline.model_training.v371_group_atomic_batch_v1 import (
    V371GroupAtomicBatchModel,
)


BATCH_SOURCE = Path(os.environ.get("V371_R1B_BATCH_SOURCE", str(r1.DEFAULT_BASE / "phase_batch_features_active_v1.csv")))
OUTPUT = Path(os.environ.get("V371_R1B_OUTPUT", str(r1.DEFAULT_BASE / "v371_r1b_group_atomic_batch")))
MODEL_OUTPUT = (
    Path(os.environ.get("V371_R1B_MODEL_OUTPUT", str(ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r1b_group_atomic_batch")))
)
R1_OUTPUT = Path(os.environ.get("V371_R1B_R1_OUTPUT", str(r1.DEFAULT_BASE / "v371_r1_temporal_group")))
GROUP_FRAME = Path(os.environ.get("V371_R1B_GROUP_FRAME", str(r1.DEFAULT_FRAME)))
METADATA = Path(os.environ.get("V371_R1B_METADATA", str(r1.DEFAULT_C3_METADATA)))
SCOPE = os.environ.get("V371_R1B_SCOPE", "GLOBAL_ACTIVE")
MODEL_NAME = os.environ.get(
    "V371_R1B_MODEL_NAME",
    "V3.7.1-Corrected-Challenger-R1B-Group-Atomic-Batch",
)


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


def score_batches(model, frame: pd.DataFrame, candidate: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    final, stage1, stage1b, stage2 = model.predict(frame, cluster_col="material_system")
    columns = [
        column for column in (
            "OrderID", "BatchNumber", "PalletID", "LabelGroupID", "CompoundName",
            "RecipeID", "ProcessVersion", "MixerLine", "material_system", "phase_route",
            "TemplateKey", "MNY", "lab_time", "split",
        ) if column in frame.columns
    ]
    batches = frame[columns].copy()
    batches["Candidate"] = candidate
    batches["ActualMNY"] = batches["MNY"]
    batches["Stage1Prediction"] = stage1
    batches["Stage1bBias"] = stage1b
    batches["Stage2Residual"] = stage2
    batches["RawPrediction"] = final
    batches["BatchErrorAgainstSharedLab"] = batches["RawPrediction"] - batches["MNY"]

    groups = aggregate_batch_predictions(
        batches,
        group_col="LabelGroupID",
        actual_col="ActualMNY",
        prediction_col="RawPrediction",
    )
    components = batches.groupby("LabelGroupID", as_index=False).agg(
        GroupStage1Prediction=("Stage1Prediction", "mean"),
        GroupStage1bBias=("Stage1bBias", "mean"),
        GroupStage2Residual=("Stage2Residual", "mean"),
    )
    groups = groups.merge(components, on="LabelGroupID", how="left", validate="one_to_one")
    groups["Candidate"] = candidate
    groups["MNY"] = groups["ActualMNY"]
    groups["RawPrediction"] = groups["GroupRawPrediction"]
    groups["Error"] = groups["RawPrediction"] - groups["MNY"]
    return batches, groups


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    group_frame, metadata = r1.load_contract(GROUP_FRAME, METADATA)
    batch_source = pd.read_csv(BATCH_SOURCE, low_memory=False)
    batch_frame, s1_features, s2_features, batch_contract = build_batch_phase_training_frame(
        batch_source, group_frame, metadata
    )
    train = batch_frame.loc[batch_frame["split"].eq("train")].copy()
    validation = batch_frame.loc[batch_frame["split"].eq("val")].copy()
    oot = batch_frame.loc[batch_frame["split"].eq("oot")].copy()

    started = time.perf_counter()
    model = V371GroupAtomicBatchModel(
        shrinkage_k=5.0,
        use_cluster_experts=True,
        use_material_route_matrix=True,
        use_silica_subsystem=True,
        use_cb_subsystem=True,
        crossfit_splits=5,
        time_col="lab_time",
        objective="regression",
    ).fit(
        train,
        s1_features,
        s2_features,
        target_col="MNY",
        cluster_col="material_system",
    )
    fit_seconds = time.perf_counter() - started

    batch_outputs = []
    group_outputs = []
    metrics = []
    for split_frame in (train, validation, oot):
        batches, groups = score_batches(model, split_frame, "R1B_GROUP_ATOMIC_BATCH")
        batch_outputs.append(batches)
        group_outputs.append(groups)
        metrics.extend(r1.metric_rows(groups, "R1B_GROUP_ATOMIC_BATCH", str(groups["split"].iloc[0])))

    r1_predictions = pd.read_csv(
        R1_OUTPUT / "r1_selected_group_predictions.csv", low_memory=False
    )
    allowed_groups = set(group_frame["LabelGroupID"].astype(str))
    r1_predictions["LabelGroupID"] = r1_predictions["LabelGroupID"].astype(str)
    r1_predictions = r1_predictions.loc[
        r1_predictions["LabelGroupID"].isin(allowed_groups)
    ].copy()
    r1_predictions["Candidate"] = "R1_GROUP_AGGREGATE_MODEL"
    r1_metrics = []
    for split in ("train", "val", "oot"):
        subset = r1_predictions.loc[r1_predictions["split"].eq(split)].copy()
        r1_metrics.extend(r1.metric_rows(subset, "R1_GROUP_AGGREGATE_MODEL", split))
    baseline_predictions, baseline_metrics = r1.baseline_metric_rows(group_frame)
    c3_predictions, c3_metrics = r1.c3_metric_rows(r1.DEFAULT_BASE)
    if "LabelGroupID" in c3_predictions.columns:
        c3_predictions["LabelGroupID"] = c3_predictions["LabelGroupID"].astype(str)
        c3_predictions = c3_predictions.loc[
            c3_predictions["LabelGroupID"].isin(allowed_groups)
        ].copy()
        c3_metrics = pd.DataFrame(
            sum(
                [
                    r1.metric_rows(c3_predictions.loc[c3_predictions["split"].eq(split)], "C3_RANDOM_KFOLD_COMPOUND_BALANCED", split)
                    for split in ("train", "val", "oot")
                    if not c3_predictions.loc[c3_predictions["split"].eq(split)].empty
                ],
                [],
            )
        )
    metric_frame = pd.concat(
        [pd.DataFrame(metrics), pd.DataFrame(r1_metrics), baseline_metrics, c3_metrics],
        ignore_index=True,
    )

    all_batches = pd.concat(batch_outputs, ignore_index=True)
    all_groups = pd.concat(
        group_outputs + [r1_predictions, baseline_predictions, c3_predictions],
        ignore_index=True,
        sort=False,
    )
    all_batches.to_csv(OUTPUT / "r1b_batch_predictions.csv", index=False, encoding="utf-8-sig")
    all_groups.to_csv(OUTPUT / "r1b_group_predictions.csv", index=False, encoding="utf-8-sig")
    metric_frame.to_csv(OUTPUT / "r1b_metrics.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "r1b_batch_feature_contract.json").write_text(
        json.dumps(batch_contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model_path = MODEL_OUTPUT / "hybrid_model_shadow.joblib"
    joblib.dump(model, model_path)
    repeat_delta = float(
        np.max(np.abs(model.predict(validation)[0] - model.predict(validation)[0]))
    )
    summary = {
        "model_name": MODEL_NAME,
        "scope": SCOPE,
        "contract_version": model.contract_version,
        "release_status": "NOT_RELEASED_DEVELOPMENT_DIAGNOSTIC",
        "training_unit": "BATCH_ROWS_WITH_ONE_EFFECTIVE_WEIGHT_PER_LAB_GROUP",
        "metric_contract": GROUP_METRIC_CONTRACT,
        "stage3_calibration_enabled": False,
        "crossfit_audit": model.crossfit_audit_,
        "batch_feature_contract": batch_contract,
        "fit_seconds": fit_seconds,
        "same_input_repeat_max_abs_delta": repeat_delta,
        "train_batches": int(len(train)),
        "validation_batches": int(len(validation)),
        "oot_batches": int(len(oot)),
        "train_groups": int(train["LabelGroupID"].nunique()),
        "validation_groups": int(validation["LabelGroupID"].nunique()),
        "oot_groups": int(oot["LabelGroupID"].nunique()),
        "oot_lock_status": "COMPROMISED_BY_PRIOR_C1_C2_C3_R1_R2_INSPECTION",
        "group_snapshot_sha256": r1.file_sha256(GROUP_FRAME),
        "batch_snapshot_sha256": r1.file_sha256(BATCH_SOURCE),
        "model_sha256": r1.file_sha256(model_path),
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "local_mms_access": False,
        "database_access_during_run": False,
        "automatic_control": "DISABLED",
        "recommendation_status": "NOT_GENERATED_MODEL_VALIDATION_ONLY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "r1b_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (MODEL_OUTPUT / "feature_metadata.json").write_text(
        json.dumps(
            {**summary, "s1_features": s1_features, "s2_features": s2_features},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    focus = metric_frame.loc[
        metric_frame["candidate"].isin([
            "R1B_GROUP_ATOMIC_BATCH", "R1_GROUP_AGGREGATE_MODEL",
            "C3_RANDOM_KFOLD_COMPOUND_BALANCED", "baseline_recipe_train_mean",
        ])
        & metric_frame["split"].isin(["val", "oot"])
        & metric_frame["scope"].isin(r1.SCOPES)
    ][[
        "candidate", "split", "scope", "groups", "mae", "rmse", "bias", "r2",
        "prediction_to_actual_std_ratio", "within_5_mny_rate",
    ]]
    report = f"""# V3.7.1 R1B 逐批预测 → Lab 组均值实验报告

## 实验问题

验证旧 V3.7 语义：模型对每个车次单独预测，然后只对同一 `OrderID+PalletID` Lab 结果覆盖的精确车次取预测均值，与唯一 Lab MNY 比较一次。

## 防泄漏合同

- {len(batch_frame):,} 个 {SCOPE} 批次对应 {batch_frame['LabelGroupID'].nunique():,} 个独立 Lab 组。
- 整个 Lab 组原子进入同一个 train/val/OOT 和同一个 expanding-window OOF 折。
- 每个批次训练权重为 `1 / 组内批次数`，所以每个 Lab 组总权重严格等于 1。
- 排除了只能在组完成后获得的 `__group_std/min/max` 特征；缺失值只使用训练折中位数并增加显式 missing flag。
- Stage3 关闭；未知专家路线硬阻断。

## 指标对照

{markdown_table(focus)}

## 解释原则

- 若 R1B 明显优于组聚合 R1，说明非线性模型必须保留批次粒度，最终服务也应逐批预测后按 Pallet 汇总。
- 若 R1B 仍不优于历史均值，问题就不在 MAE 计算方式，而在特征合同、时间漂移或模型残差结构。
- 当前 OOT 已被前序实验查看，只能作诊断；最终放行等待新时间窗。

## 状态

- `NOT_RELEASED_DEVELOPMENT_DIAGNOSTIC`
- 自动推荐、Recipe/PLC 写入：关闭。
- APAC RDS Active-only 冻结快照；本次数据库访问：`false`。
- 本地 MMS/Archive：`false`；Archive：`BLOCKED_ARCHIVE_NOT_REPLICATED`。
- 同输入重复预测最大差：`{repeat_delta}`。
- 模型 SHA256：`{summary['model_sha256']}`。
"""
    (OUTPUT / "R1B_实验报告.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "release_status": summary["release_status"],
        "fit_seconds": fit_seconds,
        "model_sha256": summary["model_sha256"],
        "report": str(OUTPUT / "R1B_实验报告.md").encode("ascii", "backslashreplace").decode("ascii"),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
