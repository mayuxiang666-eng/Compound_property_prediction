"""Train and audit R1 equal-group temporal V3.7.1 challengers.

This runner is deliberately snapshot-only. It never opens MMS, Archive, RDS,
Mustang, Lab, recipe, PLC, or production prediction databases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.group_metric_contract import (
    GROUP_METRIC_CONTRACT,
    assert_one_row_per_lab_group,
    evaluate_group_frame,
)
from Mooney_Prediction_Pipeline.model_training.v371_corrected_hybrid_v2 import (
    V371CorrectedTemporalGroupModel,
)


DEFAULT_BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
DEFAULT_FRAME = DEFAULT_BASE / "pallet_group_training_frame_c2_prepared.csv"
DEFAULT_C3_METADATA = (
    ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_c3_active"
    / "feature_metadata.json"
)
DEFAULT_OUTPUT = DEFAULT_BASE / "v371_r1_temporal_group"
DEFAULT_MODEL_OUTPUT = (
    ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r1_temporal_group"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def load_contract(frame_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(frame_path, low_memory=False)
    required = {
        "LabelGroupID", "MNY", "lab_time", "split", "CompoundName",
        "RecipeID", "ProcessVersion", "MixerLine", "material_system",
    }
    required.update(metadata["s1_features"])
    required.update(metadata["s2_features"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"R1 snapshot is missing required columns: {missing}")
    assert_one_row_per_lab_group(frame)
    if len(frame) != frame["LabelGroupID"].nunique():
        raise RuntimeError("R1 snapshot is not one row per Lab group")
    if set(frame["split"].dropna().unique()) != {"train", "val", "oot"}:
        raise ValueError("R1 snapshot must contain train/val/oot group-atomic splits")
    if frame.groupby("LabelGroupID")["split"].nunique().max() != 1:
        raise ValueError("A Lab group occurs in more than one split")
    frame["MNY"] = pd.to_numeric(frame["MNY"], errors="coerce")
    frame["lab_time"] = pd.to_datetime(frame["lab_time"], errors="coerce", utc=True)
    if frame[["MNY", "lab_time"]].isna().any().any():
        raise ValueError("R1 target or Lab event time contains missing values")
    return frame, metadata


def add_predictions(model, frame: pd.DataFrame, candidate: str) -> pd.DataFrame:
    final, stage1, stage1b, stage2 = model.predict(frame, cluster_col="material_system")
    identity = [
        column for column in (
            "LabelGroupID", "OrderID", "PalletID", "CompoundName", "RecipeID",
            "ProcessVersion", "MixerLine", "material_system", "phase_route",
            "TemplateKey", "MNY", "lab_time", "split", "batch_count", "PhaseBatchCount",
        ) if column in frame.columns
    ]
    result = frame[identity].copy()
    result["Candidate"] = candidate
    result["Stage1Prediction"] = stage1
    result["Stage1bBias"] = stage1b
    result["Stage2Residual"] = stage2
    result["RawPrediction"] = final
    result["Error"] = result["RawPrediction"] - result["MNY"]
    return result


def scope_mask(frame: pd.DataFrame, scope: str) -> pd.Series:
    compound = frame["CompoundName"].astype(str).str.upper()
    if scope == "global":
        return pd.Series(True, index=frame.index)
    if scope == "T15760":
        return compound.str.contains("T15760", regex=False)
    if scope == "T15760_M1_06_002":
        return compound.eq("M1-T15760---- 06 002")
    if scope == "T15760_M1_06_002_PV46":
        version = pd.to_numeric(frame["ProcessVersion"], errors="coerce")
        return compound.eq("M1-T15760---- 06 002") & version.eq(46.0)
    if scope == "B00458":
        return compound.str.contains("B00458", regex=False)
    raise ValueError(f"Unknown metric scope: {scope}")


SCOPES = ("global", "T15760", "T15760_M1_06_002", "T15760_M1_06_002_PV46", "B00458")


def metric_rows(predictions: pd.DataFrame, candidate: str, split: str) -> list[dict[str, Any]]:
    selected = predictions.loc[predictions["split"].eq(split)].copy()
    rows = []
    for scope in SCOPES:
        subset = selected.loc[scope_mask(selected, scope)].copy()
        metrics = evaluate_group_frame(subset) if not subset.empty else {"groups": 0}
        rows.append({"candidate": candidate, "split": split, "scope": scope, **metrics})
    for (recipe_id, version), subset in selected.groupby(["RecipeID", "ProcessVersion"], dropna=False):
        if len(subset) < 10:
            continue
        rows.append(
            {
                "candidate": candidate,
                "split": split,
                "scope": f"RecipeVersion::{recipe_id}::{version}",
                **evaluate_group_frame(subset),
            }
        )
    return rows


def baseline_predictions(train: pd.DataFrame, evaluated: pd.DataFrame) -> dict[str, np.ndarray]:
    global_mean = float(train["MNY"].mean())
    recipe_mean = train.groupby("RecipeID")["MNY"].mean()
    compound_mean = train.groupby("CompoundName")["MNY"].mean()
    return {
        "baseline_global_train_mean": np.full(len(evaluated), global_mean),
        "baseline_recipe_train_mean": evaluated["RecipeID"].map(recipe_mean).fillna(global_mean).to_numpy(),
        "baseline_compound_train_mean": evaluated["CompoundName"].map(compound_mean).fillna(global_mean).to_numpy(),
    }


def baseline_metric_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["split"].eq("train")]
    outputs = []
    metrics = []
    for split in ("val", "oot"):
        evaluated = frame.loc[frame["split"].eq(split)].copy()
        for name, predictions in baseline_predictions(train, evaluated).items():
            result = evaluated[[
                "LabelGroupID", "CompoundName", "RecipeID", "ProcessVersion", "MNY", "split"
            ]].copy()
            result["Candidate"] = name
            result["RawPrediction"] = predictions
            outputs.append(result)
            metrics.extend(metric_rows(result, name, split))
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(metrics)


def c3_metric_rows(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs = []
    metrics = []
    for split, filename in (("val", "c3_val_predictions.csv"), ("oot", "c3_oot_predictions.csv")):
        path = base_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        result = frame[[
            "LabelGroupID", "CompoundName", "RecipeID", "ProcessVersion", "MNY", "split"
        ]].copy()
        result["Candidate"] = "C3_RANDOM_KFOLD_COMPOUND_BALANCED"
        result["RawPrediction"] = pd.to_numeric(frame["C3Prediction"], errors="coerce")
        outputs.append(result)
        metrics.extend(metric_rows(result, "C3_RANDOM_KFOLD_COMPOUND_BALANCED", split))
    if not outputs:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(metrics)


def select_candidate(metrics: pd.DataFrame) -> str:
    selected = metrics.loc[
        metrics["split"].eq("val") & metrics["scope"].eq("global")
        & metrics["candidate"].str.startswith("R1_")
    ].sort_values(["mae", "rmse", "candidate"], kind="stable")
    if selected.empty:
        raise RuntimeError("No R1 candidate has validation metrics")
    return str(selected.iloc[0]["candidate"])


def render_report(
    metrics: pd.DataFrame,
    selected: str,
    models: dict[str, V371CorrectedTemporalGroupModel],
    frame_path: Path,
) -> str:
    focus = metrics.loc[
        metrics["scope"].isin(SCOPES)
        & metrics["candidate"].isin(
            [selected, "C3_RANDOM_KFOLD_COMPOUND_BALANCED", "baseline_recipe_train_mean"]
        )
    ].copy()
    columns = [
        "candidate", "split", "scope", "groups", "mae", "rmse", "bias", "r2",
        "prediction_to_actual_std_ratio", "within_5_mny_rate",
    ]
    focus = focus.reindex(columns=columns)
    if focus.empty:
        table = "No metrics."
    else:
        try:
            table = focus.to_markdown(index=False, floatfmt=".4f")
        except ImportError:
            # Keep report generation deterministic in minimal model environments.
            table = focus.to_string(index=False)
    audit = json.dumps(models[selected].crossfit_audit_, ensure_ascii=False, indent=2)
    return f"""# V3.7.1 R1 等权 Lab 组 + 时间交叉拟合实验报告

## 结论状态

- 候选：`{selected}`。
- 这是开发期 Challenger，不覆盖冻结 V3.7，也不覆盖 C3。
- Stage3 保持关闭；不生成 Step 推荐，不写 Recipe/PLC/生产表。
- 数据来源是已冻结的 APAC RDS Active-only 快照；本次训练未连接任何数据库。
- Archive 状态：`BLOCKED_ARCHIVE_NOT_REPLICATED`；Local MMS access：`false`。
- OOT 已在前序 C1/C2/C3 实验中查看过，因此这里只能标记为开发诊断，不能作为最终盲测放行证据。

## 本轮修复的设计缺陷

1. 训练帧强制一行一个 `OrderID+PalletID` Lab 组，每组损失权重严格等于 1。
2. 移除 C3 的配方频次再平衡，避免 T15760 因历史组多而被降权到约 1/3。
3. Stage1、Stage1b 以及 Silica/Carbon Black 子专家改成按 `lab_time` 的 expanding-window OOF。
4. Stage1b 只允许使用更早且已有 OOF 预测的独立 Lab 组，不再用随机未来折。
5. 未知材料/路线继续 fail closed；Stage3 不参与候选选择。
6. 主指标合同固定为 `{GROUP_METRIC_CONTRACT}`。

## 关键结果

{table}

## 如何理解旧 2.5–2.9 MAE

旧 README 的低 MAE 包含按批次行推进的 Stage3 offset；同一 Pallet 的 Lab 标签会在多行中重复，`lag=3` 实际是三个批次行，不一定是三个独立 Lab 事件。旧切分也不是当前严格时间 OOT。因此该数字只能作为历史系统结果，不能和这里的 raw、Stage3-off、时间验证结果直接等价。

## 仍需补做

- 当前 R1 输入是一行一个 Pallet 组的聚合特征。批次级 Phase 快照已经存在，下一实验将训练 group-atomic batch model，并严格执行“每批预测 -> 同 Lab 组均值 -> 比较一次”，与组级模型并行比较。
- R2 将加入 `RecipeID + ProcessVersion + MixerLine + TemplateKey` 分层 nominal，避免不同版本设定变化被粗粒度路线均值冲淡。
- R3 只在通用模型通过后增加 T15760/B00458 的局部残差专家。
- 在新的未查看时间窗到来前，任何结果都不得标记 production release。

## 可追溯信息

- 输入：`{frame_path}`
- 输入 SHA256：`{file_sha256(frame_path)}`
- 生成时间 UTC：`{datetime.now(timezone.utc).isoformat()}`
- 交叉拟合审计：

```json
{audit}
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_C3_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument(
        "--objectives", nargs="+", default=["regression", "regression_l1", "huber"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.model_output.mkdir(parents=True, exist_ok=True)
    frame, source_metadata = load_contract(args.frame, args.metadata)
    train = frame.loc[frame["split"].eq("train")].copy()
    validation = frame.loc[frame["split"].eq("val")].copy()
    oot = frame.loc[frame["split"].eq("oot")].copy()

    models: dict[str, V371CorrectedTemporalGroupModel] = {}
    prediction_frames = []
    metric_records = []
    timing_records = []
    for objective in args.objectives:
        candidate = f"R1_{objective.upper()}"
        started = time.perf_counter()
        model = V371CorrectedTemporalGroupModel(
            shrinkage_k=5.0,
            use_cluster_experts=True,
            use_material_route_matrix=True,
            use_silica_subsystem=True,
            use_cb_subsystem=True,
            crossfit_splits=5,
            time_col="lab_time",
            objective=objective,
        ).fit(
            train,
            source_metadata["s1_features"],
            source_metadata["s2_features"],
            target_col="MNY",
            cluster_col="material_system",
        )
        elapsed = time.perf_counter() - started
        models[candidate] = model
        timing_records.append(
            {"candidate": candidate, "objective": objective, "fit_seconds": elapsed}
        )
        for split_frame in (train, validation, oot):
            predictions = add_predictions(model, split_frame, candidate)
            prediction_frames.append(predictions)
            metric_records.extend(metric_rows(predictions, candidate, predictions["split"].iloc[0]))

    baseline_output, baseline_metrics = baseline_metric_rows(frame)
    c3_output, c3_metrics = c3_metric_rows(args.frame.parent)
    metrics = pd.concat(
        [pd.DataFrame(metric_records), baseline_metrics, c3_metrics], ignore_index=True
    )
    selected = select_candidate(metrics)
    selected_model = models[selected]
    repeat_a = selected_model.predict(validation)[0]
    repeat_b = selected_model.predict(validation)[0]
    repeat_delta = float(np.max(np.abs(repeat_a - repeat_b)))

    all_predictions = pd.concat(
        prediction_frames + [baseline_output, c3_output], ignore_index=True, sort=False
    )
    all_predictions.to_csv(args.output / "r1_all_group_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output / "r1_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(timing_records).to_csv(
        args.output / "r1_fit_timings.csv", index=False, encoding="utf-8-sig"
    )
    selected_predictions = all_predictions.loc[all_predictions["Candidate"].eq(selected)]
    selected_predictions.to_csv(
        args.output / "r1_selected_group_predictions.csv", index=False, encoding="utf-8-sig"
    )

    model_path = args.model_output / "hybrid_model_shadow.joblib"
    joblib.dump(selected_model, model_path)
    metadata = {
        "model_name": "V3.7.1-Corrected-Challenger-R1-Temporal-Group",
        "selected_candidate": selected,
        "contract_version": selected_model.contract_version,
        "selection_rule": "minimum global validation group MAE, then RMSE",
        "release_status": "NOT_RELEASED_DEVELOPMENT_DIAGNOSTIC",
        "group_metric_contract": GROUP_METRIC_CONTRACT,
        "training_unit": "ONE_ROW_PER_ORDERID_PALLETID_LAB_GROUP",
        "group_loss_weight": 1.0,
        "stage3_calibration_enabled": False,
        "crossfit_audit": selected_model.crossfit_audit_,
        "same_input_repeat_max_abs_delta": repeat_delta,
        "oot_lock_status": "COMPROMISED_BY_PRIOR_C1_C2_C3_INSPECTION",
        "source_snapshot": str(args.frame),
        "source_sha256": file_sha256(args.frame),
        "model_sha256": file_sha256(model_path),
        "train_groups": int(len(train)),
        "validation_groups": int(len(validation)),
        "oot_groups": int(len(oot)),
        "s1_features": source_metadata["s1_features"],
        "s2_features": source_metadata["s2_features"],
        "pid_contract": source_metadata.get("pid_contract"),
        "curve_quality_contract": source_metadata.get("curve_quality_contract"),
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "local_mms_access": False,
        "database_access_during_run": False,
        "automatic_control": "DISABLED",
        "recommendation_status": "NOT_GENERATED_MODEL_VALIDATION_ONLY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.model_output / "feature_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "r1_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = render_report(metrics, selected, models, args.frame)
    (args.output / "R1_实验报告.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "selected_candidate": selected,
        "same_input_repeat_max_abs_delta": repeat_delta,
        "metrics": metrics.loc[
            metrics["candidate"].eq(selected) & metrics["scope"].isin(SCOPES)
        ].to_dict("records"),
        "report": str(args.output / "R1_实验报告.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
