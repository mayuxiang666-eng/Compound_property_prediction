"""Train R2 hierarchical-nominal challenger on the frozen Active-only snapshot."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training import run_v371_r1_temporal_group as r1
from Mooney_Prediction_Pipeline.model_training.v371_hierarchical_nominal_v1 import (
    V371HierarchicalNominalModel,
)


OUTPUT = r1.DEFAULT_BASE / "v371_r2_hierarchical_nominal"
MODEL_OUTPUT = (
    ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r2_hierarchical_nominal"
)
R1_OUTPUT = r1.DEFAULT_BASE / "v371_r1_temporal_group"


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


def load_r1_comparison() -> tuple[pd.DataFrame, str]:
    summary = json.loads((R1_OUTPUT / "r1_summary.json").read_text(encoding="utf-8"))
    selected = str(summary["selected_candidate"])
    predictions = pd.read_csv(R1_OUTPUT / "r1_selected_group_predictions.csv", low_memory=False)
    predictions = predictions.rename(columns={"Candidate": "OriginalCandidate"})
    predictions["Candidate"] = "R1_EQUAL_GROUP_TEMPORAL"
    return predictions, selected


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, metadata = r1.load_contract(r1.DEFAULT_FRAME, r1.DEFAULT_C3_METADATA)
    train = frame.loc[frame["split"].eq("train")].copy()
    validation = frame.loc[frame["split"].eq("val")].copy()
    oot = frame.loc[frame["split"].eq("oot")].copy()

    model = V371HierarchicalNominalModel(
        shrinkage_k=5.0,
        use_cluster_experts=True,
        use_material_route_matrix=True,
        use_silica_subsystem=True,
        use_cb_subsystem=True,
        crossfit_splits=5,
        time_col="lab_time",
        objective="regression",
        template_min_groups=5,
        recipe_version_min_groups=5,
        recipe_min_groups=10,
        compound_min_groups=10,
    ).fit(
        train,
        metadata["s1_features"],
        metadata["s2_features"],
        target_col="MNY",
        cluster_col="material_system",
    )

    predictions = []
    metric_records = []
    resolution_records = []
    for split_frame in (train, validation, oot):
        scored = r1.add_predictions(model, split_frame, "R2_HIERARCHICAL_NOMINAL")
        predictions.append(scored)
        split = str(scored["split"].iloc[0])
        metric_records.extend(r1.metric_rows(scored, "R2_HIERARCHICAL_NOMINAL", split))
        source = model.nominal_resolution(split_frame)
        audit = split_frame[[
            "LabelGroupID", "CompoundName", "RecipeID", "ProcessVersion", "MixerLine",
            "TemplateKey", "split",
        ]].copy()
        audit["NominalSource"] = source.to_numpy()
        resolution_records.append(audit)

    r1_predictions, original_r1_name = load_r1_comparison()
    r1_metrics = []
    for split in ("train", "val", "oot"):
        subset = r1_predictions.loc[r1_predictions["split"].eq(split)].copy()
        r1_metrics.extend(r1.metric_rows(subset, "R1_EQUAL_GROUP_TEMPORAL", split))
    baseline_predictions, baseline_metrics = r1.baseline_metric_rows(frame)
    c3_predictions, c3_metrics = r1.c3_metric_rows(r1.DEFAULT_BASE)
    metrics = pd.concat(
        [pd.DataFrame(metric_records), pd.DataFrame(r1_metrics), baseline_metrics, c3_metrics],
        ignore_index=True,
    )
    all_predictions = pd.concat(
        predictions + [r1_predictions, baseline_predictions, c3_predictions],
        ignore_index=True,
        sort=False,
    )
    resolution = pd.concat(resolution_records, ignore_index=True)

    model_path = MODEL_OUTPUT / "hybrid_model_shadow.joblib"
    joblib.dump(model, model_path)
    repeat_delta = float(
        np.max(np.abs(model.predict(validation)[0] - model.predict(validation)[0]))
    )
    summary = {
        "model_name": "V3.7.1-Corrected-Challenger-R2-Hierarchical-Nominal",
        "contract_version": model.contract_version,
        "base_candidate": original_r1_name,
        "release_status": "NOT_RELEASED_GATES_FAILED_AND_OOT_COMPROMISED",
        "stage3_calibration_enabled": False,
        "training_unit": "ONE_ROW_PER_ORDERID_PALLETID_LAB_GROUP",
        "group_loss_weight": 1.0,
        "process_nominal_contract": model.crossfit_audit_["process_nominal_contract"],
        "nominal_resolution_by_split": resolution.groupby(
            ["split", "NominalSource"]
        ).size().rename("groups").reset_index().to_dict("records"),
        "crossfit_audit": model.crossfit_audit_,
        "same_input_repeat_max_abs_delta": repeat_delta,
        "oot_lock_status": "COMPROMISED_BY_PRIOR_C1_C2_C3_R1_INSPECTION",
        "source_snapshot": str(r1.DEFAULT_FRAME),
        "source_sha256": r1.file_sha256(r1.DEFAULT_FRAME),
        "model_sha256": r1.file_sha256(model_path),
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "local_mms_access": False,
        "database_access_during_run": False,
        "automatic_control": "DISABLED",
        "recommendation_status": "NOT_GENERATED_MODEL_VALIDATION_ONLY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    all_predictions.to_csv(OUTPUT / "r2_all_group_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT / "r2_metrics.csv", index=False, encoding="utf-8-sig")
    resolution.to_csv(OUTPUT / "r2_nominal_resolution.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "r2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (MODEL_OUTPUT / "feature_metadata.json").write_text(
        json.dumps({**summary, "s1_features": metadata["s1_features"], "s2_features": metadata["s2_features"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    focus = metrics.loc[
        metrics["candidate"].isin([
            "R2_HIERARCHICAL_NOMINAL", "R1_EQUAL_GROUP_TEMPORAL",
            "C3_RANDOM_KFOLD_COMPOUND_BALANCED", "baseline_recipe_train_mean",
        ])
        & metrics["split"].isin(["val", "oot"])
        & metrics["scope"].isin(r1.SCOPES)
    ][[
        "candidate", "split", "scope", "groups", "mae", "rmse", "bias", "r2",
        "prediction_to_actual_std_ratio", "within_5_mny_rate",
    ]]
    resolution_summary = resolution.groupby(
        ["split", "NominalSource"]
    ).size().rename("groups").reset_index()
    report = f"""# V3.7.1 R2 层级 Phase Nominal 实验报告

## 结论

R2 完成，但在当前已查看 OOT 上仍只作为开发诊断，不能放行。R2 保留四阶段专家结构和 R1 的等权 Lab 组/时间 OOF，只把 Stage2 过程偏差参照改为：

`TemplateKey → RecipeID+ProcessVersion+MixerLine → RecipeID+MixerLine → Compound+MixerLine → Material+Route → Global`

这允许同一 RecipeID 的工程师版本调整作为正常工艺状态存在；新版本不会被强行套用旧模板，又能在自身样本不足时安全回退。

## 指标对照

{markdown_table(focus)}

## Nominal 回退覆盖

{markdown_table(resolution_summary)}

## 判定规则

- 若 R2 只改善训练集、不改善验证集，说明 TemplateKey 过细或 Stage2 可学习信号不足，不进入 R3。
- 若 T15760 改善但全局/B00458 恶化，则 R3 只能做显式局部残差专家，不能替换通用模型。
- 当前 OOT 已被反复查看；最终放行必须等待新的时间窗。
- Stage3、自动推荐和 PLC/Recipe 写入保持关闭。

## 数据边界

- APAC RDS Active-only 冻结快照；运行期间数据库访问：`false`。
- 本地 MMS/Archive：`false`；Archive：`BLOCKED_ARCHIVE_NOT_REPLICATED`。
- 同输入重复预测最大差：`{repeat_delta}`。
- 模型 SHA256：`{summary['model_sha256']}`。
"""
    (OUTPUT / "R2_实验报告.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": summary["release_status"],
        "model_sha256": summary["model_sha256"],
        "report": str(OUTPUT / "R2_实验报告.md").encode("ascii", "backslashreplace").decode("ascii"),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
