"""Finite expert-architecture and shrinkage audit for the R1 raw model."""

from __future__ import annotations

import json
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
from Mooney_Prediction_Pipeline.model_training.v371_corrected_hybrid_v2 import (
    V371CorrectedTemporalGroupModel,
)


OUTPUT = r1.DEFAULT_BASE / "v371_r3_expert_ablation"
MODEL_OUTPUT = (
    ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r3_expert_ablation"
)


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


def fit_candidate(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    metadata: dict,
    *,
    silica: bool,
    carbon_black: bool,
    shrinkage_k: float,
) -> tuple[V371CorrectedTemporalGroupModel, pd.DataFrame, list[dict], float]:
    started = time.perf_counter()
    model = V371CorrectedTemporalGroupModel(
        shrinkage_k=shrinkage_k,
        use_cluster_experts=True,
        use_material_route_matrix=True,
        use_silica_subsystem=silica,
        use_cb_subsystem=carbon_black,
        crossfit_splits=5,
        time_col="lab_time",
        objective="regression",
    ).fit(
        train,
        metadata["s1_features"],
        metadata["s2_features"],
        target_col="MNY",
        cluster_col="material_system",
    )
    elapsed = time.perf_counter() - started
    predictions = r1.add_predictions(model, validation, name)
    metrics = r1.metric_rows(predictions, name, "val")
    return model, predictions, metrics, elapsed


def metric_value(metrics: pd.DataFrame, candidate: str, scope: str, column: str) -> float:
    return float(
        metrics.loc[
            metrics["candidate"].eq(candidate) & metrics["scope"].eq(scope), column
        ].iloc[0]
    )


def choose(metrics: pd.DataFrame, candidates: list[str]) -> str:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "candidate": candidate,
                "global_mae": metric_value(metrics, candidate, "global", "mae"),
                "t15760_mae": metric_value(metrics, candidate, "T15760", "mae"),
                "global_rmse": metric_value(metrics, candidate, "global", "rmse"),
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        ["global_mae", "t15760_mae", "global_rmse", "candidate"], kind="stable"
    )
    best_global = float(ranked.iloc[0]["global_mae"])
    tied = ranked.loc[ranked["global_mae"].le(best_global + 0.02)]
    return str(tied.sort_values(["t15760_mae", "global_mae"]).iloc[0]["candidate"])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, metadata = r1.load_contract(r1.DEFAULT_FRAME, r1.DEFAULT_C3_METADATA)
    train = frame.loc[frame["split"].eq("train")].copy()
    validation = frame.loc[frame["split"].eq("val")].copy()
    oot = frame.loc[frame["split"].eq("oot")].copy()

    architectures = {
        "R3_SPECIALIZED_BOTH_K5": (True, True),
        "R3_FOUR_ROUTE_ONLY_K5": (False, False),
        "R3_SILICA_SUBSYSTEM_ONLY_K5": (True, False),
        "R3_CB_SUBSYSTEM_ONLY_K5": (False, True),
    }
    models = {}
    validation_predictions = []
    metric_records = []
    timings = []
    for name, (silica, carbon_black) in architectures.items():
        model, predictions, metrics, elapsed = fit_candidate(
            name,
            train,
            validation,
            metadata,
            silica=silica,
            carbon_black=carbon_black,
            shrinkage_k=5.0,
        )
        models[name] = model
        validation_predictions.append(predictions)
        metric_records.extend(metrics)
        timings.append(
            {"candidate": name, "fit_seconds": elapsed, "silica_subsystem": silica,
             "cb_subsystem": carbon_black, "shrinkage_k": 5.0}
        )
    metrics = pd.DataFrame(metric_records)
    best_architecture = choose(metrics, list(architectures))
    silica, carbon_black = architectures[best_architecture]

    shrinkage_candidates = [best_architecture]
    for shrinkage in (2.0, 10.0, 20.0):
        name = f"R3_{'SILICA' if silica else 'NO_SILICA'}_{'CB' if carbon_black else 'NO_CB'}_K{int(shrinkage)}"
        model, predictions, candidate_metrics, elapsed = fit_candidate(
            name,
            train,
            validation,
            metadata,
            silica=silica,
            carbon_black=carbon_black,
            shrinkage_k=shrinkage,
        )
        models[name] = model
        validation_predictions.append(predictions)
        metric_records.extend(candidate_metrics)
        timings.append(
            {"candidate": name, "fit_seconds": elapsed, "silica_subsystem": silica,
             "cb_subsystem": carbon_black, "shrinkage_k": shrinkage}
        )
        shrinkage_candidates.append(name)
    metrics = pd.DataFrame(metric_records)
    selected = choose(metrics, shrinkage_candidates)
    selected_model = models[selected]
    oot_predictions = r1.add_predictions(selected_model, oot, selected)
    metrics = pd.concat(
        [metrics, pd.DataFrame(r1.metric_rows(oot_predictions, selected, "oot"))],
        ignore_index=True,
    )
    repeat_delta = float(
        np.max(np.abs(selected_model.predict(validation)[0] - selected_model.predict(validation)[0]))
    )

    pd.concat(validation_predictions, ignore_index=True).to_csv(
        OUTPUT / "r3_validation_predictions.csv", index=False, encoding="utf-8-sig"
    )
    oot_predictions.to_csv(OUTPUT / "r3_selected_oot_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT / "r3_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(timings).to_csv(OUTPUT / "r3_fit_timings.csv", index=False, encoding="utf-8-sig")
    selected_model.stage2_expert_report_.to_csv(
        OUTPUT / "r3_selected_expert_report.csv", index=False, encoding="utf-8-sig"
    )

    model_path = MODEL_OUTPUT / "hybrid_model_shadow.joblib"
    joblib.dump(selected_model, model_path)
    summary = {
        "model_name": "V3.7.1-Corrected-Challenger-R3-Expert-Ablation",
        "selected_candidate": selected,
        "selected_architecture_source": best_architecture,
        "selection_rule": "min global val MAE; within 0.02 tie by T15760 MAE then global RMSE",
        "release_status": "NOT_RELEASED_GATES_PENDING_OR_FAILED",
        "stage3_calibration_enabled": False,
        "crossfit_audit": selected_model.crossfit_audit_,
        "same_input_repeat_max_abs_delta": repeat_delta,
        "model_sha256": r1.file_sha256(model_path),
        "source_sha256": r1.file_sha256(r1.DEFAULT_FRAME),
        "oot_lock_status": "COMPROMISED_BY_PRIOR_EXPERIMENT_INSPECTION",
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "local_mms_access": False,
        "database_access_during_run": False,
        "automatic_control": "DISABLED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "r3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (MODEL_OUTPUT / "feature_metadata.json").write_text(
        json.dumps(
            {**summary, "s1_features": metadata["s1_features"], "s2_features": metadata["s2_features"]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    focus = metrics.loc[
        metrics["scope"].isin(["global", "T15760", "B00458"])
    ][[
        "candidate", "split", "scope", "groups", "mae", "rmse", "bias", "r2",
        "prediction_to_actual_std_ratio",
    ]]
    report = f"""# V3.7.1 R3 专家结构消融与 Shrinkage 优化报告

## 结论

本轮只做有限、可解释的优化：比较四路线专家本体与 Silica/Carbon Black 子专家是否在严格时间验证中产生真实增益，然后只对最佳结构比较 `k=2/5/10/20`。未扩大搜索空间，也未启用 Stage3。

- 最佳结构来源：`{best_architecture}`。
- 最终候选：`{selected}`。
- 状态：`NOT_RELEASED_GATES_PENDING_OR_FAILED`；当前 OOT 已被前序实验查看，只作诊断。

## 指标

{markdown_table(focus)}

## 决策含义

- 如果 route-only 优于子专家，说明现有物理子专家分区在当前数据量下增加了方差，应保留概念但降低复杂度。
- 如果子专家稳定改善验证但 OOT 退化，问题是时间漂移/特征稳定性，不应继续调深度。
- B00458 验证/OOT 组数不足 30 时不参与正式架构放行。
- 局部 T15760/B00458 残差专家只有在通用结构通过后才允许进入下一步。

## 边界

- Active-only 冻结快照；运行期间数据库访问：`false`。
- 本地 MMS/Archive：`false`；Stage3/自动控制：`false`。
- 模型 SHA256：`{summary['model_sha256']}`。
"""
    (OUTPUT / "R3_实验报告.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "selected_candidate": selected,
        "release_status": summary["release_status"],
        "model_sha256": summary["model_sha256"],
        "report": str(OUTPUT / "R3_实验报告.md").encode("ascii", "backslashreplace").decode("ascii"),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
