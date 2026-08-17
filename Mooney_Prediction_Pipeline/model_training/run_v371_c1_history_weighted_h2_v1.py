"""Run the target-free H2 weighted legacy-history Phase C1 diagnostic.

H2 keeps the fixed current validation/OOT groups unchanged.  Legacy history
is admitted only when its exact RecipeID, mixer, route and static process
contract are compatible with the current training period.  Admitted groups
receive time-decayed weights and an exact-RecipeID total-weight cap.

The legacy CSV does not contain the order's real ProcessVersion.  A PV token
embedded in RecipeID is not treated as ProcessVersion and is never used for
matching.  No database connection is opened.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.model_training.run_v371_c1_history_augmented_phase_v1 import (
    group_predictions,
    make_training_frame,
    markdown_table,
    metric_record,
    plot_overview,
    sha256,
    trend_records,
)
from Mooney_Prediction_Pipeline.model_training.v371_source_weighted_group_atomic_batch_v1 import (
    V371SourceWeightedGroupAtomicBatchModel,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger"
H1_DIR = BASE / "v371_c1_history_augmented_phase_v1"
DEFAULT_CURRENT = H1_DIR / "current_common_batch_features.parquet"
DEFAULT_HISTORY = H1_DIR / "history_common_batch_features.parquet"
DEFAULT_CONTRACT = H1_DIR / "common_feature_contract.json"
DEFAULT_CURRENT_GROUPS = (
    BASE / "global_rds_active_v1" / "m1_temporal_jit_v1"
    / "m1_temporal_jit_acceptance_input.csv"
)
DEFAULT_HISTORY_GROUPS = (
    BASE / "local_history_label_release_v1" / "temporal_safe_history_group_labels.csv"
)
DEFAULT_H0_PREDICTIONS = H1_DIR / "candidate_group_predictions.csv"
DEFAULT_OUTPUT = BASE / "v371_c1_history_weighted_h2_v1"
DEFAULT_MODEL_OUTPUT = (
    ROOT / "Mooney_Prediction_Pipeline" / "models"
    / "v371_c1_history_weighted_h2_v1"
)

STATIC_MATCH_COLUMNS = (
    "Target_Temperature",
    "Top_Fill_Factor",
    "Bot_Fill_Factor",
    "weight_pct_solid_elastomer",
    "weight_pct_natural_rubber",
    "weight_pct_silica",
    "weight_pct_oil",
    "weight_pct_silian",
    "weight_pct_carbon_black",
)
MATCH_KEYS = ("RecipeID", "MixerLine", "phase_route")
H0_NAME = "H0_CURRENT_ONLY_PHASE_C1"
H2_NAME = "H2_WEIGHTED_MATCHED_HISTORY_PHASE_C1"


def _numeric_group_profile(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["LabelGroupID", *MATCH_KEYS, *STATIC_MATCH_COLUMNS]
    work = frame[columns].copy()
    for column in STATIC_MATCH_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    aggregations: dict[str, Any] = {key: "first" for key in MATCH_KEYS}
    aggregations.update({column: "median" for column in STATIC_MATCH_COLUMNS})
    return work.groupby("LabelGroupID", as_index=False, dropna=False).agg(aggregations)


def _match_tolerance(column: str, low: float, high: float) -> float:
    if column == "Target_Temperature":
        return 0.5
    scale = max(abs(low), abs(high), 1.0)
    return max(0.1, 0.03 * scale)


def build_history_eligibility(
    current: pd.DataFrame,
    history: pd.DataFrame,
    current_groups: pd.DataFrame,
    history_groups: pd.DataFrame,
    *,
    minimum_current_recipe_groups: int = 10,
) -> pd.DataFrame:
    """Create a target-free legacy-history compatibility audit."""

    splits = current_groups[["LabelGroupID", "split"]].copy()
    splits["LabelGroupID"] = splits["LabelGroupID"].astype(str)
    current_profile = _numeric_group_profile(current)
    current_profile["LabelGroupID"] = current_profile["LabelGroupID"].astype(str)
    current_train = current_profile.merge(splits, on="LabelGroupID", validate="one_to_one")
    current_train = current_train[current_train["split"].eq("train")].copy()
    history_profile = _numeric_group_profile(history)
    history_profile["LabelGroupID"] = history_profile["LabelGroupID"].astype(str)

    support = current_train.groupby(list(MATCH_KEYS), dropna=False).size().rename(
        "current_train_contract_groups"
    )
    ranges = current_train.groupby(list(MATCH_KEYS), dropna=False)[
        list(STATIC_MATCH_COLUMNS)
    ].agg(["min", "max"])
    ranges.columns = [f"{column}__{stat}" for column, stat in ranges.columns]
    reference = support.to_frame().join(ranges).reset_index()
    audit = history_profile.merge(reference, on=list(MATCH_KEYS), how="left", validate="many_to_one")

    statuses: list[str] = []
    failed_columns: list[str] = []
    for row in audit.itertuples(index=False):
        row_dict = row._asdict()
        if pd.isna(row_dict["current_train_contract_groups"]):
            statuses.append("BLOCKED_NO_EXACT_RECIPE_MIXER_ROUTE_IN_CURRENT_TRAIN")
            failed_columns.append("")
            continue
        if int(row_dict["current_train_contract_groups"]) < minimum_current_recipe_groups:
            statuses.append("BLOCKED_INSUFFICIENT_CURRENT_RECIPE_SUPPORT")
            failed_columns.append("")
            continue
        failures: list[str] = []
        for column in STATIC_MATCH_COLUMNS:
            value = row_dict[column]
            low = row_dict[f"{column}__min"]
            high = row_dict[f"{column}__max"]
            if pd.isna(low) and pd.isna(high) and pd.isna(value):
                continue
            if pd.isna(value) or pd.isna(low) or pd.isna(high):
                failures.append(f"{column}:MISSING_SEMANTICS_DIFFER")
                continue
            tolerance = _match_tolerance(column, float(low), float(high))
            if not (float(low) - tolerance <= float(value) <= float(high) + tolerance):
                failures.append(column)
        if failures:
            statuses.append("BLOCKED_STATIC_PROCESS_CONTRACT_MISMATCH")
            failed_columns.append("|".join(failures))
        else:
            statuses.append("ELIGIBLE_STATIC_PROCESS_CONTRACT_MATCH")
            failed_columns.append("")
    audit["eligibility_status"] = statuses
    audit["failed_static_columns"] = failed_columns
    audit["legacy_process_version_status"] = "LEGACY_ORDER_PROCESS_VERSION_UNAVAILABLE"
    audit["matching_uses_recipeid_embedded_pv_as_process_version"] = False

    released = history_groups[[
        "LabelGroupID", "lab_time", "FixedValidationOOTStart", "temporal_cutoff_status"
    ]].copy()
    released["LabelGroupID"] = released["LabelGroupID"].astype(str)
    audit = audit.merge(released, on="LabelGroupID", validate="one_to_one")
    return audit


def assign_history_weights(
    eligibility: pd.DataFrame,
    current: pd.DataFrame,
    current_groups: pd.DataFrame,
    *,
    base_weight: float = 0.25,
    half_life_days: float = 180.0,
    recipe_cap_fraction: float = 0.25,
) -> pd.DataFrame:
    """Assign target-free time decay and exact-RecipeID aggregate caps."""

    if not (0 < base_weight <= 1 and half_life_days > 0 and 0 < recipe_cap_fraction <= 1):
        raise ValueError("INVALID_HISTORY_WEIGHT_CONFIGURATION")
    result = eligibility.copy()
    lab_time = pd.to_datetime(result["lab_time"], errors="coerce", utc=True)
    cutoff = pd.to_datetime(result["FixedValidationOOTStart"], errors="coerce", utc=True)
    result["history_age_days"] = (cutoff - lab_time).dt.total_seconds().div(86400.0)
    result["raw_time_decay_weight"] = base_weight * np.power(
        0.5, result["history_age_days"].clip(lower=0) / half_life_days
    )
    result.loc[result["history_age_days"].isna(), "eligibility_status"] = (
        "BLOCKED_MISSING_HISTORY_OR_CUTOFF_TIME"
    )
    result.loc[result["history_age_days"].lt(0), "eligibility_status"] = (
        "BLOCKED_HISTORY_AFTER_FIXED_HOLDOUT_START"
    )

    splits = current_groups[["LabelGroupID", "split"]].copy()
    splits["LabelGroupID"] = splits["LabelGroupID"].astype(str)
    current_profile = _numeric_group_profile(current)[["LabelGroupID", "RecipeID"]]
    current_profile["LabelGroupID"] = current_profile["LabelGroupID"].astype(str)
    train_recipe = current_profile.merge(splits, on="LabelGroupID", validate="one_to_one")
    train_counts = train_recipe[train_recipe["split"].eq("train")].groupby("RecipeID").size()
    result["current_train_recipe_groups"] = result["RecipeID"].map(train_counts).fillna(0).astype(int)
    result["recipe_history_effective_weight_cap"] = (
        recipe_cap_fraction * result["current_train_recipe_groups"]
    )
    eligible = result["eligibility_status"].eq("ELIGIBLE_STATIC_PROCESS_CONTRACT_MATCH")
    raw_sum = result.loc[eligible].groupby("RecipeID")["raw_time_decay_weight"].sum()
    result["recipe_raw_history_weight_sum"] = result["RecipeID"].map(raw_sum).fillna(0.0)
    scale = result["recipe_history_effective_weight_cap"].div(
        result["recipe_raw_history_weight_sum"].replace(0.0, np.nan)
    ).clip(upper=1.0).fillna(0.0)
    result["recipe_cap_scale"] = scale
    result["_group_training_weight"] = 0.0
    result.loc[eligible, "_group_training_weight"] = (
        result.loc[eligible, "raw_time_decay_weight"] * result.loc[eligible, "recipe_cap_scale"]
    )
    if result.loc[eligible, "_group_training_weight"].le(0).any():
        raise RuntimeError("ELIGIBLE_HISTORY_RECEIVED_NONPOSITIVE_WEIGHT")
    cap_audit = result.loc[eligible].groupby("RecipeID").agg(
        effective_weight=("_group_training_weight", "sum"),
        cap=("recipe_history_effective_weight_cap", "first"),
    )
    if (cap_audit["effective_weight"] > cap_audit["cap"] + 1e-10).any():
        raise RuntimeError("RECIPE_HISTORY_EFFECTIVE_WEIGHT_CAP_BREACH")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--current-groups", type=Path, default=DEFAULT_CURRENT_GROUPS)
    parser.add_argument("--history-groups", type=Path, default=DEFAULT_HISTORY_GROUPS)
    parser.add_argument("--h0-predictions", type=Path, default=DEFAULT_H0_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_output.mkdir(parents=True, exist_ok=True)
    current = pd.read_parquet(args.current)
    history = pd.read_parquet(args.history)
    current_groups = pd.read_csv(args.current_groups, low_memory=False)
    history_groups = pd.read_csv(args.history_groups, low_memory=False)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    metadata = {"s1_features": contract["stage1_features"], "s2_features": contract["stage2_features"]}

    current["LabelGroupID"] = current["LabelGroupID"].astype(str)
    history["LabelGroupID"] = history["LabelGroupID"].astype(str)
    current_groups["LabelGroupID"] = current_groups["LabelGroupID"].astype(str)
    history_groups["LabelGroupID"] = history_groups["LabelGroupID"].astype(str)
    eligibility = build_history_eligibility(current, history, current_groups, history_groups)
    weights = assign_history_weights(eligibility, current, current_groups)
    weights.to_csv(args.output_dir / "history_eligibility_and_weight_audit.csv", index=False, encoding="utf-8-sig")

    selected_weights = weights.loc[
        weights["eligibility_status"].eq("ELIGIBLE_STATIC_PROCESS_CONTRACT_MATCH"),
        ["LabelGroupID", "_group_training_weight"],
    ]
    selected_ids = set(selected_weights["LabelGroupID"])
    selected_history = history[history["LabelGroupID"].isin(selected_ids)].copy()
    selected_labels = history_groups[history_groups["LabelGroupID"].isin(selected_ids)].copy()
    selected_history = selected_history.merge(selected_weights, on="LabelGroupID", validate="many_to_one")
    current["_group_training_weight"] = 1.0

    # The only legacy version truth available is "unknown".  Preserve that
    # fact explicitly and prevent the parsed RecipeID PV token from entering
    # template/version audit semantics.
    selected_history["ProcessVersion"] = -1
    selected_history["TemplateKey"] = (
        selected_history["RecipeID"].astype(str) + "|"
        + selected_history["MixerLine"].astype(str) + "|"
        + selected_history["phase_route"].astype(str)
        + "|STATIC_TARGET_MATCHED|LEGACY_ORDER_PV_UNKNOWN"
    )

    frame, s1, s2, frame_contract = make_training_frame(
        current, selected_history, current_groups, selected_labels, metadata, include_history=True
    )
    train = frame[frame["split"].eq("train")].copy()
    validation = frame[frame["split"].eq("val")].copy()
    oot = frame[frame["split"].eq("oot")].copy()
    started = time.perf_counter()
    model = V371SourceWeightedGroupAtomicBatchModel(
        shrinkage_k=5.0,
        use_cluster_experts=True,
        use_material_route_matrix=True,
        use_silica_subsystem=True,
        use_cb_subsystem=True,
        crossfit_splits=5,
        time_col="lab_time",
        objective="regression",
    ).fit(train, s1, s2, target_col="MNY", cluster_col="material_system")
    fit_seconds = time.perf_counter() - started

    batch_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []
    trend_pairs: list[pd.DataFrame] = []
    for split_frame in (train, validation, oot):
        batch_pred, group_pred = group_predictions(model, split_frame, H2_NAME)
        batch_parts.append(batch_pred)
        group_parts.append(group_pred)
        split = str(group_pred["split"].iloc[0])
        metric_rows.append(metric_record(group_pred, H2_NAME, split))
        rows, pairs = trend_records(group_pred, H2_NAME, split)
        trend_rows.extend(rows)
        trend_pairs.extend(pairs)

    h2_batches = pd.concat(batch_parts, ignore_index=True)
    h2_groups = pd.concat(group_parts, ignore_index=True)
    h2_metrics = pd.DataFrame(metric_rows)
    h2_trends = pd.DataFrame(trend_rows)
    h2_pairs = pd.concat(trend_pairs, ignore_index=True, sort=False)
    h0_groups = pd.read_csv(args.h0_predictions, low_memory=False)
    h0_groups = h0_groups[h0_groups["Candidate"].eq(H0_NAME)].copy()
    h0_metrics = pd.DataFrame([
        metric_record(h0_groups, H0_NAME, split) for split in ("train", "val", "oot")
    ])
    h0_trend_rows: list[dict[str, Any]] = []
    h0_pair_parts: list[pd.DataFrame] = []
    for split in ("train", "val", "oot"):
        rows, pairs = trend_records(h0_groups, H0_NAME, split)
        h0_trend_rows.extend(rows)
        h0_pair_parts.extend(pairs)

    comparison_groups = pd.concat([h0_groups, h2_groups], ignore_index=True, sort=False)
    comparison_metrics = pd.concat([h0_metrics, h2_metrics], ignore_index=True)
    comparison_trends = pd.concat([pd.DataFrame(h0_trend_rows), h2_trends], ignore_index=True)
    comparison_pairs = pd.concat([*h0_pair_parts, h2_pairs], ignore_index=True, sort=False)
    h2_batches.to_parquet(args.output_dir / "h2_batch_predictions.parquet", index=False, compression="zstd")
    comparison_groups.to_csv(args.output_dir / "candidate_group_predictions.csv", index=False, encoding="utf-8-sig")
    comparison_metrics.to_csv(args.output_dir / "candidate_level_metrics.csv", index=False, encoding="utf-8-sig")
    comparison_trends.to_csv(args.output_dir / "candidate_trend_metrics.csv", index=False, encoding="utf-8-sig")
    comparison_pairs.to_csv(args.output_dir / "candidate_trend_pairs.csv", index=False, encoding="utf-8-sig")

    model_path = args.model_output / "h2_weighted_matched_history_phase_c1_shadow.joblib"
    joblib.dump(model, model_path)
    chart = args.output_dir / "actual_vs_prediction_full_overview.png"
    plot_overview(comparison_groups, comparison_metrics, chart)
    oot_metrics = comparison_metrics[comparison_metrics["split"].eq("oot")]
    h0 = oot_metrics[oot_metrics["candidate"].eq(H0_NAME)].iloc[0]
    h2 = oot_metrics[oot_metrics["candidate"].eq(H2_NAME)].iloc[0]
    oot_trends = comparison_trends[
        comparison_trends["split"].eq("oot")
        & comparison_trends["trend_scope"].eq("WITHIN_ORDER")
    ]
    promote = bool(
        h2["mae"] < h0["mae"]
        and h2["rmse"] < h0["rmse"]
        and abs(h2["bias"]) <= 1.5
        and h2["r2"] > 0
        and 0.70 <= h2["prediction_to_actual_std_ratio"] <= 1.30
    )
    selected_audit = weights[weights["_group_training_weight"].gt(0)]
    summary = {
        "experiment": "V3.7.1-M1-C1-WEIGHTED-MATCHED-HISTORY-H2-V1",
        "release_status": "NOT_RELEASED_DEVELOPMENT_DIAGNOSTIC",
        "h2_promote_over_h0": promote,
        "history_groups_considered": int(weights["LabelGroupID"].nunique()),
        "history_groups_eligible": int(selected_audit["LabelGroupID"].nunique()),
        "history_batches_eligible": int(len(selected_history)),
        "history_effective_group_weight": float(selected_audit["_group_training_weight"].sum()),
        "history_group_nominal_count_is_not_effective_vote_count": True,
        "weight_contract": {
            "base_weight": 0.25,
            "half_life_days": 180.0,
            "per_recipe_total_cap_fraction_of_current_train_groups": 0.25,
        },
        "process_version_contract": {
            "legacy_order_process_version": "UNAVAILABLE",
            "status": "LEGACY_ORDER_PROCESS_VERSION_UNAVAILABLE",
            "recipeid_embedded_pv_used_as_process_version": False,
            "consequence": "H2_IS_DIAGNOSTIC_AND_CANNOT_PROVE_PROCESSVERSION_TEMPLATE_MATCH",
        },
        "oot_mae_change_h2_minus_h0": float(h2["mae"] - h0["mae"]),
        "oot_rmse_change_h2_minus_h0": float(h2["rmse"] - h0["rmse"]),
        "model_sha256": sha256(model_path),
        "chart_sha256": sha256(chart),
        "fit_seconds": fit_seconds,
        "frame_contract": frame_contract,
        "source_weight_contract": model.source_weight_audit_,
        "stage3_calibration_enabled": False,
        "recommendation_status": "NOT_GENERATED_MODEL_VALIDATION_ONLY",
        "database_access_during_run": False,
        "local_mms_access": False,
        "archive_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "current": sha256(args.current),
            "history": sha256(args.history),
            "current_groups": sha256(args.current_groups),
            "history_groups": sha256(args.history_groups),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    metric_columns = [
        "candidate", "split", "groups", "batches", "mae", "rmse", "bias", "r2",
        "pearson", "spearman", "prediction_to_actual_std_ratio", "within_2_rate",
        "within_3_rate", "within_5_rate",
    ]
    trend_columns = [
        "candidate", "split", "pairs", "significant_pairs", "delta_pearson",
        "direction_accuracy", "delta_std_ratio", "level_gate_passed", "trend_gate_passed",
    ]
    report = f"""# V3.7.1 M1 H2 加权历史增强实验报告

## 结论

- 本轮只使用本地冻结快照，没有访问数据库、本地 MMS 或 Archive。
- H2 纳入 {summary['history_groups_eligible']} 个历史 Lab 组 / {summary['history_batches_eligible']} 个批次，但时间衰减与逐 RecipeID 限额后的总有效票数只有 {summary['history_effective_group_weight']:.3f} 组。
- 旧历史源没有订单真实 ProcessVersion；RecipeID 中的 `PVxx` 不是订单 ProcessVersion。本轮明确标记 `LEGACY_ORDER_PROCESS_VERSION_UNAVAILABLE`，因此 H2 只能用于诊断，不能证明版本级模板一致。
- H2 相对 H0 的 OOT MAE 变化为 {summary['oot_mae_change_h2_minus_h0']:+.4f} MU，RMSE 变化为 {summary['oot_rmse_change_h2_minus_h0']:+.4f} MU。
- 是否可晋级 H0：`{promote}`。Stage3、参数推荐和自动控制继续关闭。

## 组级 Level 指标

{markdown_table(comparison_metrics[metric_columns])}

## OOT 订单内趋势与波动指标

{markdown_table(oot_trends[trend_columns])}

## 历史准入与权重合同

1. 精确匹配 RecipeID、MixerLine、PhaseRoute。
2. 静态温度、填充因子和材料比例必须落在当前训练期同合同范围（含小容差）内。
3. 单个历史组基础权重 0.25，180 天半衰期。
4. 每个 RecipeID 的历史总有效权重不超过其当前训练 Lab 组数的 25%。
5. MNY 数值从未参与准入、相似性或权重计算。

## 评价口径

模型逐批次预测，再按 `OrderID+PalletID` 平均，仅与该 Pallet 唯一 Lab MNY 比较一次。散点全局相关性不能替代订单内 delta Pearson、方向准确率和 delta 标准差比。
"""
    report_path = args.output_dir / "M1_H2加权历史增强实验报告.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({
        "release_status": summary["release_status"],
        "h2_promote_over_h0": summary["h2_promote_over_h0"],
        "history_groups_eligible": summary["history_groups_eligible"],
        "history_effective_group_weight": summary["history_effective_group_weight"],
        "oot_mae_change_h2_minus_h0": summary["oot_mae_change_h2_minus_h0"],
        "model_sha256": summary["model_sha256"],
        "oot_metrics": oot_metrics.to_dict("records"),
        "oot_within_order_trend": oot_trends.to_dict("records"),
        "report": str(report_path),
        "chart": str(chart),
    }, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
