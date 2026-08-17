"""Evaluate a low-dimensional oil-post-temperature x Wet-budget H3 layer."""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

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
from Mooney_Prediction_Pipeline.model_training.v371_group_atomic_batch_v1 import (
    V371GroupAtomicBatchModel,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger"
COMMON = BASE / "v371_c1_history_augmented_phase_v1"
CURRENT = COMMON / "current_common_batch_features.parquet"
CONTRACT = COMMON / "common_feature_contract.json"
GROUPS = BASE / "global_rds_active_v1" / "m1_temporal_jit_v1" / "m1_temporal_jit_acceptance_input.csv"
H0_PREDICTIONS = COMMON / "candidate_group_predictions.csv"
OUTPUT = BASE / "v371_c1_thermal_budget_h3_v1"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_c1_thermal_budget_h3_v1"

H0_NAME = "H0_CURRENT_ONLY_PHASE_C1"
H3_NAME = "H3_THERMAL_BUDGET_PHASE_C1"
RAW_INPUTS = {
    "oil_temp": "phase_curve__Stage3_OilLoading__temp__last",
    "wet_duration": "phase_curve__Stage4_WetMixing__duration",
    "wet_rotor_integral": "phase_curve__Stage4_WetMixing__rotor__integral",
}
MECHANISM_FEATURES = (
    "mechanism_stage3_oil_temp_excess",
    "mechanism_stage4_wet_duration_deficit",
    "mechanism_stage4_wet_rotor_integral_deficit",
    "mechanism_stage3_oil_temp_x_wet_duration_deficit",
    "mechanism_stage3_oil_temp_x_wet_rotor_deficit",
    "mechanism_oil_wet_route_applicable",
)


def build_train_only_nominals(
    current: pd.DataFrame, groups: pd.DataFrame, minimum_exact_groups: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add target-free mechanism features using current-train process nominals."""

    work = current.copy()
    work["LabelGroupID"] = work["LabelGroupID"].astype(str)
    labels = groups[["LabelGroupID", "split"]].copy()
    labels["LabelGroupID"] = labels["LabelGroupID"].astype(str)
    identity = [
        "RecipeID", "ProcessVersion", "MixerLine", "material_system", "phase_route",
    ]
    aggregations = {column: "first" for column in identity}
    aggregations.update({source: "mean" for source in RAW_INPUTS.values()})
    profile = work.groupby("LabelGroupID", as_index=False, dropna=False).agg(aggregations)
    profile = profile.merge(labels, on="LabelGroupID", validate="one_to_one")
    train = profile[profile["split"].eq("train")].copy()

    exact_keys = ["RecipeID", "ProcessVersion", "MixerLine", "phase_route"]
    exact = train.groupby(exact_keys, dropna=False).agg(
        exact_groups=("LabelGroupID", "nunique"),
        **{f"nominal_{name}": (source, "median") for name, source in RAW_INPUTS.items()},
    ).reset_index()
    exact.loc[exact["exact_groups"].lt(minimum_exact_groups), [
        f"nominal_{name}" for name in RAW_INPUTS
    ]] = np.nan
    fallback_keys = ["material_system", "MixerLine", "phase_route"]
    fallback = train.groupby(fallback_keys, dropna=False).agg(
        fallback_groups=("LabelGroupID", "nunique"),
        **{f"fallback_{name}": (source, "median") for name, source in RAW_INPUTS.items()},
    ).reset_index()

    augmented = work.merge(exact, on=exact_keys, how="left", validate="many_to_one")
    augmented = augmented.merge(fallback, on=fallback_keys, how="left", validate="many_to_one")
    audit_rows = []
    for name, source in RAW_INPUTS.items():
        augmented[f"used_nominal_{name}"] = augmented[f"nominal_{name}"].fillna(
            augmented[f"fallback_{name}"]
        )
        audit_rows.append({
            "mechanism_input": name,
            "source_feature": source,
            "exact_nominal_rows": int(augmented[f"nominal_{name}"].notna().sum()),
            "fallback_nominal_rows": int(
                (
                    augmented[f"nominal_{name}"].isna()
                    & augmented[f"fallback_{name}"].notna()
                ).sum()
            ),
            "unavailable_nominal_rows": int(augmented[f"used_nominal_{name}"].isna().sum()),
        })

    oil_route = augmented["phase_route"].eq("oil_wet")
    oil = pd.to_numeric(augmented[RAW_INPUTS["oil_temp"]], errors="coerce")
    wet_duration = pd.to_numeric(augmented[RAW_INPUTS["wet_duration"]], errors="coerce")
    wet_rotor = pd.to_numeric(augmented[RAW_INPUTS["wet_rotor_integral"]], errors="coerce")
    oil_excess = (oil - augmented["used_nominal_oil_temp"]).clip(lower=0)
    duration_deficit = (augmented["used_nominal_wet_duration"] - wet_duration).clip(lower=0)
    rotor_deficit = (augmented["used_nominal_wet_rotor_integral"] - wet_rotor).clip(lower=0)
    augmented["mechanism_stage3_oil_temp_excess"] = oil_excess.where(oil_route, 0.0)
    augmented["mechanism_stage4_wet_duration_deficit"] = duration_deficit.where(oil_route, 0.0)
    augmented["mechanism_stage4_wet_rotor_integral_deficit"] = rotor_deficit.where(oil_route, 0.0)
    augmented["mechanism_stage3_oil_temp_x_wet_duration_deficit"] = (
        oil_excess * duration_deficit
    ).where(oil_route, 0.0)
    augmented["mechanism_stage3_oil_temp_x_wet_rotor_deficit"] = (
        oil_excess * rotor_deficit
    ).where(oil_route, 0.0)
    augmented["mechanism_oil_wet_route_applicable"] = oil_route.astype(float)
    for feature in MECHANISM_FEATURES:
        if augmented.loc[~oil_route, feature].fillna(0).ne(0).any():
            raise RuntimeError("NO_OIL_ROUTE_MECHANISM_MUST_BE_PHYSICAL_ZERO")
    audit = pd.DataFrame(audit_rows)
    return augmented, audit


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    current = pd.read_parquet(CURRENT)
    groups = pd.read_csv(GROUPS, low_memory=False)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    current_augmented, nominal_audit = build_train_only_nominals(current, groups)
    nominal_audit.to_csv(OUTPUT / "thermal_nominal_audit.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "s1_features": contract["stage1_features"],
        "s2_features": [*contract["stage2_features"], *MECHANISM_FEATURES],
    }
    frame, s1, s2, frame_contract = make_training_frame(
        current_augmented, current_augmented.iloc[0:0], groups, groups.iloc[0:0],
        metadata, include_history=False,
    )
    train = frame[frame["split"].eq("train")].copy()
    validation = frame[frame["split"].eq("val")].copy()
    oot = frame[frame["split"].eq("oot")].copy()
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
    ).fit(train, s1, s2, target_col="MNY", cluster_col="material_system")
    fit_seconds = time.perf_counter() - started
    h3_groups = []
    h3_batches = []
    metric_rows = []
    trend_rows = []
    pair_parts = []
    for split_frame in (train, validation, oot):
        batch_pred, group_pred = group_predictions(model, split_frame, H3_NAME)
        h3_batches.append(batch_pred)
        h3_groups.append(group_pred)
        split = str(group_pred["split"].iloc[0])
        metric_rows.append(metric_record(group_pred, H3_NAME, split))
        rows, pairs = trend_records(group_pred, H3_NAME, split)
        trend_rows.extend(rows)
        pair_parts.extend(pairs)
    h3_group = pd.concat(h3_groups, ignore_index=True)
    h3_metric = pd.DataFrame(metric_rows)
    h3_trend = pd.DataFrame(trend_rows)
    h0 = pd.read_csv(H0_PREDICTIONS, low_memory=False)
    h0 = h0[h0["Candidate"].eq(H0_NAME)].copy()
    h0_metric = pd.DataFrame([metric_record(h0, H0_NAME, split) for split in ("train", "val", "oot")])
    h0_trend_rows = []
    h0_pairs = []
    for split in ("train", "val", "oot"):
        rows, pairs = trend_records(h0, H0_NAME, split)
        h0_trend_rows.extend(rows)
        h0_pairs.extend(pairs)
    comparison_groups = pd.concat([h0, h3_group], ignore_index=True, sort=False)
    metrics = pd.concat([h0_metric, h3_metric], ignore_index=True)
    trends = pd.concat([pd.DataFrame(h0_trend_rows), h3_trend], ignore_index=True)
    pairs = pd.concat([*h0_pairs, *pair_parts], ignore_index=True, sort=False)
    pd.concat(h3_batches, ignore_index=True).to_parquet(
        OUTPUT / "h3_batch_predictions.parquet", index=False, compression="zstd"
    )
    comparison_groups.to_csv(OUTPUT / "candidate_group_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT / "candidate_level_metrics.csv", index=False, encoding="utf-8-sig")
    trends.to_csv(OUTPUT / "candidate_trend_metrics.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(OUTPUT / "candidate_trend_pairs.csv", index=False, encoding="utf-8-sig")
    model_path = MODEL_OUTPUT / "h3_thermal_budget_phase_c1_shadow.joblib"
    joblib.dump(model, model_path)
    chart = OUTPUT / "actual_vs_prediction_full_overview.png"
    plot_overview(comparison_groups, metrics, chart)
    oot_metrics = metrics[metrics["split"].eq("oot")]
    val_metrics = metrics[metrics["split"].eq("val")]
    h0_oot = oot_metrics[oot_metrics["candidate"].eq(H0_NAME)].iloc[0]
    h3_oot = oot_metrics[oot_metrics["candidate"].eq(H3_NAME)].iloc[0]
    h0_val = val_metrics[val_metrics["candidate"].eq(H0_NAME)].iloc[0]
    h3_val = val_metrics[val_metrics["candidate"].eq(H3_NAME)].iloc[0]
    oot_order_trend = trends[
        trends["split"].eq("oot") & trends["trend_scope"].eq("WITHIN_ORDER")
    ]
    promote = bool(
        h3_val["mae"] < h0_val["mae"]
        and h3_oot["mae"] < h0_oot["mae"]
        and h3_oot["rmse"] < h0_oot["rmse"]
        and abs(h3_oot["bias"]) <= 1.5
    )
    summary = {
        "experiment": "V3.7.1-M1-C1-THERMAL-BUDGET-H3-V1",
        "status": "NOT_RELEASED_DEVELOPMENT_DIAGNOSTIC",
        "promote_h3_over_h0": promote,
        "validation_mae_change_h3_minus_h0": float(h3_val["mae"] - h0_val["mae"]),
        "oot_mae_change_h3_minus_h0": float(h3_oot["mae"] - h0_oot["mae"]),
        "oot_rmse_change_h3_minus_h0": float(h3_oot["rmse"] - h0_oot["rmse"]),
        "feature_contract": list(MECHANISM_FEATURES),
        "nominal_fit_scope": "CURRENT_TRAIN_GROUPS_ONLY_TARGET_FREE",
        "fit_seconds": fit_seconds,
        "model_sha256": sha256(model_path),
        "chart_sha256": sha256(chart),
        "frame_contract": frame_contract,
        "stage3_calibration_enabled": False,
        "database_access": False,
        "local_mms_access": False,
        "archive_access": False,
        "recommendation_status": "NOT_GENERATED_MODEL_VALIDATION_ONLY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    focus = [
        "candidate", "split", "groups", "mae", "rmse", "bias", "r2", "pearson",
        "prediction_to_actual_std_ratio", "within_2_rate", "within_3_rate",
    ]
    trend_focus = oot_order_trend[[
        "candidate", "pairs", "significant_pairs", "delta_pearson", "direction_accuracy",
        "delta_std_ratio", "level_gate_passed", "trend_gate_passed",
    ]]
    report = f"""# V3.7.1 M1 H3 热预算耦合特征实验

## 结论

- H3 只新增 6 个目标无关机制特征；训练期 nominal 只从当前 train Lab 组构造。
- 机制定义：油后温度超额、Wet duration/rotor integral 缺口及两项乘积；无油路线为显式物理零。
- Validation MAE 变化 H3-H0：{summary['validation_mae_change_h3_minus_h0']:+.4f} MU。
- OOT MAE/RMSE 变化 H3-H0：{summary['oot_mae_change_h3_minus_h0']:+.4f}/{summary['oot_rmse_change_h3_minus_h0']:+.4f} MU。
- 是否晋级：`{promote}`。Stage3、推荐和控制继续关闭。

## Level 指标

{markdown_table(metrics[focus])}

## OOT 订单内趋势

{markdown_table(trend_focus)}

本轮没有访问数据库、本地 MMS 或 Archive。结果只用于判断专家热预算机制能否稳定改善四阶段模型。
"""
    report_path = OUTPUT / "M1_H3热预算耦合特征实验报告.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({
        "promote": promote,
        "validation_mae_change": summary["validation_mae_change_h3_minus_h0"],
        "oot_mae_change": summary["oot_mae_change_h3_minus_h0"],
        "oot_metrics": oot_metrics.to_dict("records"),
        "oot_order_trend": oot_order_trend.to_dict("records"),
        "chart": str(chart),
    }, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
