"""Diagnose train/validation/OOT stability of M1 within-order Phase deltas.

This is a snapshot-only diagnostic.  It does not alter the production model.
Each transition compares consecutive Lab/Pallet groups in the same physical
order and exact template, after batch features have been averaged per group.
Phase-bundle Ridge candidates are fit on train transitions only; the selected
bundle is chosen by validation delta MAE, never by OOT.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from Mooney_Prediction_Pipeline.model_training.run_v371_c1_history_augmented_phase_v1 import (
    markdown_table,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger"
COMMON = BASE / "v371_c1_history_augmented_phase_v1"
CURRENT = COMMON / "current_common_batch_features.parquet"
GROUPS = BASE / "global_rds_active_v1" / "m1_temporal_jit_v1" / "m1_temporal_jit_acceptance_input.csv"
OUTPUT = BASE / "m1_phase_delta_signal_v1"

PHASES = (
    "Stage2_DryMixing",
    "Stage3_OilLoading",
    "Stage4_WetMixing",
    "Stage5_PID",
    "Stage6_BottomMixing",
)
STATS = (
    "duration",
    "temp__mean",
    "temp__max",
    "temp__last",
    "power__integral",
    "torque__integral",
    "rotor__mean",
    "rotor__integral",
)


def phase_feature_bundles(columns: list[str]) -> dict[str, list[str]]:
    available = set(columns)
    bundles: dict[str, list[str]] = {}
    for phase in PHASES:
        features = [f"phase_curve__{phase}__{stat}" for stat in STATS]
        bundles[phase] = [feature for feature in features if feature in available]
    bundles["DryWetPIDBottom_All"] = list(dict.fromkeys(
        feature for phase in PHASES for feature in bundles[phase]
    ))
    return bundles


def build_group_frame(batch: pd.DataFrame, labels: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    label_columns = ["LabelGroupID", "MNY", "lab_time", "split"]
    labels = labels[label_columns].copy()
    labels["LabelGroupID"] = labels["LabelGroupID"].astype(str)
    if labels["LabelGroupID"].duplicated().any():
        raise ValueError("DUPLICATE_LAB_GROUP")
    batch = batch.copy()
    batch["LabelGroupID"] = batch["LabelGroupID"].astype(str)
    identities = [
        "OrderID", "RecipeID", "ProcessVersion", "MixerLine", "material_system",
        "phase_route", "TemplateKey",
    ]
    for feature in features:
        batch[feature] = pd.to_numeric(batch[feature], errors="coerce")
    aggregations = {column: "first" for column in identities}
    aggregations.update({feature: "mean" for feature in features})
    group = batch.groupby("LabelGroupID", as_index=False, dropna=False).agg(aggregations)
    group = group.merge(labels, on="LabelGroupID", validate="one_to_one")
    group["lab_time"] = pd.to_datetime(group["lab_time"], errors="raise", utc=True)
    return group


def build_transitions(group: pd.DataFrame, features: list[str], max_gap_days: float = 30.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    segment = ["OrderID", "RecipeID", "ProcessVersion", "MixerLine", "TemplateKey"]
    for _, part in group.sort_values("lab_time").groupby(segment, dropna=False, sort=False):
        ordered = part.sort_values("lab_time", kind="stable")
        records = list(ordered.to_dict("records"))
        for previous, current in zip(records, records[1:]):
            gap = (current["lab_time"] - previous["lab_time"]).total_seconds() / 86400.0
            if gap <= 0 or gap > max_gap_days or current["split"] != previous["split"]:
                continue
            row: dict[str, object] = {
                "from_group": previous["LabelGroupID"],
                "to_group": current["LabelGroupID"],
                "OrderID": current["OrderID"],
                "RecipeID": current["RecipeID"],
                "ProcessVersion": current["ProcessVersion"],
                "material_system": current["material_system"],
                "phase_route": current["phase_route"],
                "split": current["split"],
                "from_lab_time": previous["lab_time"],
                "to_lab_time": current["lab_time"],
                "gap_days": gap,
                "previous_mny": float(previous["MNY"]),
                "actual_mny": float(current["MNY"]),
                "actual_delta_mny": float(current["MNY"] - previous["MNY"]),
            }
            for feature in features:
                a = pd.to_numeric(pd.Series([previous[feature]]), errors="coerce").iloc[0]
                b = pd.to_numeric(pd.Series([current[feature]]), errors="coerce").iloc[0]
                row[f"delta__{feature}"] = b - a if pd.notna(a) and pd.notna(b) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def metric_row(frame: pd.DataFrame, prediction: np.ndarray, candidate: str, split: str) -> dict[str, object]:
    y_delta = frame["actual_delta_mny"].to_numpy(dtype=float)
    y_level = frame["actual_mny"].to_numpy(dtype=float)
    level_prediction = frame["previous_mny"].to_numpy(dtype=float) + prediction
    significant = np.abs(y_delta) >= 2.0
    direction = np.sign(prediction[significant]) == np.sign(y_delta[significant])
    actual_std = float(np.std(y_delta, ddof=0))
    prediction_std = float(np.std(prediction, ddof=0))
    return {
        "candidate": candidate,
        "split": split,
        "pairs": int(len(frame)),
        "significant_pairs": int(significant.sum()),
        "level_mae": float(mean_absolute_error(y_level, level_prediction)),
        "delta_mae": float(mean_absolute_error(y_delta, prediction)),
        "delta_rmse": float(mean_squared_error(y_delta, prediction) ** 0.5),
        "delta_bias": float(np.mean(prediction - y_delta)),
        "delta_pearson": float(pd.Series(y_delta).corr(pd.Series(prediction))),
        "direction_accuracy_abs_actual_ge_2": float(direction.mean()) if len(direction) else math.nan,
        "actual_delta_std": actual_std,
        "prediction_delta_std": prediction_std,
        "delta_std_ratio": prediction_std / actual_std if actual_std else math.nan,
    }


def fit_phase_candidates(
    transitions: pd.DataFrame, bundles: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    train = transitions[transitions["split"].eq("train")]
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    candidates: dict[str, Pipeline] = {}
    for bundle, features in bundles.items():
        delta_features = [f"delta__{feature}" for feature in features]
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler()),
            ("ridge", Ridge(alpha=100.0)),
        ])
        model.fit(train[delta_features], train["actual_delta_mny"])
        candidates[bundle] = model
        for split in ("train", "val", "oot"):
            part = transitions[transitions["split"].eq(split)].copy()
            if part.empty:
                continue
            predicted = model.predict(part[delta_features])
            metrics.append(metric_row(part, predicted, bundle, split))
            out = part[[
                "from_group", "to_group", "OrderID", "RecipeID", "ProcessVersion",
                "material_system", "phase_route", "split", "to_lab_time", "previous_mny",
                "actual_mny", "actual_delta_mny",
            ]].copy()
            out["candidate"] = bundle
            out["predicted_delta_mny"] = predicted
            out["predicted_mny"] = out["previous_mny"] + predicted
            predictions.append(out)
    for split in ("train", "val", "oot"):
        part = transitions[transitions["split"].eq(split)].copy()
        if part.empty:
            continue
        zero = np.zeros(len(part))
        metrics.append(metric_row(part, zero, "CARRY_FORWARD_ZERO_DELTA", split))
        out = part[[
            "from_group", "to_group", "OrderID", "RecipeID", "ProcessVersion",
            "material_system", "phase_route", "split", "to_lab_time", "previous_mny",
            "actual_mny", "actual_delta_mny",
        ]].copy()
        out["candidate"] = "CARRY_FORWARD_ZERO_DELTA"
        out["predicted_delta_mny"] = 0.0
        out["predicted_mny"] = out["previous_mny"]
        predictions.append(out)
    metric_frame = pd.DataFrame(metrics)
    val = metric_frame[
        metric_frame["split"].eq("val")
        & ~metric_frame["candidate"].eq("CARRY_FORWARD_ZERO_DELTA")
    ].sort_values(["delta_mae", "delta_rmse", "candidate"], kind="stable")
    selected = str(val.iloc[0]["candidate"])
    return metric_frame, pd.concat(predictions, ignore_index=True), selected


def association_audit(transitions: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        column = f"delta__{feature}"
        row: dict[str, object] = {"feature": feature}
        signs = []
        for split in ("train", "val", "oot"):
            part = transitions[[column, "actual_delta_mny", "split"]].dropna()
            part = part[part["split"].eq(split)]
            row[f"{split}_pairs"] = int(len(part))
            pearson = part[column].corr(part["actual_delta_mny"], method="pearson") if len(part) >= 8 else np.nan
            spearman = part[column].corr(part["actual_delta_mny"], method="spearman") if len(part) >= 8 else np.nan
            row[f"{split}_pearson"] = pearson
            row[f"{split}_spearman"] = spearman
            if pd.notna(spearman) and abs(spearman) > 1e-12:
                signs.append(int(np.sign(spearman)))
        row["spearman_sign_stable_across_available_splits"] = len(signs) >= 2 and len(set(signs)) == 1
        available = [abs(row[f"{split}_spearman"]) for split in ("train", "val", "oot") if pd.notna(row[f"{split}_spearman"])]
        row["minimum_abs_spearman_across_available_splits"] = min(available) if available else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["spearman_sign_stable_across_available_splits", "minimum_abs_spearman_across_available_splits"],
        ascending=[False, False], kind="stable",
    )


def plot(selected_predictions: pd.DataFrame, metrics: pd.DataFrame, selected: str, path: Path) -> None:
    oot = selected_predictions[
        selected_predictions["split"].eq("oot") & selected_predictions["candidate"].eq(selected)
    ].sort_values("to_lab_time")
    row = metrics[(metrics["split"].eq("oot")) & (metrics["candidate"].eq(selected))].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    low = min(oot["actual_mny"].min(), oot["predicted_mny"].min()) - 1
    high = max(oot["actual_mny"].max(), oot["predicted_mny"].max()) + 1
    axes[0, 0].scatter(oot["actual_mny"], oot["predicted_mny"], alpha=0.75)
    axes[0, 0].plot([low, high], [low, high], "--", color="black")
    axes[0, 0].set_title(f"OOT level: {selected} | MAE={row['level_mae']:.2f} MU")
    axes[0, 0].set_xlabel("Actual MNY")
    axes[0, 0].set_ylabel("Previous Lab + predicted process delta")
    delta_low = min(oot["actual_delta_mny"].min(), oot["predicted_delta_mny"].min()) - 1
    delta_high = max(oot["actual_delta_mny"].max(), oot["predicted_delta_mny"].max()) + 1
    axes[0, 1].scatter(oot["actual_delta_mny"], oot["predicted_delta_mny"], alpha=0.75, color="#d9792b")
    axes[0, 1].plot([delta_low, delta_high], [delta_low, delta_high], "--", color="black")
    axes[0, 1].axhline(0, color="gray", linewidth=0.8)
    axes[0, 1].axvline(0, color="gray", linewidth=0.8)
    axes[0, 1].set_title(f"OOT delta | r={row['delta_pearson']:.3f}, std ratio={row['delta_std_ratio']:.3f}")
    axes[0, 1].set_xlabel("Actual ΔMNY")
    axes[0, 1].set_ylabel("Predicted ΔMNY")
    x = np.arange(len(oot))
    axes[1, 0].plot(x, oot["actual_mny"], color="black", label="Actual")
    axes[1, 0].plot(x, oot["predicted_mny"], color="#3769a8", label="Predicted")
    axes[1, 0].set_title("OOT chronological transition endpoints")
    axes[1, 0].legend()
    compare = metrics[metrics["split"].eq("oot")].sort_values("delta_mae")
    axes[1, 1].barh(compare["candidate"], compare["delta_mae"], color="#6b8e6b")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("OOT ΔMNY MAE by train-fitted Phase bundle")
    axes[1, 1].set_xlabel("ΔMNY MAE (MU)")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("M1 within-order Phase-delta signal diagnostic", fontsize=16)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    batch = pd.read_parquet(CURRENT)
    labels = pd.read_csv(GROUPS, low_memory=False)
    bundles = phase_feature_bundles(batch.columns.tolist())
    features = bundles["DryWetPIDBottom_All"]
    group = build_group_frame(batch, labels, features)
    transitions = build_transitions(group, features)
    if set(transitions["split"]) != {"train", "val", "oot"}:
        raise ValueError("MISSING_REQUIRED_TRANSITION_SPLIT")
    metrics, predictions, selected = fit_phase_candidates(transitions, bundles)
    associations = association_audit(transitions, features)
    transitions.to_parquet(OUTPUT / "within_order_phase_transitions.parquet", index=False, compression="zstd")
    metrics.to_csv(OUTPUT / "phase_bundle_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT / "phase_bundle_predictions.csv", index=False, encoding="utf-8-sig")
    associations.to_csv(OUTPUT / "phase_delta_associations.csv", index=False, encoding="utf-8-sig")
    chart = OUTPUT / "actual_vs_prediction_full_overview.png"
    plot(predictions, metrics, selected, chart)
    selected_metrics = metrics[metrics["candidate"].eq(selected)].set_index("split")
    carry = metrics[metrics["candidate"].eq("CARRY_FORWARD_ZERO_DELTA")].set_index("split")
    release = bool(
        selected_metrics.loc["val", "delta_mae"] < carry.loc["val", "delta_mae"]
        and selected_metrics.loc["oot", "delta_mae"] < carry.loc["oot", "delta_mae"]
        and selected_metrics.loc["oot", "direction_accuracy_abs_actual_ge_2"] >= 0.60
        and 0.70 <= selected_metrics.loc["oot", "delta_std_ratio"] <= 1.30
    )
    summary = {
        "experiment": "M1-WITHIN-ORDER-PHASE-DELTA-SIGNAL-V1",
        "status": "NOT_RELEASED_DEVELOPMENT_DIAGNOSTIC",
        "selected_by_validation_delta_mae": selected,
        "promote_phase_delta_candidate": release,
        "pair_counts": transitions.groupby("split").size().to_dict(),
        "selected_metrics": selected_metrics.reset_index().to_dict("records"),
        "carry_forward_metrics": carry.reset_index().to_dict("records"),
        "stable_feature_count": int(associations["spearman_sign_stable_across_available_splits"].sum()),
        "stage3_calibration_enabled": False,
        "database_access": False,
        "local_mms_access": False,
        "archive_access": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# M1 同订单 Phase Delta 信号诊断 V1

## 结论

- 使用当前同源 Phase 快照；无数据库、本地 MMS 或 Archive 访问。
- 相邻 Lab 组必须属于同一 OrderID、RecipeID、ProcessVersion、MixerLine 和 TemplateKey，且两组位于同一时间切片。
- 训练/验证/OOT 转移数：{summary['pair_counts']}。
- 仅按验证集 ΔMNY MAE 选择的 Phase bundle：`{selected}`。
- 是否晋级：`{release}`。本实验不修改 C1，不生成 Step 推荐。

## 指标

{markdown_table(metrics)}

## 解释边界

Phase bundle 只使用当前组与前一组的实际过程特征差，不使用当前或未来 Lab 值作为输入。若某个阶段在训练、验证和 OOT 的关联方向不稳定，即使总体相关较高，也不能作为控制方向证据。
"""
    (OUTPUT / "M1同订单Phase_Delta信号诊断报告.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "selected": selected,
        "promote": release,
        "pair_counts": summary["pair_counts"],
        "oot_selected": selected_metrics.loc["oot"].to_dict(),
        "oot_carry_forward": carry.loc["oot"].to_dict(),
        "chart": str(chart),
    }, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
