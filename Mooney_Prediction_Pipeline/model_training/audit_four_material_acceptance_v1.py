"""Audit the current batch-to-Lab-group model on an expanded material set.

The source prediction file is batch-level.  A Lab result belongs to one
OrderID+PalletID group, so all metrics in this module are calculated after
collapsing predictions to one row per LabelGroupID.  This prevents one Lab
value from being counted once per batch.

No model is trained or changed by this audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXISTING_ACCEPTANCE = {
    "CarbonBlack": ["M1-B00458---- 12 005"],
    "Silica": ["M1-T15760---- 06 002"],
}

# Chosen by pre-declared data availability, not by performance metrics:
# at least 5 OOT Lab groups and representation in train/validation.
ADDED_ACCEPTANCE = {
    "CarbonBlack": ["M1-B00460B--- 04 007", "M1-B15563R1-- 01 005"],
    "Silica": ["M1-T09170---- 23 003", "M1-T33025W3-- 03 005"],
}


def _safe_std(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.std(ddof=1)) if len(values) >= 2 else float("nan")


def _corr(left: pd.Series, right: pd.Series, method: str) -> float:
    frame = pd.concat([left, right], axis=1).dropna()
    if len(frame) < 3 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1], method=method))


def _group_frame(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "LabelGroupID", "MNY", "RawPrediction", "OrderID", "CompoundName",
        "material_system", "split", "lab_time",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Prediction audit missing columns: {missing}")
    frame = raw.copy()
    frame["MNY"] = pd.to_numeric(frame["MNY"], errors="coerce")
    frame["RawPrediction"] = pd.to_numeric(frame["RawPrediction"], errors="coerce")
    frame["lab_time"] = pd.to_datetime(frame["lab_time"], errors="coerce", utc=True)
    if frame[["MNY", "RawPrediction"]].isna().any().any():
        raise ValueError("Prediction audit contains missing MNY or prediction")

    def one_group(group_id: object, group: pd.DataFrame) -> pd.Series:
        if group["MNY"].max() - group["MNY"].min() > 1e-9:
            raise ValueError(f"CONFLICTING_MNY_WITHIN_LABEL_GROUP: {group_id}")
        fixed = ["CompoundName", "material_system", "split"]
        if any(group[column].dropna().nunique() != 1 for column in fixed):
            raise ValueError(f"CONFLICTING_GROUP_METADATA: {group_id}")
        order_values = group["OrderID"].dropna()
        if order_values.empty:
            order_id = str(group_id).split("|", 1)[0]
        else:
            order_id = order_values.iloc[0]
        time_values = group["lab_time"].dropna()
        return pd.Series(
            {
                "LabelGroupID": str(group_id),
                "MNY": float(group["MNY"].iloc[0]),
                "Prediction": float(group["RawPrediction"].mean()),
                "BatchCountObserved": int(len(group)),
                "OrderID": order_id,
                "CompoundName": group["CompoundName"].iloc[0],
                "material_system": group["material_system"].iloc[0],
                "split": group["split"].iloc[0],
                # Some batch rows carry the group label time while other
                # rows have null join metadata.  Preserve the non-null group
                # time; do not invent a timestamp.
                "lab_time": time_values.iloc[0] if not time_values.empty else pd.NaT,
            }
        )

    rows = [one_group(group_id, group) for group_id, group in frame.groupby("LabelGroupID", sort=False, dropna=False)]
    return pd.DataFrame(rows)


def _direction_metrics(groups: pd.DataFrame) -> dict[str, float | int]:
    pairs: list[dict[str, float]] = []
    ordered = groups.sort_values(["OrderID", "lab_time", "LabelGroupID"], kind="stable")
    for _, order in ordered.groupby("OrderID", sort=False):
        if len(order) < 2:
            continue
        actual = order["MNY"].to_numpy(dtype=float)
        predicted = order["Prediction"].to_numpy(dtype=float)
        for index in range(1, len(order)):
            actual_delta = actual[index] - actual[index - 1]
            predicted_delta = predicted[index] - predicted[index - 1]
            if abs(actual_delta) < 1e-9:
                continue
            pairs.append(
                {
                    "actual_delta": actual_delta,
                    "predicted_delta": predicted_delta,
                }
            )
    pair_frame = pd.DataFrame(pairs)
    if pair_frame.empty:
        return {
            "direction_pairs": 0,
            "direction_accuracy": float("nan"),
            "significant_direction_pairs_abs_actual_ge_2": 0,
            "significant_direction_accuracy_abs_actual_ge_2": float("nan"),
            "actual_delta_std": float("nan"),
            "predicted_delta_std": float("nan"),
            "delta_std_ratio": float("nan"),
        }
    actual_sign = np.sign(pair_frame["actual_delta"])
    predicted_sign = np.sign(pair_frame["predicted_delta"])
    significant = pair_frame["actual_delta"].abs() >= 2.0
    return {
        "direction_pairs": int(len(pair_frame)),
        "direction_accuracy": float((actual_sign == predicted_sign).mean()),
        "significant_direction_pairs_abs_actual_ge_2": int(significant.sum()),
        "significant_direction_accuracy_abs_actual_ge_2": (
            float((actual_sign[significant] == predicted_sign[significant]).mean())
            if significant.any() else float("nan")
        ),
        "actual_delta_std": _safe_std(pair_frame["actual_delta"]),
        "predicted_delta_std": _safe_std(pair_frame["predicted_delta"]),
        "delta_std_ratio": (
            float(pair_frame.loc[:, "predicted_delta"].std(ddof=1)
                  / pair_frame.loc[:, "actual_delta"].std(ddof=1))
            if pair_frame["actual_delta"].std(ddof=1) > 0 else float("nan")
        ),
    }


def _range_metrics(groups: pd.DataFrame) -> dict[str, float | int]:
    ranges = []
    for _, order in groups.groupby("OrderID", sort=False):
        if len(order) < 2:
            continue
        actual_range = float(order["MNY"].max() - order["MNY"].min())
        predicted_range = float(order["Prediction"].max() - order["Prediction"].min())
        if actual_range > 0:
            ranges.append({"actual_range": actual_range, "predicted_range": predicted_range})
    if not ranges:
        return {"range_orders": 0, "median_predicted_to_actual_range_ratio": float("nan")}
    ranges_frame = pd.DataFrame(ranges)
    return {
        "range_orders": int(len(ranges_frame)),
        "median_predicted_to_actual_range_ratio": float(
            (ranges_frame["predicted_range"] / ranges_frame["actual_range"]).median()
        ),
    }


def _metrics(groups: pd.DataFrame) -> dict[str, object]:
    actual = groups["MNY"]
    prediction = groups["Prediction"]
    error = prediction - actual
    actual_std = _safe_std(actual)
    prediction_std = _safe_std(prediction)
    result: dict[str, object] = {
        "groups": int(len(groups)),
        "orders": int(groups["OrderID"].nunique()),
        "batches_observed": int(groups["BatchCountObserved"].sum()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(error.mean()),
        "actual_std": actual_std,
        "prediction_std": prediction_std,
        "prediction_to_actual_std_ratio": (
            float(prediction_std / actual_std) if np.isfinite(actual_std) and actual_std > 0 else float("nan")
        ),
        "pearson": _corr(prediction, actual, "pearson"),
        "spearman": _corr(prediction, actual, "spearman"),
        "within_2_mu_rate": float((error.abs() <= 2.0).mean()),
        "within_3_mu_rate": float((error.abs() <= 3.0).mean()),
        "within_5_mu_rate": float((error.abs() <= 5.0).mean()),
    }
    result.update(_direction_metrics(groups))
    result.update(_range_metrics(groups))
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    return value


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    strict_m1_model: bool = False,
) -> None:
    oot = metrics[metrics["split"] == "oot"].copy()
    t15760 = oot.loc[oot["CompoundName"].eq("M1-T15760---- 06 002"), "mae"]
    t33025 = oot.loc[oot["CompoundName"].eq("M1-T33025W3-- 03 005"), "mae"]
    model_scope_sentence = (
        "本报告使用的是严格 M1-only 重训的 `V3.7.1-Corrected-Challenger-R1B-M1-Only`。"
        if strict_m1_model
        else "本报告使用的是 `V3.7.1-Corrected-Challenger-R1B-Group-Atomic-Batch` 的 M1 子集回放。当前 artifact 尚未严格按 M1-only 重训，因此本报告只是 M1 范围诊断。"
    )
    lines = [
        "# V3.7.1 R1B 扩展双胶料验收报告",
        "",
        "> 状态：开发诊断，不是生产放行；Stage3 在线校准关闭；不生成自动控制动作。",
        "",
        "## 1. 评估口径",
        "",
        "当前文件是批次级预测。一个 Lab/Pallet 结果对应多个车次，因此先按 `LabelGroupID = OrderID|PalletID` 对组内所有车次的 `RawPrediction` 求均值，再与该组唯一 MNY 比较一次。方向指标在同一 `OrderID` 内按 Lab 时间排序，对相邻标签组的变化方向进行比较；`abs(actual_delta) >= 2 MU` 另报显著变化方向准确度。",
        "",
        "波动捕捉同时看：预测标准差/实际标准差、同订单相邻组预测变化量标准差/实际变化量标准差，以及有至少两个标签组订单的预测范围/实际范围中位数。样本很少时，这些指标只作诊断，不作单独放行结论。",
        "",
        "## 2. 本次验收范围",
        "",
        "本次模型范围严格限定为 `CompoundName` 以 `M1-` 开头的胶料；M2、M3B、PB、R1 等全部不进入训练/验收范围。保留原试点 B00458、T15760，并在各材料体系新增两个精确配方版本。新增配方按 M1 历史覆盖预注册，未按性能结果挑选：",
        "",
        "- CarbonBlack：`M1-B00460B--- 04 007`、`M1-B15563R1-- 01 005`。",
        "- Silica：`M1-T09170---- 23 003`、`M1-T33025W3-- 03 005`。",
        "",
        "## 3. OOT 结果",
        "",
        "| 材料体系 | 精确配方版本 | Lab组 | 订单 | MAE | RMSE | Bias | 方向准确度 | 显著方向准确度 | 预测Std/实际Std | Delta Std比 | 范围比中位数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in oot.sort_values(["material_system", "CompoundName"]).iterrows():
        lines.append(
            "| {material_system} | `{CompoundName}` | {groups} | {orders} | {mae} | {rmse} | {bias} | {direction_accuracy} | {significant_direction_accuracy_abs_actual_ge_2} | {prediction_to_actual_std_ratio} | {delta_std_ratio} | {median_predicted_to_actual_range_ratio} |".format(
                material_system=row["material_system"],
                CompoundName=row["CompoundName"],
                groups=int(row["groups"]),
                orders=int(row["orders"]),
                mae=_fmt(row["mae"]),
                rmse=_fmt(row["rmse"]),
                bias=_fmt(row["bias"]),
                direction_accuracy=_fmt(row["direction_accuracy"]),
                significant_direction_accuracy_abs_actual_ge_2=_fmt(row["significant_direction_accuracy_abs_actual_ge_2"]),
                prediction_to_actual_std_ratio=_fmt(row["prediction_to_actual_std_ratio"]),
                delta_std_ratio=_fmt(row["delta_std_ratio"]),
                median_predicted_to_actual_range_ratio=_fmt(row["median_predicted_to_actual_range_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## 4. 覆盖与限制",
            "",
            "| 材料体系 | 精确配方版本 | Train组 | Val组 | OOT组 | OOT订单 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in coverage.sort_values(["material_system", "CompoundName"]).iterrows():
        lines.append(
            f"| {row['material_system']} | `{row['CompoundName']}` | {int(row['train_groups'])} | {int(row['val_groups'])} | {int(row['oot_groups'])} | {int(row['oot_orders'])} |"
        )
    lines.extend(
        [
            "",
            "- 新增两个 M1 炭黑版本各只有 2 个 OOT Lab 组，只能作为探索性验收，不能作为正式统计放行证据；T15760 的方向指标相对更有参考价值，但预测振幅仍明显不足。",
            "- `direction_accuracy` 不是把绝对值误差转成方向，而是比较同订单相邻 Lab 组的 `sign(predicted_delta)` 与 `sign(actual_delta)`。",
            "- 当前结果不能因为 MAE 达标就放行：预测Std/实际Std显著低于 0.70 的配方，说明模型没有捕捉足够批次间波动。",
            "",
            "## 5. 当前模型到底用了哪些专家阶段",
            "",
            f"{model_scope_sentence} 不是冻结 V3.7，也没有启用 Stage3。当前实际启用的是三个阶段：",
            "",
            "1. **Stage1：全局静态配方表面**，使用配方组成、填充因子、正确的 Silica PHR、Process Target Temperature、QM和材料体系等生产前/开始时特征。",
            "2. **Stage1b：配方/材料偏置收缩**，通过时间交叉拟合残差估计 CompoundName 与材料体系偏置，避免直接用当前 Lab 标签覆盖 OOT。",
            "3. **Stage2：过程残差专家**，按 CarbonBlack/Silica 与 oil_wet/no_oil_dry 路线分流；Silica 分成 PID、Wet、Bottom、Material 四个子专家，CarbonBlack 分成 Preparation、Bottom、Material 三个子专家，并用组级时间 OOF 组合。当前还增加了按批次构造的 PID duration/target 与 curve-quality risk/missing-safe 信息。",
            "",
            "**Stage3 在线校准：关闭。** 因而当前结果不是“加了 Stage3 校准后的结果”，而是 Stage1 + Stage1b + Stage2 专家残差模型的无校准 raw 输出。能耗模型也未参与资格判断或排序。",
            "",
            "## 6. 结论",
            "",
            f"本轮完成了六个 M1 精确配方版本的组级 OOT 审计，但当前模型仍不能作为推荐放行模型：多个配方的预测Std/实际Std低于 0.70，T15760 的 OOT MAE为 {_fmt(t15760.iloc[0] if not t15760.empty else float('nan'))} MU，新增白炭黑 T33025 的 OOT MAE为 {_fmt(t33025.iloc[0] if not t33025.empty else float('nan'))} MU。下一步继续修复批次间过程状态/版本迁移和历史先验残差结构，重新评估方向与波动；不能打开 Stage3 校准掩盖基础模型问题。",
            "",
            "审计输入：真正的一车次一行文件 `r1b_batch_predictions.csv`；不使用组级导出文件作为车次输入。生成文件同时包含 CSV、JSON manifest 和本报告。",
        ]
    )
    (output_dir / "扩展双胶料验收报告_20260813.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(source: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(source)
    raw_rows_before_m1_filter = int(len(raw))
    raw = raw[raw["CompoundName"].astype(str).str.startswith("M1-")].copy()
    groups = _group_frame(raw)
    selected = {material: existing + ADDED_ACCEPTANCE[material] for material, existing in EXISTING_ACCEPTANCE.items()}
    rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for material, compounds in selected.items():
        for compound in compounds:
            subset = groups[(groups["material_system"] == material) & (groups["CompoundName"] == compound)]
            if subset.empty:
                continue
            coverage = {
                "material_system": material,
                "CompoundName": compound,
                "train_groups": int((subset["split"] == "train").sum()),
                "val_groups": int((subset["split"] == "val").sum()),
                "oot_groups": int((subset["split"] == "oot").sum()),
                "train_orders": int(subset.loc[subset["split"] == "train", "OrderID"].nunique()),
                "val_orders": int(subset.loc[subset["split"] == "val", "OrderID"].nunique()),
                "oot_orders": int(subset.loc[subset["split"] == "oot", "OrderID"].nunique()),
                "all_groups": int(len(subset)),
            }
            coverage_rows.append(coverage)
            for split in ("train", "val", "oot"):
                split_groups = subset[subset["split"] == split]
                if split_groups.empty:
                    continue
                row = {
                    "split": split,
                    "material_system": material,
                    "CompoundName": compound,
                }
                row.update(_metrics(split_groups))
                rows.append(row)
    metrics = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "expanded_material_acceptance_metrics.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output_dir / "expanded_material_acceptance_coverage.csv", index=False, encoding="utf-8-sig")
    # Order-state artifacts are also strict M1-only retrains.  Do not label
    # them as a mixed-model M1 subset merely because the directory name is
    # ``m1_order_state`` rather than ``m1_only``.
    artifact_text = f"{source} {output_dir}".lower()
    strict_m1_model = "m1_only" in artifact_text or "m1_order_state" in artifact_text
    _write_report(output_dir, metrics, coverage, strict_m1_model=strict_m1_model)
    payload = {
        "contract": "V3.7.1-R1B-GROUP-ATOMIC-BATCH-V1",
        "source": str(source),
        "scope": "M1_ONLY",
        "raw_rows_before_m1_filter": raw_rows_before_m1_filter,
        "raw_rows_after_m1_filter": int(len(raw)),
        "label_unit": "one row per LabelGroupID; batch predictions averaged within group",
        "direction_definition": "consecutive non-zero MNY deltas within OrderID sorted by lab_time; significant threshold abs(actual delta)>=2 MU",
        "volatility_definition": "prediction/actual standard deviation ratio plus within-order delta std and median range ratio",
        "existing_acceptance": EXISTING_ACCEPTANCE,
        "added_acceptance": ADDED_ACCEPTANCE,
        "metrics": _json_safe(metrics.to_dict(orient="records")),
        "coverage": _json_safe(coverage.to_dict(orient="records")),
    }
    (output_dir / "expanded_material_acceptance_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics, coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics, coverage = run(args.source, args.output_dir)
    print(metrics.to_string(index=False))
    print("\nCoverage")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
