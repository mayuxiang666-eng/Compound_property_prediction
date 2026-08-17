"""M1-only semi-supervised teacher-student experiment.

This is a local-snapshot-only development experiment.  It never opens a
database connection and never changes the frozen V3.7/V3.7.1 artifacts.

Contract:
- real Lab groups are the only rows used for OOT scoring;
- unlabeled groups are eligible for pseudo-labeling only before the locked OOT
  start time;
- pseudo-labels require both ensemble agreement and feature-domain proximity;
- pseudo-labels receive a small training weight and never replace real labels;
- Stage3 calibration remains disabled.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
LAB_INPUT = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1" / "m1_temporal_jit_v1" / "m1_temporal_jit_acceptance_input.csv"
BATCH_INPUT = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1" / "v371_r1b_m1_only_snapshot_v1" / "m1_only_batch_pid_quality_features.csv"
OUT = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1" / "m1_semi_supervised_teacher_student_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model(seed: int) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        ExtraTreesRegressor(
            n_estimators=240,
            min_samples_leaf=5,
            max_features=0.65,
            random_state=seed,
            n_jobs=-1,
        ),
    )


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(err ** 2))
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "actual_std": float(np.std(y, ddof=1)) if len(y) > 1 else float("nan"),
        "prediction_std": float(np.std(pred, ddof=1)) if len(pred) > 1 else float("nan"),
        "prediction_to_actual_std_ratio": float(np.std(pred, ddof=1) / np.std(y, ddof=1)) if len(y) > 1 and np.std(y, ddof=1) > 0 else float("nan"),
        "within_2_mny_rate": float(np.mean(np.abs(err) <= 2.0)),
        "within_3_mny_rate": float(np.mean(np.abs(err) <= 3.0)),
        "within_5_mny_rate": float(np.mean(np.abs(err) <= 5.0)),
    }


def trend_metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    deltas = []
    pred_deltas = []
    for _, g in frame.sort_values("lab_time").groupby("OrderID"):
        if len(g) < 2:
            continue
        actual = g["MNY"].to_numpy(dtype=float)
        predicted = g[pred_col].to_numpy(dtype=float)
        deltas.extend(np.diff(actual).tolist())
        pred_deltas.extend(np.diff(predicted).tolist())
    if len(deltas) < 2:
        return {"delta_pairs": float(len(deltas)), "direction_accuracy": float("nan"), "delta_pearson": float("nan"), "delta_std_ratio": float("nan")}
    a = np.asarray(deltas)
    p = np.asarray(pred_deltas)
    nonzero = np.abs(a) > 1e-9
    direction = float(np.mean(np.sign(a[nonzero]) == np.sign(p[nonzero]))) if np.any(nonzero) else float("nan")
    pearson = float(np.corrcoef(a, p)[0, 1]) if np.std(a) > 0 and np.std(p) > 0 else float("nan")
    return {
        "delta_pairs": float(len(a)),
        "direction_accuracy": direction,
        "delta_pearson": pearson,
        "delta_std_ratio": float(np.std(p, ddof=1) / np.std(a, ddof=1)) if np.std(a, ddof=1) > 0 else float("nan"),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(LAB_INPUT, low_memory=False)
    batches = pd.read_csv(BATCH_INPUT, low_memory=False)
    lab["LabelGroupID"] = lab["LabelGroupID"].astype(str)
    batches["LabelGroupID"] = batches["LabelGroupID"].astype(str)
    lab["lab_time"] = pd.to_datetime(lab["lab_time"], errors="coerce", utc=True)
    batches["BatchStartTime"] = pd.to_datetime(batches["BatchStartTime"], errors="coerce", utc=True)

    excluded = {"LabelGroupID", "OrderID", "PalletID", "BatchNumber", "MNY", "ActualMNY", "JITPrediction", "JITP10", "JITP90", "JITError", "RawPrediction", "Candidate", "split", "lab_time"}
    common = sorted(set(lab.columns).intersection(batches.columns) - excluded)
    feature_cols = []
    for col in common:
        if pd.api.types.is_numeric_dtype(lab[col]) and pd.api.types.is_numeric_dtype(batches[col]):
            feature_cols.append(col)
    if len(feature_cols) < 20:
        raise RuntimeError(f"BLOCKED_INSUFFICIENT_COMMON_FEATURES:{len(feature_cols)}")

    labeled_ids = set(lab["LabelGroupID"])
    unlabeled = batches.loc[~batches["LabelGroupID"].isin(labeled_ids)].copy()
    unlabeled_group = unlabeled.groupby("LabelGroupID", as_index=False)[feature_cols].mean(numeric_only=True)
    unlabeled_time = unlabeled.groupby("LabelGroupID", as_index=False)["BatchStartTime"].min()
    unlabeled_group = unlabeled_group.merge(unlabeled_time, on="LabelGroupID", how="left")
    oot_start = lab.loc[lab["split"].eq("oot"), "lab_time"].min()
    pseudo_pool = unlabeled_group.loc[unlabeled_group["BatchStartTime"] < oot_start].copy()

    train = lab.loc[lab["split"].eq("train")].copy()
    val = lab.loc[lab["split"].eq("val")].copy()
    oot = lab.loc[lab["split"].eq("oot")].copy()
    X_train = train[feature_cols]
    X_oot = oot[feature_cols]
    y_train = train["MNY"].to_numpy(dtype=float)
    y_oot = oot["MNY"].to_numpy(dtype=float)

    teachers = []
    for seed in (11, 23, 37, 53, 71):
        fitted = model(seed)
        fitted.fit(X_train, y_train)
        teachers.append(fitted)
    teacher_oot = np.mean([m.predict(X_oot) for m in teachers], axis=0)
    teacher_pool = np.vstack([m.predict(pseudo_pool[feature_cols]) for m in teachers]).T if len(pseudo_pool) else np.empty((0, len(teachers)))

    # Domain distance is computed after median imputation and standardization.
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    train_scaled = prep.fit_transform(X_train)
    pool_scaled = prep.transform(pseudo_pool[feature_cols]) if len(pseudo_pool) else np.empty((0, len(feature_cols)) )
    if len(pool_scaled):
        distances = np.sqrt(((pool_scaled[:, None, :] - train_scaled[None, :, :]) ** 2).mean(axis=2)).min(axis=1)
        pseudo_pool["teacher_prediction"] = teacher_pool.mean(axis=1)
        pseudo_pool["teacher_ensemble_std"] = teacher_pool.std(axis=1, ddof=1)
        pseudo_pool["nearest_train_distance"] = distances
        accepted = pseudo_pool.loc[(pseudo_pool["teacher_ensemble_std"] <= 1.5) & (pseudo_pool["nearest_train_distance"] <= 3.5)].copy()
    else:
        pseudo_pool["teacher_prediction"] = []
        pseudo_pool["teacher_ensemble_std"] = []
        pseudo_pool["nearest_train_distance"] = []
        accepted = pseudo_pool.copy()

    student_train = pd.concat([train[feature_cols + ["MNY"]], accepted[feature_cols].assign(MNY=accepted["teacher_prediction"])], ignore_index=True)
    weights = np.concatenate([np.ones(len(train)), np.full(len(accepted), 0.15)])
    student = model(101)
    student.fit(student_train[feature_cols], student_train["MNY"].to_numpy(dtype=float), extratreesregressor__sample_weight=weights)
    student_oot = student.predict(X_oot)

    output = oot[["LabelGroupID", "OrderID", "PalletID", "CompoundName", "RecipeID", "ProcessVersion", "MNY", "lab_time"]].copy()
    output["TeacherPrediction"] = teacher_oot
    output["StudentPrediction"] = student_oot
    output["TeacherError"] = output["TeacherPrediction"] - output["MNY"]
    output["StudentError"] = output["StudentPrediction"] - output["MNY"]
    output.to_csv(OUT / "m1_semi_supervised_oot_predictions.csv", index=False)
    pseudo_pool.to_csv(OUT / "m1_semi_supervised_pseudo_pool_audit.csv", index=False)

    rows = []
    for name, col in (("TEACHER", "TeacherPrediction"), ("STUDENT", "StudentPrediction")):
        row = {"model": name, "split": "oot", **metrics(output["MNY"].to_numpy(float), output[col].to_numpy(float)), **trend_metrics(output.rename(columns={col: "pred"}), "pred")}
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT / "m1_semi_supervised_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    ordered = output.sort_values("lab_time").reset_index(drop=True)
    axes[0].plot(ordered.index, ordered["MNY"], label="Actual Lab", color="#111827", linewidth=1.8)
    axes[0].plot(ordered.index, ordered["TeacherPrediction"], label="Teacher", color="#2563eb", alpha=0.8)
    axes[0].plot(ordered.index, ordered["StudentPrediction"], label="Semi-supervised Student", color="#dc2626", alpha=0.8)
    axes[0].set_title("M1 OOT full ordered view")
    axes[0].set_xlabel("OOT Lab group ordered by time")
    axes[0].set_ylabel("Mooney (MU)")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    lo = float(min(output["MNY"].min(), output["TeacherPrediction"].min(), output["StudentPrediction"].min()))
    hi = float(max(output["MNY"].max(), output["TeacherPrediction"].max(), output["StudentPrediction"].max()))
    axes[1].scatter(output["MNY"], output["TeacherPrediction"], s=18, alpha=0.55, label="Teacher", color="#2563eb")
    axes[1].scatter(output["MNY"], output["StudentPrediction"], s=18, alpha=0.55, label="Student", color="#dc2626")
    axes[1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[1].set_title("Actual vs predicted")
    axes[1].set_xlabel("Actual Lab MNY")
    axes[1].set_ylabel("Predicted MNY")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "m1_semi_supervised_actual_vs_predicted_fullview.png")
    plt.close(fig)

    summary = {
        "experiment": "M1_SEMI_SUPERVISED_TEACHER_STUDENT_V1",
        "scope": "M1_ONLY",
        "stage3_calibration_enabled": False,
        "database_access": False,
        "lab_groups": int(lab["LabelGroupID"].nunique()),
        "unlabeled_groups": int(unlabeled["LabelGroupID"].nunique()),
        "pseudo_pool_before_oot": int(len(pseudo_pool)),
        "pseudo_labels_accepted": int(len(accepted)),
        "pseudo_label_weight": 0.15,
        "common_numeric_features": int(len(feature_cols)),
        "oot_start": str(oot_start),
        "teacher_student_metrics": rows,
        "input_sha256": {"lab_input": sha256(LAB_INPUT), "batch_input": sha256(BATCH_INPUT)},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "release_status": "NOT_RELEASED_EXPERIMENT_ONLY",
    }
    (OUT / "m1_semi_supervised_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = [
        "# M1 半监督 Teacher–Student 实验报告",
        "",
        "本实验只使用本地快照，未访问 MMS/Archive；Stage3 校准关闭。真实 Lab 组用于 OOT 评价，未标注组只在 OOT 起点之前参与伪标签候选。",
        "",
        f"- 有 Lab 组：{summary['lab_groups']}；未标注组：{summary['unlabeled_groups']}；OOT 前伪标签池：{summary['pseudo_pool_before_oot']}；通过置信度/域距离门禁：{summary['pseudo_labels_accepted']}。",
        f"- 共同数值特征：{summary['common_numeric_features']}；伪标签权重：0.15。",
        "",
        "## OOT 指标",
        "",
        markdown_table(metrics_df),
        "",
        "解释：若伪标签通过数量很少或 Student 没有同时改善 MAE、方向和振幅，则半监督仅保留为诊断，不晋级模型。",
        "",
        "## 全览图",
        "",
        "见 `m1_semi_supervised_actual_vs_predicted_fullview.png`。",
    ]
    (OUT / "M1半监督TeacherStudent实验报告.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
