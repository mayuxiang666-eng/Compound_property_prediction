"""M1 semi-supervised process-representation experiment, V2.

Unlabeled M1 batch rows are used only to learn a pre-OOT PCA representation.
Real Lab groups then train the supervised teacher/student models on the
original common features plus group-level latent process-state mean/std.
No database access, Stage3 calibration, or production artifact mutation.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.experiment_m1_semi_supervised_teacher_student_v1 import (
    markdown_table,
    metrics,
    model,
    sha256,
    trend_metrics,
)


LAB_INPUT = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1" / "m1_temporal_jit_v1" / "m1_temporal_jit_acceptance_input.csv"
BATCH_INPUT = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1" / "v371_r1b_m1_only_snapshot_v1" / "m1_only_batch_pid_quality_features.csv"
OUT = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1" / "m1_semisupervised_representation_v2"


def fit_latent(batch: pd.DataFrame, feature_cols: list[str], cutoff: pd.Timestamp) -> tuple[pd.DataFrame, int, int]:
    pre = batch.loc[batch["BatchStartTime"] < cutoff].copy()
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X_pre = prep.fit_transform(pre[feature_cols])
    n_components = min(24, X_pre.shape[1], max(2, X_pre.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=17)
    pca.fit(X_pre)
    X_all = pca.transform(prep.transform(batch[feature_cols]))
    latent = pd.DataFrame(X_all, columns=[f"latent_process_{i:02d}" for i in range(n_components)])
    latent["LabelGroupID"] = batch["LabelGroupID"].to_numpy()
    mean = latent.groupby("LabelGroupID", as_index=False).mean(numeric_only=True)
    std = latent.groupby("LabelGroupID", as_index=False).std(numeric_only=True).fillna(0.0)
    mean = mean.rename(columns={c: f"{c}_mean" for c in mean.columns if c != "LabelGroupID"})
    std = std.rename(columns={c: f"{c}_std" for c in std.columns if c != "LabelGroupID"})
    group_latent = mean.merge(std, on="LabelGroupID", how="outer")
    return group_latent, int(len(pre)), int(n_components)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(LAB_INPUT, low_memory=False)
    batches = pd.read_csv(BATCH_INPUT, low_memory=False)
    lab["LabelGroupID"] = lab["LabelGroupID"].astype(str)
    batches["LabelGroupID"] = batches["LabelGroupID"].astype(str)
    lab["lab_time"] = pd.to_datetime(lab["lab_time"], errors="coerce", utc=True)
    batches["BatchStartTime"] = pd.to_datetime(batches["BatchStartTime"], errors="coerce", utc=True)
    cutoff = lab.loc[lab["split"].eq("oot"), "lab_time"].min()

    excluded = {"LabelGroupID", "OrderID", "PalletID", "BatchNumber", "MNY", "ActualMNY", "JITPrediction", "JITP10", "JITP90", "JITError", "RawPrediction", "Candidate", "split", "lab_time"}
    common = sorted(set(lab.columns).intersection(batches.columns) - excluded)
    feature_cols = [c for c in common if pd.api.types.is_numeric_dtype(lab[c]) and pd.api.types.is_numeric_dtype(batches[c])]
    latent, pre_rows, n_components = fit_latent(batches, feature_cols, cutoff)

    lab2 = lab.merge(latent, on="LabelGroupID", how="left")
    labeled_ids = set(lab2["LabelGroupID"])
    raw_unlabeled = batches.loc[~batches["LabelGroupID"].isin(labeled_ids)].copy()
    unlabeled = raw_unlabeled.groupby("LabelGroupID", as_index=False)[feature_cols].mean(numeric_only=True)
    unlabeled_time = raw_unlabeled.groupby("LabelGroupID", as_index=False)["BatchStartTime"].min()
    unlabeled = unlabeled.merge(unlabeled_time, on="LabelGroupID", how="left").merge(latent, on="LabelGroupID", how="left")
    latent_cols = [c for c in latent.columns if c != "LabelGroupID"]
    model_cols = feature_cols + latent_cols
    pseudo_pool = unlabeled.loc[unlabeled["BatchStartTime"] < cutoff].copy()

    train = lab2.loc[lab2["split"].eq("train")].copy()
    oot = lab2.loc[lab2["split"].eq("oot")].copy()
    teachers = []
    for seed in (11, 23, 37, 53, 71):
        fitted = model(seed)
        fitted.fit(train[model_cols], train["MNY"].to_numpy(float))
        teachers.append(fitted)
    teacher_oot = np.mean([m.predict(oot[model_cols]) for m in teachers], axis=0)

    teacher_pool = np.vstack([m.predict(pseudo_pool[model_cols]) for m in teachers]).T if len(pseudo_pool) else np.empty((0, len(teachers)))
    distance_prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    train_scaled = distance_prep.fit_transform(train[model_cols])
    pool_scaled = distance_prep.transform(pseudo_pool[model_cols]) if len(pseudo_pool) else np.empty((0, len(model_cols)))
    if len(pseudo_pool):
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

    pseudo_rows = accepted[model_cols].copy()
    pseudo_rows["MNY"] = accepted["teacher_prediction"].to_numpy(float)
    student_train = pd.concat([train[model_cols + ["MNY"]], pseudo_rows], ignore_index=True)
    weights = np.concatenate([np.ones(len(train)), np.full(len(pseudo_rows), 0.15)])
    student = model(101)
    student.fit(student_train[model_cols], student_train["MNY"].to_numpy(float), extratreesregressor__sample_weight=weights)
    student_oot = student.predict(oot[model_cols])

    predictions = oot[["LabelGroupID", "OrderID", "PalletID", "CompoundName", "RecipeID", "ProcessVersion", "MNY", "lab_time"]].copy()
    predictions["TeacherPrediction"] = teacher_oot
    predictions["StudentPrediction"] = student_oot
    predictions.to_csv(OUT / "m1_representation_oot_predictions.csv", index=False)
    pseudo_pool.to_csv(OUT / "m1_representation_pseudo_pool_audit.csv", index=False)

    rows = []
    for name, col in (("REPRESENTATION_TEACHER", "TeacherPrediction"), ("REPRESENTATION_STUDENT", "StudentPrediction")):
        rows.append({"model": name, "split": "oot", **metrics(predictions["MNY"].to_numpy(float), predictions[col].to_numpy(float)), **trend_metrics(predictions.rename(columns={col: "pred"}), "pred")})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT / "m1_representation_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    ordered = predictions.sort_values("lab_time").reset_index(drop=True)
    axes[0].plot(ordered.index, ordered["MNY"], label="Actual Lab", color="#111827", linewidth=1.8)
    axes[0].plot(ordered.index, ordered["TeacherPrediction"], label="Representation Teacher", color="#2563eb", alpha=0.8)
    axes[0].plot(ordered.index, ordered["StudentPrediction"], label="Representation Student", color="#dc2626", alpha=0.8)
    axes[0].set_title("M1 OOT ordered full view")
    axes[0].set_xlabel("OOT Lab group ordered by time")
    axes[0].set_ylabel("Mooney (MU)")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    lo = float(min(predictions["MNY"].min(), predictions["TeacherPrediction"].min(), predictions["StudentPrediction"].min()))
    hi = float(max(predictions["MNY"].max(), predictions["TeacherPrediction"].max(), predictions["StudentPrediction"].max()))
    axes[1].scatter(predictions["MNY"], predictions["TeacherPrediction"], s=18, alpha=0.55, label="Teacher", color="#2563eb")
    axes[1].scatter(predictions["MNY"], predictions["StudentPrediction"], s=18, alpha=0.55, label="Student", color="#dc2626")
    axes[1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[1].set_title("Actual vs predicted")
    axes[1].set_xlabel("Actual Lab MNY")
    axes[1].set_ylabel("Predicted MNY")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "m1_representation_actual_vs_predicted_fullview.png")
    plt.close(fig)

    summary = {
        "experiment": "M1_SEMISUPERVISED_PROCESS_REPRESENTATION_V2",
        "scope": "M1_ONLY",
        "database_access": False,
        "stage3_calibration_enabled": False,
        "representation_fit_cutoff": str(cutoff),
        "representation_fit_batch_rows": pre_rows,
        "pca_components": n_components,
        "common_numeric_features": len(feature_cols),
        "latent_group_features": len(latent_cols),
        "unlabeled_groups": int(raw_unlabeled["LabelGroupID"].nunique()),
        "pseudo_pool_before_oot": int(len(pseudo_pool)),
        "pseudo_labels_accepted": int(len(accepted)),
        "pseudo_label_weight": 0.15,
        "metrics": rows,
        "input_sha256": {"lab_input": sha256(LAB_INPUT), "batch_input": sha256(BATCH_INPUT)},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "release_status": "NOT_RELEASED_EXPERIMENT_ONLY",
    }
    (OUT / "m1_representation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = [
        "# M1 半监督过程表征 V2 实验报告",
        "",
        "本实验只使用本地快照。先用 OOT 起点之前的 M1 批次学习 PCA 过程状态表征，再把每个 LabelGroup 的表征均值/标准差加入监督 Teacher–Student 模型。",
        "",
        f"- 表征训练批次：{pre_rows}；PCA 维度：{n_components}；原始共同数值特征：{len(feature_cols)}；潜在组特征：{len(latent_cols)}。",
        f"- 未标注组：{summary['unlabeled_groups']}；OOT 前伪标签池：{summary['pseudo_pool_before_oot']}；通过门禁：{summary['pseudo_labels_accepted']}。",
        "",
        "## OOT 指标",
        "",
        markdown_table(metrics_df),
        "",
        "该实验仍是诊断版本，不覆盖冻结模型，不启用 Stage3，也不生成推荐。",
        "",
        "全览图：`m1_representation_actual_vs_predicted_fullview.png`。",
    ]
    (OUT / "M1半监督过程表征V2实验报告.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
