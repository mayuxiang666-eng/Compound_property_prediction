"""Leakage-blocked dual material expert residual challenger.

Targets T15760 (Silica) and B00460 (CarbonBlack) in parallel.  The base
prediction is the existing R1 full-group OOF prediction.  Actual Step Value
features are used only as a diagnostic residual expert; this artifact is not
an intervention/recommendation model because actual post-batch values are not
available before the next batch.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.feature_leakage_guard import assert_no_target_leakage
from Mooney_Prediction_Pipeline.model_training.order_atomic_trend_metrics import candidate_score, metric_record

BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUT = BASE / "v371_dual_material_experts_v1"
MODEL_OUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_dual_material_experts_v1"
TARGETS = ["M1-T15760---- 06 002", "M3B-B00460---- 01 002"]


def ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", RobustScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def stage(material: str, group: str, step: int) -> str:
    g = str(group).upper()
    if "MSB" in g:
        return "PID_LOWER_MSB" if material == "Silica" else "BOTTOM_MSB"
    if step <= 2:
        return "LOCKED_LOADING"
    if step == 3:
        return "DRY_MST"
    if step == 4:
        return "OIL_MST"
    if step <= 9:
        return "WET_MST"
    return "PID_UPPER_MST" if material == "Silica" else "PID_MST"


def build_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    raw = raw.copy()
    raw["StepNo"] = pd.to_numeric(raw["step_key"].str.extract(r"\|S(\d+)\|")[0], errors="coerce")
    raw["GroupName"] = raw["step_key"].str.split("|", n=1).str[0]
    raw["control_kind"] = raw["step_key"].str.rsplit("|", n=1).str[-1]
    raw["stage"] = [stage(m, g, int(s)) for m, g, s in zip(raw["CompoundName"], raw["GroupName"], raw["StepNo"])]
    # First aggregate each group's repeated batch executions by physical step.
    long = raw.groupby(["LabelGroupID", "CompoundName", "material_system", "step_key", "stage", "control_kind"], dropna=False).agg(
        actual=("actual_value_mean", "mean"), target=("target_mean", "mean"),
    ).reset_index()
    actual = long.pivot_table(index=["LabelGroupID", "CompoundName", "material_system"], columns="step_key", values="actual", aggfunc="first")
    target = long.pivot_table(index=["LabelGroupID", "CompoundName", "material_system"], columns="step_key", values="target", aggfunc="first")
    actual.columns = [f"expert_actual__{x}" for x in actual.columns]
    target.columns = [f"expert_target__{x}" for x in target.columns]
    wide = pd.concat([actual, target], axis=1).reset_index()

    # Mechanism aggregates retain the explicit silica upper/lower PID split
    # and the carbon-black Dry/Wet/PID/Bottom split.
    for material, g in long.groupby("material_system"):
        pass
    for mat in wide["material_system"].dropna().unique():
        rows = long[long["material_system"].eq(mat)]
        for st, x in rows.groupby("stage"):
            for field in ("actual", "target"):
                for k, y in x.groupby("control_kind"):
                    values = y[["LabelGroupID", field]].rename(columns={field: f"expert_{st}__{k}__{field}"})
                    values = values.groupby("LabelGroupID", as_index=False).mean(numeric_only=True)
                    wide = wide.merge(values, on="LabelGroupID", how="left")
    # Target-minus-execution gaps are diagnostics of control transfer, not
    # causal intervention features.
    for col in list(wide.columns):
        if col.startswith("expert_actual__"):
            suffix = col.replace("expert_actual__", "")
            target_col = f"expert_target__{suffix}"
            if target_col in wide:
                wide[f"expert_gap__{suffix}"] = wide[target_col] - wide[col]
    numeric = wide.select_dtypes(include=[np.number]).columns.tolist()
    usable = [c for c in numeric if wide[c].notna().sum() >= 35 and wide[c].nunique(dropna=True) >= 2]
    return wide[["LabelGroupID"] + usable], usable


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    r1_pred = pd.read_csv(BASE / "v371_r1_temporal_group" / "r1_all_group_predictions.csv", low_memory=False)
    r1_pred = r1_pred[["LabelGroupID", "split", "RawPrediction"]].rename(columns={"RawPrediction": "LevelPrediction"})
    groups = pd.read_csv(BASE / "pallet_group_training_frame_c2_prepared.csv", low_memory=False)
    groups = groups[groups["CompoundName"].isin(TARGETS)].copy()
    groups = groups.merge(r1_pred, on="LabelGroupID", how="inner", validate="one_to_one")
    phase = pd.read_csv(BASE / "phase_batch_features_active_v1.csv", usecols=["LabelGroupID", "BatchStartTime"], low_memory=False)
    phase["BatchStartTime"] = pd.to_datetime(phase["BatchStartTime"], errors="coerce")
    starts = phase.groupby("LabelGroupID", as_index=False)["BatchStartTime"].min().rename(columns={"BatchStartTime": "production_start"})
    groups = groups.merge(starts, on="LabelGroupID", how="left", validate="one_to_one")
    groups["MNY"] = pd.to_numeric(groups["MNY"], errors="coerce")
    groups["ResidualTarget"] = groups["MNY"] - pd.to_numeric(groups["LevelPrediction"], errors="coerce")

    step = pd.read_csv(OUT.parent / "dual_target_readiness" / "dual_pilot_group_step_features.csv", low_memory=False)
    step = step[step["CompoundName"].isin(TARGETS)].copy()
    expert, features = build_features(step)
    groups = groups.merge(expert, on="LabelGroupID", how="inner", validate="one_to_one")
    assert_no_target_leakage(groups, features)
    expert.to_csv(OUT / "dual_expert_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": features, "contract": "actual Step Value/Target diagnostic; no direct recommendation"}).to_csv(OUT / "dual_expert_feature_contract.csv", index=False, encoding="utf-8-sig")

    train = groups[groups["split"].eq("train") & groups["ResidualTarget"].notna()].copy()
    candidates = {}
    rows = []
    models = {}
    for alpha in (10.0, 50.0, 200.0, 500.0):
        model = ridge(alpha)
        model.fit(train[features], train["ResidualTarget"])
        residual = model.predict(groups[features])
        models[alpha] = model
        for gamma in (0.0, 0.25, 0.50, 0.75, 1.0):
            pred = groups["LevelPrediction"].to_numpy(dtype=float) + gamma * residual
            name = f"DUAL_EXPERT_A{int(alpha):03d}_G{int(gamma*100):03d}"
            candidates[name] = pred
            score, level, trend = candidate_score(groups, pred)
            rows.append({"candidate": name, "alpha": alpha, "gamma": gamma, "selection_score_val": score, **level, **trend})
    search = pd.DataFrame(rows)
    selected_row = search.sort_values(["selection_score_val", "mae", "candidate"], kind="stable").iloc[0]
    selected = str(selected_row["candidate"])
    groups["FinalPrediction"] = candidates[selected]

    metrics = []
    for split in ("val", "oot"):
        metrics.append({"scope": "POOLED", **metric_record(groups, "FinalPrediction", split, selected)})
        for compound in TARGETS:
            part = groups[groups["CompoundName"].eq(compound)].copy()
            metrics.append({"scope": compound, **metric_record(part, "FinalPrediction", split, selected)})
    metrics = pd.DataFrame(metrics)
    search.to_csv(OUT / "dual_expert_validation_search.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "dual_expert_metrics.csv", index=False, encoding="utf-8-sig")
    groups[["LabelGroupID", "OrderID", "PalletID", "CompoundName", "material_system", "RecipeID", "ProcessVersion", "split", "MNY", "LevelPrediction", "FinalPrediction", "production_start"]].to_csv(OUT / "dual_expert_group_predictions.csv", index=False, encoding="utf-8-sig")

    selected_alpha = float(selected_row["alpha"])
    selected_gamma = float(selected_row["gamma"])
    bundle = {
        "contract": "V3.7.1-DUAL-MATERIAL-EXPERT-DIAGNOSTIC-V1",
        "selected_candidate": selected, "alpha": selected_alpha, "gamma": selected_gamma,
        "model": models[selected_alpha], "features": features,
        "base_prediction": "R1 full-group OOF RawPrediction",
        "material_experts": {"Silica": "DRY_MST/OIL_MST/WET_MST/PID_UPPER_MST/PID_LOWER_MSB", "CarbonBlack": "DRY_MST/OIL_MST/WET_MST/PID_MST/BOTTOM_MSB"},
        "post_batch_actual_features": True, "recommendation_ready": False,
        "stage3_calibration_enabled": False, "automatic_control": "DISABLED",
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
    }
    joblib.dump(bundle, MODEL_OUT / "dual_material_expert_diagnostic_shadow.joblib")
    summary = {
        "contract": bundle["contract"], "groups": int(len(groups)), "train_groups": int(len(train)), "feature_count": len(features),
        "groups_by_compound": groups["CompoundName"].value_counts().to_dict(), "pair_scope": "within OrderID, production_start sorted",
        "selected_candidate": selected, "selected_alpha": selected_alpha, "selected_gamma": selected_gamma,
        "validation_metrics": metrics[metrics["split"].eq("val")].to_dict("records"),
        "oot_metrics": metrics[metrics["split"].eq("oot")].to_dict("records"),
        "selection_policy": "validation only; final OOT not used for selection",
        "target_leakage_gate": "PASSED", "recommendation_ready": False,
        "reason_not_recommendation_ready": "features include post-batch actual Step Value; Target-to-actual process response is still required",
        "database_access_during_run": False, "local_mms_access": False, "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = [
        "# 双材料阶段专家残差候选报告", "",
        "本候选只使用冻结 RDS Active 快照和 R1 全量组级 OOF 预测。Stage3 关闭，未访问本地 MMS/Archive。",
        "", f"- 选中候选：`{selected}`（仅基于验证集选择）", f"- 特征数：{len(features)}", "- 推荐就绪：否",
        "- 原因：当前专家包含批次完成后的实际 Step Value，只能用于诊断阶段敏感性；还必须建立 Target→实际过程响应模型。",
        "", "## 专家分层", "", "- T15760：Dry/Oil/Wet/上部 MST PID/下部 MSB PID continuation。", "- B00460：Dry/Oil/Wet/PID MST/Bottom MSB。", "- Step 1–2 和 ram 不作为推荐动作。",
        "", "## 评价结果", "", "详见 `dual_expert_metrics.csv`。最终 OOT 未参与参数选择。",
    ]
    (OUT / "dual_material_expert_diagnostic_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
