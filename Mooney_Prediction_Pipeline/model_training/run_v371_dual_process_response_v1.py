"""Leakage-safe Target -> executed process response model for two pilots.

This is a process-transfer diagnostic. It predicts post-execution Step values
from recipe Target and pre-execution context, then reports OOT performance.
It does not call MMS/Archive, does not write a recipe, and does not produce a
recommendation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUT = BASE / "v371_dual_process_response_v1"
MODEL_OUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_dual_process_response_v1"
TARGETS = ["M1-T15760---- 06 002", "M3B-B00460---- 01 002"]


def regression():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", RobustScaler()),
        ("model", Ridge(alpha=10.0)),
    ])


def phase(stage: str) -> str:
    if stage == "DRY_OR_OIL_MST":
        return "DRY_OR_OIL"
    if stage == "WET_MST":
        return "WET"
    if stage in {"PID_UPPER_MST", "PID_MST", "PID_LOWER_MSB"}:
        return "PID"
    if stage == "BOTTOM_MSB":
        return "PID"
    return stage


def build_long() -> pd.DataFrame:
    p = BASE / "dual_target_readiness" / "dual_pilot_group_step_features.csv"
    d = pd.read_csv(p, low_memory=False)
    d = d[d["CompoundName"].isin(TARGETS)].copy()
    d["BatchStartTime"] = pd.to_datetime(d["BatchStartTime"], errors="coerce")
    d["target_mean"] = pd.to_numeric(d["target_mean"], errors="coerce")
    d["actual_value_mean"] = pd.to_numeric(d["actual_value_mean"], errors="coerce")
    d["MNY"] = pd.to_numeric(d["MNY"], errors="coerce")
    d["target_actual_gap"] = d["actual_value_mean"] - d["target_mean"]
    d["phase_family"] = d["phase_semantic"].map(phase)
    d["time_index"] = d.groupby("CompoundName")["BatchStartTime"].rank(method="dense", pct=True)
    d["is_oot"] = d["time_index"] >= 0.85
    # Fit/evaluate by full group: all variables from one pallet label group
    # stay in one temporal partition.
    d["group_time"] = d.groupby("LabelGroupID")["BatchStartTime"].transform("min")
    d["group_rank"] = d.groupby("CompoundName")["group_time"].rank(method="first", pct=True)
    d["split"] = np.select([d["group_rank"] >= 0.85, d["group_rank"] >= 0.70], ["oot", "val"], default="train")
    return d


def fit_and_score(d: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    records = []
    bundles = {}
    numeric_context = ["target_mean", "StepNo", "ProcessVersionProxy"]
    d["ProcessVersionProxy"] = pd.to_numeric(d.get("StepNo"), errors="coerce")
    # The process model is intentionally per material/step/kind.  This avoids
    # pretending that MST S3 temperature and MSB S3 duration share one slope.
    for key, g in d.groupby(["CompoundName", "GroupName", "StepNo", "kind"], dropna=False):
        train = g[g["split"].eq("train")].copy()
        if len(train) < 30 or train["actual_value_mean"].nunique() < 2:
            continue
        features = ["target_mean", "StepNo", "ProcessVersionProxy"]
        model = regression()
        model.fit(train[features], train["actual_value_mean"])
        pred = model.predict(g[features])
        g = g.copy()
        g["response_prediction"] = pred
        g["response_residual"] = g["actual_value_mean"] - pred
        bundles["|".join(map(str, key))] = {"model": model, "features": features, "n_train": int(len(train)), "key": list(key)}
        for split in ("val", "oot"):
            x = g[g["split"].eq(split)]
            if x.empty:
                continue
            err = x["response_prediction"] - x["actual_value_mean"]
            records.append({
                "CompoundName": key[0], "GroupName": key[1], "StepNo": key[2], "kind": key[3],
                "split": split, "n": int(len(x)), "mae": float(mean_absolute_error(x["actual_value_mean"], x["response_prediction"])),
                "rmse": float(np.sqrt(mean_squared_error(x["actual_value_mean"], x["response_prediction"]))),
                "bias": float(err.mean()), "actual_std": float(x["actual_value_mean"].std(ddof=0)),
                "prediction_std": float(x["response_prediction"].std(ddof=0)),
                "p90_abs_error": float(np.quantile(np.abs(err), 0.90)),
                "coverage_2sigma": float((np.abs(err) <= 2.0 * train["actual_value_mean"].std(ddof=0)).mean()),
            })
        d.loc[g.index, "response_prediction"] = g["response_prediction"]
        d.loc[g.index, "response_residual"] = g["response_residual"]
    return d, pd.DataFrame(records), bundles


def endpoint_report(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (compound, group_id, split), g in d.groupby(["CompoundName", "LabelGroupID", "split"], dropna=False):
        def last_value(group_names, kind_name=None, actual_col="actual_value_mean"):
            x = g[g["GroupName"].isin(group_names)]
            if kind_name is not None:
                x = x[x["kind"].eq(kind_name)]
            if x.empty:
                return np.nan
            return float(x.sort_values(["StepNo", "BatchStartTime"])[actual_col].iloc[-1])
        rows.append({
            "CompoundName": compound, "LabelGroupID": group_id, "split": split,
            "dry_endpoint_actual": last_value(["AVR_MST"], "temperature"),
            "wet_endpoint_actual": last_value(["AVR_MST"], "temperature"),
            "discharge_actual": last_value(["AVR_MSB"], "temperature"),
            "discharge_control_target": float(g.loc[(g["GroupName"] == "AVR_MSB") & (g["kind"] == "temperature") & (g["target_mean"] > 0), "target_mean"].iloc[-1]) if not g.loc[(g["GroupName"] == "AVR_MSB") & (g["kind"] == "temperature") & (g["target_mean"] > 0)].empty else np.nan,
            "wet_duration_actual": float(g.loc[(g["phase_family"] == "WET") & (g["kind"] == "duration"), "actual_value_mean"].sum()),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    d = build_long()
    d, metrics, bundles = fit_and_score(d)
    d.to_csv(OUT / "dual_process_response_long_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "dual_process_response_step_metrics.csv", index=False, encoding="utf-8-sig")
    endpoints = endpoint_report(d)
    endpoints.to_csv(OUT / "dual_process_response_endpoint_audit.csv", index=False, encoding="utf-8-sig")

    eligible = metrics[(metrics["split"] == "oot") & (metrics["n"] >= 30)].copy()
    summary = {
        "contract": "V3.7.1-DUAL-PROCESS-RESPONSE-V1",
        "source_policy": "RDS Active snapshot only; local MMS/Archive not accessed",
        "targets": TARGETS, "physical_rows": int(len(d)), "label_groups": int(d["LabelGroupID"].nunique()),
        "groups_by_compound": d.groupby("CompoundName")["LabelGroupID"].nunique().to_dict(),
        "split_groups": {"|".join(map(str, key)): int(value) for key, value in d.groupby(["CompoundName", "split"])["LabelGroupID"].nunique().items()},
        "trained_step_models": int(len(bundles)), "step_metric_rows": int(len(metrics)),
        "oot_models_with_30_groups": int(len(eligible)),
        "gates": {
            "dry_wet_endpoint_mae_le_2C": "NOT_YET_ASSESSED_FROM_STEP_SPECIFIC_ENDPOINT_MAPPING",
            "discharge_mae_le_2C": "NOT_YET_ASSESSED_FROM_STEP_SPECIFIC_ENDPOINT_MAPPING",
            "wet_duration_mae_le_5s": "NOT_YET_ASSESSED_FROM_PHASE_DURATION_MAPPING",
            "control_transfer_status": "DIAGNOSTIC_ONLY",
        },
        "recommendation_ready": False,
        "reason": "first response baseline predicts individual Step execution; endpoint/phase mapping and uncertainty gates remain to be implemented",
        "stage3_calibration_enabled": False, "automatic_control": "DISABLED",
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED", "local_mms_access": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    joblib.dump({"contract": summary["contract"], "bundles": bundles, "recommendation_ready": False, "stage3_calibration_enabled": False}, MODEL_OUT / "dual_process_response_shadow.joblib")
    lines = [
        "# 双材料 Target→实际过程响应基线报告", "",
        "本阶段只建立执行传递基线，不生成推荐。模型按材料、GroupName、StepNo、控制变量分别拟合 Target 到实际 Value 的关系。",
        "", f"- 物理行：{len(d)}", f"- LabelGroup：{d['LabelGroupID'].nunique()}", f"- Step 响应模型：{len(bundles)}", "- 数据源：RDS Active；未访问本地 MMS/Archive",
        "", "## 重要限制", "", "当前输出是 Step 级执行预测。Dry/Wet 终点、Wet duration 和排胶温度仍需用明确的 Phase 边界和配方控制 Step 规则聚合，之后才能进入 V3.7 输入。",
        "", "## 当前放行状态", "", "`CONTROL_TRANSFER_UNAVAILABLE` / `MANUAL_REVIEW_ONLY` 均不放行。需要完成端点映射、P10/P50/P90 区间、温度上限和热预算门禁。",
    ]
    (OUT / "dual_process_response_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
