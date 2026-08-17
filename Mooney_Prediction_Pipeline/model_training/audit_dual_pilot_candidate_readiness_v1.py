"""Rank a CarbonBlack pilot and audit it beside T15760.

Read-only audit over the frozen RDS Active snapshot.  This deliberately does
not query MMS/Archive and does not train or overwrite a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUT = BASE / "dual_target_readiness"


def mae(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(y - p))) if len(y) else np.nan


def rmse(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else np.nan


def safe_corr(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce")
    b = pd.to_numeric(pd.Series(b), errors="coerce")
    z = pd.concat([a, b], axis=1).dropna()
    if len(z) < 3 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(z.iloc[:, 0].corr(z.iloc[:, 1]))


def pair_audit(pred: pd.DataFrame) -> dict:
    """Adjacent within-order label-group trend audit on available C1 rows."""
    work = pred.copy()
    work["production_start"] = pd.to_datetime(work["production_start"], errors="coerce")
    work["MNY"] = pd.to_numeric(work["MNY"], errors="coerce")
    work["Prediction"] = pd.to_numeric(work["Prediction"], errors="coerce")
    pairs = []
    for order_id, g in work.dropna(subset=["production_start", "MNY", "Prediction"]).groupby("OrderID"):
        g = g.sort_values(["production_start", "LabelGroupID"])
        if len(g) < 2:
            continue
        for i in range(len(g) - 1):
            x, y = g.iloc[i], g.iloc[i + 1]
            pairs.append(
                {
                    "OrderID": order_id,
                    "from_group": x["LabelGroupID"],
                    "to_group": y["LabelGroupID"],
                    "actual_delta": float(y["MNY"] - x["MNY"]),
                    "prediction_delta": float(y["Prediction"] - x["Prediction"]),
                }
            )
    pair_df = pd.DataFrame(pairs)
    if pair_df.empty:
        return {"pair_count": 0, "order_count": 0, "delta_pearson": np.nan, "direction_accuracy": np.nan}
    actual = pair_df["actual_delta"]
    pred_delta = pair_df["prediction_delta"]
    usable = (actual.abs() > 1e-9) & (pred_delta.abs() > 1e-9)
    return {
        "pair_count": int(len(pair_df)),
        "order_count": int(pair_df["OrderID"].nunique()),
        "delta_pearson": safe_corr(actual, pred_delta),
        "direction_accuracy": float((np.sign(actual[usable]) == np.sign(pred_delta[usable])).mean()) if usable.any() else np.nan,
        "actual_delta_std": float(actual.std(ddof=0)),
        "prediction_delta_std": float(pred_delta.std(ddof=0)),
        "delta_std_ratio": float(pred_delta.std(ddof=0) / actual.std(ddof=0)) if actual.std(ddof=0) > 1e-9 else np.nan,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame_path = BASE / "pallet_group_training_frame_c2_prepared.csv"
    pred_path = BASE / "v371_c1_strict_order_selected" / "c1_group_predictions.csv"
    readiness_path = BASE / "global_step_phase_template_readiness_v1" / "template_readiness.csv"

    frame = pd.read_csv(frame_path, low_memory=False)
    pred = pd.read_csv(pred_path, low_memory=False)
    readiness = pd.read_csv(readiness_path, low_memory=False)

    frame["MNY"] = pd.to_numeric(frame.get("MNY"), errors="coerce")
    frame["batch_count"] = pd.to_numeric(frame.get("batch_count"), errors="coerce")
    frame["PhaseBatchCount"] = pd.to_numeric(frame.get("PhaseBatchCount"), errors="coerce")

    def summarize(compound: str, material: str) -> dict:
        all_rows = frame[frame["CompoundName"].eq(compound)].copy()
        validation = pred[pred["CompoundName"].eq(compound)].copy()
        templates = readiness[readiness.get("CompoundName", pd.Series(dtype=object)).eq(compound)].copy() if "CompoundName" in readiness.columns else pd.DataFrame()
        if templates.empty and "RecipeID" in readiness.columns:
            templates = readiness[readiness["RecipeID"].astype(str).str.contains(compound.split("-")[1] if "-" in compound else compound, na=False)].copy()
        val_oot = validation[validation["split"].isin(["val", "oot"])] if "split" in validation.columns else validation
        return {
            "CompoundName": compound,
            "material_system": material,
            "all_label_groups": int(len(all_rows)),
            "all_orders": int(all_rows["OrderID"].nunique()) if not all_rows.empty else 0,
            "all_recipe_versions": int(all_rows["RecipeID"].nunique()) if not all_rows.empty else 0,
            "all_templates": int(all_rows["TemplateKey"].nunique()) if not all_rows.empty else 0,
            "all_batch_count_sum": float(all_rows["batch_count"].fillna(0).sum()) if not all_rows.empty else 0.0,
            "all_phase_batch_count_sum": float(all_rows["PhaseBatchCount"].fillna(0).sum()) if not all_rows.empty else 0.0,
            "c1_validation_rows": int(len(val_oot)),
            "c1_oot_rows": int((val_oot["split"] == "oot").sum()) if "split" in val_oot.columns else 0,
            "c1_mae": mae(val_oot["MNY"], val_oot["Prediction"]) if not val_oot.empty else np.nan,
            "c1_rmse": rmse(val_oot["MNY"], val_oot["Prediction"]) if not val_oot.empty else np.nan,
            "c1_bias": float((val_oot["Prediction"] - val_oot["MNY"]).mean()) if not val_oot.empty else np.nan,
            "c1_pair_audit": pair_audit(val_oot),
            "readiness_rows": int(len(templates)),
            "readiness_max_batch_count": float(pd.to_numeric(templates.get("batch_count"), errors="coerce").max()) if not templates.empty and "batch_count" in templates.columns else np.nan,
            "readiness_max_label_group_count": float(pd.to_numeric(templates.get("label_group_count"), errors="coerce").max()) if not templates.empty and "label_group_count" in templates.columns else np.nan,
        }

    t15760 = "M1-T15760---- 06 002"
    carbon = frame.loc[frame["material_system"].astype(str).str.casefold().eq("carbonblack"), "CompoundName"].dropna().value_counts()
    compounds = [t15760] + carbon.index.tolist()
    records = [summarize(c, "Silica" if c == t15760 else "CarbonBlack") for c in compounds]
    ranking = pd.DataFrame(records)
    cb = ranking[ranking["material_system"].eq("CarbonBlack")].copy()
    cb["selection_score"] = (
        cb["all_label_groups"].clip(upper=250) / 250.0
        + cb["c1_oot_rows"].clip(upper=50) / 50.0
        + cb["c1_pair_audit"].map(lambda x: x.get("pair_count", 0)).clip(upper=20) / 20.0
    )
    cb = cb.sort_values(["selection_score", "all_label_groups"], ascending=False)
    selected = str(cb.iloc[0]["CompoundName"]) if not cb.empty else None
    ranking["selected_parallel_pilot"] = ranking["CompoundName"].eq(selected)
    ranking = ranking.drop(columns=["c1_pair_audit"])
    ranking.to_csv(OUT / "carbon_black_parallel_candidate_ranking.csv", index=False, encoding="utf-8-sig")

    selected_details = next((x for x in records if x["CompoundName"] == selected), None)
    t_details = next((x for x in records if x["CompoundName"] == t15760), None)
    summary = {
        "source_policy": "RDS Active snapshot only; local MMS/Archive not accessed",
        "frame_path": str(frame_path),
        "prediction_path": str(pred_path),
        "readiness_path": str(readiness_path),
        "carbon_black_candidate_count": int(len(cb)),
        "selected_parallel_pilot": selected,
        "selection_reason": "largest combined active label-group coverage, C1 OOT coverage, and within-order pair coverage among CarbonBlack candidates",
        "t15760": t_details,
        "selected_carbon_black": selected_details,
        "B00458_reference": next((x for x in records if x["CompoundName"].startswith("M1-B00458")), None),
        "status": "CANDIDATE_SELECTED_FOR_PARALLEL_C1_AUDIT" if selected else "BLOCKED_NO_CARBONBLACK_CANDIDATE",
    }
    (OUT / "carbon_black_parallel_candidate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# 双胶料并行验证候选审计",
        "",
        "- 数据范围：冻结 RDS Active 快照；未访问本地 MMS/Archive。",
        f"- T15760：`{t15760}`。",
        f"- 推荐并行炭黑候选：`{selected}`。",
        "- 选择依据：Active 组级样本、C1 OOT 覆盖和订单内相邻组趋势对的综合覆盖量。",
        "- 当前结果是候选选择和基线审计，不代表模型或推荐已放行。",
        "",
        "## 下一步",
        "",
        "1. 对 T15760 与所选炭黑分别建立材料体系专家残差候选。",
        "2. 先做 raw/C1 的严格 OOT 与订单内变化趋势核对。",
        "3. 再将各自的 Dry/Wet/PID Step 语义接入反事实过程响应；没有 MODEL_PATH_OK 不生成推荐。",
    ]
    (OUT / "dual_pilot_candidate_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected": selected, "output": str(OUT), "t15760": t_details, "carbon": selected_details}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
