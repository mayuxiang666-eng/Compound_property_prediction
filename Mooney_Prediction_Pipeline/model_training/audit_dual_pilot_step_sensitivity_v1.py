"""Audit real Step Target/Value sensitivity for T15760 and B00460.

This is a retrospective diagnostic only. It uses the frozen RDS Active
exports, joins each batch's own BatchData rows to its own label group, and
does not query MMS/Archive or create recommendations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUT = BASE / "dual_target_readiness"
TARGETS = ["M1-T15760---- 06 002", "M3B-B00460---- 01 002"]


def corr(a: pd.Series, b: pd.Series) -> float:
    z = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(z) < 8 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(z.iloc[:, 0].corr(z.iloc[:, 1]))


def kind(path: str) -> str | None:
    p = str(path).casefold()
    if "step-time" in p or "duration" in p:
        return "duration"
    if "temperature" in p or "temp" in p:
        return "temperature"
    if "rotor-speed" in p or "rpm" in p:
        return "rotor"
    if "ram-pressure" in p or "ram" in p:
        return "ram"
    return None


def phase(group: str, step: int, material: str) -> str:
    g = str(group).upper()
    if "MSB" in g:
        return "PID_LOWER_MSB" if material == "Silica" else "BOTTOM_MSB"
    # Process semantics: early MST is loading/dry, middle is oil/wet, and
    # late MST is upper PID for silica; for carbon black it remains PID/Dry
    # depending on the actual recipe family and is reported transparently.
    if step <= 2:
        return "LOCKED_LOADING"
    if material == "Silica":
        if step in (3, 4):
            return "DRY_OR_OIL_MST"
        if step <= 9:
            return "WET_MST"
        if step == 10:
            return "PID_UPPER_MST"
        return "TAIL_MST"
    if step <= 4:
        return "DRY_OR_OIL_MST"
    if step <= 9:
        return "WET_MST"
    return "PID_MST"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    static_cols = ["OrderID", "BatchNumber", "PalletID", "LabelGroupID", "CompoundName", "material_system", "BatchStartTime"]
    static = pd.read_csv(BASE / "batch_static_template_skeleton.csv", usecols=static_cols, low_memory=False)
    static = static[static["CompoundName"].isin(TARGETS)].copy()
    static["BatchNumber"] = pd.to_numeric(static["BatchNumber"], errors="coerce")
    static["BatchStartTime"] = pd.to_datetime(static["BatchStartTime"], errors="coerce")
    static = static.drop_duplicates(["OrderID", "BatchNumber"])
    group_lab = pd.read_csv(BASE / "pallet_group_training_frame_c2_prepared.csv", usecols=["LabelGroupID", "OrderID", "PalletID", "CompoundName", "material_system", "MNY"], low_memory=False)
    group_lab = group_lab[group_lab["CompoundName"].isin(TARGETS)].copy()
    group_lab["MNY"] = pd.to_numeric(group_lab["MNY"], errors="coerce")
    group_lab = group_lab.drop_duplicates("LabelGroupID")

    batchdata = pd.read_csv(BASE / "batchdata_targets_values_dedup.csv", low_memory=False)
    batchdata["BatchNumber"] = pd.to_numeric(batchdata["BatchNumber"], errors="coerce")
    batchdata = batchdata.merge(static, on=["OrderID", "BatchNumber"], how="inner", suffixes=("", "_static"))
    batchdata["kind"] = batchdata["VariablePath"].map(kind)
    batchdata = batchdata[batchdata["kind"].notna()].copy()
    batchdata["StepNo"] = pd.to_numeric(batchdata["StepNo"], errors="coerce")
    batchdata["Target"] = pd.to_numeric(batchdata["Target"], errors="coerce")
    batchdata["Value"] = pd.to_numeric(batchdata["Value"], errors="coerce")
    batchdata["step_key"] = batchdata["GroupName"].astype(str) + "|S" + batchdata["StepNo"].fillna(-1).astype(int).astype(str) + "|" + batchdata["kind"]
    batchdata["phase_semantic"] = [phase(g, int(s) if pd.notna(s) else -1, m) for g, s, m in zip(batchdata["GroupName"], batchdata["StepNo"], batchdata["material_system"])]

    group_features = (
        batchdata.groupby(["LabelGroupID", "OrderID", "PalletID", "CompoundName", "material_system", "BatchStartTime", "GroupName", "StepNo", "kind", "step_key", "phase_semantic"], dropna=False)
        .agg(target_mean=("Target", "mean"), actual_value_mean=("Value", "mean"), batch_rows=("BatchNumber", "nunique"), raw_rows=("Value", "size"))
        .reset_index()
    )
    group_features = group_features.merge(group_lab[["LabelGroupID", "MNY"]], on="LabelGroupID", how="inner")
    group_features.to_csv(OUT / "dual_pilot_group_step_features.csv", index=False, encoding="utf-8-sig")

    pair_rows = []
    for (compound, order_id), g in group_features.dropna(subset=["BatchStartTime", "MNY"]).groupby(["CompoundName", "OrderID"]):
        groups = g[["LabelGroupID", "BatchStartTime", "MNY"]].drop_duplicates().sort_values(["BatchStartTime", "LabelGroupID"])
        if len(groups) < 2:
            continue
        for i in range(len(groups) - 1):
            a, b = groups.iloc[i], groups.iloc[i + 1]
            pair_rows.append({"CompoundName": compound, "OrderID": order_id, "from_group": a["LabelGroupID"], "to_group": b["LabelGroupID"], "from_mny": a["MNY"], "to_mny": b["MNY"], "mny_delta": b["MNY"] - a["MNY"]})
    pairs = pd.DataFrame(pair_rows)

    sensitivity = []
    for (compound, step_key, semantic, k), g in group_features.groupby(["CompoundName", "step_key", "phase_semantic", "kind"], dropna=False):
        base = g[["LabelGroupID", "OrderID", "BatchStartTime", "target_mean", "actual_value_mean"]].copy()
        base = base.merge(group_features[["LabelGroupID", "MNY"]].drop_duplicates(), on="LabelGroupID", how="left")
        rows = []
        for order_id, x in base.dropna(subset=["BatchStartTime", "MNY"]).groupby("OrderID"):
            x = x.sort_values(["BatchStartTime", "LabelGroupID"])
            for i in range(len(x) - 1):
                a, b = x.iloc[i], x.iloc[i + 1]
                rows.append({"target_delta": b["target_mean"] - a["target_mean"], "actual_value_delta": b["actual_value_mean"] - a["actual_value_mean"], "mny_delta": b["MNY"] - a["MNY"]})
        d = pd.DataFrame(rows)
        if d.empty:
            continue
        target_nonzero = d["target_delta"].abs() > 1e-9
        sensitivity.append({
            "CompoundName": compound,
            "step_key": step_key,
            "phase_semantic": semantic,
            "control_kind": k,
            "group_rows": int(len(g)),
            "within_order_pairs": int(len(d)),
            "target_changed_pairs": int(target_nonzero.sum()),
            "target_delta_std": float(d["target_delta"].std(ddof=0)),
            "actual_value_delta_std": float(d["actual_value_delta"].std(ddof=0)),
            "mny_delta_std": float(d["mny_delta"].std(ddof=0)),
            "target_delta_mny_corr": corr(d.loc[target_nonzero, "target_delta"], d.loc[target_nonzero, "mny_delta"]) if target_nonzero.sum() >= 8 else np.nan,
            "actual_value_delta_mny_corr": corr(d.loc[target_nonzero, "actual_value_delta"], d.loc[target_nonzero, "mny_delta"]) if target_nonzero.sum() >= 8 else np.nan,
        })
    sens = pd.DataFrame(sensitivity)
    sens = sens.sort_values(["CompoundName", "target_changed_pairs", "target_delta_mny_corr"], ascending=[True, False, False], na_position="last")
    sens.to_csv(OUT / "dual_pilot_step_sensitivity.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source_policy": "RDS Active snapshot only; local MMS/Archive not accessed",
        "compounds": TARGETS,
        "static_batches": int(len(static)),
        "batchdata_rows_after_join": int(len(batchdata)),
        "group_step_feature_rows": int(len(group_features)),
        "within_order_pairs": int(len(pairs)),
        "pair_counts_by_compound": pairs.groupby("CompoundName").size().to_dict() if not pairs.empty else {},
        "step_rows_by_compound": sens.groupby("CompoundName").size().to_dict() if not sens.empty else {},
        "interpretation": {
            "T15760": "白炭黑将 upper MST PID 与 lower MSB PID continuation 分开审计；不假设 duration 单调降低 MNY。",
            "B00460": "炭黑先分 Dry/Wet/PID_MST/BOTTOM_MSB；仅有组内实际 Target 变化且样本足够的 Step 才进入候选特征白名单。",
        },
    }
    (OUT / "dual_pilot_step_sensitivity_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
