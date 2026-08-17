"""Corrected read-only Step sensitivity summary using actual process changes.

The recipe Target vector can be constant within a pilot version.  In that
case Target deltas are zero even though executed Value deltas vary.  This
version reports both separately and ranks actual Value changes for diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUT = BASE / "dual_target_readiness"


def corr(a, b):
    z = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(z) < 8 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(z.iloc[:, 0].corr(z.iloc[:, 1]))


def main():
    p = OUT / "dual_pilot_group_step_features.csv"
    df = pd.read_csv(p, low_memory=False)
    df["BatchStartTime"] = pd.to_datetime(df["BatchStartTime"], errors="coerce")
    df["MNY"] = pd.to_numeric(df["MNY"], errors="coerce")
    rows = []
    for (compound, step_key, semantic, control_kind), g in df.groupby(["CompoundName", "step_key", "phase_semantic", "kind"], dropna=False):
        d = []
        for _, x in g.dropna(subset=["BatchStartTime", "MNY"]).groupby("OrderID"):
            x = x.sort_values(["BatchStartTime", "LabelGroupID"])
            for i in range(len(x) - 1):
                a, b = x.iloc[i], x.iloc[i + 1]
                d.append({
                    "target_delta": b["target_mean"] - a["target_mean"],
                    "actual_value_delta": b["actual_value_mean"] - a["actual_value_mean"],
                    "mny_delta": b["MNY"] - a["MNY"],
                })
        q = pd.DataFrame(d)
        if q.empty:
            continue
        target_changed = q["target_delta"].abs() > 1e-9
        actual_changed = q["actual_value_delta"].abs() > 1e-9
        rows.append({
            "CompoundName": compound,
            "step_key": step_key,
            "phase_semantic": semantic,
            "control_kind": control_kind,
            "within_order_pairs": int(len(q)),
            "target_changed_pairs": int(target_changed.sum()),
            "actual_value_changed_pairs": int(actual_changed.sum()),
            "target_delta_std": float(q["target_delta"].std(ddof=0)),
            "actual_value_delta_std": float(q["actual_value_delta"].std(ddof=0)),
            "mny_delta_std": float(q["mny_delta"].std(ddof=0)),
            "target_delta_mny_corr": corr(q.loc[target_changed, "target_delta"], q.loc[target_changed, "mny_delta"]) if target_changed.sum() >= 8 else np.nan,
            "actual_value_delta_mny_corr": corr(q.loc[actual_changed, "actual_value_delta"], q.loc[actual_changed, "mny_delta"]) if actual_changed.sum() >= 8 else np.nan,
        })
    out = pd.DataFrame(rows)
    out = out.sort_values(["CompoundName", "actual_value_changed_pairs", "actual_value_delta_mny_corr"], ascending=[True, False, False], na_position="last")
    out.to_csv(OUT / "dual_pilot_step_sensitivity_v2.csv", index=False, encoding="utf-8-sig")
    summary = {
        "source": str(p),
        "source_policy": "RDS Active snapshot only; local MMS/Archive not accessed",
        "note": "Target deltas measure recipe-setting changes; actual Value deltas measure executed process changes. They must not be conflated.",
        "by_compound": {},
    }
    for compound, g in out.groupby("CompoundName"):
        ranked = g[g["actual_value_changed_pairs"] >= 8].head(10)
        summary["by_compound"][compound] = {
            "step_count": int(len(g)),
            "steps_with_actual_variation_ge_8_pairs": int((g["actual_value_changed_pairs"] >= 8).sum()),
            "top_steps": ranked[["step_key", "phase_semantic", "control_kind", "actual_value_changed_pairs", "actual_value_delta_mny_corr"]].to_dict("records"),
        }
    (OUT / "dual_pilot_step_sensitivity_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
