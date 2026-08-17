"""Exact-compound Active temporal backtests for the two named pilot recipes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.model_training import train_v371_active_candidates as base
from Mooney_Prediction_Pipeline.model_training.run_v371_active_pilot_backtests import split_all_orders


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history"
SOURCE = BASE / "phase_corrected_v371_active.csv"
SCHEMA = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v37_production_model_package" / "feature_metadata.json"
ELIGIBILITY = BASE / "v371_active_candidates_v1" / "operational_core_no_rubber_qm" / "group_eligibility_and_split.csv"
OUTPUT = BASE / "v371_active_exact_pilot_backtests_v1"

PILOTS = {
    "b00458_12004": ("M1-B00458---- 12 004", "9922A00458BT_PV77_004", 75),
    "t09170ws_02004": ("M1-T09170WS-- 02 004", "9922A09170AX_PV50_004", 150),
}


def exact_cutoffs(groups: pd.DataFrame, compound: str, minimum: int):
    pilot = groups.loc[
        groups["GroupComplete"] & ~groups["MNYConflict"] & groups["CompoundName"].astype(str).eq(compound)
    ].copy()
    by_order = pilot.groupby("OrderID", as_index=False).agg(
        OrderTime=("OrderStartTime", "max"), GroupCount=("LabelGroupID", "size")
    ).sort_values(["OrderTime", "OrderID"], kind="stable").reset_index(drop=True)
    total = int(by_order["GroupCount"].sum())
    if total < minimum:
        raise ValueError(f"{compound}: {total} groups below diagnostic minimum {minimum}")
    cumulative = by_order["GroupCount"].cumsum()
    validation_start = pd.Timestamp(by_order.loc[cumulative > 0.70 * total].iloc[0]["OrderTime"])
    test_start = pd.Timestamp(by_order.loc[cumulative > 0.85 * total].iloc[0]["OrderTime"])
    counts = {
        "total": total,
        "train": int(by_order.loc[by_order["OrderTime"] < validation_start, "GroupCount"].sum()),
        "validation": int(by_order.loc[(by_order["OrderTime"] >= validation_start) & (by_order["OrderTime"] < test_start), "GroupCount"].sum()),
        "oot_test": int(by_order.loc[by_order["OrderTime"] >= test_start, "GroupCount"].sum()),
    }
    return validation_start, test_start, counts


def main() -> None:
    source = pd.read_csv(SOURCE, low_memory=False)
    source["OrderID"] = source["OrderID"].astype(str).str.strip()
    source["PalletID"] = source["PalletID"].astype(str).str.strip()
    source["BatchNumber"] = pd.to_numeric(source["BatchNumber"], errors="raise").astype(int)
    source["MNY"] = pd.to_numeric(source["MNY"], errors="coerce")
    source["OrderStartTime"] = pd.to_datetime(source["OrderStartTime"], errors="raise")
    source["LabelGroupID"] = source["OrderID"] + "|" + source["PalletID"]
    source = source.dropna(subset=["MNY"]).copy()
    groups = pd.read_csv(ELIGIBILITY, low_memory=False)
    groups["OrderID"] = groups["OrderID"].astype(str).str.strip()
    groups["OrderStartTime"] = pd.to_datetime(groups["OrderStartTime"], errors="raise")
    schema = base.load_schema(SCHEMA, ("supplier_rubber_viscosity_avg",))
    OUTPUT.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for name, (compound, recipe_id, minimum) in PILOTS.items():
        validation_start, test_start, counts = exact_cutoffs(groups, compound, minimum)
        splits = split_all_orders(source, validation_start, test_start)
        summary = base.train_candidate(f"{name}_operational_core", source, schema, splits, OUTPUT)
        predictions = pd.read_csv(OUTPUT / f"{name}_operational_core" / "group_predictions.csv", low_memory=False)
        exact_oot = predictions.loc[
            predictions["Split"].eq("oot_test") & predictions["CompoundName"].astype(str).eq(compound)
        ]
        summary.update({
            "exact_compound": compound,
            "recipe_id": recipe_id,
            "exact_group_counts": counts,
            "exact_oot_metrics": base.metric_block(exact_oot),
            "pilot_evidence_status": "PASS_MIN_150" if counts["total"] >= 150 else "DIAGNOSTIC_BELOW_150",
            "validation_start": validation_start.isoformat(),
            "oot_start": test_start.isoformat(),
        })
        (OUTPUT / f"{name}_operational_core" / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summaries[name] = summary
    result = {
        "experiment": "V3.7.1-ACTIVE-EXACT-PILOT-TEMPORAL-BACKTESTS-V1",
        "candidates": summaries,
        "automatic_control": "DISABLED",
        "recommendation_status": "NOT_GENERATED_MODEL_VALIDATION_ONLY",
    }
    (OUTPUT / "exact_pilot_backtest_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
