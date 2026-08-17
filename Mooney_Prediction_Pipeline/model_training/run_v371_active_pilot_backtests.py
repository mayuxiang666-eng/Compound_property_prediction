"""Run pilot-specific chronological backtests for B00458 and T09170.

Each pilot determines the two time cutoffs from its own complete Active label
groups.  All compounds are then split by those cutoffs, so no future order can
enter training for the selected pilot's OOT window.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.model_training import train_v371_active_candidates as base


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history"
SOURCE = BASE / "phase_corrected_v371_active.csv"
SCHEMA = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v37_production_model_package" / "feature_metadata.json"
ELIGIBILITY = BASE / "v371_active_candidates_v1" / "operational_core_no_rubber_qm" / "group_eligibility_and_split.csv"
OUTPUT = BASE / "v371_active_pilot_backtests_v1"


PILOTS = {
    "b00458": "B00458",
    "t09170": "T09170",
}


def pilot_cutoffs(groups: pd.DataFrame, pattern: str) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, int]]:
    pilot = groups.loc[
        groups["GroupComplete"]
        & ~groups["MNYConflict"]
        & groups["CompoundName"].astype(str).str.upper().str.contains(pattern, regex=False)
    ].copy()
    by_order = pilot.groupby("OrderID", as_index=False).agg(
        OrderTime=("OrderStartTime", "max"),
        GroupCount=("LabelGroupID", "size"),
    ).sort_values(["OrderTime", "OrderID"], kind="stable").reset_index(drop=True)
    total = int(by_order["GroupCount"].sum())
    if total < 150:
        raise ValueError(f"{pattern} has only {total} complete groups; minimum is 150")
    cumulative = by_order["GroupCount"].cumsum()
    validation_rows = by_order.loc[cumulative > 0.70 * total]
    test_rows = by_order.loc[cumulative > 0.85 * total]
    if validation_rows.empty or test_rows.empty:
        raise ValueError(f"{pattern} cannot form pilot-specific validation/OOT windows")
    validation_start = pd.Timestamp(validation_rows.iloc[0]["OrderTime"])
    test_start = pd.Timestamp(test_rows.iloc[0]["OrderTime"])
    counts = {
        "total": total,
        "train": int(by_order.loc[by_order["OrderTime"] < validation_start, "GroupCount"].sum()),
        "validation": int(by_order.loc[
            (by_order["OrderTime"] >= validation_start) & (by_order["OrderTime"] < test_start), "GroupCount"
        ].sum()),
        "oot_test": int(by_order.loc[by_order["OrderTime"] >= test_start, "GroupCount"].sum()),
    }
    return validation_start, test_start, counts


def split_all_orders(
    source: pd.DataFrame,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
) -> pd.DataFrame:
    orders = source.groupby("OrderID", as_index=False).agg(
        OrderTime=("OrderStartTime", "max"),
        GroupCount=("LabelGroupID", "nunique"),
    )
    orders["Split"] = "train"
    orders.loc[orders["OrderTime"] >= validation_start, "Split"] = "validation"
    orders.loc[orders["OrderTime"] >= test_start, "Split"] = "oot_test"
    return orders


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
    for name, pattern in PILOTS.items():
        validation_start, test_start, expected_counts = pilot_cutoffs(groups, pattern)
        order_splits = split_all_orders(source, validation_start, test_start)
        order_splits.to_csv(OUTPUT / f"{name}_order_time_splits.csv", index=False, encoding="utf-8-sig")
        summary = base.train_candidate(
            f"{name}_operational_core", source, schema, order_splits, OUTPUT
        )
        summary["pilot_pattern"] = pattern
        summary["pilot_validation_start"] = validation_start.isoformat()
        summary["pilot_oot_start"] = test_start.isoformat()
        summary["expected_pilot_group_counts"] = expected_counts
        candidate_dir = OUTPUT / f"{name}_operational_core"
        (candidate_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summaries[name] = summary

    result = {
        "experiment": "V3.7.1-ACTIVE-PILOT-SPECIFIC-TEMPORAL-BACKTESTS-V1",
        "contract": "ALL_TRAINING_ORDERS_STRICTLY_PRECEDE_EACH_PILOT_OOT_CUTOFF",
        "candidates": summaries,
        "automatic_control": "DISABLED",
        "recommendation_status": "NOT_GENERATED_MODEL_VALIDATION_ONLY",
    }
    (OUTPUT / "pilot_backtest_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
