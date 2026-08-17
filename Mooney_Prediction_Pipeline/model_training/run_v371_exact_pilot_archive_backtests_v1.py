"""Leak-free Archive-expanded Phase backtests for the two exact pilot recipes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.model_training import train_v371_active_candidates as base
from Mooney_Prediction_Pipeline.model_training.run_v371_active_pilot_backtests import split_all_orders


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history"
SOURCE = BASE / "phase_exact_pilots_corrected_static_v1.csv"
QUALITY = BASE / "exact_pilot_feature_contract_v1" / "batch_feature_quality.csv"
SCHEMA = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v37_production_model_package" / "feature_metadata.json"
OUTPUT = BASE / "v371_exact_pilot_archive_backtests_v1"
PILOTS = {
    "b00458_12004": ("M1-B00458---- 12 004", "9922A00458BT_PV77_004"),
    "t09170ws_02004": ("M1-T09170WS-- 02 004", "9922A09170AX_PV50_004"),
}


def _normalise(source: pd.DataFrame) -> pd.DataFrame:
    out = source.copy()
    out["OrderID"] = out["OrderID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out["PalletID"] = out["PalletID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out["BatchNumber"] = pd.to_numeric(out["BatchNumber"], errors="raise").astype(int)
    out["MNY"] = pd.to_numeric(out["MNY"], errors="coerce")
    out["OrderStartTime"] = pd.to_datetime(out["OrderStartTime"], errors="raise")
    out["LabelGroupID"] = out["OrderID"] + "|" + out["PalletID"]
    return out.dropna(subset=["MNY"]).copy()


def build_group_eligibility(source: pd.DataFrame, quality_path: Path) -> pd.DataFrame:
    quality = pd.read_csv(quality_path, low_memory=False)
    quality["OrderID"] = quality["OrderID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    quality["PalletID"] = quality["PalletID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    quality["BatchNumber"] = pd.to_numeric(quality["BatchNumber"], errors="raise").astype(int)
    quality["LabelGroupID"] = quality["OrderID"] + "|" + quality["PalletID"]
    quality["BatchAccepted"] = quality["FeatureQualityStatus"].eq("MODEL_INPUT_OK")
    quality["GroupComplete"] = quality.groupby("LabelGroupID")["BatchAccepted"].transform("all")
    groups = quality.groupby("LabelGroupID", as_index=False).agg(
        OrderID=("OrderID", "first"), PalletID=("PalletID", "first"),
        BatchCount=("BatchNumber", "size"), AcceptedBatchCount=("BatchAccepted", "sum"),
        GroupComplete=("GroupComplete", "first"),
    )
    labels = source.groupby("LabelGroupID", as_index=False).agg(
        MNY=("MNY", "median"), MNYMin=("MNY", "min"), MNYMax=("MNY", "max"),
        CompoundName=("CompoundName", "first"),
        CompoundDescription=("CompoundDescription", "first"),
        OrderStartTime=("OrderStartTime", "max"),
    )
    labels["MNYConflict"] = (labels["MNYMax"] - labels["MNYMin"]).abs().gt(1e-9)
    return groups.merge(labels, on="LabelGroupID", how="left", validate="one_to_one")


def exact_cutoffs(
    groups: pd.DataFrame, recipe_id: str,
) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, int]]:
    pilot = groups.loc[
        groups["GroupComplete"] & ~groups["MNYConflict"]
        & groups["CompoundDescription"].astype(str).eq(recipe_id)
    ].copy()
    by_order = pilot.groupby("OrderID", as_index=False).agg(
        OrderTime=("OrderStartTime", "max"), GroupCount=("LabelGroupID", "size")
    ).sort_values(["OrderTime", "OrderID"], kind="stable").reset_index(drop=True)
    total = int(by_order["GroupCount"].sum())
    if total < 150:
        raise ValueError(f"{recipe_id}: {total} complete groups below 150")
    cumulative = by_order["GroupCount"].cumsum()
    validation_start = pd.Timestamp(by_order.loc[cumulative > 0.70 * total].iloc[0]["OrderTime"])
    test_start = pd.Timestamp(by_order.loc[cumulative > 0.85 * total].iloc[0]["OrderTime"])
    counts = {
        "total": total,
        "train": int(by_order.loc[by_order["OrderTime"] < validation_start, "GroupCount"].sum()),
        "validation": int(by_order.loc[
            (by_order["OrderTime"] >= validation_start) & (by_order["OrderTime"] < test_start),
            "GroupCount",
        ].sum()),
        "oot_test": int(by_order.loc[by_order["OrderTime"] >= test_start, "GroupCount"].sum()),
    }
    return validation_start, test_start, counts


def main() -> None:
    source = _normalise(pd.read_csv(SOURCE, low_memory=False))
    groups = build_group_eligibility(source, QUALITY)
    schema = base.load_schema(SCHEMA, ("supplier_rubber_viscosity_avg",))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    groups.to_csv(OUTPUT / "pretraining_group_eligibility.csv", index=False, encoding="utf-8-sig")

    summaries: dict[str, object] = {}
    for name, (compound, recipe_id) in PILOTS.items():
        validation_start, test_start, counts = exact_cutoffs(groups, recipe_id)
        splits = split_all_orders(source, validation_start, test_start)
        splits.to_csv(OUTPUT / f"{name}_order_splits.csv", index=False, encoding="utf-8-sig")
        candidate_name = f"{name}_archive_operational_core"
        summary = base.train_candidate(candidate_name, source, schema, splits, OUTPUT)
        predictions = pd.read_csv(
            OUTPUT / candidate_name / "group_predictions.csv", low_memory=False
        )
        exact_oot = predictions.loc[
            predictions["Split"].eq("oot_test")
            & predictions["CompoundName"].astype(str).eq(compound)
        ]
        exact_validation = predictions.loc[
            predictions["Split"].eq("validation")
            & predictions["CompoundName"].astype(str).eq(compound)
        ]
        summary.update({
            "data_contract": "EXACT_PILOT_ARCHIVE_NO_FUTURE_CURVE_FILL_CORRECTED_STATIC_V1",
            "exact_compound": compound, "recipe_id": recipe_id,
            "exact_group_counts": counts,
            "exact_validation_metrics": base.metric_block(exact_validation),
            "exact_oot_metrics": base.metric_block(exact_oot),
            "validation_start": validation_start.isoformat(),
            "oot_start": test_start.isoformat(),
            "pilot_evidence_status": "PASS_MIN_150",
        })
        (OUTPUT / candidate_name / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summaries[name] = summary
    result = {
        "experiment": "V3.7.1-EXACT-PILOT-ARCHIVE-TEMPORAL-BACKTESTS-V1",
        "label_contract": "BATCH_PREDICT_THEN_ORDERID_PALLETID_MEAN",
        "split_contract": "ORDER_ATOMIC_PILOT_SPECIFIC_TEMPORAL_70_15_15",
        "source_batches": int(len(source)),
        "complete_groups_by_recipe": {
            recipe: int(groups.loc[
                groups["CompoundDescription"].eq(recipe)
                & groups["GroupComplete"] & ~groups["MNYConflict"]
            ].shape[0]) for _, recipe in PILOTS.values()
        },
        "candidates": summaries,
        "production_release": False, "database_writes": False,
        "automatic_control": "DISABLED",
        "recommendation_status": "NOT_GENERATED_MODEL_VALIDATION_ONLY",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
