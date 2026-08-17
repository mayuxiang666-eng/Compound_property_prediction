"""Create a non-destructive enriched candidate dataset with raw-lot lineage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "Mooney_Prediction_Pipeline" / "models" / "hefei_group_training_dataset_clean.csv"
PALLET_AUDIT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow" / "t09170ws_02004_raw_material_lot_by_pallet_v3.csv"
OUT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"


def main() -> None:
    history = pd.read_csv(HISTORY, low_memory=False)
    candidate = history[history["CompoundName"].astype(str).eq("M1-T09170WS-- 02 004")].copy()
    audit = pd.read_csv(PALLET_AUDIT, low_memory=False)
    for frame in (candidate, audit):
        frame["OrderID"] = frame["OrderID"].astype(str).str.strip()
        frame["PalletID"] = frame["PalletID"].astype(str).str.strip()
    keep = audit[[
        "OrderID", "PalletID", "mapped_batch_count", "raw_lot_batch_count",
        "raw_material_lot_key", "raw_material_lot_count", "raw_material_category_count",
        "raw_material_lot_status",
    ]].drop_duplicates(["OrderID", "PalletID"])
    enriched = candidate.merge(keep, on=["OrderID", "PalletID"], how="left", validate="one_to_one")
    enriched["batchheader_one_to_one_status"] = enriched["mapped_batch_count"].map(
        lambda value: "PASS" if value == 1 else "BLOCKED_BATCHHEADER_NOT_ONE_TO_ONE" if pd.notna(value) else "MISSING_BATCHHEADER"
    )
    enriched["strict_batch_alignment_status"] = enriched.apply(
        lambda row: "PASS" if row["N_batches"] == 1 and row["batchheader_one_to_one_status"] == "PASS" and row["raw_material_lot_status"] == "PASS" else "BLOCKED",
        axis=1,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    enriched_path = OUT / "t09170ws_02004_enriched_candidate_with_raw_lots.csv"
    enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")

    strat = enriched.copy()
    strat["raw_material_lot_key"] = strat["raw_material_lot_key"].fillna("MISSING_RAW_MATERIAL_LOT")
    stratification = strat.groupby("raw_material_lot_key", as_index=False).agg(
        pallet_count=("PalletID", "nunique"),
        candidate_row_count=("OrderID", "size"),
        strict_batch_count=("strict_batch_alignment_status", lambda values: int((values == "PASS").sum())),
        mny_count=("MNY", "count"),
        mny_median=("MNY", "median"),
        n_batches_min=("N_batches", "min"),
        n_batches_max=("N_batches", "max"),
    ).sort_values(["strict_batch_count", "pallet_count"], ascending=False)
    strat_path = OUT / "t09170ws_02004_raw_material_stratification.csv"
    stratification.to_csv(strat_path, index=False, encoding="utf-8-sig")
    print(f"enriched_rows={len(enriched)}")
    print(f"strict_rows={int(enriched.strict_batch_alignment_status.eq('PASS').sum())}")
    print(f"lot_strata={len(stratification)}")
    print(f"enriched={enriched_path}")
    print(f"stratification={strat_path}")


if __name__ == "__main__":
    main()
