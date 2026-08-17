"""Strictest rerun: require both label N_batches=1 and BatchHeader count=1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.optimization import run_t09170_strict_step_phase_v37_audit as audit


RAW_BATCH = audit.RAW_BATCH
PALLET_AUDIT = audit.OUT / "t09170ws_02004_raw_material_lot_by_pallet_v3.csv"
_read_csv = audit.pd.read_csv


def _read_csv_strict(path, *args, **kwargs):
    frame = _read_csv(path, *args, **kwargs)
    if Path(path).resolve() == Path(RAW_BATCH).resolve():
        pallet = _read_csv(PALLET_AUDIT, low_memory=False)
        allowed = pallet.loc[
            pallet["mapped_batch_count"].eq(1) & pallet["raw_material_lot_status"].eq("PASS"),
            ["OrderID", "PalletID"],
        ].drop_duplicates()
        frame["OrderID"] = frame["OrderID"].astype(str).str.strip()
        frame["PalletID"] = frame["PalletID"].astype(str).str.strip()
        frame = frame.merge(allowed, on=["OrderID", "PalletID"], how="inner")
    return frame


def _pair_clause(pairs: list[tuple[str, int]]) -> str:
    terms = []
    for order_id, batch_number in pairs:
        safe_order_id = str(order_id).replace("'", "''")
        terms.append(f"(bd.OrderID = '{safe_order_id}' AND bd.BatchNumber = {int(batch_number)})")
    return " OR ".join(terms) or "1 = 0"


audit.pd.read_csv = _read_csv_strict
audit._pair_clause = _pair_clause

if __name__ == "__main__":
    audit.main()
