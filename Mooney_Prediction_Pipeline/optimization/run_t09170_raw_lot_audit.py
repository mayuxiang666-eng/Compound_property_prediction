"""Read-only extraction of batch-level raw-material ChargeID lineage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection


ROOT = Path(__file__).resolve().parents[2]
COMPOUND = "M1-T09170WS-- 02 004"
HISTORY_PATH = ROOT / "Mooney_Prediction_Pipeline" / "models" / "hefei_group_training_dataset_clean.csv"
OUTPUT_DIRECTORY = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"


def _query_batches(connection, order_ids: list[str]) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in order_ids)
    query = f"""
        SELECT OrderID, BatchNumber, MaterialCode, ChargeID, Timestamp
        FROM dbo.BatchMaterials WITH (NOLOCK)
        WHERE CAST(OrderID AS VARCHAR(40)) IN ({placeholders})
          AND ChargeID IS NOT NULL
          AND ChargeID NOT LIKE 'Unknown Lot'
        UNION ALL
        SELECT OrderID, BatchNumber, MaterialCode, ChargeID, Timestamp
        FROM SFEPLANT_ARCHIVE.dbo.BatchMaterials WITH (NOLOCK)
        WHERE CAST(OrderID AS VARCHAR(40)) IN ({placeholders})
          AND ChargeID IS NOT NULL
          AND ChargeID NOT LIKE 'Unknown Lot'
    """
    params = order_ids + order_ids
    return pd.read_sql_query(query, connection, params=params)


def _category(material_code: object) -> str:
    code = str(material_code).upper()
    if code.startswith(("CE", "CN")):
        return "ELASTOMER"
    if code.startswith("CS100"):
        return "SILICA"
    if code.startswith("CC"):
        return "CARBON_BLACK"
    if code.startswith("CS"):
        return "OIL"
    return "OTHER"


def main() -> None:
    history = pd.read_csv(HISTORY_PATH, low_memory=False)
    candidate = history[history["CompoundName"].astype(str).eq(COMPOUND)].copy()
    order_ids = sorted(candidate["OrderID"].astype(str).str.strip().dropna().unique())
    connection = get_sfe_connection()
    try:
        pieces = []
        for start in range(0, len(order_ids), 100):
            pieces.append(_query_batches(connection, order_ids[start : start + 100]))
    finally:
        connection.close()
    raw = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if raw.empty:
        result = pd.DataFrame(
            columns=[
                "OrderID",
                "BatchNumber",
                "raw_material_lot_key",
                "raw_material_lot_count",
                "raw_material_lot_status",
            ]
        )
    else:
        raw["OrderID"] = raw["OrderID"].astype(str).str.strip()
        raw["BatchNumber"] = pd.to_numeric(raw["BatchNumber"], errors="coerce")
        raw["ChargeID"] = raw["ChargeID"].astype(str).str.strip()
        raw["category"] = raw["MaterialCode"].map(_category)
        raw = raw.dropna(subset=["OrderID", "BatchNumber"]).drop_duplicates(
            subset=["OrderID", "BatchNumber", "MaterialCode", "ChargeID"]
        )
        raw["lot_token"] = raw["category"] + ":" + raw["ChargeID"]
        result = (
            raw.groupby(["OrderID", "BatchNumber"], as_index=False)
            .agg(
                raw_material_lot_key=("lot_token", lambda values: "|".join(sorted(set(values)))),
                raw_material_lot_count=("ChargeID", "nunique"),
            )
        )
        result["raw_material_lot_status"] = "PASS"

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIRECTORY / "t09170ws_02004_raw_material_lot_by_batch.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(
        [
            {
                "candidate_order_ids": len(order_ids),
                "orders_with_raw_lot": int(result["OrderID"].nunique()) if not result.empty else 0,
                "batch_rows_with_raw_lot": len(result),
                "coverage_orders": float(result["OrderID"].nunique() / len(order_ids)) if order_ids else 0.0,
                "status": "PASS" if not result.empty else "BLOCKED_NO_BATCH_MATERIALS_FOR_CANDIDATE_HISTORY",
                "source": "SFEPLANT.dbo.BatchMaterials + SFEPLANT_ARCHIVE.dbo.BatchMaterials",
            }
        ]
    )
    summary_path = OUTPUT_DIRECTORY / "t09170ws_02004_raw_material_lot_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"output={output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
