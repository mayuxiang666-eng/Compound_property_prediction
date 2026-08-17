"""Efficient raw-material lot lineage audit for RecipeID 9922A09170AX_PV50_004."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection


ROOT = Path(__file__).resolve().parents[2]
RECIPE_ID = "9922A09170AX_PV50_004"
COMPOUND_NAME = "M1-T09170WS-- 02 004"
HISTORY_PATH = ROOT / "Mooney_Prediction_Pipeline" / "models" / "hefei_group_training_dataset_clean.csv"
OUTPUT_DIRECTORY = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"


def _category(material_code: object) -> str:
    code = str(material_code).upper().strip()
    if code.startswith(("CE", "CN")):
        return "ELASTOMER"
    if code.startswith("CC"):
        return "CARBON_BLACK"
    if code.startswith("CS100"):
        return "SILICA"
    if code.startswith("CS"):
        return "OIL"
    return "OTHER"


def _query_lots(connection, *, archive: bool) -> pd.DataFrame:
    prefix = "SFEPLANT_ARCHIVE.dbo" if archive else "dbo"
    query = f"""
        SELECT CAST(bm.OrderID AS VARCHAR(40)) AS OrderID,
               bm.BatchNumber, bm.MaterialCode, bm.ChargeID, bm.Timestamp
        FROM {prefix}.BatchMaterials bm WITH (NOLOCK)
        INNER JOIN {prefix}.Orders o WITH (NOLOCK) ON o.OrderID = bm.OrderID
        WHERE o.CompoundDescription = ?
          AND bm.ChargeID IS NOT NULL
          AND bm.ChargeID NOT LIKE ?
    """
    return pd.read_sql_query(query, connection, params=[RECIPE_ID, "%Unknown Lot%"])


def _query_headers(connection, *, archive: bool) -> pd.DataFrame:
    prefix = "SFEPLANT_ARCHIVE.dbo" if archive else "dbo"
    query = f"""
        SELECT CAST(bh.OrderID AS VARCHAR(40)) AS OrderID,
               bh.BatchNumber, CAST(bh.PalletID AS VARCHAR(40)) AS PalletID
        FROM {prefix}.BatchHeader bh WITH (NOLOCK)
        INNER JOIN {prefix}.Orders o WITH (NOLOCK) ON o.OrderID = bh.OrderID
        WHERE o.CompoundDescription = ?
    """
    return pd.read_sql_query(query, connection, params=[RECIPE_ID])


def _normalise_lots(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["OrderID", "BatchNumber", "raw_material_lot_key", "raw_material_lot_count", "raw_material_category_count"])
    frame = raw.copy()
    frame["OrderID"] = frame["OrderID"].astype(str).str.strip()
    frame["BatchNumber"] = pd.to_numeric(frame["BatchNumber"], errors="coerce")
    frame["ChargeID"] = frame["ChargeID"].astype(str).str.strip()
    frame["MaterialCode"] = frame["MaterialCode"].astype(str).str.strip()
    frame = frame.dropna(subset=["OrderID", "BatchNumber"])
    frame = frame[frame["ChargeID"].ne("") & frame["ChargeID"].ne("nan")]
    frame["material_category"] = frame["MaterialCode"].map(_category)
    frame["lot_token"] = frame["material_category"] + ":" + frame["ChargeID"]
    frame = frame.drop_duplicates(subset=["OrderID", "BatchNumber", "MaterialCode", "ChargeID"])
    return frame.groupby(["OrderID", "BatchNumber"], as_index=False).agg(
        raw_material_lot_key=("lot_token", lambda values: "|".join(sorted(set(values)))),
        raw_material_lot_count=("ChargeID", "nunique"),
        raw_material_category_count=("material_category", "nunique"),
    )


def main() -> None:
    history = pd.read_csv(HISTORY_PATH, low_memory=False)
    candidate = history[history["CompoundName"].astype(str).eq(COMPOUND_NAME)].copy()
    candidate["OrderID"] = candidate["OrderID"].astype(str).str.strip()
    candidate["PalletID"] = candidate["PalletID"].astype(str).str.strip()
    candidate_keys = candidate[["OrderID", "PalletID"]].drop_duplicates()

    connection = get_sfe_connection()
    try:
        online_lots = _query_lots(connection, archive=False)
        archive_lots = _query_lots(connection, archive=True)
        online_headers = _query_headers(connection, archive=False)
        archive_headers = _query_headers(connection, archive=True)
    finally:
        connection.close()

    lots = _normalise_lots(pd.concat([online_lots, archive_lots], ignore_index=True))
    headers = pd.concat([online_headers, archive_headers], ignore_index=True)
    headers["OrderID"] = headers["OrderID"].astype(str).str.strip()
    headers["PalletID"] = headers["PalletID"].astype(str).str.strip()
    headers["BatchNumber"] = pd.to_numeric(headers["BatchNumber"], errors="coerce")
    headers = headers.dropna(subset=["OrderID", "PalletID", "BatchNumber"]).drop_duplicates()

    batch_map = candidate_keys.merge(headers, on=["OrderID", "PalletID"], how="left")
    matched_batches = batch_map.merge(lots, on=["OrderID", "BatchNumber"], how="left", indicator=True)
    matched_batches["batch_lot_status"] = matched_batches["_merge"].map({"both": "PASS", "left_only": "MISSING_BATCH_MATERIALS"})
    matched_batches = matched_batches.drop(columns=["_merge"])
    detail_path = OUTPUT_DIRECTORY / "t09170ws_02004_raw_material_lot_by_batch_v3.csv"

    matched = matched_batches.groupby(["OrderID", "PalletID"], as_index=False).agg(
        mapped_batch_count=("BatchNumber", "nunique"),
        raw_lot_batch_count=("batch_lot_status", lambda values: int((values == "PASS").sum())),
        raw_material_lot_key=("raw_material_lot_key", lambda values: "|".join(sorted(set(str(v) for v in values.dropna() if str(v).strip())))),
        raw_material_lot_count=("raw_material_lot_count", "sum"),
        raw_material_category_count=("raw_material_category_count", "max"),
    )
    matched["raw_material_lot_status"] = matched.apply(
        lambda row: "PASS" if row["mapped_batch_count"] > 0 and row["mapped_batch_count"] == row["raw_lot_batch_count"] else "PARTIAL_OR_MISSING_BATCH_MATERIALS",
        axis=1,
    )
    matched = candidate_keys.merge(matched, on=["OrderID", "PalletID"], how="left")
    matched["raw_material_lot_status"] = matched["raw_material_lot_status"].fillna("MISSING_BATCH_HEADER")
    matched.to_csv(OUTPUT_DIRECTORY / "t09170ws_02004_raw_material_lot_by_pallet_v3.csv", index=False, encoding="utf-8-sig")
    matched_batches.to_csv(detail_path, index=False, encoding="utf-8-sig")

    complete = matched["raw_material_lot_status"].eq("PASS")
    summary = {
        "recipe_id": RECIPE_ID,
        "compound_name": COMPOUND_NAME,
        "candidate_rows": int(len(candidate)),
        "candidate_pallet_keys": int(len(candidate_keys)),
        "sql_online_lot_rows": int(len(online_lots)),
        "sql_archive_lot_rows": int(len(archive_lots)),
        "sql_distinct_order_batch_lots": int(len(lots)),
        "sql_distinct_order_batch_header_keys": int(len(headers)),
        "candidate_pallets_with_complete_raw_lot_coverage": int(complete.sum()),
        "candidate_pallet_coverage": float(complete.mean()) if len(complete) else 0.0,
        "status": "PASS" if bool(complete.any()) else "BLOCKED_NO_BATCH_MATERIALS",
        "source": "SFEPLANT.dbo/ARCHIVE BatchMaterials joined through exact RecipeID Orders; BatchNumber mapped by BatchHeader",
        "raw_material_lot_field": "raw_material_lot_key",
        "note": "Lineage coverage does not convert Pallet-group Lab labels into one-batch labels.",
    }
    summary_path = OUTPUT_DIRECTORY / "t09170ws_02004_raw_material_lot_summary_v3.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"detail={detail_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
