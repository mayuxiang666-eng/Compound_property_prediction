"""Efficient, read-only raw-material lot lineage audit for the exact RecipeID.

The older audit queried BatchMaterials by a large list of local OrderIDs and
timed out.  This version lets SQL use the Orders -> BatchMaterials join for
the exact recipe, then intersects the result with the local candidate history.
No credentials or raw connection strings are printed.
"""

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


def _query(connection, *, archive: bool) -> pd.DataFrame:
    prefix = "SFEPLANT_ARCHIVE.dbo" if archive else "dbo"
    query = f"""
        SELECT
            CAST(bm.OrderID AS VARCHAR(40)) AS OrderID,
            bm.BatchNumber,
            bm.MaterialCode,
            bm.ChargeID,
            bm.Timestamp
        FROM {prefix}.BatchMaterials bm WITH (NOLOCK)
        INNER JOIN {prefix}.Orders o WITH (NOLOCK)
            ON o.OrderID = bm.OrderID
        WHERE o.CompoundDescription = ?
          AND bm.ChargeID IS NOT NULL
          AND bm.ChargeID NOT LIKE ?
    """
    return pd.read_sql_query(query, connection, params=[RECIPE_ID, "%Unknown Lot%"])


def _normalise_raw(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["OrderID", "BatchNumber", "raw_material_lot_key", "raw_material_lot_count"])
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
    grouped = (
        frame.groupby(["OrderID", "BatchNumber"], as_index=False)
        .agg(
            raw_material_lot_key=("lot_token", lambda values: "|".join(sorted(set(values)))),
            raw_material_lot_count=("ChargeID", "nunique"),
            raw_material_category_count=("material_category", "nunique"),
        )
    )
    grouped["raw_material_lot_status"] = "PASS"
    return grouped


def main() -> None:
    history = pd.read_csv(HISTORY_PATH, low_memory=False)
    candidate = history[history["CompoundName"].astype(str).eq(COMPOUND_NAME)].copy()
    candidate["OrderID"] = candidate["OrderID"].astype(str).str.strip()
    candidate["BatchNumber"] = pd.to_numeric(candidate["BatchNumber"], errors="coerce")
    candidate_keys = candidate[["OrderID", "BatchNumber"]].drop_duplicates()

    connection = get_sfe_connection()
    try:
        online = _query(connection, archive=False)
        archive = _query(connection, archive=True)
    finally:
        connection.close()
    raw = pd.concat([online, archive], ignore_index=True)
    lots = _normalise_raw(raw)
    matched = candidate_keys.merge(lots, on=["OrderID", "BatchNumber"], how="left", indicator=True)
    matched["raw_material_lot_status"] = matched["_merge"].map(
        {"both": "PASS", "left_only": "MISSING_BATCH_MATERIALS"}
    )
    matched = matched.drop(columns=["_merge"])

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    detail_path = OUTPUT_DIRECTORY / "t09170ws_02004_raw_material_lot_by_batch_v2.csv"
    matched.to_csv(detail_path, index=False, encoding="utf-8-sig")

    candidate_order_count = int(candidate["OrderID"].nunique())
    matched_order_count = int(matched.loc[matched["raw_material_lot_status"].eq("PASS"), "OrderID"].nunique())
    matched_batch_count = int(matched["raw_material_lot_status"].eq("PASS").sum())
    summary = {
        "recipe_id": RECIPE_ID,
        "compound_name": COMPOUND_NAME,
        "candidate_rows": int(len(candidate)),
        "candidate_orders": candidate_order_count,
        "candidate_order_batch_keys": int(len(candidate_keys)),
        "sql_online_rows": int(len(online)),
        "sql_archive_rows": int(len(archive)),
        "sql_distinct_order_batch_lots": int(len(lots)),
        "candidate_orders_with_raw_lot": matched_order_count,
        "candidate_batch_keys_with_raw_lot": matched_batch_count,
        "candidate_batch_key_coverage": float(matched_batch_count / len(candidate_keys)) if len(candidate_keys) else 0.0,
        "status": "PASS" if matched_batch_count else "BLOCKED_NO_BATCH_MATERIALS",
        "source": "SFEPLANT.dbo.BatchMaterials + SFEPLANT_ARCHIVE.dbo.BatchMaterials joined through exact RecipeID Orders",
        "raw_material_lot_field": "raw_material_lot_key",
        "note": "Coverage is lineage coverage only; it does not convert Pallet-group Lab labels into one-batch labels.",
    }
    summary_path = OUTPUT_DIRECTORY / "t09170ws_02004_raw_material_lot_summary_v2.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"detail={detail_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
