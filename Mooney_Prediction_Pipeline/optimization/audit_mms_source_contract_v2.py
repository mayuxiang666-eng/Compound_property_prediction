"""Read-only evidence audit for recipe, ChargeID/QM and BatchCurve selection.

Credentials are loaded from the existing JSON and are never written to output.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import psycopg2
import pyodbc


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow" / "mms_source_contract_v2"
DEFAULT_CREDENTIALS = ROOT / "Mooney_Prediction_Pipeline" / "data_processing" / "credentials.json"
RECIPE_ID = os.environ.get("MOONEY_AUDIT_RECIPE_ID", "9922A09170AX_PV50_004")
ORDER_ID = os.environ.get("MOONEY_AUDIT_ORDER_ID", "2016836")
AUDIT_BATCH = int(os.environ.get("MOONEY_AUDIT_BATCH_NUMBER", "1"))


def credentials() -> dict:
    path = Path(os.environ.get("MOONEY_MMS_CREDENTIALS_PATH", DEFAULT_CREDENTIALS))
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def connect_mms(database: str):
    c = credentials()["HF_MMS"]
    return pyodbc.connect(
        f"DRIVER={c['driver']};SERVER={c['server']};DATABASE={database};"
        f"UID={c['username']};PWD={c['password']};Encrypt=yes;TrustServerCertificate=yes;",
        timeout=60,
    )


def query_mms(sql: str, params: list[object] | None = None, database: str = "SFEPLANT") -> pd.DataFrame:
    with connect_mms(database) as connection:
        return pd.read_sql_query(sql, connection, params=params or [])


def charge_hash(value: object) -> str:
    return hashlib.sha256(str(value).strip().upper().encode("utf-8")).hexdigest()[:16]


def recipe_rows() -> pd.DataFrame:
    sql = """
    SELECT N'SFEPLANT' SourceDatabase, RecipeID, StepNumber, MaterialCode, MaterialName,
           TRY_CAST(Pphr AS FLOAT) Pphr
      FROM SFEPLANT.dbo.RecipeMaterials WITH (NOLOCK) WHERE RecipeID=?
    UNION ALL
    SELECT N'SFEPLANT_ARCHIVE', RecipeID, StepNumber, MaterialCode, MaterialName,
           TRY_CAST(Pphr AS FLOAT)
      FROM SFEPLANT_ARCHIVE.dbo.RecipeMaterials WITH (NOLOCK) WHERE RecipeID=?
    """
    return query_mms(sql, [RECIPE_ID, RECIPE_ID])


def order_materials() -> pd.DataFrame:
    sql = """
    SELECT N'SFEPLANT' SourceDatabase, CAST(o.OrderID AS VARCHAR(40)) OrderID,
           o.CompoundDescription RecipeID, om.MaterialCode, om.MaterialName,
           TRY_CAST(om.BatchWeight AS FLOAT) MaterialWeight
      FROM SFEPLANT.dbo.Orders o WITH (NOLOCK)
      JOIN SFEPLANT.dbo.OrderMaterials om WITH (NOLOCK) ON om.OrderID=o.OrderID
     WHERE CAST(o.OrderID AS VARCHAR(40))=?
    UNION ALL
    SELECT N'SFEPLANT_ARCHIVE', CAST(o.OrderID AS VARCHAR(40)), o.CompoundDescription,
           om.MaterialCode, om.MaterialName, TRY_CAST(om.BatchWeight AS FLOAT)
      FROM SFEPLANT_ARCHIVE.dbo.Orders o WITH (NOLOCK)
      JOIN SFEPLANT_ARCHIVE.dbo.OrderMaterials om WITH (NOLOCK) ON om.OrderID=o.OrderID
     WHERE CAST(o.OrderID AS VARCHAR(40))=?
    """
    return query_mms(sql, [ORDER_ID, ORDER_ID])


def charge_rows() -> pd.DataFrame:
    sql = """
    SELECT N'SFEPLANT' SourceDatabase, CAST(bm.OrderID AS VARCHAR(40)) OrderID,
           bm.BatchNumber, bm.MaterialCode, bm.MaterialName, bm.ChargeID,
           TRY_CAST(bm.ActualWeight AS FLOAT) ActualWeight
      FROM SFEPLANT.dbo.BatchMaterials bm WITH (NOLOCK)
     WHERE CAST(bm.OrderID AS VARCHAR(40))=?
    UNION ALL
    SELECT N'SFEPLANT_ARCHIVE', CAST(bm.OrderID AS VARCHAR(40)), bm.BatchNumber,
           bm.MaterialCode, bm.MaterialName, bm.ChargeID, TRY_CAST(bm.ActualWeight AS FLOAT)
      FROM SFEPLANT_ARCHIVE.dbo.BatchMaterials bm WITH (NOLOCK)
     WHERE CAST(bm.OrderID AS VARCHAR(40))=?
    """
    return query_mms(sql, [ORDER_ID, ORDER_ID])


def curve_metadata() -> pd.DataFrame:
    sql = """
    SELECT TOP (50) N'SFEPLANT' SourceDatabase, CAST(OrderID AS VARCHAR(40)) OrderID,
           BatchNumber, EquipmentID, [Timestamp],
           CASE WHEN Curve1 IS NULL THEN 1 ELSE 0 END Curve1IsNull,
           CASE WHEN Curve2 IS NULL THEN 1 ELSE 0 END Curve2IsNull,
           CASE WHEN Curve5 IS NULL THEN 1 ELSE 0 END Curve5IsNull,
           CASE WHEN Curve6 IS NULL THEN 1 ELSE 0 END Curve6IsNull,
           CASE WHEN Curve7 IS NULL THEN 1 ELSE 0 END Curve7IsNull,
           DATALENGTH(Curve1) Curve1Bytes, DATALENGTH(Curve2) Curve2Bytes,
           DATALENGTH(Curve5) Curve5Bytes, DATALENGTH(Curve6) Curve6Bytes,
           DATALENGTH(Curve7) Curve7Bytes
      FROM SFEPLANT.dbo.BatchCurve WITH (NOLOCK)
     WHERE CAST(OrderID AS VARCHAR(40))=? AND BatchNumber=? ORDER BY [Timestamp] DESC
    """
    return query_mms(sql, [ORDER_ID, AUDIT_BATCH])


def qm_coverage(charges: pd.DataFrame) -> pd.DataFrame:
    valid = charges.loc[
        charges["ChargeID"].notna() & ~charges["ChargeID"].astype(str).str.contains("Unknown Lot", case=False),
        "ChargeID",
    ].astype(str).str.strip().str.upper().drop_duplicates().tolist()
    if not valid:
        return pd.DataFrame()
    c = credentials()["mustangmaster"]
    connection = psycopg2.connect(
        host=c["host"], port=c["port"], database=c["database"],
        user=c["user"], password=c["password"], sslmode="require",
    )
    sql = """
    SELECT batch, material, "short text" parameter, uom,
           AVG(CAST("mean value / s" AS DOUBLE PRECISION)) mean_value
      FROM mustangmaster.mustang_global_glue_01.sap_supplier_qm
     WHERE plant='9200' AND UPPER(TRIM(batch)) = ANY(%s)
     GROUP BY batch, material, "short text", uom
    """
    try:
        result = pd.read_sql_query(sql, connection, params=[valid])
    finally:
        connection.close()
    if not result.empty:
        result["ChargeIDHash"] = result["batch"].map(charge_hash)
        result = result.drop(columns=["batch"])
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recipe = recipe_rows()
    order_mat = order_materials()
    charges = charge_rows()
    curves = curve_metadata()
    qm_error = None
    try:
        qm = qm_coverage(charges)
    except Exception as exc:
        qm = pd.DataFrame()
        qm_error = f"{type(exc).__name__}: {str(exc)[:500]}"

    recipe.to_csv(OUT / "recipe_material_rows.csv", index=False)
    order_mat.to_csv(OUT / "order_material_rows.csv", index=False)
    public_charges = charges.copy()
    public_charges["ChargeIDHash"] = public_charges["ChargeID"].map(charge_hash)
    public_charges = public_charges.drop(columns=["ChargeID"])
    public_charges.to_csv(OUT / "batch_charge_rows_hashed.csv", index=False)
    curves.to_csv(OUT / "batch_curve_row_metadata.csv", index=False)
    qm.to_csv(OUT / "supplier_qm_rows_hashed.csv", index=False)

    recipe_key = ["RecipeID", "StepNumber", "MaterialCode", "MaterialName", "Pphr"]
    duplicate_counts = recipe.groupby(recipe_key, dropna=False).size() if not recipe.empty else pd.Series(dtype=int)
    category = recipe["MaterialCode"].fillna("").astype(str).str.upper() if not recipe.empty else pd.Series(dtype=str)
    summary = {
        "recipe_id": RECIPE_ID,
        "order_id": ORDER_ID,
        "audit_batch": AUDIT_BATCH,
        "recipe_rows_total_active_plus_archive": int(len(recipe)),
        "recipe_unique_rows": int(recipe.drop_duplicates(recipe_key).shape[0]) if not recipe.empty else 0,
        "recipe_duplicate_copies": int((duplicate_counts - 1).clip(lower=0).sum()) if len(duplicate_counts) else 0,
        "recipe_silica_rows": int(category.str.startswith("CS100").sum()) if len(category) else 0,
        "recipe_carbon_black_rows": int(category.str.startswith("CC").sum()) if len(category) else 0,
        "recipe_silica_phr_unique_source_sum": float(
            recipe.loc[category.str.startswith(("CS10", "CS12")), recipe.columns == "Pphr"].sum().iloc[0]
        ) if len(category) and category.str.startswith(("CS10", "CS12")).any() else 0.0,
        "order_material_rows": int(len(order_mat)),
        "charge_rows": int(len(charges)),
        "distinct_charge_ids": int(charges["ChargeID"].dropna().astype(str).nunique()) if not charges.empty else 0,
        "qm_rows": int(len(qm)),
        "qm_query_error": qm_error,
        "curve_rows": int(len(curves)),
        "latest_curve7_is_null": bool(curves.iloc[0]["Curve7IsNull"]) if not curves.empty else None,
        "latest_curve7_bytes": int(curves.iloc[0]["Curve7Bytes"]) if not curves.empty and pd.notna(curves.iloc[0]["Curve7Bytes"]) else None,
        "max_curve7_bytes": int(pd.to_numeric(curves["Curve7Bytes"], errors="coerce").max()) if not curves.empty and curves["Curve7Bytes"].notna().any() else None,
        "credentials_written_to_output": False,
        "read_only_audit": True,
    }
    with (OUT / "mms_source_contract_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
