"""Canonical MMS source adapter for the Phase-incremental V3.7 path.

The adapter fixes the two production-contract errors found in the shortcut:
CS1056 silica is classified as silica (not oil), and active/archive duplicates
are deduplicated before aggregation.  Curve selection prefers a complete
temperature/power row and then the newest timestamp; it never treats a later
incomplete row as the final curve.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import psycopg2
import pyodbc

from .canonical_material_contract import (
    aggregate_batch_qm,
    aggregate_order_material_percentages,
    corrected_silica_phr,
    deduplicate_cross_database_rows,
)


ROOT = Path(__file__).resolve().parents[2]
CREDENTIALS_PATH = Path(
    os.environ.get(
        "MOONEY_MMS_CREDENTIALS_PATH",
        ROOT / "Mooney_Prediction_Pipeline" / "data_processing" / "credentials.json",
    )
)


def read_credentials() -> dict:
    with CREDENTIALS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def connect_mms(database: str = "SFEPLANT") -> pyodbc.Connection:
    c = read_credentials()["HF_MMS"]
    connection_string = (
        f"DRIVER={c['driver']};SERVER={c['server']};DATABASE={database};"
        f"UID={c['username']};PWD={c['password']};"
        "Encrypt=yes;TrustServerCertificate=yes;"
    )
    return pyodbc.connect(connection_string, timeout=60)


def query_mms(sql: str, params: list[object] | None = None) -> pd.DataFrame:
    with connect_mms() as connection:
        return pd.read_sql_query(sql, connection, params=params or [])


def query_order_materials(order_id: str) -> pd.DataFrame:
    sql = """
    WITH raw AS
    (
        SELECT N'SFEPLANT' AS SourceDatabase, CAST(o.OrderID AS VARCHAR(64)) AS OrderID,
               o.CompoundDescription AS RecipeID, om.MaterialCode, om.MaterialName,
               TRY_CAST(om.BatchWeight AS FLOAT) AS MaterialWeight
          FROM SFEPLANT.dbo.Orders o WITH (NOLOCK)
          JOIN SFEPLANT.dbo.OrderMaterials om WITH (NOLOCK) ON om.OrderID=o.OrderID
         WHERE CAST(o.OrderID AS VARCHAR(64))=?
        UNION ALL
        SELECT N'SFEPLANT_ARCHIVE', CAST(o.OrderID AS VARCHAR(64)), o.CompoundDescription,
               om.MaterialCode, om.MaterialName, TRY_CAST(om.BatchWeight AS FLOAT)
          FROM SFEPLANT_ARCHIVE.dbo.Orders o WITH (NOLOCK)
          JOIN SFEPLANT_ARCHIVE.dbo.OrderMaterials om WITH (NOLOCK) ON om.OrderID=o.OrderID
         WHERE CAST(o.OrderID AS VARCHAR(64))=?
    )
    SELECT * FROM raw
    """
    return query_mms(sql, [str(order_id), str(order_id)])


def query_batch_headers(order_id: str) -> pd.DataFrame:
    sql = """
    WITH raw AS
    (
        SELECT N'SFEPLANT' AS SourceDatabase, CAST(OrderID AS VARCHAR(64)) OrderID,
               BatchNumber, CAST(PalletID AS VARCHAR(128)) PalletID,
               TRY_CAST(BatchWeight AS FLOAT) BatchWeight, EquipmentID
          FROM SFEPLANT.dbo.BatchHeader WITH (NOLOCK) WHERE CAST(OrderID AS VARCHAR(64))=?
        UNION ALL
        SELECT N'SFEPLANT_ARCHIVE', CAST(OrderID AS VARCHAR(64)), BatchNumber,
               CAST(PalletID AS VARCHAR(128)), TRY_CAST(BatchWeight AS FLOAT), EquipmentID
          FROM SFEPLANT_ARCHIVE.dbo.BatchHeader WITH (NOLOCK) WHERE CAST(OrderID AS VARCHAR(64))=?
    )
    SELECT * FROM raw
    """
    frame = query_mms(sql, [str(order_id), str(order_id)])
    if frame.empty:
        return frame
    return deduplicate_cross_database_rows(frame, ["OrderID", "BatchNumber", "EquipmentID"])


def query_batch_curves(order_id: str) -> pd.DataFrame:
    sql = """
    WITH raw AS
    (
        SELECT N'SFEPLANT' AS SourceDatabase, CAST(OrderID AS VARCHAR(64)) OrderID,
               BatchNumber, EquipmentID, [Timestamp], Curve1, Curve2, Curve5, Curve6, Curve7,
               ROW_NUMBER() OVER
               (
                   PARTITION BY OrderID, BatchNumber, EquipmentID
                   ORDER BY CASE WHEN Curve1 IS NOT NULL AND Curve2 IS NOT NULL THEN 0 ELSE 1 END,
                            [Timestamp] DESC
               ) AS CurveRank
          FROM SFEPLANT.dbo.BatchCurve WITH (NOLOCK)
         WHERE CAST(OrderID AS VARCHAR(64))=?
        UNION ALL
        SELECT N'SFEPLANT_ARCHIVE', CAST(OrderID AS VARCHAR(64)), BatchNumber, EquipmentID,
               [Timestamp], Curve1, Curve2, Curve5, Curve6, Curve7,
               ROW_NUMBER() OVER
               (
                   PARTITION BY OrderID, BatchNumber, EquipmentID
                   ORDER BY CASE WHEN Curve1 IS NOT NULL AND Curve2 IS NOT NULL THEN 0 ELSE 1 END,
                            [Timestamp] DESC
               )
          FROM SFEPLANT_ARCHIVE.dbo.BatchCurve WITH (NOLOCK)
         WHERE CAST(OrderID AS VARCHAR(64))=?
    )
    SELECT * FROM raw WHERE CurveRank=1
    """
    curves = query_mms(sql, [str(order_id), str(order_id)])
    if curves.empty:
        return curves
    return deduplicate_cross_database_rows(curves, ["OrderID", "BatchNumber", "EquipmentID"])


def query_batch_material_charges(order_id: str) -> pd.DataFrame:
    sql = """
    WITH raw AS
    (
        SELECT N'SFEPLANT' SourceDatabase, CAST(OrderID AS VARCHAR(64)) OrderID,
               BatchNumber, MaterialCode, MaterialName, ChargeID,
               TRY_CAST(ActualWeight AS FLOAT) ActualWeight
          FROM SFEPLANT.dbo.BatchMaterials WITH (NOLOCK)
         WHERE CAST(OrderID AS VARCHAR(64))=? AND ChargeID IS NOT NULL
        UNION ALL
        SELECT N'SFEPLANT_ARCHIVE', CAST(OrderID AS VARCHAR(64)), BatchNumber,
               MaterialCode, MaterialName, ChargeID, TRY_CAST(ActualWeight AS FLOAT)
          FROM SFEPLANT_ARCHIVE.dbo.BatchMaterials WITH (NOLOCK)
         WHERE CAST(OrderID AS VARCHAR(64))=? AND ChargeID IS NOT NULL
    )
    SELECT * FROM raw WHERE UPPER(LTRIM(RTRIM(CAST(ChargeID AS VARCHAR(256))))) <> 'UNKNOWN LOT'
    """
    charges = query_mms(sql, [str(order_id), str(order_id)])
    if charges.empty:
        return charges
    return deduplicate_cross_database_rows(
        charges, ["OrderID", "BatchNumber", "MaterialCode", "ChargeID"],
    )


def query_recipe_materials(recipe_ids: list[str]) -> pd.DataFrame:
    if not recipe_ids:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in recipe_ids)
    sql = f"""
    SELECT N'SFEPLANT' SourceDatabase, RecipeID, StepNumber, MaterialCode, MaterialName,
           TRY_CAST(Pphr AS FLOAT) Pphr
      FROM SFEPLANT.dbo.RecipeMaterials WITH (NOLOCK) WHERE RecipeID IN ({placeholders})
    UNION ALL
    SELECT N'SFEPLANT_ARCHIVE', RecipeID, StepNumber, MaterialCode, MaterialName,
           TRY_CAST(Pphr AS FLOAT)
      FROM SFEPLANT_ARCHIVE.dbo.RecipeMaterials WITH (NOLOCK) WHERE RecipeID IN ({placeholders})
    """
    return query_mms(sql, [*recipe_ids, *recipe_ids])


def query_supplier_qm(charges: pd.DataFrame) -> pd.DataFrame:
    if charges.empty:
        return pd.DataFrame()
    ids = charges["ChargeID"].astype(str).str.strip().str.upper().drop_duplicates().tolist()
    c = read_credentials()["mustangmaster"]
    connection = psycopg2.connect(
        host=c["host"], port=c["port"], database=c["database"],
        user=c["user"], password=c["password"], sslmode="require",
    )
    sql = """
    SELECT batch, material, "short text" AS parameter, uom,
           AVG(CAST("mean value / s" AS DOUBLE PRECISION)) AS mean_value
      FROM mustangmaster.mustang_global_glue_01.sap_supplier_qm
     WHERE plant='9200' AND UPPER(TRIM(batch)) = ANY(%s)
     GROUP BY batch, material, "short text", uom
    """
    try:
        return pd.read_sql_query(sql, connection, params=[ids])
    finally:
        connection.close()


def build_corrected_static_features(order_id: str) -> pd.DataFrame:
    """Return corrected composition/QM features keyed by completed batch."""
    materials = query_order_materials(order_id)
    headers = query_batch_headers(order_id)
    if materials.empty or headers.empty:
        raise ValueError(f"No material/header rows for OrderID={order_id}")
    percentages = aggregate_order_material_percentages(materials, headers)
    charges = query_batch_material_charges(order_id)
    qm = aggregate_batch_qm(charges, query_supplier_qm(charges)) if not charges.empty else pd.DataFrame()
    result = percentages.merge(qm, on=["OrderID", "BatchNumber"], how="left", validate="one_to_one")
    recipes = query_recipe_materials(headers["RecipeID"].dropna().astype(str).drop_duplicates().tolist()) if "RecipeID" in headers else pd.DataFrame()
    if not recipes.empty:
        phr = corrected_silica_phr(recipes)
        result = result.merge(
            headers[["OrderID", "BatchNumber", "RecipeID"]].drop_duplicates(),
            on=["OrderID", "BatchNumber"], how="left", validate="one_to_one",
        ).merge(phr, on="RecipeID", how="left", validate="many_to_one")
    result["MaterialContractVersion"] = "MATERIAL-TAXONOMY-HF-V2"
    return result
