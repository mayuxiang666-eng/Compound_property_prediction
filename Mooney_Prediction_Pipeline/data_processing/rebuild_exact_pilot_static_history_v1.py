"""Rebuild historical material/QM features for the two exact pilot recipes.

All MMS reads are index-seeking and read-only.  Active rows win exact
cross-database duplicates.  Supplier QM is linked only through each batch's
actual ChargeID; no current-period median is copied into historical batches.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.data_processing.build_corrected_phase_history_v1 import (
    MATERIAL_COLUMNS,
    QM_COLUMNS,
)
from Mooney_Prediction_Pipeline.data_processing.canonical_material_contract import (
    aggregate_batch_qm,
    aggregate_order_material_percentages,
    corrected_silica_phr,
    deduplicate_cross_database_rows,
)
from Mooney_Prediction_Pipeline.data_processing.canonical_mms_source import query_supplier_qm
from Mooney_Prediction_Pipeline.feature_engineering.corrected_phase_contract import (
    apply_corrected_phase_contract,
)
from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history"
SOURCE = BASE / "phase_curve_timesafe_physical_v3.csv"
CROSSDB = BASE / "crossdb_exact_pilot_step_history_v1"
RESOLUTION = CROSSDB / "order_resolution.csv"
OUTPUT = BASE / "phase_exact_pilots_corrected_static_v1.csv"
AUDIT = BASE / "phase_exact_pilots_corrected_static_v1_audit.json"
CACHE = BASE / "exact_pilot_static_cache_v1"
RECIPES = {"9922A00458BT_PV77_004", "9922A09170AX_PV50_004"}
DATABASES = ("SFEPLANT", "SFEPLANT_ARCHIVE")


def _cast_values(count: int, fields: int) -> str:
    if fields == 1:
        return ",".join("(CAST(? AS VARCHAR(20)))" for _ in range(count))
    return ",".join(
        "(CAST(? AS VARCHAR(20)),CAST(? AS INT))" for _ in range(count)
    )


def _read_or_query(path: Path, connection, query: str, params: list[object]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    frame = pd.read_sql_query(query, connection, params=params)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def _source_scope(source_path: Path) -> pd.DataFrame:
    phase = pd.read_csv(source_path, low_memory=False)
    phase["OrderID"] = phase["OrderID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    phase["BatchNumber"] = pd.to_numeric(phase["BatchNumber"], errors="raise").astype(int)
    selected = phase.loc[phase["CompoundDescription"].astype(str).isin(RECIPES)].copy()
    if selected.duplicated(["OrderID", "BatchNumber"]).any():
        raise ValueError("Exact pilot Phase scope is not unique by OrderID+BatchNumber")
    return selected


def _resolution(path: Path, scope: pd.DataFrame) -> pd.DataFrame:
    resolved = pd.read_csv(path, dtype=str)
    resolved["OriginalOrderID"] = resolved["OriginalOrderID"].astype(str).str.strip()
    resolved["ResolvedOrderID"] = resolved["ResolvedOrderID"].astype(str).str.strip()
    wanted = set(scope["OrderID"].unique())
    return resolved.loc[resolved["OriginalOrderID"].isin(wanted)].copy()


def _query_order_materials(
    connection, resolved: pd.DataFrame, cache_dir: Path, chunk_size: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for database in DATABASES:
        mapping = resolved.loc[resolved["SourceDatabase"].eq(database), [
            "OriginalOrderID", "ResolvedOrderID",
        ]].drop_duplicates()
        ids = sorted(mapping["ResolvedOrderID"].unique())
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start:start + chunk_size]
            values = _cast_values(len(chunk), 1)
            query = f"""
                WITH ranked AS (
                    SELECT CAST(o.OrderID AS VARCHAR(64)) ResolvedOrderID,
                           o.CompoundDescription RecipeID,
                           om.MaterialCode, om.MaterialName,
                           TRY_CAST(om.BatchWeight AS FLOAT) MaterialWeight,
                           ROW_NUMBER() OVER (
                               PARTITION BY om.OrderID, om.SequenceNumber, om.MaterialCode, om.Zone
                               ORDER BY om.[Timestamp] DESC
                           ) AS rn
                    FROM (VALUES {values}) wanted(OrderID)
                    INNER LOOP JOIN {database}.dbo.Orders o WITH (NOLOCK, FORCESEEK)
                      ON o.OrderID=wanted.OrderID
                    INNER LOOP JOIN {database}.dbo.OrderMaterials om WITH (NOLOCK, FORCESEEK)
                      ON om.OrderID=o.OrderID
                )
                SELECT ResolvedOrderID, RecipeID, MaterialCode, MaterialName, MaterialWeight
                FROM ranked WHERE rn=1 OPTION (FORCE ORDER)
            """
            frame = _read_or_query(
                cache_dir / f"{database.lower()}_{start // chunk_size:04d}.csv",
                connection, query, list(chunk),
            )
            frame["SourceDatabase"] = database
            frame["ResolvedOrderID"] = frame["ResolvedOrderID"].astype(str).str.strip()
            frame = frame.merge(mapping, on="ResolvedOrderID", how="inner", validate="many_to_one")
            frame["OrderID"] = frame["OriginalOrderID"]
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    return deduplicate_cross_database_rows(
        raw, ["OrderID", "MaterialCode", "MaterialName", "MaterialWeight"],
        source_column="SourceDatabase",
    )


def _exact_pairs(scope: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    return scope[["OrderID", "BatchNumber"]].merge(
        mapping[["OriginalOrderID", "ResolvedOrderID"]],
        left_on="OrderID", right_on="OriginalOrderID", how="inner", validate="many_to_one",
    )[["OrderID", "ResolvedOrderID", "BatchNumber"]].sort_values(
        ["ResolvedOrderID", "BatchNumber"], kind="stable"
    )


def _query_exact_table(
    connection, resolved: pd.DataFrame, scope: pd.DataFrame, cache_dir: Path,
    *, table: str, select_sql: str, partition_sql: str, chunk_size: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for database in DATABASES:
        mapping = resolved.loc[resolved["SourceDatabase"].eq(database)]
        pairs = _exact_pairs(scope, mapping)
        for start in range(0, len(pairs), chunk_size):
            chunk = pairs.iloc[start:start + chunk_size].copy()
            values = _cast_values(len(chunk), 2)
            query = f"""
                WITH ranked AS (
                    SELECT CAST(src.OrderID AS VARCHAR(64)) ResolvedOrderID,
                           src.BatchNumber, {select_sql},
                           ROW_NUMBER() OVER (
                               PARTITION BY {partition_sql}
                               ORDER BY src.[Timestamp] DESC
                           ) rn
                    FROM (VALUES {values}) wanted(OrderID, BatchNumber)
                    INNER LOOP JOIN {database}.dbo.{table} src WITH (NOLOCK, FORCESEEK)
                      ON src.OrderID=wanted.OrderID AND src.BatchNumber=wanted.BatchNumber
                )
                SELECT * FROM ranked WHERE rn=1 OPTION (FORCE ORDER)
            """
            params: list[object] = []
            for row in chunk.itertuples(index=False):
                params.extend([str(row.ResolvedOrderID), int(row.BatchNumber)])
            frame = _read_or_query(
                cache_dir / f"{database.lower()}_{start // chunk_size:04d}.csv",
                connection, query, params,
            )
            if frame.empty:
                continue
            frame["SourceDatabase"] = database
            frame["ResolvedOrderID"] = frame["ResolvedOrderID"].astype(str).str.strip()
            frame["BatchNumber"] = pd.to_numeric(frame["BatchNumber"], errors="raise").astype(int)
            frame = frame.merge(
                chunk, on=["ResolvedOrderID", "BatchNumber"], how="inner", validate="many_to_one"
            )
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _query_recipe_materials(connection, cache_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    ids = sorted(RECIPES)
    placeholders = ",".join("CAST(? AS VARCHAR(64))" for _ in ids)
    for database in DATABASES:
        query = f"""
            SELECT RecipeID, StepNumber, MaterialCode, MaterialName, TRY_CAST(Pphr AS FLOAT) Pphr
            FROM {database}.dbo.RecipeMaterials WITH (NOLOCK, FORCESEEK)
            WHERE RecipeID IN ({placeholders})
        """
        frame = _read_or_query(
            cache_path.with_name(f"{cache_path.stem}_{database.lower()}.csv"),
            connection, query, ids,
        )
        frame["SourceDatabase"] = database
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    return deduplicate_cross_database_rows(
        raw, ["RecipeID", "StepNumber", "MaterialCode", "MaterialName", "Pphr"],
        source_column="SourceDatabase",
    )


def _query_qm_cached(charges: pd.DataFrame, cache_dir: Path, chunk_size: int) -> pd.DataFrame:
    ids = sorted(charges["ChargeID"].astype(str).str.strip().str.upper().drop_duplicates())
    frames: list[pd.DataFrame] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start:start + chunk_size]
        path = cache_dir / f"qm_{start // chunk_size:04d}.csv"
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
        else:
            frame = query_supplier_qm(pd.DataFrame({"ChargeID": chunk}))
            frame.to_csv(path, index=False, encoding="utf-8-sig")
        frames.append(frame)
        print(json.dumps({
            "event": "qm_chunk", "chunk": start // chunk_size,
            "charge_ids": len(chunk), "rows": len(frame), "cache": str(path),
        }), flush=True)
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()


def run(args: argparse.Namespace) -> dict[str, object]:
    scope = _source_scope(args.source)
    resolved = _resolution(args.resolution, scope)
    connection = get_sfe_connection()
    try:
        materials = _query_order_materials(
            connection, resolved, args.cache_dir / "order_materials", args.order_chunk_size
        )
        headers_raw = _query_exact_table(
            connection, resolved, scope, args.cache_dir / "batch_headers",
            table="BatchHeader",
            select_sql="CAST(src.PalletID AS VARCHAR(128)) PalletID, TRY_CAST(src.BatchWeight AS FLOAT) BatchWeight, src.EquipmentID",
            partition_sql="src.OrderID,src.BatchNumber,src.EquipmentID",
            chunk_size=args.pair_chunk_size,
        )
        charges_raw = _query_exact_table(
            connection, resolved, scope, args.cache_dir / "batch_materials",
            table="BatchMaterials",
            select_sql="src.MaterialCode,src.MaterialName,CAST(src.ChargeID AS VARCHAR(256)) ChargeID,TRY_CAST(src.ActualWeight AS FLOAT) ActualWeight",
            partition_sql="src.OrderID,src.BatchNumber,src.EquipmentID,src.SequenceNumber,src.MaterialCode,src.ChargeID",
            chunk_size=args.pair_chunk_size,
        )
        recipe_materials = _query_recipe_materials(
            connection, args.cache_dir / "recipe_materials.csv"
        )
    finally:
        connection.close()

    headers = deduplicate_cross_database_rows(
        headers_raw, ["OrderID", "BatchNumber"], source_column="SourceDatabase"
    )
    charges_raw = charges_raw.loc[
        charges_raw["ChargeID"].notna()
        & charges_raw["ChargeID"].astype(str).str.strip().str.upper().ne("UNKNOWN LOT")
    ].copy()
    charges = deduplicate_cross_database_rows(
        charges_raw, ["OrderID", "BatchNumber", "MaterialCode", "ChargeID"],
        source_column="SourceDatabase",
    )
    composition = aggregate_order_material_percentages(materials, headers)
    qm_raw = _query_qm_cached(charges, args.cache_dir / "supplier_qm", args.qm_chunk_size)
    qm = aggregate_batch_qm(charges, qm_raw) if not qm_raw.empty else pd.DataFrame()
    phr = corrected_silica_phr(recipe_materials)
    recipe_map = scope[["OrderID", "CompoundDescription"]].drop_duplicates("OrderID").rename(
        columns={"CompoundDescription": "RecipeID"}
    )
    static = composition.merge(qm, on=["OrderID", "BatchNumber"], how="left", validate="one_to_one")
    static = static.merge(recipe_map, on="OrderID", how="left", validate="many_to_one")
    static = static.merge(phr, on="RecipeID", how="left", validate="many_to_one")

    replace = [*MATERIAL_COLUMNS, *QM_COLUMNS, "silica_phr"]
    corrected = scope.drop(columns=replace, errors="ignore").merge(
        static[["OrderID", "BatchNumber", *replace]],
        on=["OrderID", "BatchNumber"], how="left", validate="one_to_one",
    )
    silica_present = pd.to_numeric(corrected["weight_pct_silica"], errors="coerce").fillna(0).gt(0)
    corrected.loc[~silica_present, "silica_phr"] = 0.0
    corrected = apply_corrected_phase_contract(corrected)
    corrected["MaterialContractVersion"] = "MATERIAL-TAXONOMY-HF-V2"
    corrected["CurveContractVersion"] = "PHASE-CURVE-TIMESAFE-PHYSICAL-V3"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(args.output, index=False, encoding="utf-8-sig")
    materials.to_csv(args.cache_dir / "order_materials_effective.csv", index=False, encoding="utf-8-sig")
    headers.to_csv(args.cache_dir / "batch_headers_effective.csv", index=False, encoding="utf-8-sig")
    charges.to_csv(args.cache_dir / "batch_materials_effective.csv", index=False, encoding="utf-8-sig")
    recipe_materials.to_csv(args.cache_dir / "recipe_materials_effective.csv", index=False, encoding="utf-8-sig")
    audit = {
        "contract": "EXACT-PILOT-CORRECTED-STATIC-HISTORY-V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source), "output": str(args.output),
        "rows": int(len(corrected)),
        "recipes": {str(k): int(v) for k, v in corrected.groupby("CompoundDescription").size().to_dict().items()},
        "order_material_rows": int(len(materials)), "batch_headers": int(len(headers)),
        "batch_charge_rows": int(len(charges)), "unique_charge_ids": int(charges["ChargeID"].nunique()),
        "supplier_qm_rows": int(len(qm_raw)),
        "missing_by_static_column": {column: int(corrected[column].isna().sum()) for column in replace},
        "active_archive_priority": "SFEPLANT_OVER_SFEPLANT_ARCHIVE_ON_EXACT_KEY",
        "current_values_copied_to_history": False,
        "cross_batch_or_future_fill": False,
        "database_writes": False, "automatic_control": "DISABLED",
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--resolution", type=Path, default=RESOLUTION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--order-chunk-size", type=int, default=100)
    parser.add_argument("--pair-chunk-size", type=int, default=100)
    parser.add_argument("--qm-chunk-size", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
