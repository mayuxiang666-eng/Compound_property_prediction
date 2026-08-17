"""Checkpointed, index-friendly full-history builder for V3.7.1.

Uses numeric indexed OrderID predicates, caches every source chunk, records
progress after each query, and resumes after interruption. Source access is
read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2

from Mooney_Prediction_Pipeline.data_processing import build_corrected_phase_history_v1 as v1
from Mooney_Prediction_Pipeline.data_processing.canonical_mms_source import connect_mms, read_credentials


DEFAULT_CACHE = v1.ROOT / "reports" / "corrected_phase_history" / "source_cache_v2"
DEFAULT_PROGRESS = v1.ROOT / "reports" / "corrected_phase_history" / "source_cache_v2_progress.json"

MATERIAL_SQL = """
SELECT N'SFEPLANT' SourceDatabase, CAST(o.OrderID AS VARCHAR(64)) OrderID,
       o.CompoundDescription RecipeID, om.MaterialCode, om.MaterialName,
       TRY_CAST(om.BatchWeight AS FLOAT) MaterialWeight
  FROM SFEPLANT.dbo.Orders o WITH (NOLOCK)
  JOIN SFEPLANT.dbo.OrderMaterials om WITH (NOLOCK) ON om.OrderID=o.OrderID
 WHERE o.OrderID IN ({placeholders})
UNION ALL
SELECT N'SFEPLANT_ARCHIVE', CAST(o.OrderID AS VARCHAR(64)), o.CompoundDescription,
       om.MaterialCode, om.MaterialName, TRY_CAST(om.BatchWeight AS FLOAT)
  FROM SFEPLANT_ARCHIVE.dbo.Orders o WITH (NOLOCK)
  JOIN SFEPLANT_ARCHIVE.dbo.OrderMaterials om WITH (NOLOCK) ON om.OrderID=o.OrderID
 WHERE o.OrderID IN ({placeholders})
"""

HEADER_SQL = """
SELECT N'SFEPLANT' SourceDatabase, CAST(OrderID AS VARCHAR(64)) OrderID,
       BatchNumber, CAST(PalletID AS VARCHAR(128)) PalletID,
       TRY_CAST(BatchWeight AS FLOAT) BatchWeight, EquipmentID
  FROM SFEPLANT.dbo.BatchHeader WITH (NOLOCK) WHERE OrderID IN ({placeholders})
UNION ALL
SELECT N'SFEPLANT_ARCHIVE', CAST(OrderID AS VARCHAR(64)), BatchNumber,
       CAST(PalletID AS VARCHAR(128)), TRY_CAST(BatchWeight AS FLOAT), EquipmentID
  FROM SFEPLANT_ARCHIVE.dbo.BatchHeader WITH (NOLOCK) WHERE OrderID IN ({placeholders})
"""

CHARGE_SQL = """
SELECT N'SFEPLANT' SourceDatabase, CAST(OrderID AS VARCHAR(64)) OrderID,
       BatchNumber, MaterialCode, MaterialName, CAST(ChargeID AS VARCHAR(256)) ChargeID,
       TRY_CAST(ActualWeight AS FLOAT) ActualWeight
  FROM SFEPLANT.dbo.BatchMaterials WITH (NOLOCK)
 WHERE OrderID IN ({placeholders}) AND ChargeID IS NOT NULL
UNION ALL
SELECT N'SFEPLANT_ARCHIVE', CAST(OrderID AS VARCHAR(64)), BatchNumber,
       MaterialCode, MaterialName, CAST(ChargeID AS VARCHAR(256)), TRY_CAST(ActualWeight AS FLOAT)
  FROM SFEPLANT_ARCHIVE.dbo.BatchMaterials WITH (NOLOCK)
 WHERE OrderID IN ({placeholders}) AND ChargeID IS NOT NULL
"""

RECIPE_SQL = """
SELECT N'SFEPLANT' SourceDatabase, RecipeID, StepNumber, MaterialCode, MaterialName,
       TRY_CAST(Pphr AS FLOAT) Pphr
  FROM SFEPLANT.dbo.RecipeMaterials WITH (NOLOCK) WHERE RecipeID IN ({placeholders})
UNION ALL
SELECT N'SFEPLANT_ARCHIVE', RecipeID, StepNumber, MaterialCode, MaterialName,
       TRY_CAST(Pphr AS FLOAT)
  FROM SFEPLANT_ARCHIVE.dbo.RecipeMaterials WITH (NOLOCK) WHERE RecipeID IN ({placeholders})
"""

QM_SQL = """
SELECT batch, material, "short text" AS parameter, uom,
       AVG(CAST("mean value / s" AS DOUBLE PRECISION)) AS mean_value
  FROM mustangmaster.mustang_global_glue_01.sap_supplier_qm
 WHERE plant='9200' AND UPPER(TRIM(batch)) = ANY(%s)
 GROUP BY batch, material, "short text", uom
"""


class ProgressLog:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.events = json.loads(path.read_text(encoding="utf-8")).get("events", []) if path.is_file() else []
        except (json.JSONDecodeError, OSError):
            self.events = []

    def add(self, **event: object) -> None:
        self.events.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event})
        payload = {
            "contract": "CORRECTED-HISTORY-SOURCE-CACHE-V2",
            "credentials_written_to_output": False,
            "automatic_control": "DISABLED",
            "events": self.events,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def _chunk_key(values: list[object]) -> str:
    return hashlib.sha256("\n".join(str(v) for v in values).encode()).hexdigest()[:16]


def _numeric_orders(order_ids: list[str]) -> tuple[list[int], list[str]]:
    numeric, rejected = [], []
    for value in order_ids:
        text = str(value).strip()
        (numeric if text.isdigit() else rejected).append(int(text) if text.isdigit() else text)
    return sorted(set(numeric)), sorted(set(rejected))


def _read_mms(sql: str, params: list[object], timeout_seconds: int) -> pd.DataFrame:
    with connect_mms() as connection:
        connection.timeout = int(timeout_seconds)
        return pd.read_sql_query(sql, connection, params=params)


def _cached_query(source: str, values: list[object], sql_template: str, cache_dir: Path,
                  progress: ProgressLog, chunk_size: int, timeout_seconds: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    source_dir = cache_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, len(values), chunk_size)):
        chunk = values[start:start + chunk_size]
        key = _chunk_key(chunk)
        cache_path = source_dir / f"{index:04d}_{key}.parquet"
        began = time.perf_counter()
        if cache_path.is_file():
            frame, cache_hit = pd.read_parquet(cache_path), True
        else:
            placeholders = ",".join("?" for _ in chunk)
            try:
                frame = _read_mms(sql_template.format(placeholders=placeholders), [*chunk, *chunk], timeout_seconds)
                frame.to_parquet(cache_path, index=False)
            except Exception as exc:
                progress.add(source=source, chunk_index=index, chunk_key=key, input_count=len(chunk),
                             status="FAILED", error_type=type(exc).__name__,
                             seconds=round(time.perf_counter() - began, 3))
                raise
            cache_hit = False
        progress.add(source=source, chunk_index=index, chunk_key=key, input_count=len(chunk),
                     output_rows=int(len(frame)), cache_hit=cache_hit, status="COMPLETE",
                     seconds=round(time.perf_counter() - began, 3))
        print(json.dumps(progress.events[-1], ensure_ascii=False), flush=True)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _cached_qm_query(charges: pd.DataFrame, cache_dir: Path, progress: ProgressLog,
                     chunk_size: int, timeout_seconds: int) -> pd.DataFrame:
    if charges.empty:
        return pd.DataFrame()
    values = sorted(charges["ChargeID"].astype(str).str.strip().str.upper().drop_duplicates().tolist())
    source_dir = cache_dir / "supplier_qm"
    source_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    credentials = read_credentials()["mustangmaster"]
    for index, start in enumerate(range(0, len(values), chunk_size)):
        chunk = values[start:start + chunk_size]
        key = _chunk_key(chunk)
        cache_path = source_dir / f"{index:04d}_{key}.parquet"
        began = time.perf_counter()
        if cache_path.is_file():
            frame, cache_hit = pd.read_parquet(cache_path), True
        else:
            connection = psycopg2.connect(
                host=credentials["host"], port=credentials["port"], database=credentials["database"],
                user=credentials["user"], password=credentials["password"], sslmode="require",
                connect_timeout=min(int(timeout_seconds), 60),
                options=f"-c statement_timeout={int(timeout_seconds) * 1000}",
            )
            try:
                frame = pd.read_sql_query(QM_SQL, connection, params=[chunk])
                frame.to_parquet(cache_path, index=False)
            except Exception as exc:
                progress.add(source="supplier_qm", chunk_index=index, chunk_key=key,
                             input_count=len(chunk), status="FAILED", error_type=type(exc).__name__,
                             seconds=round(time.perf_counter() - began, 3))
                raise
            finally:
                connection.close()
            cache_hit = False
        progress.add(source="supplier_qm", chunk_index=index, chunk_key=key, input_count=len(chunk),
                     output_rows=int(len(frame)), cache_hit=cache_hit, status="COMPLETE",
                     seconds=round(time.perf_counter() - began, 3))
        print(json.dumps(progress.events[-1], ensure_ascii=False), flush=True)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()


def run(args: argparse.Namespace) -> dict[str, object]:
    source = pd.read_csv(args.input, usecols=["OrderID"], low_memory=False)
    all_orders = sorted(source["OrderID"].astype(str).str.strip().drop_duplicates().tolist())
    numeric, rejected = _numeric_orders(all_orders)
    if args.max_orders:
        numeric = numeric[:args.max_orders]
        selected = {str(value) for value in numeric}
        subset = pd.read_csv(args.input, low_memory=False)
        subset = subset.loc[subset["OrderID"].astype(str).str.strip().isin(selected)]
        input_path = args.cache_dir / f"pilot_input_{len(numeric)}_orders.csv"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(input_path, index=False, encoding="utf-8-sig")
    else:
        input_path = args.input

    progress = ProgressLog(args.progress)
    progress.add(source="order_scope", status="STARTED", numeric_order_count=len(numeric),
                 non_numeric_order_count=len(rejected), non_numeric_order_ids=rejected)

    v1.query_order_materials_bulk = lambda order_ids, _: _cached_query(
        "order_materials", _numeric_orders(order_ids)[0], MATERIAL_SQL, args.cache_dir,
        progress, args.chunk_size, args.query_timeout)
    v1.query_batch_headers_bulk = lambda order_ids, _: _cached_query(
        "batch_headers", _numeric_orders(order_ids)[0], HEADER_SQL, args.cache_dir,
        progress, args.chunk_size, args.query_timeout)
    v1.query_batch_charges_bulk = lambda order_ids, _: _cached_query(
        "batch_charges", _numeric_orders(order_ids)[0], CHARGE_SQL, args.cache_dir,
        progress, args.chunk_size, args.query_timeout)
    v1.query_recipe_materials = lambda recipe_ids: _cached_query(
        "recipe_materials", sorted(recipe_ids), RECIPE_SQL, args.cache_dir,
        progress, args.chunk_size, args.query_timeout)
    v1.query_qm_bulk = lambda frame, _chunk_size=0: _cached_qm_query(
        frame, args.cache_dir, progress, args.qm_chunk_size, args.query_timeout)

    audit = v1.build(input_path, args.output, args.audit, args.chunk_size)
    audit.update({
        "source_cache_contract": "CORRECTED-HISTORY-SOURCE-CACHE-V2",
        "source_cache_dir": str(args.cache_dir), "progress_audit": str(args.progress),
        "numeric_order_count": len(numeric), "non_numeric_order_count": len(rejected),
        "query_timeout_seconds": args.query_timeout,
    })
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    progress.add(source="build", status="COMPLETE", output_rows=audit["output_rows"])
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=v1.DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=v1.DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=v1.DEFAULT_AUDIT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--qm-chunk-size", type=int, default=500)
    parser.add_argument("--query-timeout", type=int, default=120)
    parser.add_argument("--max-orders", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
