"""Exact-pair query adapter for the cross-database pilot extractor.

This keeps the V1 audit contract while replacing broad order-level BatchData
reads with read-only OrderID+BatchNumber VALUES joins.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.optimization import (
    extract_crossdb_exact_pilot_step_history_v1 as base,
)


def _query_exact_pairs(connection, database: str, pairs: list[tuple[str, int]]) -> pd.DataFrame:
    values_clause = ",".join("(?,?)" for _ in pairs)
    params: list[object] = []
    for order_id, batch_number in pairs:
        params.extend([order_id, int(batch_number)])
    query = f"""
        SELECT CAST(bd.OrderID AS VARCHAR(64)) AS ResolvedOrderID,
               bd.BatchNumber, bd.EquipmentID, bd.GroupName, bd.StepNo,
               MAX(CASE WHEN bd.VariablePath='SCP-1-Step-Time-rel-s' THEN bd.Target END) AS time_SP,
               MAX(CASE WHEN bd.VariablePath='SCP-1-Step-Time-rel-s' THEN bd.Value END) AS time_AV,
               MAX(CASE WHEN bd.VariablePath='SCP-3-Temperature-C-F' THEN bd.Target END) AS temp_SP,
               MAX(CASE WHEN bd.VariablePath='SCP-3-Temperature-C-F' THEN bd.Value END) AS temp_AV,
               MAX(CASE WHEN bd.VariablePath='Rotor-Speed-1/min-Rpm' THEN bd.Target END) AS rotor_SP,
               MAX(CASE WHEN bd.VariablePath='Rotor-Speed-1/min-Rpm' THEN bd.Value END) AS rotor_AV,
               MAX(CASE WHEN bd.VariablePath='RAM-Pressure-percent' THEN bd.Target END) AS ram_SP,
               MAX(CASE WHEN bd.VariablePath='RAM-Pressure-percent' THEN bd.Value END) AS ram_AV,
               MAX(CASE WHEN bd.VariablePath='SCP-2-Energy-rel-kWh' THEN bd.Value END) AS energy_AV,
               MAX(bd.Timestamp) AS ts
        FROM {database}.dbo.BatchData bd WITH (NOLOCK)
        INNER JOIN (VALUES {values_clause}) wanted(OrderID, BatchNumber)
          ON wanted.OrderID = bd.OrderID AND wanted.BatchNumber = bd.BatchNumber
        WHERE bd.GroupName IN ('AVR_MST','AVR_MSB')
          AND bd.VariablePath IN (
              'SCP-1-Step-Time-rel-s','SCP-3-Temperature-C-F',
              'Rotor-Speed-1/min-Rpm','RAM-Pressure-percent','SCP-2-Energy-rel-kWh'
          )
        GROUP BY bd.OrderID, bd.BatchNumber, bd.EquipmentID, bd.GroupName, bd.StepNo
    """
    return pd.read_sql_query(query, connection, params=params)


def fetch_steps_exact(
    connection,
    resolved: pd.DataFrame,
    scope: pd.DataFrame,
    cache_dir: Path,
    *,
    chunk_size: int,
    refresh: bool,
) -> pd.DataFrame:
    cache_dir = cache_dir.with_name("steps_exact_pairs")
    cache_dir.mkdir(parents=True, exist_ok=True)
    exact_pairs = scope[["OrderID", "BatchNumber"]].drop_duplicates()
    frames: list[pd.DataFrame] = []
    for database in base.DATABASES:
        mapping = resolved.loc[resolved["SourceDatabase"].eq(database), [
            "OriginalOrderID", "ResolvedOrderID",
        ]].drop_duplicates()
        wanted = exact_pairs.merge(
            mapping, left_on="OrderID", right_on="OriginalOrderID",
            how="inner", validate="many_to_one",
        )[["OrderID", "ResolvedOrderID", "BatchNumber"]]
        wanted = wanted.sort_values(["ResolvedOrderID", "BatchNumber"], kind="stable")
        for start in range(0, len(wanted), chunk_size):
            chunk = wanted.iloc[start:start + chunk_size].copy()
            path = cache_dir / f"{database.lower()}_exact_{start // chunk_size:04d}.csv"
            if path.exists() and not refresh:
                fetched = pd.read_csv(path, low_memory=False)
            else:
                query_pairs = [
                    (str(row.ResolvedOrderID), int(row.BatchNumber))
                    for row in chunk.itertuples(index=False)
                ]
                fetched = _query_exact_pairs(connection, database, query_pairs)
                fetched["SourceDatabase"] = database
                fetched.to_csv(path, index=False, encoding="utf-8-sig")
            if fetched.empty:
                print(json.dumps({
                    "event": "exact_step_chunk", "database": database,
                    "chunk": start // chunk_size, "pairs": len(chunk), "rows": 0,
                }), flush=True)
                continue
            fetched["ResolvedOrderID"] = base._normalise_order_id(fetched["ResolvedOrderID"])
            fetched["BatchNumber"] = pd.to_numeric(fetched["BatchNumber"], errors="raise").astype(int)
            fetched["SourceDatabase"] = database
            fetched = fetched.merge(
                chunk, on=["ResolvedOrderID", "BatchNumber"],
                how="inner", validate="many_to_one",
            )
            frames.append(fetched)
            print(json.dumps({
                "event": "exact_step_chunk", "database": database,
                "chunk": start // chunk_size, "pairs": len(chunk),
                "rows": int(len(fetched)), "cache": str(path),
            }), flush=True)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    for column in base.NUMERIC_STEP_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def main() -> None:
    base.fetch_steps = fetch_steps_exact
    base.main()


if __name__ == "__main__":
    main()
