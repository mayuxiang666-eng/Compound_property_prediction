"""Resolve legacy OrderID aliases before running the cached V3.7.1 builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.data_processing import build_corrected_phase_history_v2 as v2
from Mooney_Prediction_Pipeline.data_processing import build_corrected_phase_history_v6 as v6  # noqa: F401
from Mooney_Prediction_Pipeline.data_processing.canonical_material_contract import (
    deduplicate_cross_database_rows,
)


RESOLUTION_SQL = """
SELECT N'SFEPLANT' SourceDatabase, CAST(OrderID AS VARCHAR(64)) ResolvedOrderID
  FROM SFEPLANT.dbo.Orders WITH (NOLOCK) WHERE OrderID IN ({placeholders})
UNION ALL
SELECT N'SFEPLANT_ARCHIVE', CAST(OrderID AS VARCHAR(64))
  FROM SFEPLANT_ARCHIVE.dbo.Orders WITH (NOLOCK) WHERE OrderID IN ({placeholders})
"""


def _candidate_map(frame: pd.DataFrame) -> tuple[dict[str, list[str]], pd.DataFrame]:
    work = frame[["OrderID", "OrderStartTime"]].copy()
    work["OrderID"] = work["OrderID"].astype(str).str.strip()
    work["year"] = pd.to_datetime(work["OrderStartTime"], errors="coerce").dt.year.astype("Int64")
    unique = work.drop_duplicates("OrderID")
    candidates: dict[str, list[str]] = {}
    rows: list[dict[str, object]] = []
    for row in unique.itertuples(index=False):
        original = str(row.OrderID)
        values = [original]
        if "~" not in original and pd.notna(row.year):
            values.append(f"{int(row.year)}~{original}")
        values = list(dict.fromkeys(values))
        candidates[original] = values
        for rank, value in enumerate(values):
            rows.append({"OriginalOrderID": original, "CandidateOrderID": value, "CandidateRank": rank})
    return candidates, pd.DataFrame(rows)


def resolve_input(args: argparse.Namespace, progress: v2.ProgressLog) -> tuple[Path, dict[str, object]]:
    full = pd.read_csv(args.input, low_memory=False)
    full["OrderID"] = full["OrderID"].astype(str).str.strip()
    if args.max_orders:
        selected = set(sorted(full["OrderID"].drop_duplicates().tolist())[:args.max_orders])
        full = full.loc[full["OrderID"].isin(selected)].copy()
    candidates, candidate_frame = _candidate_map(full)
    candidate_ids = sorted(candidate_frame["CandidateOrderID"].drop_duplicates().tolist())
    resolved_raw = v2._cached_query(
        "order_resolution", candidate_ids, RESOLUTION_SQL, args.cache_dir,
        progress, args.chunk_size, args.query_timeout,
    )
    if not resolved_raw.empty:
        resolved_raw = deduplicate_cross_database_rows(
            resolved_raw, ["ResolvedOrderID"], source_column="SourceDatabase",
        )
    existing = set(resolved_raw.get("ResolvedOrderID", pd.Series(dtype=str)).astype(str))
    resolution: dict[str, str] = {}
    unresolved: list[str] = []
    prefixed_matches = 0
    exact_matches = 0
    for original, options in candidates.items():
        match = next((candidate for candidate in options if candidate in existing), None)
        if match is None:
            resolution[original] = original
            unresolved.append(original)
        else:
            resolution[original] = match
            exact_matches += int(match == original)
            prefixed_matches += int(match != original)

    full["OriginalOrderID"] = full["OrderID"]
    full["OrderID"] = full["OrderID"].map(resolution)
    if full.duplicated(["OrderID", "BatchNumber"]).any():
        raise ValueError("OrderID resolution created duplicate OrderID+BatchNumber keys")
    resolved_path = args.cache_dir / f"resolved_input_{full['OrderID'].nunique()}_orders.csv"
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(resolved_path, index=False, encoding="utf-8-sig")
    audit = {
        "contract": "MMS-ORDERID-RESOLUTION-V1",
        "input_orders": int(len(candidates)), "resolved_exact": int(exact_matches),
        "resolved_year_prefixed": int(prefixed_matches), "unresolved_count": int(len(unresolved)),
        "unresolved_order_ids": unresolved, "resolved_input": str(resolved_path),
        "credentials_written_to_output": False, "automatic_control": "DISABLED",
    }
    resolution_audit = args.cache_dir.parent / "order_id_resolution_v7.json"
    resolution_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    progress.add(source="order_resolution", status="RESOLVED", **{
        key: value for key, value in audit.items() if key not in {"unresolved_order_ids"}
    })
    return resolved_path, audit


def run(args: argparse.Namespace) -> dict[str, object]:
    progress = v2.ProgressLog(args.progress)
    resolved_input, resolution_audit = resolve_input(args, progress)
    args.input = resolved_input
    args.max_orders = None
    result = v2.run(args)
    result["order_id_resolution"] = resolution_audit
    args.audit.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=v2.v1.DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=v2.v1.DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=v2.v1.DEFAULT_AUDIT)
    parser.add_argument("--cache-dir", type=Path, default=v2.DEFAULT_CACHE)
    parser.add_argument("--progress", type=Path, default=v2.DEFAULT_PROGRESS)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--qm-chunk-size", type=int, default=500)
    parser.add_argument("--query-timeout", type=int, default=120)
    parser.add_argument("--max-orders", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
