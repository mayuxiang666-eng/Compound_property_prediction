"""Read-only coverage audit for an active-SFEPLANT-only corrected baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.data_processing import build_corrected_phase_history_v2 as cache
from Mooney_Prediction_Pipeline.data_processing.build_corrected_phase_history_v7 import _candidate_map


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_RESOLUTION_SQL = """
SELECT N'SFEPLANT' SourceDatabase, CAST(OrderID AS VARCHAR(64)) ResolvedOrderID
  FROM SFEPLANT.dbo.Orders WITH (NOLOCK) WHERE OrderID IN ({placeholders})
UNION ALL
SELECT N'SFEPLANT' SourceDatabase, CAST(OrderID AS VARCHAR(64)) ResolvedOrderID
  FROM SFEPLANT.dbo.Orders WITH (NOLOCK) WHERE 1=0 AND OrderID IN ({placeholders})
"""


def run(args: argparse.Namespace) -> dict[str, object]:
    frame = pd.read_csv(args.input, low_memory=False)
    frame["OrderID"] = frame["OrderID"].astype(str).str.strip()
    candidates, candidate_frame = _candidate_map(frame)
    ids = sorted(candidate_frame["CandidateOrderID"].drop_duplicates().tolist())
    progress = cache.ProgressLog(args.progress)
    resolved = cache._cached_query(
        "active_order_resolution", ids, ACTIVE_RESOLUTION_SQL, args.cache_dir,
        progress, args.chunk_size, args.query_timeout,
    )
    existing = set(resolved.get("ResolvedOrderID", pd.Series(dtype=str)).astype(str))
    mapping = {}
    for original, options in candidates.items():
        mapping[original] = next((candidate for candidate in options if candidate in existing), None)
    matched_originals = {original for original, value in mapping.items() if value is not None}
    active = frame.loc[frame["OrderID"].isin(matched_originals)].copy()
    active["ResolvedOrderID"] = active["OrderID"].map(mapping)
    active["OrderStartTime"] = pd.to_datetime(active["OrderStartTime"], errors="coerce")

    compound = active["CompoundName"].astype(str).str.upper()
    b00458 = active.loc[compound.str.contains("B00458", regex=False)]
    t09170 = active.loc[compound.str.contains("T09170", regex=False)]

    def scope(value: pd.DataFrame) -> dict[str, object]:
        return {
            "batches": int(len(value)),
            "orders": int(value["OrderID"].nunique()),
            "label_groups": int(value.groupby(["OrderID", "PalletID"]).ngroups) if len(value) else 0,
            "date_min": str(value["OrderStartTime"].min()) if len(value) else None,
            "date_max": str(value["OrderStartTime"].max()) if len(value) else None,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    active.to_csv(args.output, index=False, encoding="utf-8-sig")
    mapping_path = args.output.with_name("active_order_id_mapping_v1.csv")
    pd.DataFrame([
        {"OriginalOrderID": key, "ResolvedOrderID": value, "ActiveMatch": value is not None}
        for key, value in mapping.items()
    ]).to_csv(mapping_path, index=False, encoding="utf-8-sig")
    audit = {
        "contract": "ACTIVE-SFEPLANT-COVERAGE-V1",
        "full_history": scope(frame), "active_matched": scope(active),
        "b00458_active": scope(b00458), "t09170_active": scope(t09170),
        "active_order_match_rate": float(len(matched_originals) / max(len(candidates), 1)),
        "active_batch_coverage_rate": float(len(active) / max(len(frame), 1)),
        "active_scope_output": str(args.output), "mapping_output": str(mapping_path),
        "credentials_written_to_output": False, "automatic_control": "DISABLED",
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "reports" / "corrected_phase_history" / "phase_curve_timesafe_v1.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "corrected_phase_history" / "phase_curve_timesafe_active_scope_v1.csv")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports" / "corrected_phase_history" / "active_order_coverage_v1.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "reports" / "corrected_phase_history" / "source_cache_active_v1")
    parser.add_argument("--progress", type=Path, default=ROOT / "reports" / "corrected_phase_history" / "source_cache_active_v1_progress.json")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--query-timeout", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
