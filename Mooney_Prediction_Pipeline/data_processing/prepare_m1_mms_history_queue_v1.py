"""Create a bounded M1 order queue from an existing local snapshot.

This script does not access MMS.  It only prepares exact OrderIDs for the
guarded incremental extractor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1" / "v371_r1b_m1_only_snapshot_v1" / "m1_only_batch_pid_quality_features.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "mms_incremental_history" / "m1_order_queue_v1.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-orders", type=int)
    args = parser.parse_args()
    frame = pd.read_csv(args.source, usecols=["OrderID", "CompoundName", "OrderTimestamp"], low_memory=False)
    frame["OrderID"] = frame["OrderID"].astype(str).str.strip()
    frame["CompoundName"] = frame["CompoundName"].astype(str).str.strip()
    frame = frame.loc[frame["CompoundName"].str.startswith("M1-")].copy()
    frame["OrderTimestamp"] = pd.to_datetime(frame["OrderTimestamp"], errors="coerce", utc=True)
    queue = (
        frame.sort_values(["OrderTimestamp", "OrderID"], kind="stable")
        .drop_duplicates("OrderID", keep="first")
        [["OrderID", "CompoundName", "OrderTimestamp"]]
        .reset_index(drop=True)
    )
    if args.max_orders:
        queue = queue.head(args.max_orders).copy()
    queue["QueueRank"] = range(1, len(queue) + 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(queue.head(10).to_string(index=False))
    print(f"queued_orders={len(queue)} output={args.output}")


if __name__ == "__main__":
    main()
