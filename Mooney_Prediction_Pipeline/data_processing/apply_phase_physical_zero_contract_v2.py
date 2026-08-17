"""Canonical Stage6 ram contract: MSB has no ram, therefore both features are zero."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.data_processing.apply_phase_physical_zero_contract_v1 import (
    BASE,
    PHYSICAL_ZERO_COLUMNS,
    SOURCE,
)


def run(source: Path, output: Path, audit_path: Path, anomaly_path: Path) -> dict[str, object]:
    frame = pd.read_csv(source, low_memory=False)
    missing = sorted(set(PHYSICAL_ZERO_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Physical-zero columns absent from Phase source: {missing}")
    masks = {}
    anomaly_mask = pd.Series(False, index=frame.index)
    for column in PHYSICAL_ZERO_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        masks[column] = {
            "null_to_zero": int(numeric.isna().sum()),
            "nonzero_to_zero": int(numeric.fillna(0).ne(0).sum()),
        }
        anomaly_mask |= numeric.fillna(0).ne(0)
    anomaly_columns = [
        "OrderID", "BatchNumber", "CompoundDescription", "CompoundName", "OrderStartTime",
        *PHYSICAL_ZERO_COLUMNS,
    ]
    anomalies = frame.loc[anomaly_mask, anomaly_columns].copy()
    for column in PHYSICAL_ZERO_COLUMNS:
        frame[column] = 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    anomalies.to_csv(anomaly_path, index=False, encoding="utf-8-sig")
    audit = {
        "contract": "PHASE-PHYSICAL-NONAPPLICABILITY-ZERO-V2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(source), "output": str(output),
        "rows": int(len(frame)), "columns_modified": PHYSICAL_ZERO_COLUMNS,
        "modifications": masks,
        "nonzero_physical_anomaly_rows": int(len(anomalies)),
        "anomaly_audit": str(anomaly_path),
        "interpretation": "Original Phase builder always emits zero WayofRam for Stage6 because MSB has no ram.",
        "other_missing_values_modified": False,
        "future_or_cross_batch_fill": False,
        "database_writes": False, "automatic_control": "DISABLED",
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=BASE / "phase_curve_timesafe_physical_v3.csv")
    parser.add_argument("--audit", type=Path, default=BASE / "phase_curve_timesafe_physical_v3_audit.json")
    parser.add_argument("--anomalies", type=Path, default=BASE / "phase_curve_timesafe_stage6_ram_anomalies_v2.csv")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output, args.audit, args.anomalies), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
