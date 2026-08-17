"""Apply auditable physical non-applicability zeros to no-leak Phase history."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history"
SOURCE = BASE / "phase_curve_timesafe_v1.csv"
OUTPUT = BASE / "phase_curve_timesafe_physical_v2.csv"
AUDIT = BASE / "phase_curve_timesafe_physical_v2_audit.json"
PHYSICAL_ZERO_COLUMNS = [
    "Stage6_BottomMixing_WayofRam_Mean",
    "Stage6_BottomMixing_WayofRam_Std",
]


def run(source: Path, output: Path, audit_path: Path) -> dict[str, object]:
    frame = pd.read_csv(source, low_memory=False)
    missing_columns = sorted(set(PHYSICAL_ZERO_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Physical-zero columns absent from Phase source: {missing_columns}")
    replacements: dict[str, int] = {}
    preexisting_nonzero: dict[str, int] = {}
    for column in PHYSICAL_ZERO_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        replacements[column] = int(numeric.isna().sum())
        preexisting_nonzero[column] = int(numeric.fillna(0).ne(0).sum())
        frame.loc[numeric.isna(), column] = 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    audit = {
        "contract": "PHASE-PHYSICAL-NONAPPLICABILITY-ZERO-V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(source), "output": str(output),
        "rows": int(len(frame)),
        "columns_modified": PHYSICAL_ZERO_COLUMNS,
        "null_replacements": replacements,
        "preexisting_nonzero_values_preserved": preexisting_nonzero,
        "interpretation": "MSB Stage6 has no ram; only null cells in these two columns become physical zero.",
        "other_missing_values_modified": False,
        "future_or_cross_batch_fill": False,
        "database_writes": False,
        "automatic_control": "DISABLED",
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output, args.audit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
