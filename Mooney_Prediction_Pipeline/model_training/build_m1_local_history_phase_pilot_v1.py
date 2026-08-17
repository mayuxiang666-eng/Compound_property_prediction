"""Build a deterministic current-batch-only Phase curve pilot from quarantine.

The output is diagnostic only.  It demonstrates that preserved local curves
can pass through the leakage-safe V3.7 t1--t7 curve feature path without using
Lab/MNY.  It is not promoted to training because exact historical Step/Target,
SCP-LOG and Lab disposition/revision evidence are not present in the CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.r10b_phase_curve_features import (
    extract_phase_curve_row,
)


DEFAULT_INPUT = (
    ROOT / "reports" / "v371_challenger" / "local_history_quarantine_v1"
    / "historical_m1_phase_rebuild_candidates.parquet"
)
DEFAULT_OUTPUT = (
    ROOT / "reports" / "v371_challenger" / "local_history_phase_pilot_v1"
)
PILOT_CODES = ("T15760", "B00458", "T09170", "T33025", "B00460", "B15563")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_sample(frame: pd.DataFrame, per_code: int) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    used: set[str] = set()
    for code in PILOT_CODES:
        mask = (
            frame["CompoundName"].astype(str).str.contains(code, case=False, na=False)
            | frame["RecipeID"].astype(str).str.contains(code, case=False, na=False)
        )
        subset = frame.loc[mask].sort_values(
            ["OrderStartTime", "PhysicalBatchKey"], kind="stable"
        )
        if subset.empty:
            continue
        positions = np.linspace(0, len(subset) - 1, min(per_code, len(subset)), dtype=int)
        take = subset.iloc[np.unique(positions)].copy()
        take["PilotCode"] = code
        selected.append(take)
        used.update(take["PhysicalBatchKey"].astype(str))
    if not selected:
        raise ValueError("NO_PILOT_COMPOUND_ROWS_IN_QUARANTINE")
    result = pd.concat(selected, ignore_index=True)
    return result.drop_duplicates("PhysicalBatchKey", keep="first")


def canonical_curve_row(row: pd.Series) -> pd.Series:
    canonical = row.copy()
    canonical["Curve1"] = row["temp"]
    canonical["Curve2"] = row["power"]
    canonical["Curve5"] = row["Torque"]
    canonical["Curve6"] = row["RotorSpeed"]
    canonical["Curve7"] = row["WayofRam"]
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-code", type=int, default=100)
    parser.add_argument("--points", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_parquet(args.input)
    if "MNY" not in source.columns:
        raise ValueError("QUARANTINE_SOURCE_EXPECTED_CANDIDATE_MNY_COLUMN")
    pilot = deterministic_sample(source, args.per_code)
    features: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for row in pilot.itertuples(index=False):
        series = canonical_curve_row(pd.Series(row._asdict()))
        try:
            extracted = extract_phase_curve_row(series, points=args.points)
            extracted.update({
                "PhysicalBatchKey": series["PhysicalBatchKey"],
                "LabelGroupID": series["LabelGroupID"],
                "CompoundName": series["CompoundName"],
                "RecipeID": series["RecipeID"],
                "ProcessVersion": series["ProcessVersion"],
                "MixerLine": series["MixerLine"],
                "OrderStartTime": series["OrderStartTime"],
                "PilotCode": series["PilotCode"],
                "historical_lab_status": "BLOCKED_UNVERIFIED_HISTORICAL_LAB",
                "phase_feature_training_status": "DIAGNOSTIC_ONLY_MISSING_EXACT_STEP_CONTRACT",
            })
            features.append(extracted)
            audits.append({
                "PhysicalBatchKey": series["PhysicalBatchKey"],
                "PilotCode": series["PilotCode"],
                "status": "PHASE_REBUILT_CURRENT_BATCH_ONLY",
                "error": "",
            })
        except Exception as exc:  # fail closed and preserve per-batch evidence
            audits.append({
                "PhysicalBatchKey": series["PhysicalBatchKey"],
                "PilotCode": series["PilotCode"],
                "status": "BLOCKED_PHASE_REBUILD",
                "error": f"{type(exc).__name__}:{exc}",
            })

    feature_frame = pd.DataFrame(features)
    if "MNY" in feature_frame.columns:
        raise ValueError("LAB_TARGET_LEAKED_INTO_PHASE_FEATURE_OUTPUT")
    audit_frame = pd.DataFrame(audits)
    feature_path = args.output_dir / "historical_m1_phase_pilot_features.parquet"
    audit_path = args.output_dir / "historical_m1_phase_pilot_audit.csv"
    feature_frame.to_parquet(feature_path, index=False, compression="zstd")
    audit_frame.to_csv(audit_path, index=False, encoding="utf-8-sig")

    manifest = {
        "contract_version": "M1-LOCAL-HISTORY-PHASE-PILOT-V1",
        "input_path": str(args.input),
        "input_sha256": sha256(args.input),
        "selected_batches": int(len(pilot)),
        "rebuilt_batches": int(len(feature_frame)),
        "blocked_batches": int(audit_frame["status"].ne("PHASE_REBUILT_CURRENT_BATCH_ONLY").sum()),
        "pilot_code_counts": audit_frame["PilotCode"].value_counts().sort_index().to_dict(),
        "feature_columns": int(len(feature_frame.columns)),
        "mny_in_feature_output": False,
        "curve_fill_policy": "WITHIN_CURRENT_BATCH_SIGNAL_ONLY",
        "cross_batch_fill": False,
        "phase_boundary_contract": "V3.7_T1_T7_CURRENT_BATCH_HEURISTIC_DIAGNOSTIC",
        "training_status": "BLOCKED_MISSING_EXACT_STEP_AND_VERIFIED_LAB_CONTRACT",
        "database_access_during_build": False,
        "local_mms_access": False,
        "archive_access": False,
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
        "feature_path": str(feature_path),
        "feature_sha256": sha256(feature_path),
        "audit_path": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = args.output_dir / "historical_m1_phase_pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
