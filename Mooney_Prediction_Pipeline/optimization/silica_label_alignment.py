"""Audit Lab-to-process label granularity before Step-level recommendations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MooneySpecContract:
    compound_name: str
    target: float | None
    lower: float | None
    upper: float | None
    source: str
    status: str


def audit_pallet_label_alignment(history: pd.DataFrame, *, compound_name: str) -> pd.DataFrame:
    """Classify labels as true one-batch evidence or pallet-group-only evidence."""
    required = {"OrderID", "PalletID", "MNY", "N_batches", "CompoundName", "OrderStartTime"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"History missing alignment columns: {sorted(missing)}")
    frame = history[history["CompoundName"].astype(str).eq(compound_name)].copy()
    frame["OrderStartTime"] = pd.to_datetime(frame["OrderStartTime"], errors="coerce")
    frame["N_batches"] = pd.to_numeric(frame["N_batches"], errors="coerce")
    frame["MNY"] = pd.to_numeric(frame["MNY"], errors="coerce")
    frame["alignment_unit"] = "PALLET"
    frame["alignment_status"] = "PALLET_GROUP_NOT_BATCH_ONE_TO_ONE"
    frame.loc[frame["N_batches"].eq(1.0), "alignment_status"] = "ONE_BATCH_PALLET"
    frame["eligible_for_batch_step_replay"] = frame["alignment_status"].eq("ONE_BATCH_PALLET") & frame["MNY"].notna()
    frame["alignment_key"] = frame["OrderID"].astype(str) + "|" + frame["PalletID"].astype(str)
    return frame


def alignment_summary(alignment: pd.DataFrame) -> dict[str, Any]:
    if alignment.empty:
        return {
            "records": 0,
            "one_batch_records": 0,
            "pallet_group_records": 0,
            "batch_step_replay_ready": False,
            "reason": "NO_CANDIDATE_RECORDS",
        }
    one_batch = int(alignment["eligible_for_batch_step_replay"].sum())
    records = int(len(alignment))
    return {
        "records": records,
        "one_batch_records": one_batch,
        "pallet_group_records": records - one_batch,
        "batch_step_replay_ready": one_batch >= 150,
        "reason": "PASS" if one_batch >= 150 else "BLOCKED_INSUFFICIENT_ONE_BATCH_LABELS",
    }


def resolve_mooney_spec_contract(lab_rows: pd.DataFrame, *, compound_name: str) -> MooneySpecContract:
    """Use only stable Lab target/tolerance rows; ambiguous specifications block replay."""
    required = {
        "test_target",
        "tolerance_lower",
        "tolerance_upper",
        "corp_tolerance_lower",
        "corp_tolerance_upper",
    }
    missing = required.difference(lab_rows.columns)
    if missing:
        raise ValueError(f"Lab rows missing spec columns: {sorted(missing)}")
    numeric = lab_rows.copy()
    for column in required:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    numeric["spec_lower"] = numeric["tolerance_lower"].combine_first(numeric["corp_tolerance_lower"])
    numeric["spec_upper"] = numeric["tolerance_upper"].combine_first(numeric["corp_tolerance_upper"])
    contracts = numeric[["test_target", "spec_lower", "spec_upper"]].dropna().drop_duplicates()
    if len(contracts) != 1:
        return MooneySpecContract(compound_name, None, None, None, "compound_excellence_datamart", "BLOCKED_AMBIGUOUS_OR_MISSING_SPEC")
    row = contracts.iloc[0]
    low, high = float(row["spec_lower"]), float(row["spec_upper"])
    target = float(row["test_target"]) if pd.notna(row["test_target"]) else (low + high) / 2.0
    if low >= high:
        return MooneySpecContract(compound_name, None, None, None, "compound_excellence_datamart", "BLOCKED_INVALID_SPEC_RANGE")
    return MooneySpecContract(compound_name, target, low, high, "compound_excellence_datamart", "PASS")


def write_alignment_audit(
    alignment: pd.DataFrame,
    spec: MooneySpecContract,
    *,
    output_directory: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    """Persist transparent CSV and JSON audit artifacts for later agents."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"{stem}_label_alignment.csv"
    json_path = destination / f"{stem}_label_alignment_summary.json"
    alignment.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = alignment_summary(alignment)
    summary["spec_contract"] = {
        "compound_name": spec.compound_name,
        "target": spec.target,
        "lower": spec.lower,
        "upper": spec.upper,
        "source": spec.source,
        "status": spec.status,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path
