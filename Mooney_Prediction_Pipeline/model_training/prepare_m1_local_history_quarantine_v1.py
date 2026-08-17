"""Prepare a local-only, fail-closed M1 historical curve quarantine.

This module never opens a database connection.  It inventories the preserved
``stage_statistics_enriched.csv`` file, normalises legacy ``YYYY~OrderID``
aliases, protects the fixed validation/OOT groups, and writes a local parquet
cache for current-batch-only feature rebuilding.

Historical MNY values are deliberately not released for training because the
CSV has no disposition/revision contract.  They remain candidate labels until
an independently trusted Lab snapshot confirms 001/011 and the latest valid
result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "stage_statistics_enriched.csv"
DEFAULT_CURRENT_BATCH = (
    ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
    / "v371_r1b_m1_only_snapshot_v1" / "m1_only_batch_pid_quality_features.csv"
)
DEFAULT_CURRENT_GROUP = (
    ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
    / "m1_temporal_jit_v1" / "m1_temporal_jit_acceptance_input.csv"
)
DEFAULT_OUTPUT = ROOT / "reports" / "v371_challenger" / "local_history_quarantine_v1"

CURVE_COLUMNS = ("temp", "power", "Torque", "RotorSpeed", "WayofRam")
CORE_ALIGNED_CURVES = ("temp", "power", "Torque", "RotorSpeed")
HOLDOUT_SPLITS = {"validation", "val", "oot", "oot_test", "test"}
UNVERIFIED_LAB_STATUS = "BLOCKED_UNVERIFIED_HISTORICAL_LAB"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"^\d{4}~", "", text)
    text = re.sub(r"\.0$", "", text)
    return text


def normalize_batch_number(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(number) else str(int(number))


def curve_sample_count(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip().replace(" ", "").replace("\n", "").replace("\r", "")
    if not text or len(text) % 4:
        return 0
    try:
        int(text[: min(len(text), 64)], 16)
    except ValueError:
        return 0
    return len(text) // 4


def _append_reason(reasons: pd.Series, mask: pd.Series, code: str) -> None:
    active = mask.fillna(False)
    reasons.loc[active] = reasons.loc[active].where(
        reasons.loc[active].eq(""), reasons.loc[active] + ";"
    ) + code


def _normalised_keys(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    order = frame["OrderID"].map(normalize_identifier)
    batch = frame["BatchNumber"].map(normalize_batch_number)
    pallet = frame.get("PalletID", pd.Series("", index=frame.index)).map(normalize_identifier)
    return order + "|" + batch, order + "|" + pallet, order


def _normalised_group_keys(frame: pd.DataFrame) -> pd.Series:
    order = frame["OrderID"].map(normalize_identifier)
    pallet = frame["PalletID"].map(normalize_identifier)
    return order + "|" + pallet


def build_quarantine(
    source: pd.DataFrame,
    current_batch: pd.DataFrame,
    current_group: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = {
        "OrderID", "BatchNumber", "PalletID", "CompoundName",
        "CompoundDescription", "OrderStartTime", "Target_Temperature", "MNY",
        *CURVE_COLUMNS,
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"HISTORICAL_SOURCE_MISSING_COLUMNS: {missing}")

    work = source.copy()
    work.insert(0, "SourceOrderID", work["OrderID"].astype(str))
    batch_key, group_key, normal_order = _normalised_keys(work)
    work["OrderID"] = normal_order
    work["BatchNumber"] = work["BatchNumber"].map(normalize_batch_number)
    work["PalletID"] = work["PalletID"].map(normalize_identifier)
    work["PhysicalBatchKey"] = batch_key
    work["LabelGroupID"] = group_key
    work["RecipeID"] = work["CompoundDescription"].astype(str).str.strip()
    work["ProcessVersion"] = pd.to_numeric(
        work["RecipeID"].str.extract(r"_PV(\d+)_", expand=False), errors="coerce"
    ).astype("Int64")
    work["OrderStartTime"] = pd.to_datetime(work["OrderStartTime"], errors="coerce")
    work["lab_time"] = pd.to_datetime(work.get("test_result_start_time"), errors="coerce")

    current_batch_keys, _, _ = _normalised_keys(current_batch)
    current_keys = set(current_batch_keys)
    current_groups = _normalised_group_keys(current_group)
    split = current_group.get("split", pd.Series("", index=current_group.index)).astype(str).str.lower()
    fixed_holdout_groups = set(current_groups.loc[split.isin(HOLDOUT_SPLITS)])

    work["is_m1"] = work["CompoundName"].astype(str).str.startswith("M1-")
    work["overlaps_current_corrected_snapshot"] = work["PhysicalBatchKey"].isin(current_keys)
    work["fixed_holdout_group"] = work["LabelGroupID"].isin(fixed_holdout_groups)
    work["historical_lab_status"] = UNVERIFIED_LAB_STATUS
    work["historical_lab_candidate_value"] = pd.to_numeric(work["MNY"], errors="coerce")

    length_columns: list[str] = []
    for column in CURVE_COLUMNS:
        output = f"{column}_sample_count"
        work[output] = work[column].map(curve_sample_count)
        length_columns.append(output)
    core_lengths = work[[f"{column}_sample_count" for column in CORE_ALIGNED_CURVES]]
    work["curve_all_signals_present"] = work[length_columns].gt(0).all(axis=1)
    work["curve_core_length_match"] = core_lengths.nunique(axis=1).eq(1)
    work["curve_min_core_samples"] = core_lengths.min(axis=1)
    target_temperature = pd.to_numeric(work["Target_Temperature"], errors="coerce")

    reasons = pd.Series("", index=work.index, dtype=object)
    _append_reason(reasons, ~work["is_m1"], "BLOCKED_NON_M1")
    _append_reason(
        reasons,
        work["PhysicalBatchKey"].str.startswith("|") | work["PhysicalBatchKey"].str.endswith("|"),
        "BLOCKED_INVALID_BATCH_KEY",
    )
    _append_reason(
        reasons,
        work["LabelGroupID"].str.startswith("|") | work["LabelGroupID"].str.endswith("|"),
        "BLOCKED_INVALID_LABEL_GROUP_KEY",
    )
    _append_reason(reasons, work["overlaps_current_corrected_snapshot"], "BLOCKED_CURRENT_SNAPSHOT_PREFERRED")
    _append_reason(reasons, work["fixed_holdout_group"], "BLOCKED_FIXED_VALIDATION_OOT_GROUP")
    _append_reason(reasons, ~work["curve_all_signals_present"], "BLOCKED_MISSING_CURVE_SIGNAL")
    _append_reason(reasons, ~work["curve_core_length_match"], "BLOCKED_CORE_SIGNAL_LENGTH_MISMATCH")
    _append_reason(reasons, work["curve_min_core_samples"].lt(20), "BLOCKED_SHORT_CURVE")
    _append_reason(reasons, target_temperature.isna() | target_temperature.le(0), "BLOCKED_MISSING_PROCESS_TARGET_TEMPERATURE")
    work["phase_rebuild_block_reasons"] = reasons
    work["phase_rebuild_eligible"] = reasons.eq("")

    training_reasons = reasons.copy()
    _append_reason(training_reasons, work["historical_lab_candidate_value"].isna(), "BLOCKED_MISSING_HISTORICAL_MNY")
    _append_reason(training_reasons, pd.Series(True, index=work.index), UNVERIFIED_LAB_STATUS)
    work["training_block_reasons"] = training_reasons
    work["training_eligible"] = False
    work["source_preference"] = np.where(
        work["overlaps_current_corrected_snapshot"],
        "CURRENT_CORRECTED_SNAPSHOT",
        "LOCAL_HISTORICAL_RAW_CURVE",
    )

    if work["PhysicalBatchKey"].duplicated().any():
        duplicate_count = int(work["PhysicalBatchKey"].duplicated(keep=False).sum())
        raise ValueError(f"HISTORICAL_DUPLICATE_PHYSICAL_BATCH_KEY: {duplicate_count}")

    raw_candidates = work.loc[work["phase_rebuild_eligible"]].copy()
    manifest = {
        "contract_version": "M1-LOCAL-HISTORY-QUARANTINE-V1",
        "source_rows": int(len(source)),
        "m1_rows": int(work["is_m1"].sum()),
        "unique_physical_batches": int(work["PhysicalBatchKey"].nunique()),
        "unique_lab_groups": int(work.loc[work["is_m1"], "LabelGroupID"].nunique()),
        "current_snapshot_overlap_batches": int(work["overlaps_current_corrected_snapshot"].sum()),
        "fixed_holdout_rows": int(work["fixed_holdout_group"].sum()),
        "missing_curve_signal_rows": int((~work["curve_all_signals_present"]).sum()),
        "core_length_mismatch_rows": int((~work["curve_core_length_match"]).sum()),
        "phase_rebuild_candidate_batches": int(len(raw_candidates)),
        "training_eligible_batches": 0,
        "historical_lab_status": UNVERIFIED_LAB_STATUS,
        "database_access_during_build": False,
        "local_mms_access": False,
        "archive_access": False,
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
    }
    return work, raw_candidates, manifest


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--current-batch", type=Path, default=DEFAULT_CURRENT_BATCH)
    parser.add_argument("--current-group", type=Path, default=DEFAULT_CURRENT_GROUP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.source, low_memory=False)
    current_batch = pd.read_csv(
        args.current_batch, usecols=["OrderID", "BatchNumber", "PalletID"], low_memory=False
    )
    current_group = pd.read_csv(
        args.current_group, usecols=["OrderID", "PalletID", "split"], low_memory=False
    )
    audit, candidates, manifest = build_quarantine(source, current_batch, current_group)

    audit_columns = [
        "SourceOrderID", "OrderID", "BatchNumber", "PalletID", "PhysicalBatchKey",
        "LabelGroupID", "CompoundName", "RecipeID", "ProcessVersion", "MixerLine",
        "OrderStartTime", "lab_time", "historical_lab_candidate_value",
        "is_m1", "overlaps_current_corrected_snapshot", "fixed_holdout_group",
        *[f"{column}_sample_count" for column in CURVE_COLUMNS],
        "curve_all_signals_present", "curve_core_length_match", "curve_min_core_samples",
        "phase_rebuild_eligible", "phase_rebuild_block_reasons",
        "historical_lab_status", "training_eligible", "training_block_reasons",
        "source_preference",
    ]
    audit_path = args.output_dir / "historical_m1_quarantine_index.csv"
    candidate_key_path = args.output_dir / "historical_m1_phase_rebuild_candidate_keys.csv"
    candidate_parquet_path = args.output_dir / "historical_m1_phase_rebuild_candidates.parquet"
    audit[audit_columns].to_csv(audit_path, index=False, encoding="utf-8-sig")
    candidates[audit_columns].to_csv(candidate_key_path, index=False, encoding="utf-8-sig")
    _atomic_parquet(candidates, candidate_parquet_path)

    manifest.update({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(args.source),
        "source_sha256": sha256(args.source),
        "current_batch_path": str(args.current_batch),
        "current_batch_sha256": sha256(args.current_batch),
        "current_group_path": str(args.current_group),
        "current_group_sha256": sha256(args.current_group),
        "quarantine_index_path": str(audit_path),
        "quarantine_index_sha256": sha256(audit_path),
        "candidate_key_path": str(candidate_key_path),
        "candidate_key_sha256": sha256(candidate_key_path),
        "candidate_parquet_path": str(candidate_parquet_path),
        "candidate_parquet_sha256": sha256(candidate_parquet_path),
    })
    manifest_path = args.output_dir / "historical_m1_quarantine_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
