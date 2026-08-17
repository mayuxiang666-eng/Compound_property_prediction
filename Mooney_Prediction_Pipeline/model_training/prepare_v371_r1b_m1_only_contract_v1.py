"""Build strict M1-only snapshots for the V3.7.1 R1B challenger.

This script reads only existing cloud-RDS Active snapshot artifacts.  It does
not access MMS or Archive and does not alter the original Global R1B inputs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
SOURCE_BATCH = BASE / "v371_r1b_batch_pid_quality_snapshot_v1" / "v371_r1b_batch_pid_quality_features.csv"
SOURCE_GROUP = BASE / "v371_r1b_batch_pid_quality_snapshot_v1" / "v371_r1b_group_label_snapshot_with_pid_quality.csv"
SOURCE_METADATA = BASE / "v371_r1b_batch_pid_quality_snapshot_v1" / "v371_r1b_batch_pid_quality_metadata_v1.json"
OUT = BASE / "v371_r1b_m1_only_snapshot_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    batch = pd.read_csv(SOURCE_BATCH, low_memory=False)
    group = pd.read_csv(SOURCE_GROUP, low_memory=False)
    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    if "CompoundName" not in batch.columns or "CompoundName" not in group.columns:
        raise ValueError("M1_ONLY_SCOPE_REQUIRES_COMPOUND_NAME")

    batch_mask = batch["CompoundName"].astype(str).str.startswith("M1-")
    group_mask = group["CompoundName"].astype(str).str.startswith("M1-")
    batch_m1 = batch.loc[batch_mask].copy()
    group_m1 = group.loc[group_mask].copy()
    batch_groups = set(batch_m1["LabelGroupID"].astype(str))
    group_groups = set(group_m1["LabelGroupID"].astype(str))
    if batch_groups != group_groups:
        raise ValueError(
            f"M1_ONLY_GROUP_ALIGNMENT_ERROR: batch_only={len(batch_groups - group_groups)} "
            f"group_only={len(group_groups - batch_groups)}"
        )
    if not set(batch_m1["material_system"].dropna().astype(str)).issubset({"CarbonBlack", "Silica"}):
        raise ValueError("M1_ONLY_UNKNOWN_MATERIAL_SYSTEM")
    physical_key = batch_m1[["OrderID", "BatchNumber"]].astype(str).agg("|".join, axis=1)
    duplicate_physical_batches = int(physical_key.duplicated().sum())
    if duplicate_physical_batches:
        raise ValueError(f"M1_ONLY_DUPLICATE_PHYSICAL_BATCHES: {duplicate_physical_batches}")

    batch_path = OUT / "m1_only_batch_pid_quality_features.csv"
    group_path = OUT / "m1_only_group_label_snapshot_with_pid_quality.csv"
    metadata_path = OUT / "m1_only_metadata_v1.json"
    batch_m1.to_csv(batch_path, index=False, encoding="utf-8-sig")
    group_m1.to_csv(group_path, index=False, encoding="utf-8-sig")

    metadata["contract_version"] = "V3.7.1-R1B-M1-ONLY-BATCH-PID-QUALITY-V1"
    metadata["scope"] = "M1_ONLY"
    metadata["excluded_compound_prefixes"] = ["M2-", "M3B-", "PB-", "R1-"]
    metadata["source_contract"] = {
        "source_batch": str(SOURCE_BATCH),
        "source_group": str(SOURCE_GROUP),
        "source_metadata": str(SOURCE_METADATA),
        "source_batch_sha256": sha256(SOURCE_BATCH),
        "source_group_sha256": sha256(SOURCE_GROUP),
        "source_metadata_sha256": sha256(SOURCE_METADATA),
        "database_access_during_build": False,
        "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
    }
    metadata["m1_filter_audit"] = {
        "batch_rows_before": int(len(batch)),
        "batch_rows_after": int(len(batch_m1)),
        "group_rows_before": int(len(group)),
        "group_rows_after": int(len(group_m1)),
        "lab_groups_after": int(group_m1["LabelGroupID"].nunique()),
        "material_systems_after": sorted(group_m1["material_system"].dropna().astype(str).unique().tolist()),
        "compound_prefix_rule": "CompoundName startswith M1-",
    }
    metadata["generated_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "contract_version": metadata["contract_version"],
        "scope": "M1_ONLY",
        "batch_path": str(batch_path),
        "group_path": str(group_path),
        "metadata_path": str(metadata_path),
        "batch_sha256": sha256(batch_path),
        "group_sha256": sha256(group_path),
        "metadata_sha256": sha256(metadata_path),
        "batch_rows": int(len(batch_m1)),
        "lab_groups": int(group_m1["LabelGroupID"].nunique()),
        "duplicate_physical_batches": duplicate_physical_batches,
        "excluded_non_m1_batch_rows": int(len(batch) - len(batch_m1)),
        "excluded_non_m1_group_rows": int(len(group) - len(group_m1)),
        "database_access_during_build": False,
        "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
    }
    (OUT / "m1_only_snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
