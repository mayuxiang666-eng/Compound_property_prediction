"""Release verified local historical labels without weakening fixed OOT.

Only the already materialised canonical 001/011 Lab snapshot is trusted.
Legacy MNY values embedded in the raw curve CSV are retained for discrepancy
audit but never used as the released target.  An exact RecipeID cutoff derived
from the existing fixed validation/OOT groups prevents future-history leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.prepare_m1_local_history_quarantine_v1 import (
    normalize_identifier,
)


BASE = ROOT / "reports" / "v371_challenger"
DEFAULT_QUARANTINE = BASE / "local_history_quarantine_v1" / "historical_m1_phase_rebuild_candidates.parquet"
DEFAULT_LAB = BASE / "global_rds_active_v1" / "lab_group_canonical_001_011.csv"
DEFAULT_CURRENT_BATCH = (
    BASE / "global_rds_active_v1" / "v371_r1b_m1_only_snapshot_v1"
    / "m1_only_batch_pid_quality_features.csv"
)
DEFAULT_CURRENT_GROUP = (
    BASE / "global_rds_active_v1" / "m1_temporal_jit_v1"
    / "m1_temporal_jit_acceptance_input.csv"
)
DEFAULT_OUTPUT = BASE / "local_history_label_release_v1"
HOLDOUT_SPLITS = {"validation", "val", "oot", "oot_test", "test"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_disposition(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\.0$", "", str(value).strip())
    return text.zfill(3) if text.isdigit() else text.upper()


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _group_id(order: pd.Series, pallet: pd.Series) -> pd.Series:
    return order.map(normalize_identifier) + "|" + pallet.map(normalize_identifier)


def release_labels(
    quarantine: pd.DataFrame,
    canonical_lab: pd.DataFrame,
    current_batch: pd.DataFrame,
    current_group: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required_lab = {
        "order_id", "pallet_id", "test_result", "test_result_start_time",
        "disposition_code", "status_valid",
    }
    missing = sorted(required_lab.difference(canonical_lab.columns))
    if missing:
        raise ValueError(f"CANONICAL_LAB_MISSING_COLUMNS: {missing}")

    lab = canonical_lab.copy()
    lab["LabelGroupID"] = _group_id(lab["order_id"], lab["pallet_id"])
    lab["disposition_norm"] = lab["disposition_code"].map(normalize_disposition)
    lab["status_valid_norm"] = _truthy(lab["status_valid"])
    lab["canonical_mny"] = pd.to_numeric(lab["test_result"], errors="coerce")
    lab["canonical_lab_time"] = pd.to_datetime(lab["test_result_start_time"], errors="coerce")
    if lab["LabelGroupID"].duplicated().any():
        raise ValueError("CANONICAL_LAB_NOT_ONE_ROW_PER_LABEL_GROUP")
    canonical_valid = lab[
        lab["disposition_norm"].isin({"001", "011"})
        & lab["status_valid_norm"]
        & lab["canonical_mny"].notna()
        & lab["canonical_lab_time"].notna()
    ].copy()

    group = quarantine.groupby("LabelGroupID", as_index=False).agg(
        OrderID=("OrderID", "first"),
        PalletID=("PalletID", "first"),
        CompoundName=("CompoundName", "first"),
        RecipeID=("RecipeID", "first"),
        ProcessVersion=("ProcessVersion", "first"),
        MixerLine=("MixerLine", "first"),
        GroupProductionStart=("OrderStartTime", "min"),
        HistoricalCandidateMNY=("historical_lab_candidate_value", "first"),
        HistoricalCandidateMNYCount=("historical_lab_candidate_value", "nunique"),
        PhaseBatchCount=("PhysicalBatchKey", "nunique"),
        RecipeCount=("RecipeID", "nunique"),
        MixerCount=("MixerLine", "nunique"),
    )
    if group["HistoricalCandidateMNYCount"].gt(1).any():
        raise ValueError("HISTORICAL_GROUP_HAS_MULTIPLE_EMBEDDED_MNY_VALUES")
    group["GroupProductionStart"] = pd.to_datetime(group["GroupProductionStart"], errors="coerce")

    lab_columns = [
        "LabelGroupID", "canonical_mny", "canonical_lab_time", "disposition_norm",
        "retest_number_numeric", "last_modified_timestamp", "datamart_pk",
    ]
    lab_columns = [column for column in lab_columns if column in canonical_valid.columns]
    verified = group.merge(
        canonical_valid[lab_columns], on="LabelGroupID", how="inner", validate="one_to_one"
    )
    verified["MNY"] = verified["canonical_mny"]
    verified["lab_time"] = verified["canonical_lab_time"]
    verified["label_verification_status"] = "VERIFIED_CANONICAL_LAB_001_011"
    verified["legacy_mny_abs_difference"] = (
        pd.to_numeric(verified["HistoricalCandidateMNY"], errors="coerce") - verified["MNY"]
    ).abs()

    current = current_batch.copy()
    current["BatchStartTime"] = pd.to_datetime(current["BatchStartTime"], errors="coerce")
    current_meta = current.groupby("LabelGroupID", as_index=False).agg(
        RecipeID=("RecipeID", "first"),
        FixedGroupProductionStart=("BatchStartTime", "min"),
        FixedRecipeCount=("RecipeID", "nunique"),
    )
    split = current_group[["LabelGroupID", "split"]].drop_duplicates("LabelGroupID")
    current_meta = current_meta.merge(split, on="LabelGroupID", how="inner", validate="one_to_one")
    holdout = current_meta[
        current_meta["split"].astype(str).str.lower().isin(HOLDOUT_SPLITS)
    ].copy()
    cutoffs = holdout.groupby("RecipeID", as_index=False).agg(
        FixedValidationOOTStart=("FixedGroupProductionStart", "min"),
        FixedHoldoutGroupCount=("LabelGroupID", "nunique"),
    )
    verified = verified.merge(cutoffs, on="RecipeID", how="left", validate="many_to_one")
    verified["temporal_cutoff_status"] = "BLOCKED_NO_EXACT_RECIPE_FIXED_HOLDOUT"
    has_cutoff = verified["FixedValidationOOTStart"].notna()
    before = has_cutoff & verified["GroupProductionStart"].lt(verified["FixedValidationOOTStart"])
    verified.loc[has_cutoff & ~before, "temporal_cutoff_status"] = "BLOCKED_NOT_BEFORE_FIXED_VALIDATION_OOT"
    verified.loc[before, "temporal_cutoff_status"] = "SAFE_BEFORE_EXACT_RECIPE_FIXED_VALIDATION_OOT"
    verified["history_training_release"] = before

    safe_groups = verified.loc[verified["history_training_release"]].copy()
    safe_ids = set(safe_groups["LabelGroupID"].astype(str))
    safe_process = quarantine[quarantine["LabelGroupID"].astype(str).isin(safe_ids)].copy()
    target_like = {
        "MNY", "MNY target", "test_result_start_time", "historical_lab_candidate_value",
        "historical_lab_status", "training_eligible", "training_block_reasons", "lab_time",
    }
    safe_process = safe_process.drop(columns=[c for c in target_like if c in safe_process.columns])
    safe_process["history_split"] = "train_history_candidate"
    if safe_process["PhysicalBatchKey"].duplicated().any():
        raise ValueError("SAFE_HISTORY_PROCESS_ROWS_HAVE_DUPLICATE_PHYSICAL_BATCHES")
    if set(safe_process["LabelGroupID"].astype(str)) != safe_ids:
        raise ValueError("SAFE_HISTORY_GROUP_PROCESS_ALIGNMENT_ERROR")

    manifest = {
        "contract_version": "M1-LOCAL-HISTORY-LABEL-RELEASE-V1",
        "quarantine_groups": int(group["LabelGroupID"].nunique()),
        "canonical_lab_groups": int(canonical_valid["LabelGroupID"].nunique()),
        "verified_overlap_groups": int(len(verified)),
        "verified_overlap_batches": int(verified["PhaseBatchCount"].sum()),
        "exact_recipe_cutoff_groups": int(has_cutoff.sum()),
        "temporally_safe_groups": int(len(safe_groups)),
        "temporally_safe_batches": int(len(safe_process)),
        "temporally_safe_recipes": int(safe_groups["RecipeID"].nunique()),
        "legacy_mny_exact_match_groups": int(verified["legacy_mny_abs_difference"].lt(1e-9).sum()),
        "legacy_mny_difference_gt_0_1_groups": int(verified["legacy_mny_abs_difference"].gt(0.1).sum()),
        "legacy_mny_difference_gt_1_groups": int(verified["legacy_mny_abs_difference"].gt(1.0).sum()),
        "legacy_mny_max_abs_difference": float(verified["legacy_mny_abs_difference"].max()),
        "released_target_source": "CANONICAL_LAB_001_011_ONLY",
        "embedded_legacy_mny_used_as_target": False,
        "fixed_validation_oot_preserved": True,
        "database_access_during_build": False,
        "local_mms_access": False,
        "archive_access": False,
        "automatic_control": "DISABLED",
    }
    return verified, safe_groups, safe_process, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_QUARANTINE)
    parser.add_argument("--canonical-lab", type=Path, default=DEFAULT_LAB)
    parser.add_argument("--current-batch", type=Path, default=DEFAULT_CURRENT_BATCH)
    parser.add_argument("--current-group", type=Path, default=DEFAULT_CURRENT_GROUP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    quarantine = pd.read_parquet(args.quarantine)
    canonical_lab = pd.read_csv(args.canonical_lab, low_memory=False)
    current_batch = pd.read_csv(
        args.current_batch,
        usecols=["LabelGroupID", "RecipeID", "BatchStartTime"],
        low_memory=False,
    )
    current_group = pd.read_csv(
        args.current_group, usecols=["LabelGroupID", "split"], low_memory=False
    )
    verified, safe_groups, safe_process, manifest = release_labels(
        quarantine, canonical_lab, current_batch, current_group
    )

    verified_path = args.output_dir / "verified_history_group_labels.csv"
    safe_group_path = args.output_dir / "temporal_safe_history_group_labels.csv"
    safe_process_path = args.output_dir / "temporal_safe_history_process_rows.parquet"
    mismatch_path = args.output_dir / "legacy_vs_canonical_mny_audit.csv"
    verified.to_csv(verified_path, index=False, encoding="utf-8-sig")
    safe_groups.to_csv(safe_group_path, index=False, encoding="utf-8-sig")
    safe_process.to_parquet(safe_process_path, index=False, compression="zstd")
    verified.sort_values("legacy_mny_abs_difference", ascending=False).to_csv(
        mismatch_path, index=False, encoding="utf-8-sig"
    )
    manifest.update({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "quarantine_sha256": sha256(args.quarantine),
        "canonical_lab_sha256": sha256(args.canonical_lab),
        "current_batch_sha256": sha256(args.current_batch),
        "current_group_sha256": sha256(args.current_group),
        "verified_group_path": str(verified_path),
        "verified_group_sha256": sha256(verified_path),
        "safe_group_path": str(safe_group_path),
        "safe_group_sha256": sha256(safe_group_path),
        "safe_process_path": str(safe_process_path),
        "safe_process_sha256": sha256(safe_process_path),
        "mismatch_audit_path": str(mismatch_path),
        "mismatch_audit_sha256": sha256(mismatch_path),
    })
    manifest_path = args.output_dir / "history_label_release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
