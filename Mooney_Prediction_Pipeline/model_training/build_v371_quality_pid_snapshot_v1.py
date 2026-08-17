"""Attach curve-quality audit fields to the corrected PID retrain snapshot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
SOURCE = BASE / "v371_retrain_pid_snapshot_v1" / "v371_retrain_group_feature_snapshot.csv"
QUALITY = BASE / "v371_curve_quality_v1" / "group_curve_quality_audit.csv"
OUT = BASE / "v371_retrain_quality_pid_snapshot_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE, low_memory=False)
    quality = pd.read_csv(QUALITY, low_memory=False)
    quality_columns = [
        "LabelGroupID", "curve_batch_count", "curve_physical_invalid_batch_count",
        "curve_statistical_outlier_batch_count", "curve_if_score_mean",
        "curve_if_score_min", "curve_physical_invalid_rate",
        "curve_statistical_outlier_rate", "curve_valid_batch_rate",
        "curve_quality_risk_flag",
    ]
    quality = quality[quality_columns].drop_duplicates("LabelGroupID")
    merged = frame.merge(quality, on="LabelGroupID", how="left", validate="one_to_one")
    if merged["curve_quality_risk_flag"].isna().any():
        raise ValueError("Missing curve quality audit for a Lab group")
    merged["curve_model_include_group"] = merged["curve_physical_invalid_batch_count"].eq(0)
    # Statistical outliers remain in the training population by contract.
    merged["curve_statistical_outlier_retained"] = True
    output = OUT / "v371_retrain_quality_pid_group_feature_snapshot.csv"
    merged.to_csv(output, index=False, encoding="utf-8-sig")
    manifest = {
        "contract_version": "V3.7.1-QUALITY-PID-GROUP-SNAPSHOT-V1",
        "source_snapshot": str(SOURCE),
        "source_snapshot_sha256": sha256(SOURCE),
        "quality_audit": str(QUALITY),
        "quality_audit_sha256": sha256(QUALITY),
        "group_rows": int(len(merged)),
        "groups_excluded_for_physical_invalidity": int((~merged["curve_model_include_group"]).sum()),
        "statistical_outlier_groups_retained": int(merged["curve_quality_risk_flag"].sum()),
        "mny_used_for_quality_flags": False,
        "lab_used_for_quality_flags": False,
        "source_policy": "Cloud RDS Active snapshot only; local MMS/Archive not accessed",
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
