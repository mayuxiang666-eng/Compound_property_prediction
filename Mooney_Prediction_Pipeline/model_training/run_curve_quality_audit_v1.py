"""Run the leakage-safe Active snapshot curve quality audit."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.curve_quality_contract_v1 import build_quality_audit


BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUT = BASE / "v371_curve_quality_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    batch_path = BASE / "phase_batch_features_active_v1.csv"
    group_path = BASE / "pallet_group_training_frame_c2_prepared.csv"
    batch = pd.read_csv(batch_path, low_memory=False)
    group = pd.read_csv(group_path, usecols=["LabelGroupID", "split"], low_memory=False)
    batch_audit, group_audit, manifest = build_quality_audit(batch, group)
    batch_out = OUT / "batch_curve_quality_audit.csv"
    group_out = OUT / "group_curve_quality_audit.csv"
    manifest["source_sha256"] = {batch_path.name: sha256(batch_path), group_path.name: sha256(group_path)}
    manifest["generated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["source_policy"] = "Cloud RDS Active snapshot only; local MMS/Archive not accessed"
    batch_audit.to_csv(batch_out, index=False, encoding="utf-8-sig")
    group_audit.to_csv(group_out, index=False, encoding="utf-8-sig")
    manifest["batch_audit"] = str(batch_out)
    manifest["group_audit"] = str(group_out)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
