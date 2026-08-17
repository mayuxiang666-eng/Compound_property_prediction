"""Add target-independent process mechanism features to the R1B snapshot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
SOURCE = BASE / "v371_r1b_batch_pid_quality_snapshot_v1" / "v371_r1b_batch_pid_quality_features.csv"
OUT = BASE / "v371_r1b_batch_pid_mechanism_snapshot_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, np.nan), errors="coerce")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE, low_memory=False)
    dry_temp = num(frame, "Stage2_DryMixing_temp_Mean")
    oil_temp = num(frame, "Stage3_OilLoading_temp_Mean")
    wet_temp = num(frame, "Stage4_WetMixing_temp_Mean")
    wet_duration = num(frame, "Stage4_WetMixing_Duration")
    wet_rotor = num(frame, "Stage4_WetMixing_RotorSpeed_Mean")
    dry_duration = num(frame, "Stage2_DryMixing_Duration")

    frame["mech_dry_to_oil_temp_rise"] = oil_temp - dry_temp
    frame["mech_oil_to_wet_temp_rise"] = wet_temp - oil_temp
    frame["mech_dry_thermal_exposure"] = dry_duration * (dry_temp - 100.0).clip(lower=0.0)
    frame["mech_wet_thermal_exposure"] = wet_duration * (wet_temp - 100.0).clip(lower=0.0)
    frame["mech_wet_rotor_time"] = wet_duration * wet_rotor
    frame["mech_pid_msb_total_duration"] = num(frame, "pid_actual_total_duration_s")
    # High oil temperature combined with short Wet exposure is a risk proxy,
    # not a causal label and not a recommendation rule by itself.
    frame["mech_oil_hot_wet_short_interaction"] = (oil_temp - 140.0).clip(lower=0.0) * (60.0 - wet_duration).clip(lower=0.0)

    output = OUT / "v371_r1b_batch_pid_mechanism_features.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    manifest = {
        "contract_version": "V3.7.1-R1B-BATCH-PID-MECHANISM-V1",
        "source": str(SOURCE), "source_sha256": sha256(SOURCE), "output_sha256": sha256(output),
        "mechanism_features": [c for c in frame.columns if c.startswith("mech_")],
        "mny_used": False, "lab_used": False, "causal_claim": False,
        "source_policy": "Cloud RDS Active snapshot only; local MMS/Archive not accessed",
        "stage3_calibration_enabled": False, "automatic_control": "DISABLED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
