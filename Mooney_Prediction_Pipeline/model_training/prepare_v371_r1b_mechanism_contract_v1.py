"""Prepare R1B metadata and labels for mechanism-feature ablation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
SOURCE_METADATA = BASE / "v371_r1b_batch_pid_quality_snapshot_v1" / "v371_r1b_batch_pid_quality_metadata_v1.json"
BATCH = BASE / "v371_r1b_batch_pid_mechanism_snapshot_v1" / "v371_r1b_batch_pid_mechanism_features.csv"
GROUP = BASE / "v371_r1b_batch_pid_quality_snapshot_v1" / "v371_r1b_group_label_snapshot_with_pid_quality.csv"
OUT = BASE / "v371_r1b_batch_pid_mechanism_snapshot_v1"


def main() -> None:
    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    mechanism = [
        "mech_dry_to_oil_temp_rise", "mech_oil_to_wet_temp_rise",
        "mech_dry_thermal_exposure", "mech_wet_thermal_exposure",
        "mech_wet_rotor_time", "mech_pid_msb_total_duration",
        "mech_oil_hot_wet_short_interaction",
    ]
    metadata["s2_features"] = list(dict.fromkeys(metadata["s2_features"] + mechanism))
    metadata["contract_version"] = "V3.7.1-R1B-BATCH-PID-MECHANISM-V1"
    metadata["mechanism_feature_contract"] = {"features": mechanism, "mny_used": False, "lab_used": False, "causal_claim": False}
    metadata_path = OUT / "v371_r1b_batch_pid_mechanism_metadata_v1.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    batch = pd.read_csv(BATCH, usecols=["LabelGroupID", *mechanism], low_memory=False)
    batch["LabelGroupID"] = batch["LabelGroupID"].astype(str)
    group_features = batch.groupby("LabelGroupID", as_index=False)[mechanism].mean(numeric_only=True)
    group = pd.read_csv(GROUP, low_memory=False)
    group["LabelGroupID"] = group["LabelGroupID"].astype(str)
    group = group.merge(group_features, on="LabelGroupID", how="left", validate="one_to_one")
    group_path = OUT / "v371_r1b_group_label_snapshot_with_mechanism.csv"
    group.to_csv(group_path, index=False, encoding="utf-8-sig")
    print(json.dumps({"metadata": str(metadata_path), "group_frame": str(group_path), "mechanism_count": len(mechanism), "s2_count": len(metadata["s2_features"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
