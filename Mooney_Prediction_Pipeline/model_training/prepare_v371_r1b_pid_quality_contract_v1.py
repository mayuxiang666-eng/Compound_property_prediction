"""Prepare metadata and group label snapshot for corrected R1B training."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
SOURCE_METADATA = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_c3_active" / "feature_metadata.json"
BATCH = BASE / "v371_r1b_batch_pid_quality_snapshot_v1" / "v371_r1b_batch_pid_quality_features.csv"
GROUP = BASE / "pallet_group_training_frame_c2_prepared.csv"
OUT = BASE / "v371_r1b_batch_pid_quality_snapshot_v1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    pid_features = [
        "pid_target_top_duration_s", "pid_target_bottom_duration_s", "pid_target_total_duration_s",
        "pid_actual_top_duration_s", "pid_actual_bottom_duration_s", "pid_actual_total_duration_s",
        "pid_start_step", "pid_start_found_flag", "pid_target_temperature",
    ]
    quality_features = ["curve_statistical_outlier", "curve_if_score", "curve_risk_flag"]
    metadata["s2_features"] = list(dict.fromkeys(metadata["s2_features"] + pid_features + quality_features))
    metadata["contract_version"] = "V3.7.1-R1B-BATCH-PID-QUALITY-V1"
    metadata["pid_contract"] = {
        "duration_semantics": "within_mixer_first_difference_of_SCP-1-Step-Time",
        "start_semantics": "first_AVR_MST_actual_temperature_reaches_Target_Temperature_minus_3C",
        "end_semantics": "final_AVR_MSB_discharge_step",
        "silica_total_time_bands_s": {"165": [80, 140], "155": [130, 230], "142": [270, 330]},
    }
    metadata["curve_quality_contract"] = {
        "physical_invalid": "EXCLUDE",
        "statistical_outlier": "RETAIN_WITH_RISK_FLAG",
        "mny_used": False,
        "lab_used": False,
        "detector_fit": "TRAIN_ONLY_BY_MATERIAL_SYSTEM_AND_PHASE_ROUTE",
    }
    metadata_path = OUT / "v371_r1b_batch_pid_quality_metadata_v1.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    batch = pd.read_csv(BATCH, usecols=["LabelGroupID", *pid_features, *quality_features], low_memory=False)
    batch["LabelGroupID"] = batch["LabelGroupID"].astype(str)
    numeric = [c for c in batch.columns if c != "LabelGroupID"]
    group_features = batch.groupby("LabelGroupID", as_index=False)[numeric].mean(numeric_only=True)
    group = pd.read_csv(GROUP, low_memory=False)
    group["LabelGroupID"] = group["LabelGroupID"].astype(str)
    group = group.merge(group_features, on="LabelGroupID", how="left", validate="one_to_one")
    if group[[*pid_features, *quality_features]].isna().all(axis=0).any():
        missing = group[[*pid_features, *quality_features]].columns[group[[*pid_features, *quality_features]].isna().all(axis=0)].tolist()
        raise ValueError(f"Missing group contract features: {missing}")
    group_path = OUT / "v371_r1b_group_label_snapshot_with_pid_quality.csv"
    group.to_csv(group_path, index=False, encoding="utf-8-sig")
    print(json.dumps({"metadata": str(metadata_path), "group_frame": str(group_path), "s1_count": len(metadata["s1_features"]), "s2_count": len(metadata["s2_features"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
