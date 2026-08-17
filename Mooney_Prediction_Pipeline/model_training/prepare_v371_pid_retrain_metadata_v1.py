"""Create an explicit metadata contract for the corrected PID C1 retrain."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
SOURCE = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_c3_active" / "feature_metadata.json"
OUTPUT = BASE / "v371_retrain_pid_snapshot_v1" / "v371_pid_retrain_metadata_v1.json"


def main():
    metadata = json.loads(SOURCE.read_text(encoding="utf-8"))
    pid_features = [
        "pid_target_top_duration_s", "pid_target_bottom_duration_s", "pid_target_total_duration_s",
        "pid_actual_top_duration_s", "pid_actual_bottom_duration_s", "pid_actual_total_duration_s",
        "pid_start_step_mean", "pid_start_found_rate", "pid_target_temperature",
    ]
    metadata["s2_features"] = list(dict.fromkeys(metadata["s2_features"] + pid_features))
    metadata["contract_version"] = "V3.7.1-C1-PID-DURATION-CORRECTED-V1"
    metadata["pid_contract"] = {
        "duration_semantics": "within_mixer_first_difference_of_SCP-1-Step-Time",
        "start_semantics": "first_AVR_MST_actual_temperature_reaches_Target_Temperature_minus_3C",
        "end_semantics": "final_AVR_MSB_discharge_step",
        "silica_total_time_bands_s": {"165": [80, 140], "155": [130, 230], "142": [270, 330]},
        "stage3_calibration_enabled": False,
    }
    metadata["source_metadata_sha256"] = "see_training_manifest"
    OUTPUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "s2_feature_count": len(metadata["s2_features"]), "pid_features": pid_features}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
