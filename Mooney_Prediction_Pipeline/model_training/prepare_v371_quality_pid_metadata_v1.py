"""Create the V3.7.1 quality-risk feature metadata contract."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
SOURCE = BASE / "v371_retrain_pid_snapshot_v1" / "v371_pid_retrain_metadata_v1.json"
OUTPUT = BASE / "v371_retrain_quality_pid_snapshot_v1" / "v371_quality_pid_retrain_metadata_v1.json"


def main() -> None:
    metadata = json.loads(SOURCE.read_text(encoding="utf-8"))
    quality_features = [
        "curve_batch_count", "curve_statistical_outlier_batch_count",
        "curve_if_score_mean", "curve_if_score_min",
        "curve_statistical_outlier_rate", "curve_valid_batch_rate",
        "curve_quality_risk_flag",
    ]
    metadata["s2_features"] = list(dict.fromkeys(metadata["s2_features"] + quality_features))
    metadata["contract_version"] = "V3.7.1-C1-PID-CURVE-QUALITY-RISK-V1"
    metadata["curve_quality_contract"] = {
        "version": "CURVE-QUALITY-V1-PHYSICAL-FAIL-CLOSED-STATISTICAL-RISK",
        "physical_invalid_curve_policy": "EXCLUDE_FROM_MODEL_POPULATION",
        "statistical_outlier_policy": "RETAIN_AND_ADD_RISK_FEATURES",
        "mny_used": False,
        "lab_used": False,
        "isolation_forest_fit": "TRAIN_ONLY_BY_MATERIAL_SYSTEM_AND_PHASE_ROUTE",
        "detector_scaling": "TRAIN_ONLY_ROBUST_SCALER_FOR_ISOLATION_FOREST",
    }
    OUTPUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "s2_feature_count": len(metadata["s2_features"]), "quality_features": quality_features}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
