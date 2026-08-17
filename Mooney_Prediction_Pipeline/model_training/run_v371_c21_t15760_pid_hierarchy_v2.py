"""C2.1 v2: explicit white-silica upper/lower PID hierarchy contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training import run_v371_c21_t15760_expert_step_delta as v1


OUTPUT = v1.BASE / "v371_c21_t15760_pid_hierarchy_v2"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_c21_t15760_pid_hierarchy_v2"
PID_CONTRACT = (
    "MST1-2 Loading; MST3 Dry; MST4 Oil; MST5-9 Wet; "
    "MST10 upper-mixer PID; MST11 post-PID tail; "
    "MSB1-3 lower-mixer PID continuation substeps"
)


_original_build = v1.build_expert_features


def build_pid_hierarchy_features(frame):
    result, features = _original_build(frame)
    rename = {}
    for feature in features:
        if feature.startswith("expert__pid__"):
            rename[feature] = feature.replace("expert__pid__", "expert__pid_upper_mst__", 1)
        elif feature.startswith("expert__msb_s"):
            rename[feature] = feature.replace("expert__msb_s", "expert__pid_lower_msb_s", 1)
    result = result.rename(columns=rename)
    return result, [rename.get(feature, feature) for feature in features]


def main() -> None:
    v1.OUTPUT = OUTPUT
    v1.MODEL_OUTPUT = MODEL_OUTPUT
    v1.build_expert_features = build_pid_hierarchy_features
    v1.main()

    feature_path = OUTPUT / "c21_feature_contract.csv"
    import pandas as pd
    feature = pd.read_csv(feature_path)
    feature["expert_phase_contract"] = PID_CONTRACT
    feature.to_csv(feature_path, index=False)

    summary_path = OUTPUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["contract"] = "V3.7.1-C2.1-T15760-PID-HIERARCHY-V2"
    summary["pid_hierarchy"] = {
        "overall": "Silica PID includes both upper and lower mixing",
        "upper_pid": "MST Step10; conservative adjustment policy",
        "lower_pid": "MSB Step1-3; primary silica duration/discharge-temperature adjustment area",
        "tail": "MST Step11 remains locked post-PID auxiliary step",
    }
    summary["expert_mapping"] = PID_CONTRACT
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    model_path = MODEL_OUTPUT / "c21_t15760_expert_step_delta_shadow.joblib"
    bundle = joblib.load(model_path)
    bundle["contract"] = "V3.7.1-C2.1-T15760-PID-HIERARCHY-V2"
    bundle["expert_mapping"] = PID_CONTRACT
    bundle["pid_hierarchy"] = summary["pid_hierarchy"]
    joblib.dump(bundle, MODEL_OUTPUT / "c21_t15760_pid_hierarchy_shadow.joblib")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
