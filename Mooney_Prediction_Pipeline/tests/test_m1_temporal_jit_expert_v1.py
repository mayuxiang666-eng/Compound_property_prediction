import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1" / "m1_temporal_jit_v1"


def test_jit_contract_is_temporal_m1_only():
    metadata = json.loads((OUTPUT / "m1_temporal_jit_metadata_v1.json").read_text(encoding="utf-8"))
    assert metadata["scope"] == "M1_ONLY"
    assert metadata["uses_lab_value_for_current_query"] is False
    assert metadata["uses_future_group"] is False
    assert metadata["uses_local_mms"] is False
    assert metadata["stage3_calibration_enabled"] is False

    frame = pd.read_csv(OUTPUT / "m1_temporal_jit_group_predictions.csv", low_memory=False)
    assert frame["CompoundName"].astype(str).str.startswith("M1-").all()
    assert frame["LabelGroupID"].is_unique
    assert frame["neighbor_scope"].notna().all()
    assert {"JITPrediction", "JITP10", "JITP90", "ood_status"}.issubset(frame.columns)

