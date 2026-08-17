import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1" / "m1_regularized_transition_residual_v1"


def test_regularized_residual_contract_is_m1_only_and_causal():
    metadata = json.loads((OUTPUT / "m1_regularized_transition_residual_metadata_v1.json").read_text(encoding="utf-8"))
    assert metadata["scope"] == "M1_ONLY"
    assert metadata["uses_current_lab_value"] is False
    assert metadata["uses_future_group"] is False
    assert metadata["local_mms_access"] is False
    assert metadata["transition_available_groups"] == 239
    assert metadata["model_path_ok_groups"] == 219

    frame = pd.read_csv(OUTPUT / "m1_regularized_transition_residual_predictions.csv", low_memory=False)
    assert frame["CompoundName"].astype(str).str.startswith("M1-").all()
    assert frame["LabelGroupID"].is_unique
    assert frame.loc[frame["transition_available"], "previous_MNY"].notna().all()

