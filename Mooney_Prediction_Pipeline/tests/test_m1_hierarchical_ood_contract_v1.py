import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1" / "m1_hierarchical_ood_residual_contract_v1"


def test_ood_contract_is_strict_and_recommendation_closed():
    metadata = json.loads((OUTPUT / "m1_hierarchical_ood_metadata_v1.json").read_text(encoding="utf-8"))
    assert metadata["scope"] == "M1_ONLY"
    assert metadata["transition_groups"] == 239
    assert metadata["ood_eligible_groups"] == 153
    assert metadata["recommendation_gate_open_groups"] == 0
    assert metadata["uses_current_lab_value"] is False
    assert metadata["uses_future_group"] is False
    assert metadata["local_mms_access"] is False

    frame = pd.read_csv(OUTPUT / "m1_hierarchical_ood_contract.csv", low_memory=False)
    assert frame["CompoundName"].astype(str).str.startswith("M1-").all()
    assert frame["LabelGroupID"].is_unique
    assert (frame.loc[frame["ood_eligible"], "recommendation_gate"] == "BLOCKED_MODEL_NOT_PASSED").all()


def test_ood_contract_fullview_plot_exists():
    plot = OUTPUT / "m1_hierarchical_ood_actual_vs_predicted_fullview.png"
    assert plot.exists()
    assert plot.stat().st_size > 10_000
