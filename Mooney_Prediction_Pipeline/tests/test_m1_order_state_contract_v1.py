import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "reports"
    / "v371_challenger"
    / "global_rds_active_v1"
    / "v371_r1b_m1_order_state_snapshot_v1"
    / "m1_only_order_state_manifest.json"
)


def test_order_state_manifest_is_m1_only_and_causal():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["scope"] == "M1_ONLY"
    assert manifest["uses_lab_value"] is False
    assert manifest["uses_future_batch"] is False
    assert manifest["database_access_during_build"] is False
    assert manifest["local_mms_access"] is False
    assert manifest["archive_status"] == "BLOCKED_ARCHIVE_NOT_REPLICATED"
    assert 0.0 < manifest["prior_available_rate"] < 1.0
    assert manifest["prior_same_template_rate_among_available"] == 1.0

