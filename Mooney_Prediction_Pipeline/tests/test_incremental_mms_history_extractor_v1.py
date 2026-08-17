import json
from pathlib import Path

import pytest

from Mooney_Prediction_Pipeline.data_processing.incremental_mms_history_extractor_v1 import read_credentials


def test_missing_local_mms_credentials_fail_closed(tmp_path: Path):
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"RDS_APAC": {"host": "example", "user": "u", "password": "p"}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="BLOCKED_MISSING_CREDENTIAL_SECTION:HF_MMS"):
        read_credentials(path, "HF_MMS")


def test_mms_extractor_sources_are_not_production_write_targets():
    source = Path(__file__).resolve().parents[1] / "data_processing" / "incremental_mms_history_extractor_v1.py"
    text = source.read_text(encoding="utf-8")
    assert "INSERT INTO" not in text.upper()
    assert "UPDATE " not in text.upper()
    assert "DELETE FROM" not in text.upper()
    assert "SFEPLANT_ARCHIVE" in text


def test_batchdata_path_is_compact_and_step_phase_scoped():
    source = Path(__file__).resolve().parents[1] / "data_processing" / "incremental_mms_history_extractor_v1.py"
    text = source.read_text(encoding="utf-8")
    assert "query_batchdata_compact" in text
    assert "STEP_PHASE_RELEVANT_EAV_COLUMNS_ONLY" in text
    assert "t.OrderID, t.BatchNumber, t.EquipmentID, t.GroupName, t.StepNo" in text
