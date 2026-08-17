from __future__ import annotations

import pandas as pd

from Mooney_Prediction_Pipeline.v371_challenger.source_contract import deduplicate_source_rows


def test_active_precedes_newer_archive_for_same_physical_record():
    frame = pd.DataFrame([
        {"OrderID": "1", "BatchNumber": 1, "SourceDatabase": "SFEPLANT_ARCHIVE", "Timestamp": "2026-02-01", "Value": 2},
        {"OrderID": "1", "BatchNumber": 1, "SourceDatabase": "SFEPLANT", "Timestamp": "2026-01-01", "Value": 1},
    ])
    result = deduplicate_source_rows(frame, ["OrderID", "BatchNumber"])
    assert len(result) == 1
    assert result.iloc[0]["Value"] == 1


def test_newest_timestamp_wins_within_same_source():
    frame = pd.DataFrame([
        {"OrderID": "1", "BatchNumber": 1, "SourceDatabase": "ACTIVE", "Timestamp": "2026-01-01", "Value": 1},
        {"OrderID": "1", "BatchNumber": 1, "SourceDatabase": "ACTIVE", "Timestamp": "2026-02-01", "Value": 2},
    ])
    result = deduplicate_source_rows(frame, ["OrderID", "BatchNumber"])
    assert result.iloc[0]["Value"] == 2
