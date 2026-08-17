import pandas as pd

from Mooney_Prediction_Pipeline.optimization.extract_crossdb_exact_pilot_step_history_v1 import (
    STEP_KEY,
    build_duplicate_conflict_audit,
    build_order_candidates,
    deduplicate_steps,
)


def _row(source: str, *, temp_av: float, time_av: float = 20.0) -> dict:
    return {
        "OrderID": "123",
        "ResolvedOrderID": "123",
        "BatchNumber": 1,
        "EquipmentID": "MB02",
        "GroupName": "AVR_MST",
        "StepNo": 3,
        "time_SP": 20.0,
        "time_AV": time_av,
        "temp_SP": 110.0,
        "temp_AV": temp_av,
        "rotor_SP": 50.0,
        "rotor_AV": 50.0,
        "ram_SP": 100.0,
        "ram_AV": 100.0,
        "energy_AV": 3.0,
        "ts": "2025-01-01 00:00:00",
        "SourceDatabase": source,
    }


def test_cross_database_duplicate_keeps_active_and_audits_conflict():
    raw = pd.DataFrame([
        _row("SFEPLANT", temp_av=111.0),
        _row("SFEPLANT_ARCHIVE", temp_av=112.0),
    ])
    audit = build_duplicate_conflict_audit(raw)
    assert len(audit) == 1
    assert bool(audit.iloc[0]["HasValueConflict"])
    assert "temp_AV" in audit.iloc[0]["ConflictColumns"]

    result = deduplicate_steps(raw)
    assert len(result) == 1
    assert result.iloc[0]["SourceDatabase"] == "SFEPLANT"
    assert result.iloc[0]["temp_AV"] == 111.0
    assert result[STEP_KEY].duplicated().sum() == 0


def test_order_candidates_include_year_prefixed_alias_without_replacing_original():
    scope = pd.DataFrame({
        "OrderID": ["123", "2025~456"],
        "OrderStartTime": pd.to_datetime(["2024-06-01", "2025-07-01"]),
    })
    result = build_order_candidates(scope)
    first = result.loc[result["OriginalOrderID"].eq("123"), "CandidateOrderID"].tolist()
    second = result.loc[result["OriginalOrderID"].eq("2025~456"), "CandidateOrderID"].tolist()
    assert first == ["123", "2024~123"]
    assert second == ["2025~456"]
