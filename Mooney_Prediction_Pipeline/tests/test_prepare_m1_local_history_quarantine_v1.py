import pandas as pd

from Mooney_Prediction_Pipeline.model_training.prepare_m1_local_history_quarantine_v1 import (
    UNVERIFIED_LAB_STATUS,
    build_quarantine,
    normalize_identifier,
)


def _curve(samples=30):
    return "".join(f"{value:04X}" for value in range(samples))


def _source_row(order="2024~100", batch=1, pallet=10, compound="M1-T15760---- 01 001"):
    return {
        "OrderID": order,
        "BatchNumber": batch,
        "PalletID": pallet,
        "CompoundName": compound,
        "CompoundDescription": "9922A1576030_PV27_002",
        "MixerLine": "MB02",
        "OrderStartTime": "2024-06-01 00:00:00",
        "Target_Temperature": 165,
        "MNY": 55.0,
        "test_result_start_time": "2024-06-02 00:00:00",
        "temp": _curve(),
        "power": _curve(),
        "Torque": _curve(),
        "RotorSpeed": _curve(),
        "WayofRam": _curve(),
    }


def test_normalize_identifier_removes_legacy_year_prefix():
    assert normalize_identifier("2024~1273702") == "1273702"
    assert normalize_identifier(1273702.0) == "1273702"


def test_quarantine_protects_current_snapshot_and_unverified_lab():
    source = pd.DataFrame([_source_row(), _source_row(order="2024~101", batch=2, pallet=11)])
    current_batch = pd.DataFrame([{"OrderID": "100", "BatchNumber": 1, "PalletID": "10"}])
    current_group = pd.DataFrame([
        {"OrderID": "999", "PalletID": "99", "split": "oot_test"},
    ])
    audit, candidates, manifest = build_quarantine(source, current_batch, current_group)
    overlap = audit.loc[audit["PhysicalBatchKey"].eq("100|1")].iloc[0]
    assert "BLOCKED_CURRENT_SNAPSHOT_PREFERRED" in overlap["phase_rebuild_block_reasons"]
    assert set(candidates["PhysicalBatchKey"]) == {"101|2"}
    assert not audit["training_eligible"].any()
    assert audit["historical_lab_status"].eq(UNVERIFIED_LAB_STATUS).all()
    assert manifest["database_access_during_build"] is False


def test_quarantine_blocks_fixed_holdout_and_curve_length_mismatch():
    row_holdout = _source_row(order="2024~200", pallet=20)
    row_bad = _source_row(order="2024~201", batch=2, pallet=21)
    row_bad["power"] = _curve(29)
    source = pd.DataFrame([row_holdout, row_bad])
    current_batch = pd.DataFrame(columns=["OrderID", "BatchNumber", "PalletID"])
    current_group = pd.DataFrame([
        {"OrderID": "200", "PalletID": "20", "split": "validation"},
    ])
    audit, candidates, _ = build_quarantine(source, current_batch, current_group)
    holdout = audit.loc[audit["PhysicalBatchKey"].eq("200|1")].iloc[0]
    mismatch = audit.loc[audit["PhysicalBatchKey"].eq("201|2")].iloc[0]
    assert "BLOCKED_FIXED_VALIDATION_OOT_GROUP" in holdout["phase_rebuild_block_reasons"]
    assert "BLOCKED_CORE_SIGNAL_LENGTH_MISMATCH" in mismatch["phase_rebuild_block_reasons"]
    assert candidates.empty
