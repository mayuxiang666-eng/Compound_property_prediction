import pandas as pd

from Mooney_Prediction_Pipeline.model_training.prospective_blind_window import (
    BlindWindowContract,
    audit_blind_window,
)


def _contract(**changes):
    values = dict(
        window_id="BW1",
        cutoff_lab_time="2026-08-11 11:40:59",
        frozen_model_id="R3",
        frozen_model_sha256="abc",
        min_global_groups=2,
        min_t15760_groups=1,
        min_t15760_batches=1,
        min_b00458_groups=1,
        min_b00458_batches=1,
    )
    values.update(changes)
    return BlindWindowContract(**values)


def _rows():
    return pd.DataFrame([
        {"OrderID": 1, "PalletID": "A", "BatchNumber": 1, "lab_time": "2026-08-12", "CompoundName": "T15760"},
        {"OrderID": 2, "PalletID": "B", "BatchNumber": 1, "lab_time": "2026-08-13", "CompoundName": "B00458"},
    ])


def test_blind_window_unlocks_only_after_all_support_gates():
    audit = audit_blind_window(_rows(), _contract())
    assert audit["status"] == "READY_FOR_ONE_TIME_EVALUATION"


def test_pre_cutoff_and_training_overlap_are_blocked():
    rows = _rows()
    rows.loc[0, "lab_time"] = "2026-08-01"
    audit = audit_blind_window(rows, _contract(), training_group_ids={"2|B"})
    assert "BLOCKED_WINDOW_CONTAINS_PRE_CUTOFF_ROWS" in audit["reason_codes"]
    assert "BLOCKED_TRAIN_HOLDOUT_GROUP_OVERLAP" in audit["reason_codes"]


def test_empty_window_waits_for_all_pilot_support():
    empty = _rows().iloc[:0]
    audit = audit_blind_window(empty, _contract())
    assert audit["status"] == "WAITING_NEW_BLIND_WINDOW"
    assert "WAITING_T15760_BLIND_SUPPORT" in audit["reason_codes"]
    assert "WAITING_B00458_BLIND_SUPPORT" in audit["reason_codes"]
