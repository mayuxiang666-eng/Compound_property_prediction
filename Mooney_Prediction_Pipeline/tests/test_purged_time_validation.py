import pandas as pd

from Mooney_Prediction_Pipeline.model_training.purged_time_validation import (
    assign_purged_global_time_split,
    audit_purged_time_split,
    expanding_purged_order_folds,
)


def _frame():
    rows = []
    start = pd.Timestamp("2026-01-01", tz="UTC")
    for order in range(20):
        rows.append({"OrderID": str(order), "lab_time": start + pd.Timedelta(days=order)})
    # This order crosses both normal quantile boundaries and must be purged.
    rows.extend(
        [
            {"OrderID": "cross", "lab_time": start + pd.Timedelta(days=3)},
            {"OrderID": "cross", "lab_time": start + pd.Timedelta(days=18)},
        ]
    )
    return pd.DataFrame(rows)


def test_global_split_purges_crossing_order_and_is_chronological():
    frame = _frame()
    split = assign_purged_global_time_split(frame)
    assert set(split.loc[frame["OrderID"].eq("cross")]) == {"purged"}
    audit = audit_purged_time_split(frame, split)
    assert audit["passed"]
    assert audit["train_before_validation"]
    assert audit["validation_before_oot"]


def test_expanding_folds_have_no_order_or_time_overlap():
    frame = _frame()
    for train_index, valid_index in expanding_purged_order_folds(frame, n_splits=3):
        train = frame.iloc[train_index]
        valid = frame.iloc[valid_index]
        assert set(train["OrderID"]).isdisjoint(set(valid["OrderID"]))
        assert pd.to_datetime(train["lab_time"], utc=True).max() < pd.to_datetime(
            valid["lab_time"], utc=True
        ).min()
