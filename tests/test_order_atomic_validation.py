import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.model_training.order_atomic_validation import (
    assign_recipe_order_atomic_splits,
    audit_order_split,
    expanding_window_order_folds,
)


def _frame():
    rows = []
    for recipe in ("R1", "R2"):
        for order_no in range(10):
            for group_no in range(2):
                rows.append(
                    {
                        "RecipeID": recipe,
                        "OrderID": f"{recipe}-O{order_no}",
                        "LabelGroupID": f"{recipe}-O{order_no}-G{group_no}",
                        "lab_time": pd.Timestamp("2026-01-01")
                        + pd.Timedelta(days=order_no, hours=group_no),
                    }
                )
    return pd.DataFrame(rows)


def test_outer_split_keeps_complete_orders_together():
    frame = _frame()
    split = assign_recipe_order_atomic_splits(frame)
    audit = audit_order_split(frame, split)
    assert audit["passed"]
    assert set(split) == {"train", "val", "oot"}
    assert frame.assign(split=split).groupby("OrderID")["split"].nunique().max() == 1


def test_expanding_folds_keep_complete_orders_together():
    frame = _frame().sort_values("lab_time").reset_index(drop=True)
    folds = expanding_window_order_folds(frame, n_splits=3)
    assert folds
    for train, valid in folds:
        train_orders = set(frame.iloc[train]["OrderID"])
        valid_orders = set(frame.iloc[valid]["OrderID"])
        assert train_orders.isdisjoint(valid_orders)


def test_outer_split_rejects_multi_recipe_order():
    frame = _frame()
    frame.loc[0, "OrderID"] = frame.loc[20, "OrderID"]
    try:
        assign_recipe_order_atomic_splits(frame)
    except ValueError as error:
        assert "multiple RecipeID" in str(error)
    else:
        raise AssertionError("Expected multi-recipe physical order to fail")
