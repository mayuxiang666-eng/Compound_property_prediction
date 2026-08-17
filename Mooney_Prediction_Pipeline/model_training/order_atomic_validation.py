"""Order-atomic chronological contracts for R5 Mooney experiments."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def expanding_window_order_folds(
    frame: pd.DataFrame,
    *,
    time_col: str = "lab_time",
    order_col: str = "OrderID",
    n_splits: int = 5,
    min_train_fraction: float = 0.20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding OOF folds without splitting any physical order."""

    required = {time_col, order_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing order fold columns: {missing}")
    work = frame.loc[:, [time_col, order_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    if work.isna().any().any():
        raise ValueError("Order fold input contains missing order or time")
    order_events = (
        work.groupby(order_col, as_index=False)
        .agg(_event_time=(time_col, "max"))
        .sort_values(["_event_time", order_col], kind="stable")
        .reset_index(drop=True)
    )
    n_orders = len(order_events)
    if n_orders < 3:
        raise ValueError("Order-atomic expanding validation requires at least three orders")
    warmup = max(2, int(math.ceil(n_orders * float(min_train_fraction))))
    warmup = min(warmup, n_orders - 1)
    validation_positions = np.arange(warmup, n_orders, dtype=int)
    blocks = np.array_split(validation_positions, min(int(n_splits), len(validation_positions)))
    values = work[order_col].astype(str)
    order_events[order_col] = order_events[order_col].astype(str)
    folds = []
    for block in blocks:
        if not len(block):
            continue
        train_ids = set(order_events.iloc[: int(block[0])][order_col])
        valid_ids = set(order_events.iloc[block][order_col])
        if train_ids & valid_ids:
            raise AssertionError("Order overlap in expanding fold")
        train = np.flatnonzero(values.isin(train_ids).to_numpy())
        valid = np.flatnonzero(values.isin(valid_ids).to_numpy())
        if not len(train) or not len(valid):
            raise ValueError("Order fold produced an empty partition")
        folds.append((train, valid))
    if not folds:
        raise ValueError("No order-atomic folds were generated")
    return folds


def assign_recipe_order_atomic_splits(
    frame: pd.DataFrame,
    *,
    time_col: str = "lab_time",
    order_col: str = "OrderID",
    recipe_col: str = "RecipeID",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    minimum_three_way_orders: int = 7,
) -> pd.Series:
    """Assign whole orders to per-recipe chronological train/val/OOT splits."""

    required = {time_col, order_col, recipe_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing outer split columns: {missing}")
    work = frame.loc[:, [time_col, order_col, recipe_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    if work.isna().any().any():
        raise ValueError("Outer split input contains missing order, recipe, or time")
    recipe_per_order = work.groupby(order_col)[recipe_col].nunique()
    if int(recipe_per_order.max()) != 1:
        raise ValueError("A physical OrderID maps to multiple RecipeID values")

    order_events = (
        work.groupby([recipe_col, order_col], as_index=False)
        .agg(_event_time=(time_col, "max"))
        .sort_values([recipe_col, "_event_time", order_col], kind="stable")
    )
    global_train_cut = order_events["_event_time"].quantile(train_fraction)
    global_val_cut = order_events["_event_time"].quantile(
        train_fraction + validation_fraction
    )
    assignment: dict[str, str] = {}
    for _, recipe_orders in order_events.groupby(recipe_col, sort=False):
        recipe_orders = recipe_orders.reset_index(drop=True)
        n_orders = len(recipe_orders)
        if n_orders >= minimum_three_way_orders:
            n_train = max(1, int(math.floor(n_orders * train_fraction)))
            n_val = max(1, int(math.floor(n_orders * validation_fraction)))
            if n_train + n_val >= n_orders:
                n_train = max(1, n_orders - 2)
                n_val = 1
            labels = np.full(n_orders, "oot", dtype=object)
            labels[:n_train] = "train"
            labels[n_train : n_train + n_val] = "val"
        else:
            labels = np.select(
                [
                    recipe_orders["_event_time"].le(global_train_cut),
                    recipe_orders["_event_time"].le(global_val_cut),
                ],
                ["train", "val"],
                default="oot",
            )
        for order_id, label in zip(recipe_orders[order_col].astype(str), labels):
            assignment[order_id] = str(label)

    result = work[order_col].astype(str).map(assignment)
    if result.isna().any():
        raise AssertionError("An order was not assigned to a split")
    audit = frame.assign(_split=result).groupby(order_col)["_split"].nunique()
    if int(audit.max()) != 1:
        raise AssertionError("Order-atomic outer split contract failed")
    return result


def audit_order_split(
    frame: pd.DataFrame,
    split: pd.Series | str,
    *,
    order_col: str = "OrderID",
) -> dict[str, object]:
    """Return overlap evidence for an outer order-atomic split."""

    split_values = frame[split] if isinstance(split, str) else split
    work = pd.DataFrame({order_col: frame[order_col].astype(str), "split": split_values})
    order_sets = {
        name: set(work.loc[work["split"].eq(name), order_col])
        for name in ("train", "val", "oot")
    }
    overlaps = {
        "train_val": len(order_sets["train"] & order_sets["val"]),
        "train_oot": len(order_sets["train"] & order_sets["oot"]),
        "val_oot": len(order_sets["val"] & order_sets["oot"]),
    }
    return {
        "rows": int(len(work)),
        "orders": int(work[order_col].nunique()),
        "split_rows": work["split"].value_counts().to_dict(),
        "split_orders": {
            name: len(values) for name, values in order_sets.items()
        },
        "overlaps": overlaps,
        "passed": all(value == 0 for value in overlaps.values()),
    }


__all__ = [
    "assign_recipe_order_atomic_splits",
    "audit_order_split",
    "expanding_window_order_folds",
]
