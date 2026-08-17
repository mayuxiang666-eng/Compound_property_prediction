"""Strict chronological splits for grouped Mooney experiments.

The unit being assigned is a physical OrderID, while the time boundary is
evaluated against the complete interval occupied by all Lab/Pallet groups in
that order.  Orders crossing a boundary are purged instead of being allowed to
put an early group in validation and a later group in training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _order_intervals(
    frame: pd.DataFrame,
    *,
    time_col: str,
    order_col: str,
) -> pd.DataFrame:
    required = {time_col, order_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing strict time split columns: {missing}")
    work = frame.loc[:, [time_col, order_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    if work[time_col].isna().any() or work[order_col].isna().any():
        raise ValueError("Strict time split input contains missing order or time")
    work[order_col] = work[order_col].astype(str)
    return (
        work.groupby(order_col, as_index=False)
        .agg(order_start=(time_col, "min"), order_end=(time_col, "max"))
        .sort_values(["order_start", "order_end", order_col], kind="stable")
        .reset_index(drop=True)
    )


def assign_purged_global_time_split(
    frame: pd.DataFrame,
    *,
    time_col: str = "lab_time",
    order_col: str = "OrderID",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> pd.Series:
    """Assign whole orders to train/val/OOT and purge boundary crossers.

    Boundaries are selected from order start times.  The resulting contract is:

    ``max(train order_end) < min(val order_start)`` and
    ``max(val order_end) < min(oot order_start)``.
    """

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("validation_fraction leaves no OOT interval")
    intervals = _order_intervals(frame, time_col=time_col, order_col=order_col)
    if len(intervals) < 10:
        raise ValueError("Strict global split requires at least ten orders")
    train_cut = intervals["order_start"].quantile(train_fraction)
    oot_cut = intervals["order_start"].quantile(train_fraction + validation_fraction)

    intervals["split"] = "purged"
    intervals.loc[intervals["order_end"].lt(train_cut), "split"] = "train"
    intervals.loc[
        intervals["order_start"].ge(train_cut)
        & intervals["order_end"].lt(oot_cut),
        "split",
    ] = "val"
    intervals.loc[intervals["order_start"].ge(oot_cut), "split"] = "oot"
    mapping = dict(zip(intervals[order_col], intervals["split"]))
    result = frame[order_col].astype(str).map(mapping)
    if result.isna().any():
        raise AssertionError("An OrderID was not assigned by the strict time split")
    return result


def expanding_purged_order_folds(
    frame: pd.DataFrame,
    *,
    time_col: str = "lab_time",
    order_col: str = "OrderID",
    n_splits: int = 5,
    min_train_fraction: float = 0.20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding folds with strict event time and whole-order purging."""

    intervals = _order_intervals(frame, time_col=time_col, order_col=order_col)
    n_orders = len(intervals)
    if n_orders < 10:
        raise ValueError("Purged expanding validation requires at least ten orders")
    starts = intervals["order_start"]
    first = float(min_train_fraction)
    edges = np.linspace(first, 1.0, int(n_splits) + 1)
    order_values = frame[order_col].astype(str)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for lower_q, upper_q in zip(edges[:-1], edges[1:]):
        lower = starts.quantile(float(lower_q))
        upper = starts.quantile(float(upper_q))
        train_ids = set(
            intervals.loc[intervals["order_end"].lt(lower), order_col]
        )
        if upper_q >= 1.0:
            valid_mask = intervals["order_start"].ge(lower)
        else:
            valid_mask = intervals["order_start"].ge(lower) & intervals[
                "order_end"
            ].lt(upper)
        valid_ids = set(intervals.loc[valid_mask, order_col])
        if train_ids & valid_ids:
            raise AssertionError("Order overlap in purged expanding fold")
        train_index = np.flatnonzero(order_values.isin(train_ids).to_numpy())
        valid_index = np.flatnonzero(order_values.isin(valid_ids).to_numpy())
        if len(train_index) and len(valid_index):
            folds.append((train_index, valid_index))
    if not folds:
        raise ValueError("No purged expanding folds were generated")
    return folds


def audit_purged_time_split(
    frame: pd.DataFrame,
    split: pd.Series | str,
    *,
    time_col: str = "lab_time",
    order_col: str = "OrderID",
) -> dict[str, object]:
    """Return order overlap, chronology, and purged-row evidence."""

    split_values = frame[split] if isinstance(split, str) else split
    work = frame.loc[:, [time_col, order_col]].copy()
    work["split"] = split_values.to_numpy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    order_split_counts = work.groupby(order_col)["split"].nunique()
    intervals = (
        work.loc[work["split"].isin(["train", "val", "oot"])]
        .groupby(["split", order_col], as_index=False)
        .agg(order_start=(time_col, "min"), order_end=(time_col, "max"))
    )
    bounds = {}
    for name in ("train", "val", "oot"):
        part = intervals.loc[intervals["split"].eq(name)]
        bounds[name] = {
            "orders": int(part[order_col].nunique()),
            "min_start": part["order_start"].min().isoformat() if len(part) else None,
            "max_end": part["order_end"].max().isoformat() if len(part) else None,
        }
    train_val_ok = bool(
        len(intervals.loc[intervals["split"].eq("train")])
        and len(intervals.loc[intervals["split"].eq("val")])
        and intervals.loc[intervals["split"].eq("train"), "order_end"].max()
        < intervals.loc[intervals["split"].eq("val"), "order_start"].min()
    )
    val_oot_ok = bool(
        len(intervals.loc[intervals["split"].eq("val")])
        and len(intervals.loc[intervals["split"].eq("oot")])
        and intervals.loc[intervals["split"].eq("val"), "order_end"].max()
        < intervals.loc[intervals["split"].eq("oot"), "order_start"].min()
    )
    return {
        "rows": int(len(work)),
        "order_atomic": bool(int(order_split_counts.max()) == 1),
        "split_rows": work["split"].value_counts().to_dict(),
        "split_orders": work.groupby("split")[order_col].nunique().to_dict(),
        "purged_rows": int(work["split"].eq("purged").sum()),
        "bounds": bounds,
        "train_before_validation": train_val_ok,
        "validation_before_oot": val_oot_ok,
        "passed": bool(
            int(order_split_counts.max()) == 1 and train_val_ok and val_oot_ok
        ),
    }


__all__ = [
    "assign_purged_global_time_split",
    "audit_purged_time_split",
    "expanding_purged_order_folds",
]
