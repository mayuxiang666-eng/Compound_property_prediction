"""Chronological validation helpers shared by corrected Mooney experts."""

from __future__ import annotations

import math

import numpy as np


def expanding_window_folds(
    n_samples: int,
    n_splits: int = 5,
    min_train_fraction: float = 0.20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic expanding-window folds for time-sorted rows."""

    n_samples = int(n_samples)
    n_splits = int(n_splits)
    if n_samples < 3:
        raise ValueError("Expanding-window validation requires at least three rows")
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    if not 0.0 < float(min_train_fraction) < 1.0:
        raise ValueError("min_train_fraction must be between zero and one")

    warmup = max(2, int(math.ceil(n_samples * float(min_train_fraction))))
    if warmup >= n_samples:
        warmup = n_samples - 1
    validation_indices = np.arange(warmup, n_samples, dtype=int)
    blocks = np.array_split(validation_indices, min(n_splits, len(validation_indices)))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for block in blocks:
        if len(block) == 0:
            continue
        train = np.arange(0, int(block[0]), dtype=int)
        valid = np.asarray(block, dtype=int)
        if len(train) == 0 or int(train.max()) >= int(valid.min()):
            raise RuntimeError("Invalid expanding-window chronology")
        folds.append((train, valid))
    if not folds:
        raise ValueError("No expanding-window folds could be constructed")
    return folds


def fold_audit(folds: list[tuple[np.ndarray, np.ndarray]]) -> list[dict[str, int]]:
    """Produce a JSON-safe chronology audit for a fold collection."""

    return [
        {
            "fold": fold_no,
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "train_max_index": int(train.max()),
            "valid_min_index": int(valid.min()),
            "valid_max_index": int(valid.max()),
        }
        for fold_no, (train, valid) in enumerate(folds, start=1)
    ]


__all__ = ["expanding_window_folds", "fold_audit"]
