"""Canonical laboratory-group evaluation contract for Mooney prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


GROUP_METRIC_CONTRACT = (
    "PREDICT_EACH_BATCH_THEN_MEAN_BY_ORDERID_PALLETID_AND_COMPARE_ONCE_TO_LAB"
)


def aggregate_batch_predictions(
    batch_rows: pd.DataFrame,
    group_col: str = "LabelGroupID",
    actual_col: str = "ActualMNY",
    prediction_col: str = "RawPrediction",
    conflict_tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Aggregate exact batch predictions to the physical unit measured by Lab."""

    required = {group_col, actual_col, prediction_col}
    missing = sorted(required.difference(batch_rows.columns))
    if missing:
        raise ValueError(f"Missing group metric columns: {missing}")
    if batch_rows.empty:
        return pd.DataFrame(
            columns=[group_col, "ActualMNY", "GroupRawPrediction", "BatchCount", "BatchPredictionStd"]
        )

    work = batch_rows.copy()
    work[actual_col] = pd.to_numeric(work[actual_col], errors="coerce")
    work[prediction_col] = pd.to_numeric(work[prediction_col], errors="coerce")
    if work[group_col].isna().any():
        raise ValueError("LabelGroupID contains missing values")
    if work[[actual_col, prediction_col]].isna().any().any():
        raise ValueError("Group metric inputs contain missing/non-numeric values")

    actual_span = work.groupby(group_col)[actual_col].agg(lambda x: float(x.max() - x.min()))
    conflicts = actual_span[actual_span.abs() > float(conflict_tolerance)]
    if not conflicts.empty:
        raise ValueError(
            "CONFLICTING_LAB_LABEL_WITHIN_GROUP: " + ", ".join(map(str, conflicts.index[:10]))
        )

    identity_columns = [
        column
        for column in (
            "OrderID", "PalletID", "CompoundName", "RecipeID", "ProcessVersion",
            "MixerLine", "material_system", "phase_route", "TemplateKey", "split",
        )
        if column in work.columns
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "ActualMNY": (actual_col, "first"),
        "GroupRawPrediction": (prediction_col, "mean"),
        "BatchCount": (prediction_col, "size"),
        "BatchPredictionStd": (prediction_col, "std"),
    }
    aggregations.update({column: (column, "first") for column in identity_columns})
    grouped = work.groupby(group_col, as_index=False).agg(**aggregations)
    grouped["BatchPredictionStd"] = grouped["BatchPredictionStd"].fillna(0.0)
    return grouped


def assert_one_row_per_lab_group(
    group_rows: pd.DataFrame,
    group_col: str = "LabelGroupID",
) -> None:
    """Fail closed unless a frame is one row per independent Lab group."""

    if group_col not in group_rows.columns:
        raise ValueError(f"Missing required group key: {group_col}")
    if group_rows[group_col].isna().any():
        raise ValueError("LabelGroupID contains missing values")
    duplicates = group_rows[group_col].duplicated(keep=False)
    if duplicates.any():
        ids = group_rows.loc[duplicates, group_col].astype(str).drop_duplicates().head(10)
        raise ValueError("DUPLICATE_LAB_GROUP_ROWS: " + ", ".join(ids))


def metric_block(
    group_rows: pd.DataFrame,
    actual_col: str = "ActualMNY",
    prediction_col: str = "GroupRawPrediction",
) -> dict[str, Any]:
    """Calculate release metrics at Lab-group granularity only."""

    if group_rows.empty:
        return {
            "groups": 0, "mae": None, "rmse": None, "bias": None, "r2": None,
            "actual_std": None, "prediction_std": None,
            "prediction_to_actual_std_ratio": None, "pearson": None, "spearman": None,
            "within_2_mny_rate": None, "within_3_mny_rate": None,
            "within_5_mny_rate": None,
        }
    required = {actual_col, prediction_col}
    missing = sorted(required.difference(group_rows.columns))
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")
    y = pd.to_numeric(group_rows[actual_col], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(group_rows[prediction_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metric inputs contain non-finite values")
    error = p - y
    actual_std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    prediction_std = float(np.std(p, ddof=1)) if len(p) > 1 else 0.0
    std_ratio = prediction_std / actual_std if actual_std > 0 else None
    pearson = float(pd.Series(y).corr(pd.Series(p), method="pearson")) if len(y) > 1 else None
    spearman = float(pd.Series(y).corr(pd.Series(p), method="spearman")) if len(y) > 1 else None
    return {
        "groups": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "bias": float(np.mean(error)),
        "r2": float(r2_score(y, p)) if len(y) > 1 else None,
        "actual_std": actual_std,
        "prediction_std": prediction_std,
        "prediction_to_actual_std_ratio": float(std_ratio) if std_ratio is not None else None,
        "pearson": pearson if pearson is None or np.isfinite(pearson) else None,
        "spearman": spearman if spearman is None or np.isfinite(spearman) else None,
        "within_2_mny_rate": float(np.mean(np.abs(error) <= 2.0)),
        "within_3_mny_rate": float(np.mean(np.abs(error) <= 3.0)),
        "within_5_mny_rate": float(np.mean(np.abs(error) <= 5.0)),
    }


def evaluate_group_frame(
    group_rows: pd.DataFrame,
    actual_col: str = "MNY",
    prediction_col: str = "RawPrediction",
    group_col: str = "LabelGroupID",
) -> dict[str, Any]:
    """Evaluate an already aggregated one-row-per-group model frame."""

    assert_one_row_per_lab_group(group_rows, group_col=group_col)
    renamed = group_rows.rename(
        columns={actual_col: "ActualMNY", prediction_col: "GroupRawPrediction"}
    )
    return metric_block(renamed)


__all__ = [
    "GROUP_METRIC_CONTRACT", "aggregate_batch_predictions",
    "assert_one_row_per_lab_group", "evaluate_group_frame", "metric_block",
]
