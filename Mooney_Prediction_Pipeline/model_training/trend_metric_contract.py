"""Formal Lab-group level and temporal-trend validation contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrendMetrics:
    groups: int
    pairs: int
    significant_pairs: int
    level_mae: float
    delta_mae: float
    delta_rmse: float
    delta_bias: float
    delta_pearson: float
    delta_spearman: float
    direction_accuracy: float
    actual_delta_std: float
    prediction_delta_std: float
    delta_std_ratio: float
    level_gate_passed: bool
    trend_gate_passed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    if len(left) < 2 or float(left.std(ddof=0)) <= 1e-12 or float(right.std(ddof=0)) <= 1e-12:
        return np.nan
    return float(left.corr(right, method=method))


def build_trend_pairs(
    groups: pd.DataFrame,
    *,
    actual_col: str,
    prediction_col: str,
    time_col: str,
    segment_cols: Iterable[str],
    significant_delta_mny: float = 2.0,
    max_gap_days: float = 30.0,
) -> pd.DataFrame:
    """Pair only consecutive independent Lab groups in the same production segment."""

    columns = [actual_col, prediction_col, time_col, *segment_cols]
    missing = sorted(set(columns).difference(groups.columns))
    if missing:
        raise ValueError(f"Trend contract missing columns: {missing}")
    frame = groups.copy()
    if "LabelGroupID" in frame and frame["LabelGroupID"].astype(str).duplicated().any():
        raise ValueError("Trend metrics require exactly one row per Lab group")
    frame[actual_col] = pd.to_numeric(frame[actual_col], errors="coerce")
    frame[prediction_col] = pd.to_numeric(frame[prediction_col], errors="coerce")
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce", utc=True)
    frame = frame.dropna(subset=columns).sort_values([*segment_cols, time_col], kind="stable")
    grouped = frame.groupby(list(segment_cols), dropna=False, sort=False)
    frame["previous_actual"] = grouped[actual_col].shift(1)
    frame["previous_prediction"] = grouped[prediction_col].shift(1)
    frame["previous_time"] = grouped[time_col].shift(1)
    frame["gap_days"] = (frame[time_col] - frame["previous_time"]).dt.total_seconds() / 86400.0
    pairs = frame.loc[
        frame["previous_actual"].notna()
        & frame["previous_prediction"].notna()
        & frame["gap_days"].between(0.0, max_gap_days, inclusive="both")
    ].copy()
    pairs["actual_delta"] = pairs[actual_col] - pairs["previous_actual"]
    pairs["prediction_delta"] = pairs[prediction_col] - pairs["previous_prediction"]
    pairs["delta_error"] = pairs["prediction_delta"] - pairs["actual_delta"]
    pairs["significant_actual_change"] = pairs["actual_delta"].abs().ge(significant_delta_mny)
    pairs["direction_correct"] = np.sign(pairs["actual_delta"]).eq(np.sign(pairs["prediction_delta"]))
    return pairs


def evaluate_level_and_trend(
    groups: pd.DataFrame,
    *,
    actual_col: str,
    prediction_col: str,
    time_col: str,
    segment_cols: Iterable[str],
    significant_delta_mny: float = 2.0,
    max_gap_days: float = 30.0,
    level_mae_limit: float = 2.5,
    minimum_pairs: int = 30,
    minimum_significant_pairs: int = 20,
    minimum_delta_pearson: float = 0.50,
    minimum_direction_accuracy: float = 0.70,
    delta_std_ratio_low: float = 0.70,
    delta_std_ratio_high: float = 1.30,
) -> tuple[TrendMetrics, pd.DataFrame]:
    valid = groups.dropna(subset=[actual_col, prediction_col]).copy()
    actual = pd.to_numeric(valid[actual_col], errors="coerce")
    prediction = pd.to_numeric(valid[prediction_col], errors="coerce")
    level_mae = float((prediction - actual).abs().mean()) if len(valid) else np.nan
    pairs = build_trend_pairs(
        valid,
        actual_col=actual_col,
        prediction_col=prediction_col,
        time_col=time_col,
        segment_cols=segment_cols,
        significant_delta_mny=significant_delta_mny,
        max_gap_days=max_gap_days,
    )
    if pairs.empty:
        values = dict(
            delta_mae=np.nan, delta_rmse=np.nan, delta_bias=np.nan,
            delta_pearson=np.nan, delta_spearman=np.nan, direction_accuracy=np.nan,
            actual_delta_std=np.nan, prediction_delta_std=np.nan, delta_std_ratio=np.nan,
        )
    else:
        significant = pairs.loc[pairs["significant_actual_change"]]
        actual_std = float(pairs["actual_delta"].std(ddof=0))
        prediction_std = float(pairs["prediction_delta"].std(ddof=0))
        values = dict(
            delta_mae=float(pairs["delta_error"].abs().mean()),
            delta_rmse=float(np.sqrt(np.mean(np.square(pairs["delta_error"])))),
            delta_bias=float(pairs["delta_error"].mean()),
            delta_pearson=_safe_corr(pairs["actual_delta"], pairs["prediction_delta"], "pearson"),
            delta_spearman=_safe_corr(pairs["actual_delta"], pairs["prediction_delta"], "spearman"),
            direction_accuracy=float(significant["direction_correct"].mean()) if len(significant) else np.nan,
            actual_delta_std=actual_std,
            prediction_delta_std=prediction_std,
            delta_std_ratio=prediction_std / actual_std if actual_std > 1e-12 else np.nan,
        )
    significant_count = int(pairs["significant_actual_change"].sum()) if not pairs.empty else 0
    ratio = values["delta_std_ratio"]
    corr = values["delta_pearson"]
    direction = values["direction_accuracy"]
    trend_passed = bool(
        len(pairs) >= minimum_pairs
        and significant_count >= minimum_significant_pairs
        and np.isfinite(corr) and corr >= minimum_delta_pearson
        and np.isfinite(direction) and direction >= minimum_direction_accuracy
        and np.isfinite(ratio) and delta_std_ratio_low <= ratio <= delta_std_ratio_high
    )
    metrics = TrendMetrics(
        groups=int(len(valid)), pairs=int(len(pairs)), significant_pairs=significant_count,
        level_mae=level_mae, **values,
        level_gate_passed=bool(np.isfinite(level_mae) and level_mae <= level_mae_limit),
        trend_gate_passed=trend_passed,
    )
    return metrics, pairs


__all__ = ["TrendMetrics", "build_trend_pairs", "evaluate_level_and_trend"]
