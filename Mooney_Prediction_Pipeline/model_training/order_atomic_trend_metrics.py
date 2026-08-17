"""Strict within-OrderID Lab/Pallet trend metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.model_training.run_v371_r6_recipe_delta import level_metrics


def within_order_pairs(frame: pd.DataFrame, prediction_col: str, significant_delta: float = 2.0) -> pd.DataFrame:
    time_col = "production_start" if "production_start" in frame.columns else "lab_time"
    ordered = frame.sort_values(["OrderID", time_col, "PalletID", "LabelGroupID"], kind="stable")
    rows = []
    for order_id, part in ordered.groupby("OrderID", sort=False):
        previous = None
        for current in part.itertuples(index=False):
            if previous is not None:
                actual = float(current.MNY) - float(previous.MNY)
                predicted = float(getattr(current, prediction_col)) - float(getattr(previous, prediction_col))
                rows.append(
                    {
                        "OrderID": order_id, "from_group": previous.LabelGroupID,
                        "to_group": current.LabelGroupID, "split": current.split,
                        "actual_delta": actual, "prediction_delta": predicted,
                        "significant": abs(actual) >= significant_delta,
                    }
                )
            previous = current
    return pd.DataFrame(rows)


def order_trend_metrics(frame: pd.DataFrame, prediction_col: str) -> dict[str, float | int]:
    pairs = within_order_pairs(frame, prediction_col)
    if pairs.empty:
        return {"pairs": 0, "significant_pairs": 0, "delta_pearson": np.nan, "direction_accuracy": np.nan, "delta_std_ratio": np.nan}
    actual = pairs["actual_delta"].to_numpy(dtype=float)
    predicted = pairs["prediction_delta"].to_numpy(dtype=float)
    significant = pairs["significant"].to_numpy(dtype=bool)
    return {
        "pairs": int(len(pairs)),
        "significant_pairs": int(significant.sum()),
        "delta_pearson": float(np.corrcoef(actual, predicted)[0, 1]) if np.std(actual) > 0 and np.std(predicted) > 0 else np.nan,
        "direction_accuracy": float(np.mean(np.sign(actual[significant]) == np.sign(predicted[significant]))) if significant.any() else np.nan,
        "delta_std_ratio": float(np.std(predicted, ddof=0) / np.std(actual, ddof=0)) if np.std(actual) > 0 else np.nan,
    }


def candidate_score(frame: pd.DataFrame, prediction) -> tuple[float, dict, dict]:
    work = frame.copy()
    work["Prediction"] = prediction
    val = work.loc[work["split"].eq("val")]
    level = level_metrics(val, "Prediction")
    trend = order_trend_metrics(val, "Prediction")
    corr = trend["delta_pearson"] if pd.notna(trend["delta_pearson"]) else -1.0
    direction = trend["direction_accuracy"] if pd.notna(trend["direction_accuracy"]) else 0.0
    ratio = trend["delta_std_ratio"] if pd.notna(trend["delta_std_ratio"]) else 0.0
    score = float(level["mae"])
    score += 1.5 * max(0.0, 0.50 - float(corr))
    score += 1.5 * max(0.0, 0.65 - float(direction))
    score += 0.75 * max(0.0, 0.70 - float(ratio))
    score += 0.75 * max(0.0, float(ratio) - 1.30)
    return score, level, trend


def metric_record(frame: pd.DataFrame, prediction_col: str, split: str, candidate: str) -> dict:
    part = frame.loc[frame["split"].eq(split)]
    return {"candidate": candidate, "split": split, **level_metrics(part, prediction_col), **order_trend_metrics(part, prediction_col)}


__all__ = ["candidate_score", "metric_record", "order_trend_metrics", "within_order_pairs"]
