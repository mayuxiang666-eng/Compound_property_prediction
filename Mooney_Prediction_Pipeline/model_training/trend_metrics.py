# ============================================================================
# V3.2 Trend Preservation Metrics (Phase 0.5 & Phase 4.1)
# ============================================================================
# Computes:
# 1. Overall & Compound-level Weighted MAE, RMSE, R2
# 2. Intra-Order Variance Ratio: Var(pred_within_order) / Var(true_within_order)
# 3. Pairwise Intra-Order Direction Accuracy (%)
# 4. Intra-Order Spearman Rank Correlation
# 5. High-Deviation Batch MAE (|y_i - y_order_mean| > 1.5 * std)
# ============================================================================

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_mooney_predictions(
    y_true,
    y_pred,
    df_meta,
    w_metric=None,
    order_col="OrderID",
    label_group_col="_label_group_id",
    sequence_col="BatchNumber",
):
    """Evaluate label-level accuracy, group trends, and batch process risk."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred) or len(y_true) != len(df_meta):
        raise ValueError("y_true, y_pred, and df_meta must have identical lengths.")

    if w_metric is None and "_w_metric" in df_meta.columns:
        w_metric = df_meta["_w_metric"].to_numpy()
    elif w_metric is None:
        w_metric = np.ones(len(y_true))
    w_metric = np.asarray(w_metric, dtype=float)

    metadata_columns = [column for column in [order_col, sequence_col, label_group_col] if column in df_meta.columns]
    row_work = df_meta[metadata_columns].copy()
    row_work["y_true"] = y_true
    row_work["y_pred"] = y_pred
    row_work["w_metric"] = w_metric
    work = row_work.copy()

    if label_group_col in work.columns:
        aggregation = {"y_true": "mean", "y_pred": "mean", "w_metric": "sum"}
        if order_col in work.columns:
            aggregation[order_col] = "first"
        if sequence_col in work.columns:
            aggregation[sequence_col] = "min"
        work = work.groupby(label_group_col, as_index=False).agg(aggregation)
        metric_weights = np.ones(len(work))
    else:
        metric_weights = work["w_metric"].to_numpy()

    mae = mean_absolute_error(work["y_true"], work["y_pred"], sample_weight=metric_weights)
    rmse = np.sqrt(mean_squared_error(work["y_true"], work["y_pred"], sample_weight=metric_weights))
    r2 = r2_score(work["y_true"], work["y_pred"], sample_weight=metric_weights)

    if order_col not in work.columns:
        work[order_col] = 0

    var_ratios = []
    direction_accuracies = []
    spearman_rhos = []
    for _, group in work.groupby(order_col, sort=False):
        if len(group) < 3:
            continue
        if sequence_col in group.columns:
            group = group.sort_values(sequence_col, kind="stable")
        actual = group["y_true"].to_numpy()
        predicted = group["y_pred"].to_numpy()
        actual_variance = np.var(actual)
        predicted_variance = np.var(predicted)
        if actual_variance > 1e-5:
            var_ratios.append(predicted_variance / actual_variance)
        if actual_variance > 1e-5 and predicted_variance > 1e-5:
            rho, _ = spearmanr(actual, predicted)
            if not np.isnan(rho):
                spearman_rhos.append(rho)

        correct_directions = 0
        total_pairs = 0
        for first_index in range(len(actual)):
            for second_index in range(first_index + 1, len(actual)):
                actual_delta = actual[first_index] - actual[second_index]
                predicted_delta = predicted[first_index] - predicted[second_index]
                if abs(actual_delta) > 0.3:
                    total_pairs += 1
                    correct_directions += int(np.sign(actual_delta) == np.sign(predicted_delta))
        if total_pairs > 0:
            direction_accuracies.append(correct_directions / total_pairs)

    if order_col in row_work.columns:
        row_mean = row_work.groupby(order_col)["y_true"].transform("mean")
        row_std = row_work.groupby(order_col)["y_true"].transform("std").fillna(0.0)
        high_deviation_mask = (row_std > 0.5) & ((row_work["y_true"] - row_mean).abs() > 1.5 * row_std)
    else:
        high_deviation_mask = pd.Series(False, index=row_work.index)

    if high_deviation_mask.any():
        high_deviation_mae = mean_absolute_error(
            row_work.loc[high_deviation_mask, "y_true"],
            row_work.loc[high_deviation_mask, "y_pred"],
            sample_weight=row_work.loc[high_deviation_mask, "w_metric"],
        )
    else:
        high_deviation_mae = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Variance_Ratio": float(np.mean(var_ratios)) if var_ratios else 1.0,
        "Direction_Accuracy": float(np.mean(direction_accuracies)) if direction_accuracies else 0.5,
        "Spearman_Rho": float(np.mean(spearman_rhos)) if spearman_rhos else 0.0,
        "High_Dev_MAE": float(high_deviation_mae),
        "High_Dev_Count": int(high_deviation_mask.sum()),
        "N_Evaluation_Groups": int(len(work)),
    }


if __name__ == '__main__':
    print("Trend Metrics Module ready.")
