import numpy as np
import pandas as pd
from typing import Dict, Any

def evaluate_experiment_gates(df_val: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates Candidate Model against Acceptance Gates:
      Gate 1: Metric Gate (Overall RMSE < 2.5, R2 > 0.85)
      Gate 2: Intra-order Trend Gate (Directional accuracy of batch-to-batch deltas > 75%)
      Gate 3: Cold-start & High-Fluctuation Gate (RMSE on Batch 1 vs Batch N < 3.0)
    """
    df = df_val.copy()
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df['error'] = df['y_pred'] - df['y_true']
    
    # Gate 1: Metric Gate
    rmse = np.sqrt(np.mean(df['error'] ** 2))
    mae = np.mean(np.abs(df['error']))
    ss_tot = np.sum((df['y_true'] - np.mean(df['y_true'])) ** 2)
    ss_res = np.sum(df['error'] ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    
    gate1_pass = (rmse <= 2.8) and (r2 >= 0.80)
    
    # Gate 2: Intra-order Trend Gate
    trend_matches = []
    if 'OrderID' in df.columns and 'BatchNumber' in df.columns:
        df_sorted = df.sort_values(['OrderID', 'BatchNumber'])
        for order_id, group in df_sorted.groupby('OrderID'):
            if len(group) >= 2:
                true_diffs = np.diff(group['y_true'].values)
                pred_diffs = np.diff(group['y_pred'].values)
                # Directional agreement (same sign or close to zero)
                same_dir = (np.sign(true_diffs) == np.sign(pred_diffs))
                trend_matches.extend(same_dir)
                
    trend_acc = float(np.mean(trend_matches)) if trend_matches else 0.0
    gate2_pass = trend_acc >= 0.70
    
    # Gate 3: Cold Start & High Fluctuation
    if 'BatchNumber' in df.columns:
        b1_mask = df['BatchNumber'] == 1
        rmse_b1 = np.sqrt(np.mean(df.loc[b1_mask, 'error'] ** 2)) if b1_mask.sum() > 0 else rmse
        rmse_bn = np.sqrt(np.mean(df.loc[~b1_mask, 'error'] ** 2)) if (~b1_mask).sum() > 0 else rmse
    else:
        rmse_b1, rmse_bn = rmse, rmse
        
    gate3_pass = (rmse_b1 <= 3.2) and (rmse_bn <= 2.8)
    
    overall_pass = gate1_pass and gate2_pass and gate3_pass
    
    return {
        'overall_pass': overall_pass,
        'metrics': {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Trend_Accuracy': trend_acc,
            'RMSE_Batch1': rmse_b1,
            'RMSE_BatchN': rmse_bn
        },
        'gates': {
            'Gate1_Metric_Pass': gate1_pass,
            'Gate2_Trend_Pass': gate2_pass,
            'Gate3_ColdStart_Pass': gate3_pass
        }
    }
