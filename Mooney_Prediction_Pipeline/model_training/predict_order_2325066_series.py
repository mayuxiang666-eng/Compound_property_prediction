# ============================================================================
# Predict All Batches for Order 2325066 (Carbon Black Compound M1-B00458)
# ============================================================================
# Extracts the complete time-series prediction sequence for Order 2325066.
# Shows: BatchNumber, Actual Lab MNY, Stage 1 Baseline, Stage 2 Process Delta,
# Stage 3 Calibrated Prediction, and Absolute Error.
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

from feature_engineering.clustering import cluster_silica_carbon_black
from feature_engineering.stage1_recipe_features import extract_stage1_recipe_features
from feature_engineering.stage2_process_features import extract_stage2_process_features
from feature_engineering.silica_pid_feature_builder import build_silica_pid_features
from feature_engineering.cb_dispersion_feature_builder import build_cb_dispersion_features
from model_training.effective_weighting import compute_effective_sample_weights
from model_training.hybrid_unified_model import HybridUnifiedMooneyModel
from model_training.label_group_handler import add_label_group_information
from model_training.split_builder import generate_stratified_recipe_splits
from model_training.stage3_online_calibration import Stage3DelayedFeedbackCalibrator


def predict_order_2325066_full():
    print("=" * 95)
    print("  PREDICTING ALL BATCHES FOR ORDER 2325066 (M1-B00458---- 12 005)")
    print("=" * 95)

    out_dir = os.path.join(pipeline_root, 'reports', 'v37_targeted_batches')
    os.makedirs(out_dir, exist_ok=True)

    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../../data/stage_statistics_enriched_all_features_weather_v4.csv',
    ))
    if not os.path.exists(data_path):
        data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '../../data/enriched_mny_all.csv',
        ))

    df_raw = pd.read_csv(data_path, low_memory=False)
    if 'MNY' not in df_raw.columns and 'Mooney_Viscosity' in df_raw.columns:
        df_raw['MNY'] = df_raw['Mooney_Viscosity']
    df_clean = df_raw.dropna(subset=['MNY']).copy()

    if 'CompoundName' not in df_clean.columns and 'Compound' in df_clean.columns:
        df_clean['CompoundName'] = df_clean['Compound']
    if 'OrderID' not in df_clean.columns and 'Order_No' in df_clean.columns:
        df_clean['OrderID'] = df_clean['Order_No']
    if 'BatchNumber' not in df_clean.columns and 'Batch_No' in df_clean.columns:
        df_clean['BatchNumber'] = df_clean['Batch_No']

    # Build Features
    pid_feats = build_silica_pid_features(df_clean)
    cb_feats = build_cb_dispersion_features(df_clean)

    for c in pid_feats.columns:
        df_clean[c] = pid_feats[c]
    for c in cb_feats.columns:
        df_clean[c] = cb_feats[c]

    df_clean = cluster_silica_carbon_black(df_clean)
    df_clean = add_label_group_information(df_clean)
    df_clean = compute_effective_sample_weights(df_clean)
    df_clean = generate_stratified_recipe_splits(df_clean, test_size=0.15, val_size=0.15)

    s1_cols = extract_stage1_recipe_features(df_clean)
    s2_cols_base = extract_stage2_process_features(df_clean)
    s2_cols = list(set(s2_cols_base + list(pid_feats.columns) + list(cb_feats.columns)))

    for col in set(s1_cols + s2_cols):
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)

    df_tr = df_clean[df_clean['_split'] == 'train'].copy()

    model = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True, use_cb_subsystem=True)
    model.fit(df_tr, s1_cols, s2_cols, target_col='MNY', cluster_col='material_system')

    uncal_preds, s1_preds, s1b_biases, s2_res = model.predict(df_clean, cluster_col='material_system')
    cal_preds, _ = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha_base=0.3, adaptive=True).calibrate_time_series(df_clean, uncal_preds, target_col='MNY', group_col='CompoundName')

    df_clean['s1_baseline'] = s1_preds
    df_clean['s2_delta'] = s2_res
    df_clean['calibrated_pred'] = cal_preds
    df_clean['abs_error'] = np.abs(df_clean['MNY'] - cal_preds)

    # Filter Order 2325066
    order_sub = df_clean[df_clean['OrderID'].astype(str).str.contains('2325066')].copy()
    order_sub['BatchNumber_num'] = pd.to_numeric(order_sub['BatchNumber'], errors='coerce')
    order_sub = order_sub.sort_values(by='BatchNumber_num').reset_index(drop=True)

    summary_rows = []
    for idx, r in order_sub.iterrows():
        summary_rows.append({
            'batch_number': int(r['BatchNumber_num']),
            'actual_lab_mny': round(float(r['MNY']), 2),
            'stage1_recipe_baseline': round(float(r['s1_baseline']), 2),
            'stage2_process_delta': round(float(r['s2_delta']), 2),
            'calibrated_pred': round(float(r['calibrated_pred']), 2),
            'abs_error': round(float(r['abs_error']), 2),
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(os.path.join(out_dir, 'order_2325066_full_batch_series.csv'), index=False, encoding='utf-8-sig')

    print("\n" + "=" * 95)
    print("      ORDER 2325066 FULL BATCH PREDICTION SEQUENCE")
    print("=" * 95)
    print(f"{'Batch No.':<10} | {'Actual MNY':<12} | {'S1 Baseline':<12} | {'S2 Delta':<10} | {'Calibrated Pred':<16} | {'Abs Error':<10}")
    print("-" * 95)
    for _, r in sum_df.iterrows():
        print(f"Batch {int(r['batch_number']):<4} | {r['actual_lab_mny']:<12.2f} | {r['stage1_recipe_baseline']:<12.2f} | {r['stage2_process_delta']:<10.2f} | {r['calibrated_pred']:<16.2f} | {r['abs_error']:<10.2f}")
    print("=" * 95)

    print(f"Order MAE across all {len(sum_df)} batches: {sum_df['abs_error'].mean():.2f} MNY")
    return sum_df


if __name__ == '__main__':
    predict_order_2325066_full()
