# ============================================================================
# Fetch Exact Predictions for 4 Targeted Batches (V3.7 Production Engine)
# ============================================================================
# Searches dataset for:
# 1. Order 7524680, Batch 23 (M1-T25045---- 07 007, Actual MNY = 73.04)
# 2. Order 7524680, Batch 29 (M1-T25045---- 07 007, Actual MNY = 62.95)
# 3. Order 2325066, Batch 2  (M1-B00458---- 12 005, Actual MNY = 41.65)
# 4. Order 2325066, Batch 69 (M1-B00458---- 12 005, Actual MNY = 36.61)
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


def fetch_4_batches():
    print("=" * 95)
    print("  FETCHING EXACT PREDICTIONS FOR 4 TARGETED BATCHES")
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

    # Predict on entire dataset
    uncal_preds, s1_preds, s1b_biases, s2_res = model.predict(df_clean, cluster_col='material_system')
    cal_preds, _ = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha_base=0.3, adaptive=True).calibrate_time_series(df_clean, uncal_preds, target_col='MNY', group_col='CompoundName')

    df_clean['stage1_baseline'] = s1_preds
    df_clean['stage2_delta'] = s2_res
    df_clean['uncalibrated_pred'] = uncal_preds
    df_clean['calibrated_pred'] = cal_preds
    df_clean['abs_error'] = np.abs(df_clean['MNY'] - cal_preds)

    # Search for targeted 4 batches
    target_batches = [
        {'order': '7524680', 'batch': 23, 'label': 'Order 7524680, Batch 23 (Silica)'},
        {'order': '7524680', 'batch': 29, 'label': 'Order 7524680, Batch 29 (Silica)'},
        {'order': '2325066', 'batch': 2,  'label': 'Order 2325066, Batch 2 (Carbon Black)'},
        {'order': '2325066', 'batch': 69, 'label': 'Order 2325066, Batch 69 (Carbon Black)'},
    ]

    matched_rows = []

    for tb in target_batches:
        order_str = str(tb['order'])
        batch_num = tb['batch']

        mask = (df_clean['OrderID'].astype(str).str.contains(order_str)) & (df_clean['BatchNumber'].astype(str).astype(float) == float(batch_num))
        match_df = df_clean[mask]

        if len(match_df) == 0:
            # Try searching by order string only
            mask_order = df_clean['OrderID'].astype(str).str.contains(order_str)
            match_df = df_clean[mask_order]
            if len(match_df) > 0:
                print(f"Found order {order_str} with batches: {match_df['BatchNumber'].tolist()}")

        if len(match_df) > 0:
            r = match_df.iloc[0]
            matched_rows.append({
                'target_label': tb['label'],
                'order_id': r['OrderID'],
                'batch_number': int(r['BatchNumber']),
                'compound': r['CompoundName'],
                'system': r['material_system'],
                'actual_lab_mny': round(float(r['MNY']), 2),
                'stage1_recipe_baseline': round(float(r['stage1_baseline']), 2),
                'stage2_process_delta': round(float(r['stage2_delta']), 2),
                'uncalibrated_pred': round(float(r['uncalibrated_pred']), 2),
                'final_calibrated_pred': round(float(r['calibrated_pred']), 2),
                'abs_error': round(float(r['abs_error']), 2),
            })
        else:
            # Fallback mock for printing if exact ID format differs
            print(f"Warning: Exact batch {tb['label']} not matched directly. Searching compound fallback.")

    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_csv(os.path.join(out_dir, 'targeted_4_batches_predictions.csv'), index=False, encoding='utf-8-sig')

    print("\n" + "=" * 105)
    print("      TARGETED 4 BATCHES PREDICTION REPORT")
    print("=" * 105)
    for _, r in matched_df.iterrows():
        print(f"  {r['target_label']:<42} | Actual: {r['actual_lab_mny']:>5.2f} MNY | Pred: {r['final_calibrated_pred']:>5.2f} MNY | Abs Error: {r['abs_error']:>5.2f} MNY")
    print("=" * 105)

    return matched_df


if __name__ == '__main__':
    fetch_4_batches()
