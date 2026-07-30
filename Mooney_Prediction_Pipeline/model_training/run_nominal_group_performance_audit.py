# ============================================================================
# Nominal Group-Aggregated Performance Audit Engine (V3.7 Production)
# ============================================================================
# Addresses the industrial reality that multiple production batches (cars) share 
# a single physical Lab MNY sample value.
#
# Computes TWO sets of metrics:
# 1. Per-Batch Raw Metrics (Historical Comparison)
# 2. Nominal Group-Aggregated Metrics (Nominal Group MAE & Nominal R2)
#    - Grouping by _label_group_id (OrderID + PalletID / LabSampleID)
#    - Group Mean Prediction vs Physical Lab Mooney
#
# Output:
# - reports/v37_nominal_audit/nominal_group_performance_report.csv
# - reports/v37_nominal_audit/nominal_vs_batch_comparison_summary.md
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

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


def clean_compound_name(comp_name):
    comp = str(comp_name).strip()
    if '---' in comp:
        return comp.split('---')[0].strip()
    elif '--' in comp:
        return comp.split('--')[0].strip()
    return comp


def calculate_dir_acc(y_true, y_pred, min_delta=0.3):
    correct, total = 0, 0
    n = len(y_true)
    for i in range(n):
        for j in range(i + 1, n):
            d_true = y_true[i] - y_true[j]
            d_pred = y_pred[i] - y_pred[j]
            if abs(d_true) >= min_delta:
                total += 1
                if np.sign(d_true) == np.sign(d_pred):
                    correct += 1
    return (correct / total * 100.0) if total > 0 else 50.0, total


def run_nominal_group_performance_audit():
    print("=" * 95)
    print("  RUNNING NOMINAL GROUP-AGGREGATED PERFORMANCE AUDIT")
    print("=" * 95)

    out_dir = os.path.join(pipeline_root, 'reports', 'v37_nominal_audit')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Data
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

    df_clean = df_clean.sort_values(by=['OrderID'] if 'OrderID' in df_clean.columns else df_clean.index).reset_index(drop=True)

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
    df_te = df_clean[df_clean['_split'] == 'test'].copy().reset_index(drop=True)

    # Fit Full Production Model
    model = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True, use_cb_subsystem=True)
    model.fit(df_tr, s1_cols, s2_cols, target_col='MNY', cluster_col='material_system')

    uncal_preds, s1_preds, s1b_biases, s2_res = model.predict(df_te, cluster_col='material_system')
    cal_preds, offsets = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha_base=0.3, adaptive=True).calibrate_time_series(df_te, uncal_preds, target_col='MNY', group_col='CompoundName')

    df_te['pred'] = cal_preds
    df_te['CleanCompound'] = df_te['CompoundName'].apply(clean_compound_name)

    results = []

    for system_name in ['Silica', 'CarbonBlack', 'Overall_All']:
        if system_name == 'Silica':
            sub_df = df_te[df_te['material_system'] == 'Silica'].copy()
        elif system_name == 'CarbonBlack':
            sub_df = df_te[df_te['material_system'] == 'CarbonBlack'].copy()
        else:
            sub_df = df_te.copy()

        if len(sub_df) == 0:
            continue

        # 1. Per-Batch Raw Metrics
        y_batch_act = sub_df['MNY'].values
        y_batch_pred = sub_df['pred'].values

        batch_mae = np.mean(np.abs(y_batch_act - y_batch_pred))
        batch_rmse = np.sqrt(np.mean((y_batch_act - y_batch_pred) ** 2))
        batch_ss_tot = np.sum((y_batch_act - np.mean(y_batch_act)) ** 2)
        batch_ss_res = np.sum((y_batch_act - y_batch_pred) ** 2)
        batch_r2 = 1.0 - (batch_ss_res / (batch_ss_tot + 1e-6))

        # 2. Nominal Group-Aggregated Metrics
        # Group by _label_group_id
        group_rows = []
        for grp_id, grp_data in sub_df.groupby('_label_group_id'):
            group_rows.append({
                '_label_group_id': grp_id,
                'compound': grp_data['CleanCompound'].iloc[0],
                'group_batch_count': len(grp_data),
                'act_lab_mny': grp_data['MNY'].iloc[0],  # Single Physical Lab Mooney Value
                'pred_mean_mny': grp_data['pred'].mean(),  # Mean prediction across group batches
            })
        grp_df = pd.DataFrame(group_rows)

        y_grp_act = grp_df['act_lab_mny'].values
        y_grp_pred = grp_df['pred_mean_mny'].values

        nominal_group_mae = np.mean(np.abs(y_grp_act - y_grp_pred))
        nominal_group_rmse = np.sqrt(np.mean((y_grp_act - y_grp_pred) ** 2))
        grp_ss_tot = np.sum((y_grp_act - np.mean(y_grp_act)) ** 2)
        grp_ss_res = np.sum((y_grp_act - y_grp_pred) ** 2)
        nominal_group_r2 = 1.0 - (grp_ss_res / (grp_ss_tot + 1e-6))

        results.append({
            'system': system_name,
            'test_batch_count_N': len(sub_df),
            'test_label_group_count_K': len(grp_df),
            'avg_batches_per_sample_group': round(len(sub_df) / max(len(grp_df), 1), 2),
            'per_batch_raw_mae': round(float(batch_mae), 4),
            'nominal_group_mae': round(float(nominal_group_mae), 4),
            'per_batch_raw_r2': round(float(batch_r2), 4),
            'nominal_group_r2': round(float(nominal_group_r2), 4),
            'per_batch_raw_rmse': round(float(batch_rmse), 4),
            'nominal_group_rmse': round(float(nominal_group_rmse), 4),
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(out_dir, 'nominal_group_performance_report.csv'), index=False, encoding='utf-8-sig')

    print("\n" + "=" * 105)
    print("      NOMINAL GROUP-AGGREGATED VS PER-BATCH RAW METRICS COMPARISON")
    print("=" * 105)
    print(f"{'System':<15} | {'N Batches':<9} | {'K Groups':<8} | {'Batch MAE':<10} | {'Nominal MAE':<12} | {'Batch R2':<9} | {'Nominal R2':<10}")
    print("-" * 105)
    for _, r in res_df.iterrows():
        print(f"{r['system']:<15} | {r['test_batch_count_N']:<9} | {r['test_label_group_count_K']:<8} | {r['per_batch_raw_mae']:<10.4f} | {r['nominal_group_mae']:<12.4f} | {r['per_batch_raw_r2']:<9.4f} | {r['nominal_group_r2']:<10.4f}")
    print("=" * 105)

    print(f"\nNominal group performance report saved to: {out_dir}\n")
    return res_df


if __name__ == '__main__':
    run_nominal_group_performance_audit()
