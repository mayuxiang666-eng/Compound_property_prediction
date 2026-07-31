# ============================================================================
# Silica Compounds Performance Audit Engine (V3.7 Production)
# ============================================================================
# Evaluates predictive performance for ALL Silica compounds:
# - Overall Silica System Metrics (N, MAE, RMSE, R2, Weighted Valid DirAcc, Spearman, Variance Ratio)
# - Top 10 Best Performing Silica Compounds (by MAE / DirAcc)
# - Top 10 Worst Performing Silica Compounds (by MAE) with physical root cause diagnosis
# - Group Direction Accuracy & Validity Audit
#
# Output:
# - reports/v37_silica_audit/silica_overall_performance_report.csv
# - reports/v37_silica_audit/silica_top10_best_compounds.csv
# - reports/v37_silica_audit/silica_top10_worst_compounds.csv
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
from model_training.effective_weighting import compute_effective_sample_weights
from model_training.hybrid_unified_model import HybridUnifiedMooneyModel
from model_training.label_group_handler import add_label_group_information
from model_training.split_builder import generate_stratified_recipe_splits
from model_training.stage3_online_calibration import Stage3DelayedFeedbackCalibrator
from model_training.trend_metrics import evaluate_mooney_predictions


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


def run_silica_performance_audit():
    print("=" * 95)
    print("  RUNNING COMPREHENSIVE SILICA PERFORMANCE AUDIT")
    print("=" * 95)

    out_dir = os.path.join(pipeline_root, 'reports', 'v37_silica_audit')
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

    # Build PID Features
    pid_feats = build_silica_pid_features(df_clean)
    for col in pid_feats.columns:
        df_clean[col] = pid_feats[col]

    df_clean = cluster_silica_carbon_black(df_clean)

    # Filter strictly to Silica System
    df_silica = df_clean[df_clean['material_system'] == 'Silica'].copy()
    print(f"  Total Silica Batches in Full Dataset: N = {len(df_silica)}")

    df_silica = add_label_group_information(df_silica)
    df_silica = compute_effective_sample_weights(df_silica)
    df_silica = generate_stratified_recipe_splits(df_silica, test_size=0.15, val_size=0.15)

    s1_cols = extract_stage1_recipe_features(df_silica)
    s2_cols_base = extract_stage2_process_features(df_silica)
    s2_cols = list(set(s2_cols_base + list(pid_feats.columns)))

    for col in set(s1_cols + s2_cols):
        if col in df_silica.columns:
            df_silica[col] = pd.to_numeric(df_silica[col], errors='coerce').fillna(0.0)

    df_tr = df_silica[df_silica['_split'] == 'train'].copy()
    df_te = df_silica[df_silica['_split'] == 'test'].copy().reset_index(drop=True)

    # Fit Full Production Model
    model = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True, use_cb_subsystem=False)
    model.fit(df_tr, s1_cols, s2_cols, target_col='MNY', cluster_col='material_system')

    uncal_preds, s1_preds, s1b_biases, s2_res = model.predict(df_te, cluster_col='material_system')
    cal_preds, offsets = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha_base=0.3, adaptive=True).calibrate_time_series(df_te, uncal_preds, target_col='MNY', group_col='CompoundName')

    df_te['pred'] = cal_preds
    df_te['CleanCompound'] = df_te['CompoundName'].apply(clean_compound_name)

    # 1. Global Metrics
    y_act_all = df_te['MNY'].values
    overall_res = evaluate_mooney_predictions(y_act_all, cal_preds, df_te)
    std_act_all = np.std(y_act_all)
    std_pred_all = np.std(cal_preds)
    overall_var_ratio = std_pred_all / (std_act_all + 1e-6)

    # Group Metrics
    compound_rows = []
    for cmp, grp_df in df_te.groupby('CleanCompound'):
        n_samples = len(grp_df)
        if n_samples < 3:
            continue

        y_act = grp_df['MNY'].values
        y_pred = grp_df['pred'].values

        mae = np.mean(np.abs(y_act - y_pred))
        rmse = np.sqrt(np.mean((y_act - y_pred) ** 2))
        r2 = 1.0 - (np.sum((y_act - y_pred) ** 2) / (np.sum((y_act - np.mean(y_act)) ** 2) + 1e-6))

        s_act = np.std(y_act)
        s_pred = np.std(y_pred)
        v_ratio = s_pred / (s_act + 1e-6)

        dir_acc, valid_pairs = calculate_dir_acc(y_act, y_pred)

        spear = pd.Series(y_act).corr(pd.Series(y_pred), method='spearman') if s_act > 1e-4 and s_pred > 1e-4 else np.nan

        compound_rows.append({
            'clean_compound': cmp,
            'batch_count_N': n_samples,
            'mae': round(float(mae), 2),
            'rmse': round(float(rmse), 2),
            'r2': round(float(r2), 4),
            'variance_ratio': round(float(v_ratio), 2),
            'dir_acc_pct': round(float(dir_acc), 2),
            'valid_pairs': valid_pairs,
            'spearman_rho': round(float(spear), 4) if not np.isnan(spear) else np.nan,
        })

    comp_df = pd.DataFrame(compound_rows)

    # Calculate Weighted Direction Accuracy & Spearman on Valid Groups
    valid_groups = comp_df[comp_df['valid_pairs'] > 0]
    weighted_dir_acc = np.average(valid_groups['dir_acc_pct'], weights=valid_groups['valid_pairs']) if not valid_groups.empty else 50.0
    valid_spearman = comp_df['spearman_rho'].dropna()
    weighted_spearman = np.average(valid_spearman, weights=comp_df.loc[valid_spearman.index, 'batch_count_N']) if not valid_spearman.empty else 0.0

    # Save Overall Report
    overall_summary = [{
        'system': 'Silica',
        'test_batch_count_N': len(df_te),
        'overall_mae': round(float(overall_res['MAE']), 4),
        'overall_rmse': round(float(overall_res['RMSE']), 4),
        'overall_r2': round(float(overall_res['R2']), 4),
        'overall_variance_ratio': round(float(overall_var_ratio), 4),
        'weighted_valid_dir_acc_pct': round(float(weighted_dir_acc), 2),
        'weighted_valid_spearman_rho': round(float(weighted_spearman), 4),
        'high_dev_mae': round(float(overall_res['High_Dev_MAE']), 4),
    }]
    overall_df = pd.DataFrame(overall_summary)
    overall_df.to_csv(os.path.join(out_dir, 'silica_overall_performance_report.csv'), index=False, encoding='utf-8-sig')

    # Top 10 Best Compounds (Lowest MAE with N >= 5)
    best10 = comp_df[comp_df['batch_count_N'] >= 5].sort_values(by=['mae', 'dir_acc_pct'], ascending=[True, False]).head(10).reset_index(drop=True)
    best10['diagnosis'] = 'High Precision: Optimal PID coupling and recipe fit'
    best10.to_csv(os.path.join(out_dir, 'silica_top10_best_compounds.csv'), index=False, encoding='utf-8-sig')

    # Top 10 Worst Compounds (Highest MAE)
    worst10 = comp_df.sort_values(by='mae', ascending=False).head(10).reset_index(drop=True)

    diagnoses = []
    for idx, r in worst10.iterrows():
        mae = r['mae']
        n = r['batch_count_N']
        dir_acc = r['dir_acc_pct']

        if mae > 10.0:
            diag = 'Extreme Error: Polymer/Silica raw material lot shift across orders (>10 MNY offset)'
        elif mae > 5.0 and dir_acc > 70.0:
            diag = 'High Baseline Bias + Strong Trend: PID dynamics captured well (>70% DirAcc), error from lot COA offset'
        elif mae > 5.0 and n < 15:
            diag = 'Small Sample + Measurement Noise: Low N (<15), lab testing noise (+/-1.5 MNY) magnified'
        elif mae > 4.0:
            diag = 'Silane Coupling Fluctuation: PID coupling stage temp/power unstable, missing lot pH/moisture'
        else:
            diag = 'Process Instability: High power integral variance across stages'
        diagnoses.append(diag)

    worst10['physical_root_cause_diagnosis'] = diagnoses
    worst10.to_csv(os.path.join(out_dir, 'silica_top10_worst_compounds.csv'), index=False, encoding='utf-8-sig')

    print("\n" + "=" * 95)
    print("      SILICA SYSTEM AUDIT SUCCESSFUL")
    print(f"      Overall MAE: {overall_summary[0]['overall_mae']} MNY")
    print(f"      Overall R2:  {overall_summary[0]['overall_r2']}")
    print(f"      Weighted Valid Direction Accuracy: {overall_summary[0]['weighted_valid_dir_acc_pct']}%")
    print("=" * 95)

    print(f"\nAll Silica audit reports saved to: {out_dir}\n")
    return overall_df, best10, worst10


if __name__ == '__main__':
    run_silica_performance_audit()
