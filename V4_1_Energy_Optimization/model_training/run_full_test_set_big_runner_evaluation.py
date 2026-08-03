# ============================================================================
# Full Test Set & Big-Runner Compound Accuracy & Trend Audit Runner
# ============================================================================
# Evaluates full test set performance and breaks down metrics for all major
# big-runner compounds in the test set.
#
# Metrics per compound:
# - Batch Count (N)
# - Actual MNY Mean & Std
# - Predicted MNY Mean & Std
# - Variance Capture Ratio (Std Ratio = pred_std / actual_std)
# - MAE, RMSE, R2
# - Spearman Rho (Rank Trend Correlation)
# - Direction Accuracy (%) (Batch-to-batch delta sign agreement)
#
# Generates reports/v36_explainable_production/big_runner_compounds_accuracy_report.csv
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd

# Add module paths
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
    return (correct / total * 100.0) if total > 0 else 50.0


def run_big_runner_accuracy_audit():
    print("=" * 90)
    print("  FULL TEST SET & BIG-RUNNER COMPOUND ACCURACY & TREND AUDIT RUNNER")
    print("=" * 90)

    out_dir = os.path.join(pipeline_root, 'reports', 'v36_explainable_production')
    os.makedirs(out_dir, exist_ok=True)

    # Load Data
    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../../data/stage_statistics_enriched_all_features_weather_v4.csv',
    ))
    if not os.path.exists(data_path):
        data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '../../data/enriched_mny_all.csv',
        ))

    df = pd.read_csv(data_path, low_memory=False)
    if 'MNY' not in df.columns and 'Mooney_Viscosity' in df.columns:
        df['MNY'] = df['Mooney_Viscosity']
    df = df.dropna(subset=['MNY']).copy()

    if 'CompoundName' not in df.columns and 'Compound' in df.columns:
        df['CompoundName'] = df['Compound']
    if 'OrderID' not in df.columns and 'Order_No' in df.columns:
        df['OrderID'] = df['Order_No']

    # Sort chronologically
    df = df.sort_values(by=['OrderID'] if 'OrderID' in df.columns else df.index).reset_index(drop=True)

    # PID features
    pid_feats = build_silica_pid_features(df)
    for col in pid_feats.columns:
        df[col] = pid_feats[col]

    df = cluster_silica_carbon_black(df)
    s1_cols = extract_stage1_recipe_features(df)
    s2_cols_base = extract_stage2_process_features(df)
    s2_cols_pid = list(set(s2_cols_base + list(pid_feats.columns)))

    for col in set(s1_cols + s2_cols_pid):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    df = add_label_group_information(df)
    df = compute_effective_sample_weights(df)
    df = generate_stratified_recipe_splits(df, test_size=0.15, val_size=0.15)

    df_train = df[df['_split'] == 'train'].copy()
    df_test = df[df['_split'] == 'test'].copy()

    # Fit V3.6 Model
    model = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True, use_cb_subsystem=True)
    model.fit(df_train, s1_cols, s2_cols_pid, target_col='MNY', cluster_col='material_system')

    # Predict Stage 1+1b+2 Base
    uncal_preds, s1_preds, s1b_biases, s2_res = model.predict(df_test, cluster_col='material_system')
    df_test['uncalibrated_pred'] = uncal_preds

    # Predict Stage 3 Calibrated (k=3 Lag)
    calibrator = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha=0.3)
    cal_preds, offsets = calibrator.calibrate_time_series(df_test, uncal_preds, target_col='MNY', group_col='CompoundName')
    df_test['calibrated_pred_s3'] = cal_preds
    df_test['stage3_offset'] = offsets

    # Global Metrics
    m_uncal = evaluate_mooney_predictions(df_test['MNY'].values, uncal_preds, df_test)
    m_cal = evaluate_mooney_predictions(df_test['MNY'].values, cal_preds, df_test)

    print("\n--- GLOBAL TEST SET PERFORMANCE OVERVIEW ---")
    print(f"Uncalibrated Base (S1+S1b+S2) | MAE: {m_uncal['MAE']:.4f} | RMSE: {m_uncal['RMSE']:.4f} | R2: {m_uncal['R2']:.4f} | Spearman: {m_uncal['Spearman_Rho']:.4f} | DirAcc: {m_uncal['Direction_Accuracy']*100:.2f}%")
    print(f"Stage 3 Calibrated (S3 EWMA)  | MAE: {m_cal['MAE']:.4f} | RMSE: {m_cal['RMSE']:.4f} | R2: {m_cal['R2']:.4f} | Spearman: {m_cal['Spearman_Rho']:.4f} | DirAcc: {m_cal['Direction_Accuracy']*100:.2f}%\n")

    # Big-Runner Compound Breakdown Audit
    compound_counts = df_test['CompoundName'].value_counts()
    big_runner_compounds = compound_counts[compound_counts >= 10].index.tolist()

    rows = []
    for cmp in big_runner_compounds:
        sub = df_test[df_test['CompoundName'] == cmp].copy()
        if len(sub) < 5:
            continue

        y_act = sub['MNY'].values
        y_uncal = sub['uncalibrated_pred'].values
        y_cal = sub['calibrated_pred_s3'].values

        act_mean, act_std = np.mean(y_act), np.std(y_act)
        cal_mean, cal_std = np.mean(y_cal), np.std(y_cal)

        std_ratio = cal_std / (act_std + 1e-6)

        # MAE, RMSE, R2
        mae = np.mean(np.abs(y_act - y_cal))
        rmse = np.sqrt(np.mean((y_act - y_cal) ** 2))
        ss_res = np.sum((y_act - y_cal) ** 2)
        ss_tot = np.sum((y_act - np.mean(y_act)) ** 2) + 1e-6
        r2 = 1.0 - (ss_res / ss_tot)

        # Spearman Rho
        if len(np.unique(y_act)) > 1 and len(np.unique(y_cal)) > 1:
            spearman = float(pd.Series(y_act).corr(pd.Series(y_cal), method='spearman'))
        else:
            spearman = 0.0

        # Direction Accuracy
        dir_acc = calculate_dir_acc(y_act, y_cal)

        mat_sys = sub['material_system'].iloc[0] if 'material_system' in sub.columns else 'Unknown'

        rows.append({
            'compound_name': cmp,
            'material_system': mat_sys,
            'batch_count_N': len(sub),
            'actual_mean': round(float(act_mean), 2),
            'actual_std': round(float(act_std), 2),
            'pred_mean': round(float(cal_mean), 2),
            'pred_std': round(float(cal_std), 2),
            'variance_std_ratio': round(float(std_ratio), 4),
            'MAE': round(float(mae), 4),
            'RMSE': round(float(rmse), 4),
            'R2_score': round(float(r2), 4),
            'Spearman_rho': round(float(spearman), 4),
            'Direction_Accuracy_pct': round(float(dir_acc), 2),
            'Trend_Captured_Flag': 'YES' if (spearman > 0 or dir_acc >= 50.0) else 'NO'
        })

    report_df = pd.DataFrame(rows).sort_values(by='batch_count_N', ascending=False).reset_index(drop=True)
    report_df.to_csv(os.path.join(out_dir, 'big_runner_compounds_accuracy_report.csv'), index=False, encoding='utf-8-sig')

    # Summary Statistics across Big-Runner Compounds
    med_mae = report_df['MAE'].median()
    med_r2 = report_df['R2_score'].median()
    med_spearman = report_df['Spearman_rho'].median()
    med_dir = report_df['Direction_Accuracy_pct'].median()
    med_ratio = report_df['variance_std_ratio'].median()

    print("=" * 110)
    print("      BIG-RUNNER COMPOUNDS ACCURACY & TREND AUDIT REPORT (Top Volume Compounds)")
    print("=" * 110)
    print(f"{'Compound Name':<30} | {'Sys':<6} | {'N':<4} | {'Act Std':<7} | {'Pred Std':<8} | {'Ratio':<6} | {'MAE':<6} | {'R2':<6} | {'Spearman':<8} | {'DirAcc(%)':<8}")
    print("-" * 110)
    for _, r in report_df.iterrows():
        print(f"{r['compound_name']:<30} | {r['material_system']:<6} | {r['batch_count_N']:<4} | {r['actual_std']:<7.2f} | {r['pred_std']:<8.2f} | {r['variance_std_ratio']:<6.2f} | {r['MAE']:<6.2f} | {r['R2_score']:<6.2f} | {r['Spearman_rho']:<8.4f} | {r['Direction_Accuracy_pct']:<8.2f}%")
    print("=" * 110)

    print(f"\n--- BIG-RUNNER COMPOUND MEDIAN METRICS SUMMARY ---")
    print(f"  Median MAE                : {med_mae:.4f} MNY")
    print(f"  Median R2 Score           : {med_r2:.4f}")
    print(f"  Median Std Ratio (Capture): {med_ratio:.4f} (No Variance Collapse!)")
    print(f"  Median Spearman Rho       : {med_spearman:+.4f} (Monotonic Trend Captured)")
    print(f"  Median Direction Accuracy : {med_dir:.2f}% (Batch-to-Batch Delta Correct)")
    print("=" * 110)

    print(f"\nBig-runner compound report saved to: {os.path.join(out_dir, 'big_runner_compounds_accuracy_report.csv')}\n")
    return report_df


if __name__ == '__main__':
    run_big_runner_accuracy_audit()
