# ============================================================================
# M1-T15760 Continuous Order Batch Trend Plotter
# ============================================================================
# Generates a high-resolution comparison line chart plotting Actual Lab Mooney
# vs V3.6 A5 Calibrated Prediction on continuous orders of M1-T15760.
# Saves plot PNG into artifact directory for embedding.
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# Artifact directory path
artifact_dir = r"C:\Users\uif35346\.gemini\antigravity\brain\fe15231d-68aa-4dc6-8573-0cac58c9de89"


def generate_t15760_trend_chart():
    print("=" * 80)
    print("  GENERATING M1-T15760 CONTINUOUS ORDER TREND COMPARISON CHART")
    print("=" * 80)

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

    # Fit V3.6 A5 Model
    model = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True, use_cb_subsystem=True)
    model.fit(df_train, s1_cols, s2_cols_pid, target_col='MNY', cluster_col='material_system')

    uncal_preds, _, _, _ = model.predict(df_test, cluster_col='material_system')
    df_test['uncalibrated_pred'] = uncal_preds

    calibrator = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha=0.3)
    cal_preds, offsets = calibrator.calibrate_time_series(df_test, uncal_preds, target_col='MNY', group_col='CompoundName')
    df_test['calibrated_pred_s3'] = cal_preds

    # Filter M1-T15760
    target_cmp = [c for c in df_test['CompoundName'].unique() if 'T15760' in c][0]
    t15760_df = df_test[df_test['CompoundName'] == target_cmp].copy()

    # Take a continuous sequence of 35 batches
    sub_df = t15760_df.tail(35).copy().reset_index(drop=True)

    # Plot Setup (Dark Theme)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    x_indices = np.arange(len(sub_df)) + 1
    y_actual = sub_df['MNY'].values
    y_uncal = sub_df['uncalibrated_pred'].values
    y_cal = sub_df['calibrated_pred_s3'].values

    # Plot Lines
    ax.plot(x_indices, y_actual, label='Actual Lab Mooney (MNY)', color='#38bdf8', linewidth=2.5, marker='o', markersize=6, alpha=0.9)
    ax.plot(x_indices, y_cal, label='V3.5_A5 + Stage 3 Calibrated Pred', color='#22c55e', linewidth=2.5, marker='s', markersize=5, alpha=0.95)
    ax.plot(x_indices, y_uncal, label='Base Model (Stage 1+1b+2 A5)', color='#f59e0b', linewidth=1.8, linestyle='--', marker='^', markersize=4, alpha=0.7)

    # Calculate metrics for this sequence
    spearman = pd.Series(y_actual).corr(pd.Series(y_cal), method='spearman')
    mae = np.mean(np.abs(y_actual - y_cal))
    std_ratio = np.std(y_cal) / np.std(y_actual)

    ax.set_title(f"M1-T15760 Continuous Order Batch Mooney Trend Comparison\nSpearman Rho = +{spearman:.4f} | MAE = {mae:.2f} MNY | Variance Capture = {std_ratio*100:.1f}%", fontsize=14, fontweight='bold', pad=15, color='#f8fafc')
    ax.set_xlabel("Continuous Batch Sequence Number (Production Order Stream)", fontsize=12, labelpad=10, color='#94a3b8')
    ax.set_ylabel("Mooney Viscosity (MNY)", fontsize=12, labelpad=10, color='#94a3b8')

    ax.grid(True, linestyle=':', alpha=0.3, color='#475569')
    ax.legend(loc='upper right', frameon=True, facecolor='#1e293b', edgecolor='#334155', fontsize=11)

    # Highlight order transition boundaries
    orders = sub_df['OrderID'].values
    unique_orders = []
    order_start_idx = []
    for idx, ord_id in enumerate(orders):
        if idx == 0 or ord_id != orders[idx-1]:
            unique_orders.append(ord_id)
            order_start_idx.append(idx + 1)
            if idx > 0:
                ax.axvline(x=idx + 0.5, color='#e2e8f0', linestyle='-', alpha=0.4)
                ax.text(idx + 0.7, ax.get_ylim()[0] + 0.5, f"Order #{ord_id}", color='#cbd5e1', fontsize=9, rotation=90)

    plt.tight_layout()

    # Save to artifacts directory
    output_png_path = os.path.join(artifact_dir, "m1_t15760_trend_chart.png")
    plt.savefig(output_png_path, dpi=300)
    plt.close()

    print(f"Trend plot saved to artifact path: {output_png_path}\n")
    return output_png_path


if __name__ == '__main__':
    generate_t15760_trend_chart()
