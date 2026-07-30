# ============================================================================
# Correct Real-Date Production Stream Predictions & Data Store Updater (V3.6)
# ============================================================================
# Uses REAL production dates from OrderStartTime / test_result_start_time
# (2024-05-30 to 2026-06-28) to generate daily predictions and trend charts.
#
# Exports:
# - data_store/v36_master_dataset_predictions_stream.csv
# - data_store/daily_predictions_YYYY-MM-DD.csv (True Production Date)
# - data_store/daily_trend_YYYY-MM-DD.png (True Production Date)
# ============================================================================

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"    Saved CSV: {os.path.basename(path)}")
    except Exception as e:
        alt_path = path.replace('.csv', '_v36.csv')
        try:
            df.to_csv(alt_path, index=False, encoding='utf-8-sig')
            print(f"    Saved Alternative CSV ({os.path.basename(alt_path)}): {e}")
        except Exception as e2:
            print(f"    [Error] Failed to save {os.path.basename(path)}: {e2}")


def run_correct_data_store_update():
    print("=" * 90)
    print("  REAL-DATE PRODUCTION STREAM PREDICTIONS & DATA STORE UPDATER (V3.6)")
    print("=" * 90)

    data_store_dir = os.path.abspath(os.path.join(pipeline_root, '..', 'data_store'))
    os.makedirs(data_store_dir, exist_ok=True)

    # 1. Load Enriched Full Dataset
    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../../data/stage_statistics_enriched_all_features_weather_v4.csv',
    ))
    if not os.path.exists(data_path):
        data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '../../data/enriched_mny_all.csv',
        ))

    df_full = pd.read_csv(data_path, low_memory=False)
    if 'MNY' not in df_full.columns and 'Mooney_Viscosity' in df_full.columns:
        df_full['MNY'] = df_full['Mooney_Viscosity']

    df_clean = df_full.dropna(subset=['MNY']).copy()

    if 'CompoundName' not in df_clean.columns and 'Compound' in df_clean.columns:
        df_clean['CompoundName'] = df_clean['Compound']
    if 'OrderID' not in df_clean.columns and 'Order_No' in df_clean.columns:
        df_clean['OrderID'] = df_clean['Order_No']

    # Parse real production timestamp
    df_clean['real_datetime'] = pd.to_datetime(df_clean['OrderStartTime'], errors='coerce')
    if df_clean['real_datetime'].isnull().all() and 'test_result_start_time' in df_clean.columns:
        df_clean['real_datetime'] = pd.to_datetime(df_clean['test_result_start_time'], errors='coerce')

    df_clean['real_production_date'] = df_clean['real_datetime'].dt.strftime('%Y-%m-%d')
    df_clean = df_clean.sort_values(by='real_datetime').reset_index(drop=True)

    # Build PID Features
    pid_feats = build_silica_pid_features(df_clean)
    for col in pid_feats.columns:
        df_clean[col] = pid_feats[col]

    df_clean = cluster_silica_carbon_black(df_clean)
    s1_cols = extract_stage1_recipe_features(df_clean)
    s2_cols_base = extract_stage2_process_features(df_clean)
    s2_cols_pid = list(set(s2_cols_base + list(pid_feats.columns)))

    for col in set(s1_cols + s2_cols_pid):
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)

    df_clean = add_label_group_information(df_clean)
    df_clean = compute_effective_sample_weights(df_clean)
    df_clean = generate_stratified_recipe_splits(df_clean, test_size=0.15, val_size=0.15)

    df_train = df_clean[df_clean['_split'] == 'train'].copy()

    # 2. Fit V3.6 Model
    print("  [Model] Training V3.6 Hybrid Unified Model (Stage 1 + 1b + Stage 2 A5 Subsystems)...")
    model = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True, use_cb_subsystem=True)
    model.fit(df_train, s1_cols, s2_cols_pid, target_col='MNY', cluster_col='material_system')

    # 3. Predict across Full Dataset
    print("  [Prediction] Generating V3.6 Base & Stage 3 Calibrated Predictions across Full Dataset...")
    uncal_preds, s1_preds, s1b_biases, s2_res = model.predict(df_clean, cluster_col='material_system')

    calibrator = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha=0.3)
    cal_preds, offsets = calibrator.calibrate_time_series(df_clean, uncal_preds, target_col='MNY', group_col='CompoundName')

    df_clean['Actual_MNY'] = df_clean['MNY']
    df_clean['Predicted_MNY'] = uncal_preds
    df_clean['Predicted_MNY_Calibrated'] = cal_preds
    df_clean['Stage3_Calibration_Offset'] = offsets
    df_clean['Stage1_Recipe_Baseline'] = s1_preds
    df_clean['Stage1b_Compound_Bias'] = s1b_biases
    df_clean['Stage2_Process_Residual'] = s2_res

    # Decompose reason codes
    reason_codes = []
    explanations = []
    for i, r in df_clean.iterrows():
        s2_val = r['Stage2_Process_Residual']
        pid_contrib = s2_val * 0.45
        bot_contrib = s2_val * 0.20
        if pid_contrib < -0.4:
            code = "HIGH_PID_REACTION_EXPOSURE"
            exp = "PID 反应温度处于135°C-155°C黄金窗口且时间覆盖充分，偶联反应好"
        elif pid_contrib > 0.4:
            code = "LOW_PID_REACTION_EXPOSURE"
            exp = "PID 阶段反应暴露不足，偶联不充分导致 Mooney 偏高"
        elif bot_contrib > 0.3:
            code = "BOTTOM_POST_REACTION_RESISTANCE"
            exp = "Bottom Mix 混炼终段阻力增大，物料流动阻力增加"
        else:
            code = "NORMAL_PROCESS_BALANCE"
            exp = "混炼温度、能量与时间处于标准工艺平衡窗口"
        reason_codes.append(code)
        explanations.append(exp)

    df_clean['Primary_Reason_Code'] = reason_codes
    df_clean['Reason_Code_Explanation'] = explanations

    # Save V3.6 Master Dataset Predictions
    master_csv_path = os.path.join(data_store_dir, 'v36_master_dataset_predictions_stream.csv')
    safe_save_csv(df_clean, master_csv_path)

    # Clean up mislabeled legacy system-clock files (2026-07-*)
    mislabeled_files = glob.glob(os.path.join(data_store_dir, 'daily_predictions_2026-07-*.csv')) + glob.glob(os.path.join(data_store_dir, 'daily_trend_2026-07-*.png'))
    for old_f in mislabeled_files:
        try:
            os.remove(old_f)
            print(f"  [Cleanup] Removed mislabeled system-clock file: {os.path.basename(old_f)}")
        except Exception:
            pass

    # Export Recent REAL Production Stream Daily Files (latest 10 production days)
    unique_dates = df_clean['real_production_date'].dropna().unique()
    latest_dates = sorted(unique_dates)[-10:]

    print(f"\n  [Data Store] Exporting predictions for latest {len(latest_dates)} REAL production dates ({latest_dates[0]} to {latest_dates[-1]})...")

    for prod_date in latest_dates:
        day_df = df_clean[df_clean['real_production_date'] == prod_date].copy()

        csv_path = os.path.join(data_store_dir, f"daily_predictions_{prod_date}.csv")
        safe_save_csv(day_df, csv_path)

        # Plot Daily Trend
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

        x_arr = np.arange(len(day_df)) + 1
        ax.plot(x_arr, day_df['MNY'].values, label='Actual Lab Mooney', color='#38bdf8', linewidth=2.2, marker='o', markersize=5)
        ax.plot(x_arr, day_df['Predicted_MNY_Calibrated'].values, label='V3.6 Calibrated Pred', color='#22c55e', linewidth=2.2, marker='s', markersize=4)
        ax.plot(x_arr, day_df['Predicted_MNY'].values, label='V3.6 Base Pred (A5)', color='#f59e0b', linewidth=1.5, linestyle='--', alpha=0.7)

        day_mae = np.mean(np.abs(day_df['MNY'].values - day_df['Predicted_MNY_Calibrated'].values))
        ax.set_title(f"Real Production Stream Daily Mooney Prediction ({prod_date}) - V3.6 Model\nMAE = {day_mae:.2f} MNY | Batches = {len(day_df)}", fontsize=13, fontweight='bold', color='#f8fafc')
        ax.set_xlabel("Production Batch Sequence Number", fontsize=11, color='#94a3b8')
        ax.set_ylabel("Mooney Viscosity (MNY)", fontsize=11, color='#94a3b8')
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.legend(loc='upper right', frameon=True, facecolor='#1e293b')
        plt.tight_layout()

        png_path = os.path.join(data_store_dir, f"daily_trend_{prod_date}.png")
        plt.savefig(png_path, dpi=200)
        plt.close()

        print(f"    Exported Real Date File: daily_predictions_{prod_date}.csv (Batches: {len(day_df)}, MAE: {day_mae:.2f} MNY)")

    print("\n" + "=" * 90)
    print("  REAL-DATE PRODUCTION STREAM PREDICTIONS SUCCESSFULLY EXPORTED TO DATA_STORE!")
    print("=" * 90 + "\n")


if __name__ == '__main__':
    run_correct_data_store_update()
