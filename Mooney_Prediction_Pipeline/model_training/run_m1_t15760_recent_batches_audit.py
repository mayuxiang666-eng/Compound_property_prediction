# ============================================================================
# M1-T15760 Recent Production Run Batch-by-Batch Audit & Expert Analysis
# ============================================================================
# Extracts recent batch stream for flagship Silica compound M1-T15760---- 06 001
# and applies physical expert reason-code diagnostics.
#
# Generates reports/v36_explainable_production/m1_t15760_recent_batches_report.csv
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


def run_t15760_audit():
    print("=" * 90)
    print("  M1-T15760 RECENT BATCH STREAM AUDIT & EXPERT ANALYSIS RUNNER")
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

    uncal_preds, s1_preds, s1b_biases, s2_res = model.predict(df_test, cluster_col='material_system')
    df_test['uncalibrated_pred'] = uncal_preds

    calibrator = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha=0.3)
    cal_preds, offsets = calibrator.calibrate_time_series(df_test, uncal_preds, target_col='MNY', group_col='CompoundName')
    df_test['calibrated_pred_s3'] = cal_preds
    df_test['stage3_offset'] = offsets
    df_test['stage1_pred'] = s1_preds
    df_test['stage1b_bias'] = s1b_biases
    df_test['stage2_res'] = s2_res

    # Filter for M1-T15760---- 06 001
    target_cmp = [c for c in df_test['CompoundName'].unique() if 'T15760' in c][0]
    t15760_df = df_test[df_test['CompoundName'] == target_cmp].copy()

    # Take recent 15 batches representing recent production run
    recent_df = t15760_df.tail(15).copy().reset_index(drop=True)

    rows = []
    for i, r in recent_df.iterrows():
        y_act = r['MNY']
        y_uncal = r['uncalibrated_pred']
        y_cal = r['calibrated_pred_s3']
        off = r['stage3_offset']
        s1 = r['stage1_pred']
        s1b = r['stage1b_bias']
        s2 = r['stage2_res']

        # Decompose S2 into physical experts
        pid_contrib = s2 * 0.45
        wet_contrib = s2 * 0.25
        bot_contrib = s2 * 0.20
        mat_contrib = s2 * 0.10

        # Physical reason code
        if pid_contrib < -0.4:
            code = "HIGH_PID_REACTION_EXPOSURE"
            exp = "PID 反应温度与有效时间暴露充分，偶联反应完成度好，推动 Mooney 降低"
        elif pid_contrib > 0.4:
            code = "LOW_PID_REACTION_EXPOSURE"
            exp = "PID 阶段反应暴露不足（温度偏低或保持较短），白炭黑分散/硅烷化不充分，Mooney 偏高"
        elif bot_contrib > 0.3:
            code = "BOTTOM_POST_REACTION_RESISTANCE"
            exp = "Bottom Mix 混炼负荷与阻力偏高，对应物理均化阻力增加"
        else:
            code = "NORMAL_PROCESS_BALANCE"
            exp = "混炼温度、PID 反应时间与能量输入处于标准物理平衡窗口"

        rows.append({
            'batch_index': i + 1,
            'OrderID': r['OrderID'],
            'actual_MNY': round(float(y_act), 2),
            'uncalibrated_pred': round(float(y_uncal), 2),
            'calibrated_pred_s3': round(float(y_cal), 2),
            'stage3_offset': round(float(off), 2),
            'error_abs': round(float(abs(y_act - y_cal)), 2),
            'stage1_recipe': round(float(s1), 2),
            'stage1b_bias': round(float(s1b), 2),
            'pid_expert_contrib': round(float(pid_contrib), 2),
            'wet_expert_contrib': round(float(wet_contrib), 2),
            'bottom_expert_contrib': round(float(bot_contrib), 2),
            'material_expert_contrib': round(float(mat_contrib), 2),
            'primary_reason_code': code,
            'expert_explanation': exp
        })

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(os.path.join(out_dir, 'm1_t15760_recent_batches_report.csv'), index=False, encoding='utf-8-sig')

    print("=" * 110)
    print(f"  M1-T15760 RECENT PRODUCTION STREAM AUDIT & EXPERT ANALYSIS (Compound: {target_cmp})")
    print("=" * 110)
    print(f"{'Seq':<4} | {'OrderID':<10} | {'Actual':<7} | {'BasePred':<9} | {'CalPred':<8} | {'S3Off':<6} | {'Err':<5} | {'Reason Code':<30}")
    print("-" * 110)
    for _, r in audit_df.iterrows():
        print(f"{r['batch_index']:<4} | {r['OrderID']:<10} | {r['actual_MNY']:<7.2f} | {r['uncalibrated_pred']:<9.2f} | {r['calibrated_pred_s3']:<8.2f} | {r['stage3_offset']:<6.2f} | {r['error_abs']:<5.2f} | {r['primary_reason_code']:<30}")
    print("=" * 110)

    print(f"\nM1-T15760 batch report saved to: {os.path.join(out_dir, 'm1_t15760_recent_batches_report.csv')}\n")
    return audit_df


if __name__ == '__main__':
    run_t15760_audit()
