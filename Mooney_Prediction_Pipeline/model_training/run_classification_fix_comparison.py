# ============================================================================
# Classification Fix Side-by-Side Comparison Audit Runner
# ============================================================================
# Evaluates exact performance BEFORE plant prefix fix vs AFTER plant prefix fix
# on identical test set split.
#
# Generates reports/v36_explainable_production/classification_fix_comparison_report.csv
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


def legacy_cluster_silica_cb(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy buggy classifier that misclassified T0 compounds as CarbonBlack."""
    df = df.copy()
    silica_cols = [c for c in df.columns if 'silica' in c.lower() or 'SiO2' in c]
    cb_cols = [c for c in df.columns if 'carbon_black' in c.lower() or 'N220' in c or 'N330' in c or 'CB' in c]
    
    def classify_name(row):
        comp = str(row.get('CompoundName', ''))
        s_val = sum(row[c] for c in silica_cols if pd.notnull(row[c])) if silica_cols else 0
        c_val = sum(row[c] for c in cb_cols if pd.notnull(row[c])) if cb_cols else 0
        if s_val > 0 or c_val > 0:
            return 'Silica' if s_val >= c_val else 'CarbonBlack'
        if any(k in comp.upper() for k in ['SILICA', 'SIL', 'SFE', 'T2', 'T1', 'T3', 'S1', 'S2']):
            return 'Silica'
        return 'CarbonBlack'
        
    df['material_system'] = df.apply(classify_name, axis=1)
    df['compound_cluster'] = df['material_system']
    return df


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


def run_comparison():
    print("=" * 90)
    print("  CLASSIFICATION FIX COMPARISON AUDIT RUNNER (BEFORE VS AFTER)")
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

    df_raw = pd.read_csv(data_path, low_memory=False)
    if 'MNY' not in df_raw.columns and 'Mooney_Viscosity' in df_raw.columns:
        df_raw['MNY'] = df_raw['Mooney_Viscosity']
    df_raw = df_raw.dropna(subset=['MNY']).copy()

    if 'CompoundName' not in df_raw.columns and 'Compound' in df_raw.columns:
        df_raw['CompoundName'] = df_raw['Compound']
    if 'OrderID' not in df_raw.columns and 'Order_No' in df_raw.columns:
        df_raw['OrderID'] = df_raw['Order_No']

    # Sort chronologically
    df_raw = df_raw.sort_values(by=['OrderID'] if 'OrderID' in df_raw.columns else df_raw.index).reset_index(drop=True)

    # PID features
    pid_feats = build_silica_pid_features(df_raw)
    for col in pid_feats.columns:
        df_raw[col] = pid_feats[col]

    # --- MODEL 1: BEFORE FIX (Legacy Buggy Classifier) ---
    df_legacy = legacy_cluster_silica_cb(df_raw)
    s1_cols = extract_stage1_recipe_features(df_legacy)
    s2_cols_base = extract_stage2_process_features(df_legacy)
    s2_cols_pid = list(set(s2_cols_base + list(pid_feats.columns)))

    for col in set(s1_cols + s2_cols_pid):
        if col in df_legacy.columns:
            df_legacy[col] = pd.to_numeric(df_legacy[col], errors='coerce').fillna(0.0)

    df_legacy = add_label_group_information(df_legacy)
    df_legacy = compute_effective_sample_weights(df_legacy)
    df_legacy = generate_stratified_recipe_splits(df_legacy, test_size=0.15, val_size=0.15)

    df_tr_leg = df_legacy[df_legacy['_split'] == 'train'].copy()
    df_te_leg = df_legacy[df_legacy['_split'] == 'test'].copy()

    model_leg = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True, use_cb_subsystem=True)
    model_leg.fit(df_tr_leg, s1_cols, s2_cols_pid, target_col='MNY', cluster_col='material_system')

    uncal_leg, _, _, _ = model_leg.predict(df_te_leg, cluster_col='material_system')
    cal_leg, _ = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha=0.3).calibrate_time_series(df_te_leg, uncal_leg, target_col='MNY', group_col='CompoundName')
    df_te_leg['pred_legacy'] = cal_leg

    # --- MODEL 2: AFTER FIX (Strict Plant Prefix Classifier) ---
    df_fix = cluster_silica_carbon_black(df_raw)
    for col in set(s1_cols + s2_cols_pid):
        if col in df_fix.columns:
            df_fix[col] = pd.to_numeric(df_fix[col], errors='coerce').fillna(0.0)

    df_fix = add_label_group_information(df_fix)
    df_fix = compute_effective_sample_weights(df_fix)
    df_fix = generate_stratified_recipe_splits(df_fix, test_size=0.15, val_size=0.15)

    df_tr_fix = df_fix[df_fix['_split'] == 'train'].copy()
    df_te_fix = df_fix[df_fix['_split'] == 'test'].copy()

    model_fix = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True, use_cb_subsystem=True)
    model_fix.fit(df_tr_fix, s1_cols, s2_cols_pid, target_col='MNY', cluster_col='material_system')

    uncal_fix, _, _, _ = model_fix.predict(df_te_fix, cluster_col='material_system')
    cal_fix, _ = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha=0.3).calibrate_time_series(df_te_fix, uncal_fix, target_col='MNY', group_col='CompoundName')
    df_te_fix['pred_fixed'] = cal_fix

    # Comparison per Compound
    compound_counts = df_te_fix['CompoundName'].value_counts()
    big_runner_compounds = compound_counts[compound_counts >= 10].index.tolist()

    rows = []
    for cmp in big_runner_compounds:
        sub_leg = df_te_leg[df_te_leg['CompoundName'] == cmp]
        sub_fix = df_te_fix[df_te_fix['CompoundName'] == cmp]
        if len(sub_fix) < 5:
            continue

        y_act = sub_fix['MNY'].values
        p_leg = sub_leg['pred_legacy'].values
        p_fix = sub_fix['pred_fixed'].values

        sys_leg = sub_leg['material_system'].iloc[0]
        sys_fix = sub_fix['material_system'].iloc[0]

        mae_leg = np.mean(np.abs(y_act - p_leg))
        mae_fix = np.mean(np.abs(y_act - p_fix))

        std_act = np.std(y_act)
        ratio_leg = np.std(p_leg) / (std_act + 1e-6)
        ratio_fix = np.std(p_fix) / (std_act + 1e-6)

        spearman_leg = float(pd.Series(y_act).corr(pd.Series(p_leg), method='spearman')) if len(np.unique(y_act)) > 1 else 0.0
        spearman_fix = float(pd.Series(y_act).corr(pd.Series(p_fix), method='spearman')) if len(np.unique(y_act)) > 1 else 0.0

        dir_leg = calculate_dir_acc(y_act, p_leg)
        dir_fix = calculate_dir_acc(y_act, p_fix)

        rows.append({
            'compound_name': cmp,
            'batch_count_N': len(sub_fix),
            'system_before': sys_leg,
            'system_after': sys_fix,
            'mae_before': round(float(mae_leg), 4),
            'mae_after': round(float(mae_fix), 4),
            'mae_diff': round(float(mae_fix - mae_leg), 4),
            'variance_ratio_before': round(float(ratio_leg), 4),
            'variance_ratio_after': round(float(ratio_fix), 4),
            'spearman_before': round(float(spearman_leg), 4),
            'spearman_after': round(float(spearman_fix), 4),
            'dir_acc_before_pct': round(float(dir_leg), 2),
            'dir_acc_after_pct': round(float(dir_fix), 2),
            'pid_expert_activated': 'YES' if (sys_fix == 'Silica') else 'NO'
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(out_dir, 'classification_fix_comparison_report.csv'), index=False, encoding='utf-8-sig')

    print("=" * 115)
    print("      CLASSIFICATION FIX SIDE-BY-SIDE COMPARISON (BEFORE VS AFTER PLANT PREFIX FIX)")
    print("=" * 115)
    print(f"{'Compound Name':<28} | {'Sys Before':<10} | {'Sys After':<11} | {'MAE Before':<10} | {'MAE After':<10} | {'Ratio After':<11} | {'DirAcc After':<12}")
    print("-" * 115)
    for _, r in comp_df.iterrows():
        print(f"{r['compound_name']:<28} | {r['system_before']:<10} | {r['system_after']:<11} | {r['mae_before']:<10.2f} | {r['mae_after']:<10.2f} | {r['variance_ratio_after']:<11.2f} | {r['dir_acc_after_pct']:<12.2f}%")
    print("=" * 115)

    print(f"\nComparison report saved to: {os.path.join(out_dir, 'classification_fix_comparison_report.csv')}\n")
    return comp_df


if __name__ == '__main__':
    run_comparison()
