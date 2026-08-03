# ============================================================================
# Priority 1: Arrhenius Proxy Controlled Validation Runner
# ============================================================================
# Evaluates if Arrhenius kinetic proxy is physically and empirically superior
# to traditional linear exposure proxy (Duration * Temperature).
#
# Experiments:
#   A0: Linear Exposure Proxy Only (Control)
#   A1: Arrhenius Kinetic Exposure Proxy Only
#   A2: Combined (Linear + Arrhenius Exposure Proxies)
#
# Metrics Evaluated:
#   - Silica Spearman Rho
#   - Silica Direction Accuracy (%)
#   - High-Deviation MAE
#   - Silica MAE / RMSE / R2
#
# Output:
#   reports/v35_silica_pid_expert_validation/arrhenius_proxy_validation_report.csv
#   reports/arrhenius_validation/arrhenius_proxy_validation_report.csv
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

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
from model_training.silica_subsystem import SilicaSubsystemPredictor
from model_training.trend_metrics import evaluate_mooney_predictions


def run_arrhenius_validation():
    print("=" * 80)
    print("  PRIORITY 1: ARRHENIUS PROXY CONTROLLED VALIDATION RUNNER")
    print("=" * 80)

    val_dir = os.path.join(pipeline_root, 'reports', 'v35_silica_pid_expert_validation')
    arr_dir = os.path.join(pipeline_root, 'reports', 'arrhenius_validation')
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(arr_dir, exist_ok=True)

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

    df = pd.read_csv(data_path, low_memory=False)
    if 'MNY' not in df.columns and 'Mooney_Viscosity' in df.columns:
        df['MNY'] = df['Mooney_Viscosity']
    df = df.dropna(subset=['MNY']).copy()

    if 'CompoundName' not in df.columns and 'Compound' in df.columns:
        df['CompoundName'] = df['Compound']
    if 'OrderID' not in df.columns and 'Order_No' in df.columns:
        df['OrderID'] = df['Order_No']

    # Build PID features with linear & Arrhenius proxies
    pid_feats = build_silica_pid_features(df)
    for col in pid_feats.columns:
        df[col] = pid_feats[col]

    df = cluster_silica_carbon_black(df)
    s1_cols = extract_stage1_recipe_features(df)
    s2_cols_base = extract_stage2_process_features(df)

    df = add_label_group_information(df)
    df = compute_effective_sample_weights(df)
    df = generate_stratified_recipe_splits(df, test_size=0.15, val_size=0.15)

    df_train = df[df['_split'] == 'train'].copy()
    df_test = df[df['_split'] == 'test'].copy()

    silica_test_mask = df_test['material_system'] == 'Silica'
    df_silica_test = df_test[silica_test_mask].copy()
    y_test_silica = df_silica_test['MNY'].values

    # Define 3 Controlled Validation Experiments
    experiments = [
        {
            'id': 'A0_Linear_Exposure_Only',
            'desc': 'Linear Exposure Proxy Only (Control)',
            'pid_cols': ['pid_linear_exposure_proxy', 'pid_high_temperature_risk_proxy', 'pid_control_instability_proxy', 'pid_mechanical_work_proxy'],
        },
        {
            'id': 'A1_Arrhenius_Exposure_Only',
            'desc': 'Arrhenius Kinetic Exposure Proxy Only',
            'pid_cols': ['pid_arrhenius_silanization_exposure_proxy', 'pid_high_temperature_risk_proxy', 'pid_control_instability_proxy', 'pid_mechanical_work_proxy'],
        },
        {
            'id': 'A2_Combined_Linear_And_Arrhenius',
            'desc': 'Combined (Linear + Arrhenius Exposure Proxies)',
            'pid_cols': ['pid_linear_exposure_proxy', 'pid_arrhenius_silanization_exposure_proxy', 'pid_high_temperature_risk_proxy', 'pid_control_instability_proxy', 'pid_mechanical_work_proxy'],
        },
    ]

    report_rows = []

    for exp in experiments:
        exp_id = exp['id']
        print(f"\nRunning Experiment {exp_id}: {exp['desc']}...")

        s2_features_curr = list(set(s2_cols_base + exp['pid_cols']))
        for c in s1_cols + s2_features_curr:
            if c in df.columns:
                df_train[c] = pd.to_numeric(df_train[c], errors='coerce').fillna(0.0)
                df_test[c] = pd.to_numeric(df_test[c], errors='coerce').fillna(0.0)

        # Fit model
        model = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True)
        model.fit(df_train, s1_cols, s2_features_curr, target_col='MNY', cluster_col='material_system')

        # Predict
        final_preds, s1_preds, s1b_biases, s2_res_preds = model.predict(df_test, cluster_col='material_system')

        # Silica test metrics
        silica_preds = final_preds[silica_test_mask]
        m_silica = evaluate_mooney_predictions(y_test_silica, silica_preds, df_silica_test)

        # Overall test metrics
        m_overall = evaluate_mooney_predictions(df_test['MNY'].values, final_preds, df_test)

        report_rows.append({
            'experiment_id': exp_id,
            'description': exp['desc'],
            'silica_Spearman': m_silica['Spearman_Rho'],
            'silica_Direction_Accuracy_pct': m_silica['Direction_Accuracy'] * 100.0,
            'high_deviation_MAE': m_silica['High_Dev_MAE'],
            'silica_MAE': m_silica['MAE'],
            'silica_RMSE': m_silica['RMSE'],
            'silica_R2': m_silica['R2'],
            'silica_Variance_Ratio': m_silica['Variance_Ratio'],
            'overall_MAE': m_overall['MAE'],
            'overall_Direction_Accuracy_pct': m_overall['Direction_Accuracy'] * 100.0,
        })

    val_df = pd.DataFrame(report_rows)
    val_df.to_csv(os.path.join(val_dir, 'arrhenius_proxy_validation_report.csv'), index=False, encoding='utf-8-sig')
    val_df.to_csv(os.path.join(arr_dir, 'arrhenius_proxy_validation_report.csv'), index=False, encoding='utf-8-sig')

    # Print Comparison Table
    print("\n" + "=" * 95)
    print("        ARRHENIUS PROXY CONTROLLED VALIDATION SUMMARY (A0 vs A1 vs A2)")
    print("=" * 95)
    print(f"{'Experiment':<32} | {'Silica Spearman':<15} | {'Silica DirAcc(%)':<16} | {'HighDev MAE':<12} | {'Silica MAE':<10}")
    print("-" * 95)
    for _, r in val_df.iterrows():
        print(f"{r['experiment_id']:<32} | {r['silica_Spearman']:<15.4f} | {r['silica_Direction_Accuracy_pct']:<16.2f}% | {r['high_deviation_MAE']:<12.4f} | {r['silica_MAE']:<10.4f}")
    print("=" * 95)

    a0_sp = val_df.loc[val_df['experiment_id'] == 'A0_Linear_Exposure_Only', 'silica_Spearman'].values[0]
    a1_sp = val_df.loc[val_df['experiment_id'] == 'A1_Arrhenius_Exposure_Only', 'silica_Spearman'].values[0]
    a2_sp = val_df.loc[val_df['experiment_id'] == 'A2_Combined_Linear_And_Arrhenius', 'silica_Spearman'].values[0]

    print("\n" + "=" * 80)
    print("        ARRHENIUS PROXY PHYSICAL VALIDATION VERDICT")
    print("=" * 80)
    if a1_sp > a0_sp or a2_sp > a0_sp:
        best_exp = 'A1_Arrhenius_Exposure_Only' if a1_sp >= a2_sp else 'A2_Combined_Linear_And_Arrhenius'
        print(f"  VERDICT: Arrhenius Proxy VALIDATED! ({best_exp} outperforms Linear Proxy A0).")
        print(f"  Silica Spearman improved from {a0_sp:.4f} (Linear) to {max(a1_sp, a2_sp):.4f} (Arrhenius).\n")
    else:
        print("  VERDICT: Linear Exposure Proxy A0 remains superior or equivalent on current split.\n")

    print(f"Validation report saved to:")
    print(f"  1. {os.path.join(val_dir, 'arrhenius_proxy_validation_report.csv')}")
    print(f"  2. {os.path.join(arr_dir, 'arrhenius_proxy_validation_report.csv')}\n")

    return val_df


if __name__ == '__main__':
    run_arrhenius_validation()
