# ============================================================================
# V3.6 Priority 5: Carbon Black Expert V1 Evaluation Runner
# ============================================================================
# Evaluates the specialized 3-expert Carbon Black Subsystem Predictor V1
# (CB Prep Expert, CB Bottom Mixer Response Expert, CB Material Expert)
# against baseline GBDT on Carbon Black compounds.
#
# Generates carbon_black_expert_v1_benchmark.csv in reports/v36_explainable_production/
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
from model_training.trend_metrics import evaluate_mooney_predictions


def run_cb_expert_validation():
    print("=" * 80)
    print("  V3.6 PRIORITY 5: CARBON BLACK EXPERT V1 EVALUATION RUNNER")
    print("=" * 80)

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

    cb_mask = df_test['material_system'] == 'CarbonBlack'
    df_cb_test = df_test[cb_mask].copy()
    y_cb_actual = df_cb_test['MNY'].values

    # 1. Baseline Model (Standard GBDT Stage 2 for Carbon Black)
    model_base = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True, use_cb_subsystem=False)
    model_base.fit(df_train, s1_cols, s2_cols_pid, target_col='MNY', cluster_col='material_system')
    preds_base, _, _, _ = model_base.predict(df_test, cluster_col='material_system')
    preds_cb_base = preds_base[cb_mask]
    m_cb_base = evaluate_mooney_predictions(y_cb_actual, preds_cb_base, df_cb_test)

    # 2. V3.6 Model (With Specialized 3-Expert Carbon Black Subsystem Predictor V1)
    model_v36 = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True, use_cb_subsystem=True)
    model_v36.fit(df_train, s1_cols, s2_cols_pid, target_col='MNY', cluster_col='material_system')
    preds_v36, _, _, _ = model_v36.predict(df_test, cluster_col='material_system')
    preds_cb_v36 = preds_v36[cb_mask]
    m_cb_v36 = evaluate_mooney_predictions(y_cb_actual, preds_cb_v36, df_cb_test)

    # Overall metrics
    m_overall_base = evaluate_mooney_predictions(df_test['MNY'].values, preds_base, df_test)
    m_overall_v36 = evaluate_mooney_predictions(df_test['MNY'].values, preds_v36, df_test)

    bench_rows = [
        {
            'model_version': 'V3.5_Baseline_No_CB_Subsystem',
            'cb_subset_MAE': m_cb_base['MAE'],
            'cb_subset_RMSE': m_cb_base['RMSE'],
            'cb_subset_R2': m_cb_base['R2'],
            'cb_subset_Spearman': m_cb_base['Spearman_Rho'],
            'cb_subset_Direction_Accuracy_pct': m_cb_base['Direction_Accuracy'] * 100.0,
            'overall_MAE': m_overall_base['MAE'],
            'overall_Direction_Accuracy_pct': m_overall_base['Direction_Accuracy'] * 100.0,
        },
        {
            'model_version': 'V3.6_CarbonBlackExpertV1',
            'cb_subset_MAE': m_cb_v36['MAE'],
            'cb_subset_RMSE': m_cb_v36['RMSE'],
            'cb_subset_R2': m_cb_v36['R2'],
            'cb_subset_Spearman': m_cb_v36['Spearman_Rho'],
            'cb_subset_Direction_Accuracy_pct': m_cb_v36['Direction_Accuracy'] * 100.0,
            'overall_MAE': m_overall_v36['MAE'],
            'overall_Direction_Accuracy_pct': m_overall_v36['Direction_Accuracy'] * 100.0,
        },
    ]

    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_csv(os.path.join(out_dir, 'carbon_black_expert_v1_benchmark.csv'), index=False, encoding='utf-8-sig')

    # Print Summary Table
    print("\n" + "=" * 90)
    print("        CARBON BLACK EXPERT V1 BENCHMARK SUMMARY (V3.5 vs V3.6)")
    print("=" * 90)
    print(f"{'Model Version':<32} | {'CB MAE':<8} | {'CB RMSE':<8} | {'CB R2':<7} | {'CB Spearman':<12} | {'CB DirAcc(%)':<12}")
    print("-" * 90)
    for _, r in bench_df.iterrows():
        print(f"{r['model_version']:<32} | {r['cb_subset_MAE']:<8.4f} | {r['cb_subset_RMSE']:<8.4f} | {r['cb_subset_R2']:<7.4f} | {r['cb_subset_Spearman']:<12.4f} | {r['cb_subset_Direction_Accuracy_pct']:<12.2f}%")
    print("=" * 90)

    print(f"\nCarbon Black Expert V1 benchmark report saved to:")
    print(f"  {os.path.join(out_dir, 'carbon_black_expert_v1_benchmark.csv')}\n")

    return bench_df


if __name__ == '__main__':
    run_cb_expert_validation()
