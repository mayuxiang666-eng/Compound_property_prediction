# ============================================================================
# Silica PID Residual Expert V1 Evaluation & Benchmark Runner
# ============================================================================
# Evaluates the Silica PID Residual Subsystem V1 (4 parallel sub-experts + OOF combiner)
# against baseline configurations on identical leak-free recipe splits.
#
# Generates:
#   1. silica_expert_contribution_report.csv
#   2. silica_reason_codes_audit.csv
#   3. silica_pid_v1_benchmark_summary.csv
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


def run_silica_pid_expert_v1():
    print("=" * 80)
    print("      SILICA PID RESIDUAL EXPERT V1 EVALUATION & BENCHMARK")
    print("=" * 80)

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

    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    if 'MNY' not in df.columns and 'Mooney_Viscosity' in df.columns:
        df['MNY'] = df['Mooney_Viscosity']
    df = df.dropna(subset=['MNY']).copy()

    if 'CompoundName' not in df.columns and 'Compound' in df.columns:
        df['CompoundName'] = df['Compound']
    if 'OrderID' not in df.columns and 'Order_No' in df.columns:
        df['OrderID'] = df['Order_No']

    # Pre-compute Silica PID v0 features
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

    print(f"Dataset split: Train = {len(df_train)} rows | Test = {len(df_test)} rows\n")

    # Benchmarks to compare
    configs = [
        {
            'id': 'V3.3_Baseline',
            'desc': 'V3.3 Phase Route Baseline (2 LGBM experts)',
            'use_matrix': False,
            'use_silica_subsystem': False,
        },
        {
            'id': 'V3.4_E1_4Experts',
            'desc': 'Material x Route 4 LGBM Experts',
            'use_matrix': True,
            'use_silica_subsystem': False,
        },
        {
            'id': 'Silica_PID_Expert_V1',
            'desc': 'Silica PID Residual Subsystem V1 (4 sub-experts + OOF combiner)',
            'use_matrix': True,
            'use_silica_subsystem': True,
        },
    ]

    benchmark_rows = []
    output_dir = os.path.join(pipeline_root, 'reports', 'silica_pid_expert_v1')
    os.makedirs(output_dir, exist_ok=True)

    silica_test_mask = df_test['material_system'] == 'Silica'

    for cfg in configs:
        model_id = cfg['id']
        print(f"--- Training Model: {model_id} ({cfg['desc']}) ---")

        model = HybridUnifiedMooneyModel(
            use_material_route_matrix=cfg['use_matrix'],
            use_silica_subsystem=cfg['use_silica_subsystem'],
        )
        model.fit(
            df_train,
            s1_cols,
            s2_cols_pid,
            target_col='MNY',
            cluster_col='material_system',
        )

        final_preds, s1_preds, s1b_biases, s2_res_preds = model.predict(
            df_test,
            cluster_col='material_system',
        )

        # 1. Overall Evaluation
        m_overall = evaluate_mooney_predictions(df_test['MNY'], final_preds, df_test)

        # 2. Silica Subset Evaluation
        df_silica_test = df_test[silica_test_mask].copy()
        y_silica_test = df_test.loc[silica_test_mask, 'MNY']
        preds_silica_test = final_preds[silica_test_mask]
        m_silica = evaluate_mooney_predictions(y_silica_test, preds_silica_test, df_silica_test)

        benchmark_rows.append({
            'model_id': model_id,
            'description': cfg['desc'],
            # Overall Metrics
            'overall_MAE': m_overall['MAE'],
            'overall_RMSE': m_overall['RMSE'],
            'overall_R2': m_overall['R2'],
            'overall_VarRatio': m_overall['Variance_Ratio'],
            'overall_DirAcc_pct': m_overall['Direction_Accuracy'] * 100.0,
            'overall_Spearman_Rho': m_overall['Spearman_Rho'],
            'overall_HighDev_MAE': m_overall['High_Dev_MAE'],
            # Silica Subset Metrics
            'silica_MAE': m_silica['MAE'],
            'silica_RMSE': m_silica['RMSE'],
            'silica_R2': m_silica['R2'],
            'silica_VarRatio': m_silica['Variance_Ratio'],
            'silica_DirAcc_pct': m_silica['Direction_Accuracy'] * 100.0,
            'silica_Spearman_Rho': m_silica['Spearman_Rho'],
            'silica_HighDev_MAE': m_silica['High_Dev_MAE'],
        })

        # Save contribution and reason code audit for Silica PID Expert V1
        if cfg['use_silica_subsystem']:
            silica_expert_obj = model.stage2_experts_.get(('Silica', 'oil_wet')) or model.stage2_experts_.get(('Silica', 'no_oil_dry'))
            if hasattr(silica_expert_obj, 'generate_reason_codes'):
                reason_df = silica_expert_obj.generate_reason_codes(df_silica_test)
                reason_df.to_csv(os.path.join(output_dir, 'silica_reason_codes_audit.csv'), index=False, encoding='utf-8-sig')

                # Predict sub-expert components
                X_s2_delta_test, route_test = model._transform_process_deltas(df_silica_test, 'material_system')
                sub_preds = silica_expert_obj.predict_experts(X_s2_delta_test)
                contrib_df = pd.DataFrame({
                    'CompoundName': df_silica_test['CompoundName'].values,
                    'OrderID': df_silica_test['OrderID'].values,
                    'actual_MNY': y_silica_test.values,
                    'stage1_pred': s1_preds[silica_test_mask],
                    'bias_pred': s1b_biases[silica_test_mask],
                    'pid_expert_pred': sub_preds['pid'],
                    'wet_expert_pred': sub_preds['wet'],
                    'bottom_expert_pred': sub_preds['bottom'],
                    'material_expert_pred': sub_preds['material'],
                    'final_pred': preds_silica_test,
                })
                contrib_df.to_csv(os.path.join(output_dir, 'silica_expert_contribution_report.csv'), index=False, encoding='utf-8-sig')

    bench_df = pd.DataFrame(benchmark_rows)
    bench_df.to_csv(os.path.join(output_dir, 'silica_pid_v1_benchmark_summary.csv'), index=False, encoding='utf-8-sig')

    # Print Summary Tables
    print("\n" + "=" * 90)
    print("                 OVERALL DATASET BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Model ID':<25} | {'MAE':<7} | {'RMSE':<7} | {'R2':<7} | {'VarRatio':<8} | {'Spearman':<8} | {'DirAcc(%)':<8}")
    print("-" * 90)
    for _, r in bench_df.iterrows():
        print(f"{r['model_id']:<25} | {r['overall_MAE']:<7.4f} | {r['overall_RMSE']:<7.4f} | {r['overall_R2']:<7.4f} | {r['overall_VarRatio']:<8.4f} | {r['overall_Spearman_Rho']:<8.4f} | {r['overall_DirAcc_pct']:<8.2f}%")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("                 SILICA SUBSET BENCHMARK SUMMARY (Key ROI Target)")
    print("=" * 90)
    print(f"{'Model ID':<25} | {'MAE':<7} | {'RMSE':<7} | {'R2':<7} | {'VarRatio':<8} | {'Spearman':<8} | {'DirAcc(%)':<8}")
    print("-" * 90)
    for _, r in bench_df.iterrows():
        print(f"{r['model_id']:<25} | {r['silica_MAE']:<7.4f} | {r['silica_RMSE']:<7.4f} | {r['silica_R2']:<7.4f} | {r['silica_VarRatio']:<8.4f} | {r['silica_Spearman_Rho']:<8.4f} | {r['silica_DirAcc_pct']:<8.2f}%")
    print("=" * 90)
    print(f"Reports saved to: {output_dir}\n")

    return bench_df


if __name__ == '__main__':
    run_silica_pid_expert_v1()
