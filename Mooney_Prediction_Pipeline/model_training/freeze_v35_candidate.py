# ============================================================================
# P0: Freeze Baseline Candidate V3.5_SilicaPIDExpertV1_candidate
# ============================================================================
# Freezes model configuration, feature lists, recipe split maps, benchmark
# summary, expert contribution reports, and reason code audits.
# Saves outputs to reports/v35_silica_pid_expert_validation/
# ============================================================================

import json
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


def freeze_v35_candidate():
    print("=" * 80)
    print("   P0: FREEZING MODEL CANDIDATE V3.5_SilicaPIDExpertV1_candidate")
    print("=" * 80)

    # 1. Output directory setup
    out_dir = os.path.join(pipeline_root, 'reports', 'v35_silica_pid_expert_validation')
    os.makedirs(out_dir, exist_ok=True)

    # 2. Load dataset
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

    # Save 1: split_recipe_map.csv
    r_col = 'Recipe_Code' if 'Recipe_Code' in df.columns else ('RecipeCode' if 'RecipeCode' in df.columns else ('Recipe' if 'Recipe' in df.columns else 'CompoundName'))
    split_map = df[[r_col, 'CompoundName', 'material_system', '_split', '_label_group_id']].drop_duplicates()
    split_map.to_csv(os.path.join(out_dir, 'split_recipe_map.csv'), index=False, encoding='utf-8-sig')

    # Fit Candidate Model V3.5
    model = HybridUnifiedMooneyModel(
        use_material_route_matrix=True,
        use_silica_subsystem=True,
    )
    model.fit(df_train, s1_cols, s2_cols_pid, target_col='MNY', cluster_col='material_system')

    # Save 2: model_config_snapshot.json
    config_snapshot = {
        'model_version': 'V3.5_SilicaPIDExpertV1_candidate',
        'shrinkage_k': model.shrinkage_k,
        'use_cluster_experts': model.use_cluster_experts,
        'use_material_route_matrix': model.use_material_route_matrix,
        'no_oil_expert_type': model.no_oil_expert_type,
        'use_silica_subsystem': model.use_silica_subsystem,
        'stage1_params': model.stage1_params,
        'stage2_params': model.stage2_params,
        'n_train_samples': len(df_train),
        'n_test_samples': len(df_test),
        'n_stage1_features': len(s1_cols),
        'n_stage2_features': len(s2_cols_pid),
    }
    with open(os.path.join(out_dir, 'model_config_snapshot.json'), 'w', encoding='utf-8') as f:
        json.dump(config_snapshot, f, indent=2)

    # Save 3: expert_feature_list.json
    silica_expert = model.stage2_experts_.get(('Silica', 'oil_wet')) or model.stage2_experts_.get(('Silica', 'no_oil_dry'))
    feature_list = {
        'stage1_features': s1_cols,
        'stage2_base_features': s2_cols_base,
        'stage2_pid_features': list(pid_feats.columns),
        'silica_sub_experts': {
            'pid_expert_features': getattr(silica_expert, 'feature_names_pid_', []),
            'wet_expert_features': getattr(silica_expert, 'feature_names_wet_', []),
            'bottom_expert_features': getattr(silica_expert, 'feature_names_bottom_', []),
            'material_expert_features': getattr(silica_expert, 'feature_names_material_', []),
        }
    }
    with open(os.path.join(out_dir, 'expert_feature_list.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_list, f, indent=2)

    # Predict
    final_preds, s1_preds, s1b_biases, s2_res_preds = model.predict(df_test, cluster_col='material_system')
    m_overall = evaluate_mooney_predictions(df_test['MNY'], final_preds, df_test)

    silica_mask = df_test['material_system'] == 'Silica'
    m_silica = evaluate_mooney_predictions(df_test.loc[silica_mask, 'MNY'], final_preds[silica_mask], df_test[silica_mask])

    # Save 4: benchmark_summary.csv
    bench_df = pd.DataFrame([{
        'model_version': 'V3.5_SilicaPIDExpertV1_candidate',
        'overall_MAE': m_overall['MAE'],
        'overall_RMSE': m_overall['RMSE'],
        'overall_R2': m_overall['R2'],
        'overall_VarRatio': m_overall['Variance_Ratio'],
        'overall_DirAcc_pct': m_overall['Direction_Accuracy'] * 100.0,
        'overall_Spearman_Rho': m_overall['Spearman_Rho'],
        'overall_HighDev_MAE': m_overall['High_Dev_MAE'],
        'silica_MAE': m_silica['MAE'],
        'silica_RMSE': m_silica['RMSE'],
        'silica_R2': m_silica['R2'],
        'silica_VarRatio': m_silica['Variance_Ratio'],
        'silica_DirAcc_pct': m_silica['Direction_Accuracy'] * 100.0,
        'silica_Spearman_Rho': m_silica['Spearman_Rho'],
        'silica_HighDev_MAE': m_silica['High_Dev_MAE'],
    }])
    bench_df.to_csv(os.path.join(out_dir, 'benchmark_summary.csv'), index=False, encoding='utf-8-sig')

    # Save 5: silica_expert_contribution_report.csv
    df_silica_test = df_test[silica_mask].copy()
    X_s2_delta_test, _ = model._transform_process_deltas(df_silica_test, 'material_system')
    sub_preds = silica_expert.predict_experts(X_s2_delta_test)
    contrib_df = pd.DataFrame({
        'CompoundName': df_silica_test['CompoundName'].values,
        'OrderID': df_silica_test['OrderID'].values,
        'actual_MNY': df_silica_test['MNY'].values,
        'stage1_pred': s1_preds[silica_mask],
        'bias_pred': s1b_biases[silica_mask],
        'pid_expert_pred': sub_preds['pid'],
        'wet_expert_pred': sub_preds['wet'],
        'bottom_expert_pred': sub_preds['bottom'],
        'material_expert_pred': sub_preds['material'],
        'final_pred': final_preds[silica_mask],
    })
    contrib_df.to_csv(os.path.join(out_dir, 'silica_expert_contribution_report.csv'), index=False, encoding='utf-8-sig')

    # Save 6: silica_reason_codes_audit.csv
    reason_df = silica_expert.generate_reason_codes(df_silica_test)
    reason_df.to_csv(os.path.join(out_dir, 'silica_reason_codes_audit.csv'), index=False, encoding='utf-8-sig')

    print(f"\nP0 baseline frozen successfully in: {out_dir}")
    print("Saved files:")
    print("  1. model_config_snapshot.json")
    print("  2. expert_feature_list.json")
    print("  3. split_recipe_map.csv")
    print("  4. benchmark_summary.csv")
    print("  5. silica_expert_contribution_report.csv")
    print("  6. silica_reason_codes_audit.csv\n")


if __name__ == '__main__':
    freeze_v35_candidate()
