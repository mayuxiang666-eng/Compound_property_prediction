# ============================================================================
# V3.4 Recommended Experiment Matrix Runner (E0 - E4)
# ============================================================================
# Evaluates 5 targeted experiments on identical leak-free recipe splits:
#   E0: V3.3 Phase Route Baseline (2 LGBM experts)
#   E1: Material x Route 4 LGBM experts
#   E2: No-Oil-Dry Linear Expert (Ridge/Huber fallback)
#   E3: Expert-specific Variance Gate Audit
#   E4: Silica PID v0 Features enabled
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


def run_experiment_matrix():
    print("=" * 80)
    print("      V3.4 RECOMMENDED EXPERIMENT MATRIX RUNNER (E0 - E4)")
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

    # Cluster material system
    df = cluster_silica_carbon_black(df)
    s1_cols = extract_stage1_recipe_features(df)
    s2_cols_base = extract_stage2_process_features(df)
    s2_cols_pid = list(set(s2_cols_base + list(pid_feats.columns)))

    for col in set(s1_cols + s2_cols_pid):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Split dataset
    df = add_label_group_information(df)
    df = compute_effective_sample_weights(df)
    df = generate_stratified_recipe_splits(df, test_size=0.15, val_size=0.15)

    df_train = df[df['_split'] == 'train'].copy()
    df_test = df[df['_split'] == 'test'].copy()

    print(f"Dataset split: Train = {len(df_train)} rows | Test = {len(df_test)} rows\n")

    # Define Experiment Configurations
    experiments = [
        {
            'exp_id': 'E0_V33_Baseline',
            'desc': 'V3.3 Phase Route Baseline (2 LGBM experts)',
            'use_matrix': False,
            'no_oil_type': 'lightgbm',
            'use_pid': False,
        },
        {
            'exp_id': 'E1_Material_Route_4Experts',
            'desc': 'Material System x Route 4 LGBM experts',
            'use_matrix': True,
            'no_oil_type': 'lightgbm',
            'use_pid': False,
        },
        {
            'exp_id': 'E2_NoOil_Linear_Expert',
            'desc': 'Material x Route 4 experts (No-Oil Huber Linear)',
            'use_matrix': True,
            'no_oil_type': 'huber',
            'use_pid': False,
        },
        {
            'exp_id': 'E3_Expert_Variance_Gate',
            'desc': 'Material x Route 4 experts (Route Ridge fallback)',
            'use_matrix': True,
            'no_oil_type': 'ridge',
            'use_pid': False,
        },
        {
            'exp_id': 'E4_Silica_PID_v0',
            'desc': 'Material x Route 4 experts + Silica PID v0 features',
            'use_matrix': True,
            'no_oil_type': 'lightgbm',
            'use_pid': True,
        },
    ]

    exp_results = []
    output_dir = os.path.join(pipeline_root, 'reports', 'v34_experiment_matrix')
    os.makedirs(output_dir, exist_ok=True)

    for cfg in experiments:
        exp_id = cfg['exp_id']
        print(f"--- Running Experiment: {exp_id} ({cfg['desc']}) ---")
        
        s2_features = s2_cols_pid if cfg['use_pid'] else s2_cols_base

        model = HybridUnifiedMooneyModel(
            use_material_route_matrix=cfg['use_matrix'],
            no_oil_expert_type=cfg['no_oil_type'],
        )
        model.fit(
            df_train,
            s1_cols,
            s2_features,
            target_col='MNY',
            cluster_col='material_system',
        )

        final_preds, s1_preds, s1b_biases, s2_res_preds = model.predict(
            df_test,
            cluster_col='material_system',
        )

        metrics = evaluate_mooney_predictions(df_test['MNY'], final_preds, df_test)

        # Compute intra-order residual variance capture ratio for Stage 2
        orders = df_test['OrderID'].values if 'OrderID' in df_test.columns else np.zeros(len(df_test))
        res_for_s2 = df_test['MNY'].values - (s1_preds + s1b_biases)
        var_res_s2 = float(np.mean([np.var(g['v'].values) for _, g in pd.DataFrame({'v': res_for_s2, 'o': orders}).groupby('o') if len(g) >= 3])) if len(orders) > 0 else 1.0
        var_s2_out = float(np.mean([np.var(g['v'].values) for _, g in pd.DataFrame({'v': s2_res_preds, 'o': orders}).groupby('o') if len(g) >= 3])) if len(orders) > 0 else 0.0
        s2_capture_pct = (var_s2_out / var_res_s2 * 100.0) if var_res_s2 > 1e-5 else 0.0

        capture_tier = (
            "Strong Pass (>= 35%)" if s2_capture_pct >= 35.0 else
            ("Target Pass (>= 25%)" if s2_capture_pct >= 25.0 else
             ("Minimum Pass (>= 10%)" if s2_capture_pct >= 10.0 else "FAIL (< 10%)"))
        )

        exp_results.append({
            'experiment_id': exp_id,
            'description': cfg['desc'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'R2': metrics['R2'],
            'Variance_Ratio': metrics['Variance_Ratio'],
            'Direction_Accuracy_pct': metrics['Direction_Accuracy'] * 100.0,
            'Spearman_Rho': metrics['Spearman_Rho'],
            'High_Dev_MAE': metrics['High_Dev_MAE'],
            'Stage2_Capture_pct': s2_capture_pct,
            'Stage2_Capture_Tier': capture_tier,
        })

    results_df = pd.DataFrame(exp_results)
    results_df.to_csv(os.path.join(output_dir, 'v34_experiment_matrix_summary.csv'), index=False, encoding='utf-8-sig')

    # Print summary table
    print("\n" + "=" * 90)
    print("                 V3.4 EXPERIMENT MATRIX EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'Exp ID':<25} | {'MAE':<7} | {'RMSE':<7} | {'R2':<7} | {'VarRatio':<8} | {'S2 Capture':<15} | {'DirAcc(%)':<8}")
    print("-" * 90)
    for _, r in results_df.iterrows():
        print(f"{r['experiment_id']:<25} | {r['MAE']:<7.4f} | {r['RMSE']:<7.4f} | {r['R2']:<7.4f} | {r['Variance_Ratio']:<8.4f} | {r['Stage2_Capture_pct']:>5.2f}% [{r['Stage2_Capture_Tier'].split()[0]}] | {r['Direction_Accuracy_pct']:<8.2f}%")
    print("=" * 90)
    print(f"Results saved to: {output_dir}\n")

    return results_df


if __name__ == '__main__':
    run_experiment_matrix()
