# ============================================================================
# V3.2 Architecture Evaluation & Acceptance Gate Verification Script
# ============================================================================
# Runs benchmark comparison between V2.0 baseline and V3.2 Hybrid Architecture
# Checks all 4 acceptance gates defined in Mooney_Prediction_next plan_utlimate.docx
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
from model_training.effective_weighting import compute_effective_sample_weights
from model_training.hybrid_unified_model import HybridUnifiedMooneyModel
from model_training.label_group_handler import add_label_group_information
from model_training.split_builder import generate_stratified_recipe_splits, generate_time_holdout_split
from model_training.trend_metrics import evaluate_mooney_predictions


def run_evaluation():
    print("=" * 80)
    print("Starting V3.2 Hybrid Unified Architecture Evaluation Pipeline")
    print("=" * 80)
    
    # 1. Load Data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/stage_statistics_enriched_all_features_weather_v4.csv'))
    if not os.path.exists(data_path):
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/enriched_mny_all.csv'))
        
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Filter valid target
    if 'MNY' not in df.columns and 'Mooney_Viscosity' in df.columns:
        df['MNY'] = df['Mooney_Viscosity']
    df = df.dropna(subset=['MNY']).copy()
    
    # Ensure mandatory identifiers exist
    if 'CompoundName' not in df.columns and 'Compound' in df.columns:
        df['CompoundName'] = df['Compound']
    if 'OrderID' not in df.columns and 'Order_No' in df.columns:
        df['OrderID'] = df['Order_No']
        
    # 2. Phase 0.1: System Clustering
    df = cluster_silica_carbon_black(df)
    print(
        "Material-system classification applied. "
        f"Silica: {(df['material_system'] == 'Silica').sum()}, "
        f"Carbon Black: {(df['material_system'] == 'CarbonBlack').sum()}"
    )
    
    # 3. Phase 0.2: Feature Engineering
    s1_cols = extract_stage1_recipe_features(df)
    s2_cols = extract_stage2_process_features(df)
    
    # Ensure numeric types
    for col in s1_cols + s2_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
    # 4. Phase 0.3: Label groups, weights, and strict recipe-level split
    df = add_label_group_information(df)
    df = compute_effective_sample_weights(df)
    df = generate_stratified_recipe_splits(df, test_size=0.15, val_size=0.15)

    output_dir = os.path.join(pipeline_root, "reports", "architecture_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    split_columns = ["_recipe_code", "_split", "_label_group_id", "MNY"]
    df[split_columns].to_csv(
        os.path.join(output_dir, "split_recipe_map.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    leakage_audit = (
        pd.concat(
            [
                df.groupby("_recipe_code")["_split"].nunique().rename("n_splits"),
                df.groupby("_label_group_id")["_split"].nunique().rename("n_splits"),
            ],
            keys=["recipe_code", "label_group_id"],
        )
        .rename_axis(["group_col", "group_id"])
        .reset_index()
    )
    leakage_audit["status"] = np.where(leakage_audit["n_splits"] == 1, "PASS", "FAIL")
    leakage_audit.to_csv(
        os.path.join(output_dir, "split_leakage_audit.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    label_group_summary = (
        df.groupby("_label_group_id", as_index=False)
        .agg(
            n_rows=("_label_group_id", "size"),
            n_compounds=("CompoundName", "nunique"),
            mny_mean=("MNY", "mean"),
            mny_std=("MNY", "std"),
            metric_weight=("_w_metric", "sum"),
        )
    )
    label_group_summary.to_csv(
        os.path.join(output_dir, "label_group_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    df = generate_time_holdout_split(df)
    
    # Validation stays untouched for later model selection; P0 benchmark uses
    # strict train/test partitions only.
    df_train = df[df["_split"] == "train"].copy()
    df_test = df[df["_split"] == "test"].copy()
    print(
        f"Dataset split: Train = {len(df_train)} rows, "
        f"Validation = {(df['_split'] == 'val').sum()} rows, "
        f"Test = {len(df_test)} rows"
    )
    
    # -------------------------------------------------------------------------
    # Baseline Model (V2.0 Single GBDT on all features)
    # -------------------------------------------------------------------------
    print("\n--- Training V2.0 Baseline Model ---")
    from lightgbm import LGBMRegressor
    all_features = list(set(s1_cols + s2_cols))
    
    v2_model = LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42, verbose=-1)
    v2_model.fit(df_train[all_features], df_train['MNY'], sample_weight=df_train['_w_loss'])
    
    v2_test_preds = v2_model.predict(df_test[all_features])
    v2_metrics = evaluate_mooney_predictions(df_test['MNY'], v2_test_preds, df_test)
    
    # -------------------------------------------------------------------------
    # V3.2 Hybrid Unified Model (Stages 1, 1b, 2)
    # -------------------------------------------------------------------------
    print("\n--- Training V3.2 Hybrid Unified Architecture Model ---")
    v3_model = HybridUnifiedMooneyModel(shrinkage_k=5.0, use_cluster_experts=True)
    v3_model.fit(
        df_train,
        s1_cols,
        s2_cols,
        target_col='MNY',
        cluster_col='material_system',
    )
    
    v3_test_preds, s1_preds, s1b_biases, s2_res_preds = v3_model.predict(
        df_test,
        cluster_col='material_system',
    )
    s1b_test_preds = s1_preds + s1b_biases
    v3_metrics = evaluate_mooney_predictions(df_test['MNY'], v3_test_preds, df_test)
    stage_metrics = pd.DataFrame([
        {'stage': 'stage1_baseline', **evaluate_mooney_predictions(df_test['MNY'], s1_preds, df_test)},
        {'stage': 'stage1_plus_bias', **evaluate_mooney_predictions(df_test['MNY'], s1b_test_preds, df_test)},
        {'stage': 'stage1_bias_plus_process', **v3_metrics},
    ])
    stage_metrics.to_csv(
        os.path.join(output_dir, "stage_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    v3_model.get_bias_report().to_csv(
        os.path.join(output_dir, "compound_bias_report.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    v3_model.get_stage1_feature_importance().to_csv(
        os.path.join(output_dir, "feature_importance_stage1_recipe.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    v3_model.get_stage2_expert_report().to_csv(
        os.path.join(output_dir, "stage2_expert_report.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Out-of-time holdout: fit only on past recipes and score future recipes.
    # -------------------------------------------------------------------------
    df_oot_train = df[df["_time_split"] == "train"].copy()
    df_oot_test = df[df["_time_split"] == "oot_test"].copy()
    if df_oot_train.empty or df_oot_test.empty:
        raise ValueError("Time holdout must contain both train and oot_test rows.")

    oot_v2_model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        random_state=42,
        verbose=-1,
    )
    oot_v2_model.fit(
        df_oot_train[all_features],
        df_oot_train["MNY"],
        sample_weight=df_oot_train["_w_loss"],
    )
    oot_v2_preds = oot_v2_model.predict(df_oot_test[all_features])

    oot_v3_model = HybridUnifiedMooneyModel(
        shrinkage_k=5.0,
        use_cluster_experts=True,
    )
    oot_v3_model.fit(
        df_oot_train,
        s1_cols,
        s2_cols,
        target_col="MNY",
        cluster_col="material_system",
    )
    oot_final_preds, oot_s1_preds, oot_s1b_biases, _ = oot_v3_model.predict(
        df_oot_test,
        cluster_col="material_system",
    )
    time_holdout_metrics = pd.DataFrame([
        {
            "stage": "v2_all_features_baseline",
            "split_time": "oot",
            **evaluate_mooney_predictions(df_oot_test["MNY"], oot_v2_preds, df_oot_test),
        },
        {
            "stage": "stage1_baseline",
            "split_time": "oot",
            **evaluate_mooney_predictions(df_oot_test["MNY"], oot_s1_preds, df_oot_test),
        },
        {
            "stage": "stage1_plus_bias",
            "split_time": "oot",
            **evaluate_mooney_predictions(
                df_oot_test["MNY"],
                oot_s1_preds + oot_s1b_biases,
                df_oot_test,
            ),
        },
        {
            "stage": "stage1_bias_plus_process",
            "split_time": "oot",
            **evaluate_mooney_predictions(df_oot_test["MNY"], oot_final_preds, df_oot_test),
        },
    ])
    time_holdout_metrics.to_csv(
        os.path.join(output_dir, "time_holdout_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    
    # -------------------------------------------------------------------------
    # Print Comparison Table & Acceptance Gates
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("               MODEL PERFORMANCE COMPARISON (V2.0 vs V3.2)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'V2.0 Baseline':<15} | {'V3.2 Hybrid Unified':<20} | {'Improvement':<15}")
    print("-" * 80)
    
    def percentage_improvement(baseline, candidate):
        if not np.isfinite(baseline) or baseline == 0 or not np.isfinite(candidate):
            return np.nan
        return (baseline - candidate) / baseline * 100

    mae_imprv = percentage_improvement(v2_metrics['MAE'], v3_metrics['MAE'])
    rmse_imprv = percentage_improvement(v2_metrics['RMSE'], v3_metrics['RMSE'])
    r2_diff = v3_metrics['R2'] - v2_metrics['R2']
    var_rat_diff = v3_metrics['Variance_Ratio'] - v2_metrics['Variance_Ratio']
    dir_acc_diff = (v3_metrics['Direction_Accuracy'] - v2_metrics['Direction_Accuracy']) * 100
    spearman_diff = v3_metrics['Spearman_Rho'] - v2_metrics['Spearman_Rho']
    high_dev_imprv = percentage_improvement(v2_metrics['High_Dev_MAE'], v3_metrics['High_Dev_MAE'])
    
    print(f"{'MAE (Weighted)':<25} | {v2_metrics['MAE']:<15.4f} | {v3_metrics['MAE']:<20.4f} | {mae_imprv:+.2f}%")
    print(f"{'RMSE (Weighted)':<25} | {v2_metrics['RMSE']:<15.4f} | {v3_metrics['RMSE']:<20.4f} | {rmse_imprv:+.2f}%")
    print(f"{'R² Score':<25} | {v2_metrics['R2']:<15.4f} | {v3_metrics['R2']:<20.4f} | {r2_diff:+.4f}")
    print(f"{'Intra-Order Var Ratio':<25} | {v2_metrics['Variance_Ratio']:<15.4f} | {v3_metrics['Variance_Ratio']:<20.4f} | {var_rat_diff:+.4f}")
    print(f"{'Direction Acc (%)':<25} | {v2_metrics['Direction_Accuracy']*100:<15.2f}% | {v3_metrics['Direction_Accuracy']*100:<20.2f}% | {dir_acc_diff:+.2f}%")
    print(f"{'Spearman Rank Rho':<25} | {v2_metrics['Spearman_Rho']:<15.4f} | {v3_metrics['Spearman_Rho']:<20.4f} | {spearman_diff:+.4f}")
    print(f"{'High-Dev Batch MAE':<25} | {v2_metrics['High_Dev_MAE']:<15.4f} | {v3_metrics['High_Dev_MAE']:<20.4f} | {high_dev_imprv:+.2f}%")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Check Acceptance Gates
    # -------------------------------------------------------------------------
    print("\n--- PHASE 1.1 ACCEPTANCE GATES EVALUATION ---")
    oot_stage1b = time_holdout_metrics.loc[
        time_holdout_metrics['stage'] == 'stage1_plus_bias'
    ].iloc[0]
    oot_final = time_holdout_metrics.loc[
        time_holdout_metrics['stage'] == 'stage1_bias_plus_process'
    ].iloc[0]
    gate1_pass = mae_imprv >= 8.0
    gate2_pass = np.isfinite(high_dev_imprv) and high_dev_imprv >= 12.0
    gate3_pass = (
        v3_metrics['Direction_Accuracy'] >= v2_metrics['Direction_Accuracy']
        and v3_metrics['Spearman_Rho'] >= v2_metrics['Spearman_Rho']
    )
    gate4_pass = 0.70 <= v3_metrics['Variance_Ratio'] <= 1.20
    gate5_pass = oot_final['MAE'] <= oot_stage1b['MAE'] and oot_final['RMSE'] <= oot_stage1b['RMSE']
    gate_results = pd.DataFrame([
        {'gate': 'overall_mae_gain', 'passed': gate1_pass, 'value': mae_imprv, 'threshold': '>= 8.0%'},
        {'gate': 'high_deviation_mae_gain', 'passed': gate2_pass, 'value': high_dev_imprv, 'threshold': '>= 12.0%'},
        {'gate': 'trend_non_degradation', 'passed': gate3_pass, 'value': v3_metrics['Direction_Accuracy'], 'threshold': 'direction and rank >= baseline'},
        {'gate': 'variance_ratio', 'passed': gate4_pass, 'value': v3_metrics['Variance_Ratio'], 'threshold': '[0.70, 1.20]'},
        {'gate': 'oot_stage2_gain', 'passed': gate5_pass, 'value': oot_final['MAE'] - oot_stage1b['MAE'], 'threshold': '<= 0.0 MAE and RMSE'},
    ])
    gate_results.to_csv(
        os.path.join(output_dir, 'acceptance_gate_results.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    print(f"[Gate 1] Overall MAE Improvement (Target >= 8%): {'PASSED' if gate1_pass else 'FAILED'} ({mae_imprv:+.2f}%)")
    gate2_status = "NOT EVALUABLE" if not np.isfinite(high_dev_imprv) else ("PASSED" if gate2_pass else "FAILED")
    print(f"[Gate 2] High-Deviation Batch MAE (Target >= 12%): {gate2_status} ({high_dev_imprv:+.2f}%)")
    print(f"[Gate 3] Trend does not degrade vs baseline: {'PASSED' if gate3_pass else 'FAILED'}")
    print(f"[Gate 4] Variance Ratio inside [0.70, 1.20]: {'PASSED' if gate4_pass else 'FAILED'} ({v3_metrics['Variance_Ratio']:.4f})")
    print(f"[Gate 5] OOT Stage 2 improves on Stage 1 + Bias: {'PASSED' if gate5_pass else 'FAILED'}")
    
    return v2_metrics, v3_metrics


if __name__ == '__main__':
    run_evaluation()
