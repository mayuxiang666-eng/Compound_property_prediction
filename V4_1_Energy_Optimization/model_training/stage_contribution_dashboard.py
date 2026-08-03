# ============================================================================
# V3.6 Explainability Engine: Stage Contribution Dashboard & Reason Confidence
# ============================================================================
# Implements:
#   P1: Stage Contribution Dashboard (Layer-by-layer breakdown: S1, S1b, S2 Experts)
#   P2: Reason Code Confidence Audit (Multi-tier HIGH/MEDIUM/LOW confidence mapping)
#   P3: PID Window Mapping (Physical temperature/time window rules)
# Saves outputs to reports/v36_explainable_production/
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


def build_explainability_dashboard():
    print("=" * 80)
    print("  V3.6 EXPLAINABILITY ENGINE: STAGE CONTRIBUTION & CONFIDENCE DASHBOARD")
    print("=" * 80)

    out_dir = os.path.join(pipeline_root, 'reports', 'v36_explainable_production')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Dataset
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

    # Fit Production Candidate V3.5
    model = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True)
    model.fit(df_train, s1_cols, s2_cols_pid, target_col='MNY', cluster_col='material_system')

    final_preds, s1_preds, s1b_biases, s2_res_preds = model.predict(df_test, cluster_col='material_system')

    # Sub-expert predictions for Silica test set
    silica_test_mask = df_test['material_system'] == 'Silica'
    df_silica = df_test[silica_test_mask].copy()

    silica_expert = model.stage2_experts_.get(('Silica', 'oil_wet')) or model.stage2_experts_.get(('Silica', 'no_oil_dry'))
    X_s2_delta_test, _ = model._transform_process_deltas(df_silica, 'material_system')
    sub_preds = silica_expert.predict_experts(X_s2_delta_test)
    reason_df = silica_expert.generate_reason_codes(df_silica)

    # -------------------------------------------------------------------------
    # P1: Stage Contribution Dashboard (stage_contribution_summary.csv)
    # -------------------------------------------------------------------------
    y_silica = df_silica['MNY'].values
    p_s1 = s1_preds[silica_test_mask]
    p_s1b = s1b_biases[silica_test_mask]
    p_final = final_preds[silica_test_mask]

    contrib_df = pd.DataFrame({
        'OrderID': df_silica['OrderID'].values,
        'CompoundName': df_silica['CompoundName'].values,
        'actual_MNY': y_silica,
        'stage1_recipe_pred': p_s1,
        'stage1b_bias_pred': p_s1b,
        'pid_expert_pred': sub_preds['pid'],
        'wet_expert_pred': sub_preds['wet'],
        'bottom_expert_pred': sub_preds['bottom'],
        'material_expert_pred': sub_preds['material'],
        'final_prediction': p_final,
        'residual': y_silica - p_final,
        'stage1_pct': np.abs(p_s1) / (np.abs(p_s1) + np.abs(p_s1b) + np.abs(p_final - p_s1 - p_s1b) + 1e-5) * 100.0,
        'stage1b_pct': np.abs(p_s1b) / (np.abs(p_s1) + np.abs(p_s1b) + np.abs(p_final - p_s1 - p_s1b) + 1e-5) * 100.0,
        'stage2_expert_pct': np.abs(p_final - p_s1 - p_s1b) / (np.abs(p_s1) + np.abs(p_s1b) + np.abs(p_final - p_s1 - p_s1b) + 1e-5) * 100.0,
    })
    contrib_df.to_csv(os.path.join(out_dir, 'stage_contribution_summary.csv'), index=False, encoding='utf-8-sig')

    # -------------------------------------------------------------------------
    # P2: Reason Code Confidence Audit (reason_code_confidence_audit.csv)
    # -------------------------------------------------------------------------
    df_silica['pred_MNY'] = p_final
    df_silica['primary_reason_code'] = reason_df['primary_reason_code'].values
    df_silica['confidence_level'] = reason_df['confidence_level'].values

    conf_audit_rows = []
    for (code, conf), group in df_silica.groupby(['primary_reason_code', 'confidence_level'], observed=True):
        if len(group) == 0:
            continue
        y_act = group['MNY'].values
        y_pr = group['pred_MNY'].values
        m_eval = evaluate_mooney_predictions(y_act, y_pr, group)
        conf_audit_rows.append({
            'primary_reason_code': code,
            'confidence_level': conf,
            'n_batches': len(group),
            'pct_of_total': len(group) / len(df_silica) * 100.0,
            'MAE': m_eval['MAE'],
            'RMSE': m_eval['RMSE'],
            'Spearman': m_eval['Spearman_Rho'],
            'Direction_Accuracy_pct': m_eval['Direction_Accuracy'] * 100.0,
            'mean_residual': float(np.mean(y_act - y_pr)),
        })

    conf_df = pd.DataFrame(conf_audit_rows)
    conf_df.to_csv(os.path.join(out_dir, 'reason_code_confidence_audit.csv'), index=False, encoding='utf-8-sig')

    # -------------------------------------------------------------------------
    # P3: PID Window Mapping Rules (pid_window_mapping_rules.csv)
    # -------------------------------------------------------------------------
    rules = [
        {
            'window_id': 'W1_Normal_Silanization',
            'temperature_range_c': '135°C - 155°C',
            'exposure_range': '10 - 25',
            'control_stability': 'temp_std < 1.5°C',
            'actionable_rule': 'Reaction window optimal. Maintain current mixing speed and ram pressure.',
            'target_mooney_effect': 'Nominal baseline (0.0 ± 0.5 MNY)',
        },
        {
            'window_id': 'W2_Low_Exposure_Reaction_Deficit',
            'temperature_range_c': '< 135°C',
            'exposure_range': '< 10',
            'control_stability': 'temp_std < 2.0°C',
            'actionable_rule': 'Under-reaction risk! Increase mixing duration by +15s or raise dump temp setpoint.',
            'target_mooney_effect': 'Elevated Mooney (+1.5 ~ +3.0 MNY due to incomplete coupling)',
        },
        {
            'window_id': 'W3_Thermal_Scorch_Risk',
            'temperature_range_c': '> 160°C',
            'exposure_range': '> 25',
            'control_stability': 'temp_std > 2.5°C',
            'actionable_rule': 'Thermal overshoot hazard! Reduce rotor RPM immediately or discharge batch early.',
            'target_mooney_effect': 'Scorched Mooney shift (+2.0 ~ +4.5 MNY due to premature crosslinking)',
        },
        {
            'window_id': 'W4_Control_Instability_Fluctuation',
            'temperature_range_c': 'Any',
            'exposure_range': 'Any',
            'control_stability': 'temp_std > 2.5°C',
            'actionable_rule': 'PID control instability. Inspect TCU water valve response and oil injection rate.',
            'target_mooney_effect': 'Increased intra-order variance (std > 2.5 MNY)',
        },
    ]

    rules_df = pd.DataFrame(rules)
    rules_df.to_csv(os.path.join(out_dir, 'pid_window_mapping_rules.csv'), index=False, encoding='utf-8-sig')

    # Print Summary Tables
    print("\n" + "=" * 90)
    print("            REASON CODE CONFIDENCE AUDIT SUMMARY (V3.6)")
    print("=" * 90)
    print(f"{'Reason Code':<30} | {'Conf':<6} | {'n':<5} | {'Pct(%)':<7} | {'MAE':<7} | {'Spearman':<8} | {'DirAcc(%)':<8}")
    print("-" * 90)
    for _, r in conf_df.iterrows():
        print(f"{r['primary_reason_code']:<30} | {r['confidence_level']:<6} | {r['n_batches']:<5} | {r['pct_of_total']:<7.1f}% | {r['MAE']:<7.4f} | {r['Spearman']:<8.4f} | {r['Direction_Accuracy_pct']:<8.2f}%")
    print("=" * 90)

    print(f"\nExplainability reports generated in: {out_dir}")
    print("Files created:")
    print("  1. stage_contribution_summary.csv")
    print("  2. reason_code_confidence_audit.csv")
    print("  3. pid_window_mapping_rules.csv\n")

    return contrib_df


if __name__ == '__main__':
    build_explainability_dashboard()
