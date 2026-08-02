# ============================================================================
# Master Orchestrator: Comprehensive V4.1 Energy Experiment & Report Suite
# ============================================================================
# Generates:
# 1. energy_prediction_metrics.csv
# 2. mode_b_branch_metrics.csv
# 3. high_energy_subset_metrics.csv
# 4. recipe_energy_benchmark.csv (with P10, P50, P90 & estimated saving potential)
# 5. shadow_recommendation_summary.csv
# 6. shadow_recommendation_rejection_reasons.csv
# 7. energy_mooney_tradeoff_report.csv
# 8. safe_parameter_bounds.csv
# 9. safe_bounds_coverage_summary.csv
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

from feature_engineering.clustering import cluster_silica_carbon_black
from feature_engineering.energy_label_builder import build_energy_labels_and_features, audit_energy_labels
from feature_engineering.silica_pid_feature_builder import build_silica_pid_features
from feature_engineering.cb_dispersion_feature_builder import build_cb_dispersion_features
from feature_engineering.stage1_recipe_features import extract_stage1_recipe_features
from feature_engineering.stage2_process_features import extract_stage2_process_features
from model_training.effective_weighting import compute_effective_sample_weights
from model_training.energy_model import MixingEnergyPredictionModel, evaluate_energy_model_performance
from model_training.hybrid_unified_model import HybridUnifiedMooneyModel
from model_training.label_group_handler import add_label_group_information
from model_training.split_builder import generate_stratified_recipe_splits
from optimization.safe_bounds_builder import build_safe_parameter_bounds
from optimization.mooney_constraint_checker import MooneyQualityConstraintChecker
from optimization.energy_optimizer import EnergyMooneyOptimizer


def run_v41_master_experiment():
    print("=" * 95, flush=True)
    print("  RUNNING V4.1 COMPREHENSIVE MIXING ENERGY PREDICTION & OPTIMIZATION EXPERIMENTS", flush=True)
    print("=" * 95, flush=True)

    out_dir = os.path.join(pipeline_root, 'reports', 'v41_energy_optimization')
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------------
    # STEP 1 & 2: Load Data & Build Energy Labels / Segregation Features
    # ------------------------------------------------------------------------
    print("\n[Task 1 & 2] Loading Dataset & Building Labels & Features...", flush=True)
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
    df_clean = df_raw.dropna(subset=['MNY']).copy()

    if 'CompoundName' not in df_clean.columns and 'Compound' in df_clean.columns:
        df_clean['CompoundName'] = df_clean['Compound']
    if 'OrderID' not in df_clean.columns and 'Order_No' in df_clean.columns:
        df_clean['OrderID'] = df_clean['Order_No']
    if 'MixerLine' not in df_clean.columns and 'Mixer_ID' in df_clean.columns:
        df_clean['MixerLine'] = df_clean['Mixer_ID']

    # Extract PID & CB features
    pid_feats = build_silica_pid_features(df_clean)
    cb_feats = build_cb_dispersion_features(df_clean)
    for c in pid_feats.columns:
        df_clean[c] = pid_feats[c]
    for c in cb_feats.columns:
        df_clean[c] = cb_feats[c]

    # Build Energy Labels & Section 13 Segregation
    df_clean = build_energy_labels_and_features(df_clean)
    df_clean = add_label_group_information(df_clean)
    df_clean = compute_effective_sample_weights(df_clean)
    df_clean = generate_stratified_recipe_splits(df_clean, test_size=0.15, val_size=0.15)

    label_audit_df = audit_energy_labels(df_clean)
    print("\n--- Energy Labels Summary Audit ---", flush=True)
    print(label_audit_df.to_string(index=False), flush=True)

    # ------------------------------------------------------------------------
    # STEP 3: Train & Validate Energy Prediction Models (Mode A & Mode B)
    # ------------------------------------------------------------------------
    print("\n[Task 3] Training & Validating Dual Energy Models (Mode A & Mode B)...", flush=True)

    s1_cols = extract_stage1_recipe_features(df_clean)
    s2_cols_base = extract_stage2_process_features(df_clean)
    s2_cols = list(set(s2_cols_base + list(pid_feats.columns) + list(cb_feats.columns)))

    df_tr = df_clean[df_clean['_split'] == 'train'].copy()
    df_te = df_clean[df_clean['_split'] == 'test'].copy()

    # Fit Mode A (Post-batch Diagnosis)
    model_mode_a = MixingEnergyPredictionModel(mode='mode_a')
    model_mode_a.fit(df_tr, target_col='total_kwh_per_batch')
    pred_a_kwh, pred_a_kwh_ton = model_mode_a.predict(df_te)

    # Fit Mode B (Pre-batch Recommendation)
    model_mode_b = MixingEnergyPredictionModel(mode='mode_b')
    model_mode_b.fit(df_tr, target_col='total_kwh_per_batch')
    pred_b_kwh, pred_b_kwh_ton = model_mode_b.predict(df_te)

    metrics_a = evaluate_energy_model_performance(
        df_te['total_kwh_per_batch'].values, pred_a_kwh, df_te['batch_weight_ton'].values, df_te['kwh_per_ton'].values
    )
    metrics_b = evaluate_energy_model_performance(
        df_te['total_kwh_per_batch'].values, pred_b_kwh, df_te['batch_weight_ton'].values, df_te['kwh_per_ton'].values
    )

    pd.DataFrame([
        {'model_mode': 'Mode_A_PostBatch_Diagnosis', **metrics_a},
        {'model_mode': 'Mode_B_PreBatch_Recommendation', **metrics_b},
    ]).to_csv(
        os.path.join(out_dir, 'energy_prediction_metrics.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    print(f"  Mode A (Post-batch Diagnosis) Test Metrics : MAE={metrics_a['MAE_kWh']} kWh, R2={metrics_a['R2']}, Hit_2kWh={metrics_a['Hit_2kWh_pct']}%", flush=True)
    print(f"  Mode B (Pre-batch Recommend) Test Metrics  : MAE={metrics_b['MAE_kWh']} kWh, R2={metrics_b['R2']}, Hit_2kWh={metrics_b['Hit_2kWh_pct']}%", flush=True)

    # 1. Mode B Branch Metrics Export
    branch_rows = []
    for branch, grp in df_te.groupby('system_route_branch'):
        if len(grp) >= 5:
            b_pred_kwh, _ = model_mode_b.predict(grp)
            b_m = evaluate_energy_model_performance(
                grp['total_kwh_per_batch'].values, b_pred_kwh, grp['batch_weight_ton'].values, grp['kwh_per_ton'].values
            )
            branch_rows.append({
                'material_system': grp['material_system'].iloc[0],
                'phase_route': grp['phase_route'].iloc[0],
                'system_route_branch': branch,
                **b_m,
            })

    pd.DataFrame(branch_rows).to_csv(os.path.join(out_dir, 'mode_b_branch_metrics.csv'), index=False, encoding='utf-8-sig')

    # 2. High-Energy Subset Metrics Export
    high_e_rows = []
    kwh_90th = np.percentile(df_te['total_kwh_per_batch'].values, 90)
    kwh_ton_90th = np.percentile(df_te['kwh_per_ton'].values, 90)

    top_batch_df = df_te[df_te['total_kwh_per_batch'] >= kwh_90th]
    top_ton_df = df_te[df_te['kwh_per_ton'] >= kwh_ton_90th]

    top_b_kwh, _ = model_mode_b.predict(top_batch_df)
    m_top_b = evaluate_energy_model_performance(
        top_batch_df['total_kwh_per_batch'].values, top_b_kwh, top_batch_df['batch_weight_ton'].values, top_batch_df['kwh_per_ton'].values
    )
    high_e_rows.append({'subset': 'Top_10pct_kWh_per_batch', 'count': len(top_batch_df), **m_top_b})

    top_ton_kwh, _ = model_mode_b.predict(top_ton_df)
    m_top_ton = evaluate_energy_model_performance(
        top_ton_df['total_kwh_per_batch'].values, top_ton_kwh, top_ton_df['batch_weight_ton'].values, top_ton_df['kwh_per_ton'].values
    )
    high_e_rows.append({'subset': 'Top_10pct_kWh_per_ton', 'count': len(top_ton_df), **m_top_ton})

    pd.DataFrame(high_e_rows).to_csv(os.path.join(out_dir, 'high_energy_subset_metrics.csv'), index=False, encoding='utf-8-sig')

    # 3. Recipe Energy Benchmark Export
    bench_rows = []
    for (cmp, mix, material_system, route), grp in df_clean.groupby(
        ['CompoundName', 'MixerLine', 'material_system', 'phase_route']
    ):
        ton_vals = grp['kwh_per_ton'].values
        p10 = float(np.percentile(ton_vals, 10))
        p50 = float(np.median(ton_vals))
        p90 = float(np.percentile(ton_vals, 90))
        mean_ton = float(np.mean(ton_vals))
        saving_pot = round(p90 - p10, 2)

        bench_rows.append({
            'RecipeCode': cmp,
            'material_system': material_system,
            'phase_route': route,
            'system_route_branch': f"{material_system}_{route}",
            'MixerID': mix,
            'n_batches': len(grp),
            'mean_kwh_per_ton': round(mean_ton, 2),
            'p10_kwh_per_ton': round(p10, 2),
            'p50_kwh_per_ton': round(p50, 2),
            'p90_kwh_per_ton': round(p90, 2),
            'estimated_saving_potential': saving_pot,
        })
    pd.DataFrame(bench_rows).to_csv(os.path.join(out_dir, 'recipe_energy_benchmark.csv'), index=False, encoding='utf-8-sig')

    # ------------------------------------------------------------------------
    # STEP 4 & 5: Mooney Model & Safe Bounds & Optimization Engine
    # ------------------------------------------------------------------------
    print("\n[Task 4 & 5] Fitting V3.7 Mooney Model & Building Safe Parameter Bounds...", flush=True)

    for col in set(s1_cols + s2_cols):
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)

    mooney_model = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True, use_cb_subsystem=True)
    mooney_model.fit(df_tr, s1_cols, s2_cols, target_col='MNY', cluster_col='material_system')

    bounds_df = build_safe_parameter_bounds(df_tr)
    bounds_df.to_csv(os.path.join(out_dir, 'safe_parameter_bounds.csv'), index=False, encoding='utf-8-sig')

    mooney_checker = MooneyQualityConstraintChecker(mooney_model)
    optimizer = EnergyMooneyOptimizer(model_mode_b, mooney_checker)

    # ------------------------------------------------------------------------
    # STEP 6: Execute Gated Shadow Mode Simulation
    # ------------------------------------------------------------------------
    print("\n[Task 6 & 7] Running Gated Shadow Mode Simulation on Test Set...", flush=True)

    # Fix pandas warning using groupby(...).head(3)
    sample_te = df_te.groupby('CompoundName', group_keys=False).head(3).reset_index(drop=True)

    shadow_summary_rows = []
    rejection_rows = []
    tradeoff_rows = []
    shadow_audit_rows = []

    for idx, r in sample_te.iterrows():
        cmp = r['CompoundName']
        mix = r.get('MixerLine', 'ALL')
        material_system = r.get('material_system', 'Silica')
        route = r.get('phase_route', 'OilWet')

        b_sub = bounds_df[
            (bounds_df['CompoundName'] == cmp)
            & (bounds_df['MixerLine'] == mix)
            & (bounds_df['material_system'] == material_system)
            & (bounds_df['phase_route'] == route)
        ]
        bounds_match_level = 'EXACT_RECIPE_MIXER_ROUTE'
        if len(b_sub) == 0:
            b_sub = bounds_df[
                (bounds_df['CompoundName'] == cmp)
                & (bounds_df['material_system'] == material_system)
                & (bounds_df['phase_route'] == route)
                & (bounds_df['MixerLine'] == 'ALL')
            ]
            bounds_match_level = 'SAME_RECIPE_ROUTE_ANY_MIXER'
        if len(b_sub) == 0:
            bounds_match_level = 'UNAVAILABLE'
            recipe_route_history_exists = len(bounds_df[
                (bounds_df['CompoundName'] == cmp)
                & (bounds_df['material_system'] == material_system)
                & (bounds_df['phase_route'] == route)
            ]) > 0
            bounds_rejection_reason = (
                'SAFE_BOUNDS_UNAVAILABLE'
                if recipe_route_history_exists
                else 'COLD_START_RECIPE_NO_HISTORY'
            )
        else:
            bounds_rejection_reason = 'SAFE_BOUNDS_UNAVAILABLE'

        bounds_row = b_sub.iloc[0] if len(b_sub) > 0 else pd.Series()

        mny_actual = float(r['MNY'])
        spec_low = mny_actual - 3.5
        spec_high = mny_actual + 3.5

        res = optimizer.optimize_batch(
            r,
            bounds_row,
            spec_low,
            spec_high,
            n_candidates=25,
            bounds_rejection_reason=bounds_rejection_reason,
        )

        log_item = {
            'OrderID': r.get('OrderID', f'ORD_{idx}'),
            'BatchNumber': r.get('BatchNumber', idx),
            'CompoundName': cmp,
            'MixerLine': mix,
            'material_system': material_system,
            'phase_route': route,
            'system_route_branch': f"{material_system}_{route}",
            'safe_bounds_match_level': bounds_match_level,
            'recommendation_status': res['recommendation_status'],
            'rejection_reason': res['rejection_reason'],
            'actual_mny': round(mny_actual, 2),
            'actual_kwh_per_batch': res['actual_kwh_per_batch'],
            'recommended_kwh_per_batch': res['recommended_kwh_per_batch'],
            'actual_kwh_per_ton': res['actual_kwh_per_ton'],
            'recommended_kwh_per_ton': res['recommended_kwh_per_ton'],
            'predicted_baseline_kwh_per_batch': res['predicted_baseline_kwh_per_batch'],
            'predicted_baseline_kwh_per_ton': res['predicted_baseline_kwh_per_ton'],
            'estimated_saving_kwh': res['estimated_saving_kwh'],
            'estimated_saving_pct': res['estimated_saving_pct'],
            'mooney_pred': res['mooney_pred'],
            'confidence_label': res['confidence_label'],
            'valid_candidates_count': res['valid_candidates_count'],
        }

        if res['recommendation_status'] == 'RECOMMENDED':
            shadow_summary_rows.append(log_item)
        else:
            rejection_rows.append(log_item)
        shadow_audit_rows.append(log_item)

        tradeoff_rows.append({
            'CompoundName': cmp,
            'material_system': material_system,
            'phase_route': route,
            'system_route_branch': f"{material_system}_{route}",
            'actual_kwh_per_ton': res['actual_kwh_per_ton'],
            'recommended_kwh_per_ton': res['recommended_kwh_per_ton'],
            'predicted_baseline_kwh_per_ton': res['predicted_baseline_kwh_per_ton'],
            'mooney_pred': res['mooney_pred'],
            'mooney_lower': res.get('mooney_pred_lower', round(mny_actual - 1.5, 2)),
            'mooney_upper': res.get('mooney_pred_upper', round(mny_actual + 1.5, 2)),
            'recommendation_status': res['recommendation_status'],
        })

    # Export Shadow Reports
    pd.DataFrame(shadow_summary_rows).to_csv(os.path.join(out_dir, 'shadow_recommendation_summary.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(rejection_rows).to_csv(os.path.join(out_dir, 'shadow_recommendation_rejection_reasons.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(tradeoff_rows).to_csv(os.path.join(out_dir, 'energy_mooney_tradeoff_report.csv'), index=False, encoding='utf-8-sig')

    shadow_audit_df = pd.DataFrame(shadow_audit_rows)
    coverage_df = (
        shadow_audit_df.groupby(
            ['material_system', 'phase_route', 'system_route_branch', 'safe_bounds_match_level'],
            dropna=False,
        )
        .size()
        .reset_index(name='batch_count')
    )
    coverage_df['route_batch_count'] = coverage_df.groupby('system_route_branch')['batch_count'].transform('sum')
    coverage_df['batch_pct_within_route'] = (
        coverage_df['batch_count'] / coverage_df['route_batch_count'].clip(lower=1) * 100.0
    ).round(2)
    coverage_df.to_csv(os.path.join(out_dir, 'safe_bounds_coverage_summary.csv'), index=False, encoding='utf-8-sig')

    rec_count = len(shadow_summary_rows)
    rej_count = len(rejection_rows)
    tot_count = rec_count + rej_count

    print("\n" + "=" * 95, flush=True)
    print("      V4.1 COMPREHENSIVE ENERGY EXPERIMENTS COMPLETED SUCCESSFULLY", flush=True)
    print(f"      Total Evaluated Batches        : {tot_count}", flush=True)
    print(f"      Recommended Batches (Passed Gate): {rec_count} ({rec_count / max(tot_count, 1) * 100:.1f}%)", flush=True)
    print(f"      Rejected Batches (Failed Gate)  : {rej_count} ({rej_count / max(tot_count, 1) * 100:.1f}%)", flush=True)
    print(f"      Reports Exported To            : {out_dir}", flush=True)
    print("=" * 95, flush=True)


if __name__ == '__main__':
    run_v41_master_experiment()
