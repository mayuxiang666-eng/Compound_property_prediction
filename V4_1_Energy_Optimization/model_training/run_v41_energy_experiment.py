# ============================================================================
# Master Orchestrator: Comprehensive V4.1 Energy Experiment & Report Suite
# ============================================================================
# Generates:
# 1. energy_prediction_metrics.csv
# 2. mode_b_branch_metrics.csv
# 3. high_energy_subset_metrics.csv
# 4. high_energy_model_gap_report.csv
# 4. recipe_energy_benchmark.csv (with P10, P50, P90 & estimated saving potential)
# 5. shadow_recommendation_summary.csv
# 6. shadow_recommendation_rejection_reasons.csv
# 7. energy_mooney_tradeoff_report.csv
# 8. safe_parameter_bounds.csv
# 9. safe_bounds_coverage_summary.csv
# 10. stage_energy_share.csv
# 11. mode_b_clean_metrics.csv
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
from model_training.energy_model import (
    MixingEnergyPredictionModel,
    build_energy_feature_purge_audit,
    evaluate_energy_model_performance,
)
from model_training.high_energy_specialist import HighEnergySpecialist
from model_training.hybrid_unified_model import HybridUnifiedMooneyModel
from model_training.label_group_handler import add_label_group_information
from model_training.split_builder import generate_stratified_recipe_splits
from optimization.safe_bounds_builder import build_safe_parameter_bounds
from optimization.mooney_constraint_checker import MooneyQualityConstraintChecker
from optimization.energy_optimizer import EnergyMooneyOptimizer
from optimization.candidate_generator import derive_route_stage_mask
from optimization.historical_candidate_builder import build_historical_best_reference_cohort

SHADOW_PRIORITY_RECIPES = {'B00458', 'B00163R4', 'B00163RT'}
SHADOW_PRIORITY_MIXER = 'MB02'
SHADOW_STANDARD_BATCHES_PER_RECIPE = 3
SHADOW_PRIORITY_BATCHES_PER_RECIPE = 10


def classify_recipe_maturity(history_n_batches: int, has_exact_safe_bounds: bool, gates_pass: bool) -> str:
    if history_n_batches == 0:
        return 'L0_NEW_RECIPE_COLD_START'
    if history_n_batches < 10 or not has_exact_safe_bounds:
        return 'L1_SIMILAR_RECIPE_REFERENCE'
    if history_n_batches < 30:
        return 'L2_PRELIMINARY_RECIPE_WINDOW'
    if history_n_batches < 50:
        return 'L3_SHADOW_REVIEW_READY'
    return 'L4_PILOT_READY' if gates_pass else 'L3_SHADOW_REVIEW_READY'


def _similarity_score(target: pd.Series, candidate: pd.Series) -> float:
    numeric_features = [
        'MNY',
        'batch_weight_ton',
        'Top_Fill_Factor',
        'weight_pct_solid_elastomer',
        'weight_pct_natural_rubber',
        'weight_pct_silica',
        'weight_pct_oil',
        'weight_pct_carbon_black',
    ]
    distances = []
    for feature in numeric_features:
        left = pd.to_numeric(target.get(feature, np.nan), errors='coerce')
        right = pd.to_numeric(candidate.get(feature, np.nan), errors='coerce')
        if pd.notna(left) and pd.notna(right):
            scale = max(abs(float(left)), abs(float(right)), 1.0)
            distances.append(abs(float(left) - float(right)) / scale)
    if not distances:
        return 0.0
    return float(max(0.0, 1.0 - np.mean(distances)))


def find_similar_recipes(target: pd.Series, history_df: pd.DataFrame, limit: int = 5) -> tuple[int, str, float]:
    required_columns = {'CompoundName', 'material_system', 'phase_route'}
    if not required_columns.issubset(history_df.columns):
        return 0, '', 0.0
    candidates = history_df[
        (history_df['material_system'] == target.get('material_system'))
        & (history_df['phase_route'] == target.get('phase_route'))
        & (history_df['CompoundName'] != target.get('CompoundName'))
    ].copy()
    if 'MixerLine' in history_df.columns and pd.notna(target.get('MixerLine')):
        same_mixer = candidates[candidates['MixerLine'] == target.get('MixerLine')]
        if not same_mixer.empty:
            candidates = same_mixer
    if candidates.empty:
        return 0, '', 0.0

    grouped = []
    for recipe_name, recipe_rows in candidates.groupby('CompoundName'):
        representative = recipe_rows.iloc[0]
        score = _similarity_score(target, representative)
        grouped.append((score, str(recipe_name)))
    grouped.sort(reverse=True)
    selected = grouped[:limit]
    return len(selected), ';'.join(name for _, name in selected), round(selected[0][0], 3) if selected else 0.0


def run_v41_master_experiment():
    print("=" * 95, flush=True)
    print("  RUNNING V4.1 COMPREHENSIVE MIXING ENERGY PREDICTION & OPTIMIZATION EXPERIMENTS", flush=True)
    print("=" * 95, flush=True)

    out_dir = os.path.join(pipeline_root, 'reports', 'v41_energy_optimization')
    os.makedirs(out_dir, exist_ok=True)
    feature_audit_dir = os.path.join(pipeline_root, 'reports', 'v41_energy_model')
    os.makedirs(feature_audit_dir, exist_ok=True)

    # ------------------------------------------------------------------------
    # STEP 1 & 2: Load Data & Build Energy Labels / Segregation Features
    # ------------------------------------------------------------------------
    print("\n[Task 1 & 2] Loading Dataset & Building Labels & Features...", flush=True)
    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../data/stage_statistics_enriched_all_features_weather_v4.csv',
    ))
    if not os.path.exists(data_path):
        data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '../data/enriched_mny_all.csv',
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
    df_tr = df_tr.drop(columns=[
        '_label_group_id',
        '_label_group_size',
        '_w_label_raw',
        '_w_compound_balance',
        '_w_risk',
        '_w_model_raw',
        '_w_loss',
        '_w_metric',
        '_sample_weight',
    ], errors='ignore')
    df_tr = compute_effective_sample_weights(df_tr)
    df_val = df_clean[df_clean['_split'] == 'val'].copy()
    df_te = df_clean[df_clean['_split'] == 'test'].copy()

    batch_energy_threshold = float(df_tr['total_kwh_per_batch'].quantile(0.90))
    ton_energy_threshold = float(df_tr['kwh_per_ton'].quantile(0.90))
    for frame in (df_tr, df_te):
        frame['energy_bucket'] = np.select(
            [
                frame['total_kwh_per_batch'] >= batch_energy_threshold,
                frame['kwh_per_ton'] >= ton_energy_threshold,
            ],
            ['TOP_10PCT_KWH_PER_BATCH', 'TOP_10PCT_KWH_PER_TON'],
            default='STANDARD',
        )
        frame['high_energy_flag'] = (frame['energy_bucket'] != 'STANDARD').astype(int)

    # Fit Mode A (Post-batch Diagnosis)
    model_mode_a = MixingEnergyPredictionModel(mode='mode_a')
    model_mode_a.fit(df_tr, target_col='total_kwh_per_batch')
    pred_a_kwh, pred_a_kwh_ton = model_mode_a.predict(df_te)

    # Fit Mode B (Pre-batch Recommendation)
    model_mode_b = MixingEnergyPredictionModel(mode='mode_b')
    model_mode_b.fit(df_tr, target_col='total_kwh_per_batch')
    pred_b_kwh, pred_b_kwh_ton = model_mode_b.predict(df_te)

    high_energy_specialist = HighEnergySpecialist(target_branch='Silica_OilWet')
    high_energy_specialist.fit(model_mode_b, df_tr, df_val)
    high_energy_components = high_energy_specialist.predict_components(model_mode_b, df_te)

    build_energy_feature_purge_audit().to_csv(
        os.path.join(feature_audit_dir, 'energy_feature_purge_audit.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame({'feature_name': model_mode_a.feature_names}).assign(model_mode='Mode_A_PostBatch_Diagnosis').to_csv(
        os.path.join(feature_audit_dir, 'mode_a_feature_list.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame({'feature_name': model_mode_b.feature_names}).assign(model_mode='Mode_B_PreBatch_Recommendation').to_csv(
        os.path.join(feature_audit_dir, 'mode_b_feature_list.csv'),
        index=False,
        encoding='utf-8-sig',
    )

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
    pd.DataFrame([{
        'model_mode': 'Mode_B_Clean_PreBatch_Recommendation',
        'feature_policy': 'STATIC_PRE_BATCH_AND_CONTROLLABLE_SETPOINTS_ONLY',
        **metrics_b,
    }]).to_csv(
        os.path.join(out_dir, 'mode_b_clean_metrics.csv'),
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
    top_batch_df = df_te[df_te['total_kwh_per_batch'] >= batch_energy_threshold]
    top_ton_df = df_te[df_te['kwh_per_ton'] >= ton_energy_threshold]

    top_b_kwh, _ = model_mode_b.predict(top_batch_df)
    m_top_b = evaluate_energy_model_performance(
        top_batch_df['total_kwh_per_batch'].values, top_b_kwh, top_batch_df['batch_weight_ton'].values, top_batch_df['kwh_per_ton'].values
    )
    high_e_rows.append({
        'subset': 'Top_10pct_kWh_per_batch',
        'energy_bucket': 'TOP_10PCT_KWH_PER_BATCH',
        'high_energy_flag': 1,
        'threshold': round(batch_energy_threshold, 3),
        'count': len(top_batch_df),
        **m_top_b,
    })

    top_ton_kwh, _ = model_mode_b.predict(top_ton_df)
    m_top_ton = evaluate_energy_model_performance(
        top_ton_df['total_kwh_per_batch'].values, top_ton_kwh, top_ton_df['batch_weight_ton'].values, top_ton_df['kwh_per_ton'].values
    )
    high_e_rows.append({
        'subset': 'Top_10pct_kWh_per_ton',
        'energy_bucket': 'TOP_10PCT_KWH_PER_TON',
        'high_energy_flag': 1,
        'threshold': round(ton_energy_threshold, 3),
        'count': len(top_ton_df),
        **m_top_ton,
    })

    pd.DataFrame(high_e_rows).to_csv(os.path.join(out_dir, 'high_energy_subset_metrics.csv'), index=False, encoding='utf-8-sig')

    # Branch-normalized high-energy specialist evaluation. It is diagnostic only.
    specialist_labels = high_energy_specialist.label_high_energy(df_te)
    specialist_metric_rows = []
    for subset_name, subset_mask in [
        ('ALL_TEST', np.ones(len(df_te), dtype=bool)),
        ('BRANCH_P90_KWH_PER_BATCH', specialist_labels['high_energy_batch_flag'].to_numpy(dtype=bool)),
        ('BRANCH_P90_KWH_PER_TON', specialist_labels['high_energy_ton_flag'].to_numpy(dtype=bool)),
        ('BRANCH_P90_UNION', specialist_labels['high_energy_union_flag'].to_numpy(dtype=bool)),
    ]:
        subset_df = df_te.loc[subset_mask]
        if len(subset_df) == 0:
            continue
        for model_variant, prediction in [
            ('MODE_B_BASE', pred_b_kwh[subset_mask]),
            ('MODE_B_PLUS_HIGH_ENERGY_SPECIALIST', high_energy_components['high_energy_adjusted_prediction_kwh'][subset_mask]),
        ]:
            specialist_metric_rows.append({
                'report_type': 'ENERGY_METRICS',
                'subset': subset_name,
                'model_variant': model_variant,
                'sample_count': len(subset_df),
                'specialist_activation_pct': round(
                    float(high_energy_components['high_energy_specialist_active'][subset_mask].mean() * 100.0), 2
                ),
                **evaluate_energy_model_performance(
                    subset_df['total_kwh_per_batch'].to_numpy(),
                    prediction,
                    subset_df['batch_weight_ton'].to_numpy(),
                    subset_df['kwh_per_ton'].to_numpy(),
                ),
            })

    actual_high_energy = specialist_labels['high_energy_ton_flag'].to_numpy(dtype=bool)
    predicted_high_energy = high_energy_components['high_energy_specialist_active']
    true_positive = int(np.sum(actual_high_energy & predicted_high_energy))
    false_positive = int(np.sum(~actual_high_energy & predicted_high_energy))
    false_negative = int(np.sum(actual_high_energy & ~predicted_high_energy))
    specialist_metric_rows.append({
        'report_type': 'DETECTOR_METRICS',
        'subset': high_energy_specialist.target_branch,
        'model_variant': 'HIGH_ENERGY_DETECTOR',
        'sample_count': len(df_te),
        'target_branch': high_energy_specialist.target_branch,
        'risk_threshold': high_energy_specialist.risk_threshold,
        'calibrator_sample_count': high_energy_specialist.calibrator_sample_count,
        'high_energy_precision_pct': round(true_positive / max(true_positive + false_positive, 1) * 100.0, 2),
        'high_energy_recall_pct': round(true_positive / max(true_positive + false_negative, 1) * 100.0, 2),
        'specialist_activation_pct': round(float(predicted_high_energy.mean() * 100.0), 2),
    })
    target_branch_metrics = [
        row for row in specialist_metric_rows
        if row.get('subset') == f'BRANCH_P90_KWH_PER_BATCH'
        and row.get('model_variant') in {'MODE_B_BASE', 'MODE_B_PLUS_HIGH_ENERGY_SPECIALIST'}
    ]
    target_batch_metrics = {
        row['model_variant']: row for row in target_branch_metrics
    }
    base_target = target_batch_metrics.get('MODE_B_BASE', {})
    specialist_target = target_batch_metrics.get('MODE_B_PLUS_HIGH_ENERGY_SPECIALIST', {})
    specialist_metric_rows.append({
        'report_type': 'ACCEPTANCE_GATE',
        'subset': high_energy_specialist.target_branch,
        'model_variant': 'HIGH_ENERGY_SILICA_OILWET_CALIBRATOR',
        'target_branch': high_energy_specialist.target_branch,
        'acceptance_target': 'Improve Top10% kWh/batch MAE, R2, and Hit_2kWh over Mode B base',
        'base_MAE_kWh': base_target.get('MAE_kWh', np.nan),
        'specialist_MAE_kWh': specialist_target.get('MAE_kWh', np.nan),
        'base_R2': base_target.get('R2', np.nan),
        'specialist_R2': specialist_target.get('R2', np.nan),
        'base_Hit_2kWh_pct': base_target.get('Hit_2kWh_pct', np.nan),
        'specialist_Hit_2kWh_pct': specialist_target.get('Hit_2kWh_pct', np.nan),
        'acceptance_status': (
            'PASS'
            if specialist_target.get('MAE_kWh', np.inf) < base_target.get('MAE_kWh', -np.inf)
            and specialist_target.get('R2', -np.inf) > base_target.get('R2', np.inf)
            and specialist_target.get('Hit_2kWh_pct', -np.inf) > base_target.get('Hit_2kWh_pct', np.inf)
            else 'NOT_MET'
        ),
    })
    pd.DataFrame(specialist_metric_rows).to_csv(
        os.path.join(out_dir, 'high_energy_specialist_metrics.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    # 2b. High-energy gap audit: diagnostic only, with no effect on models or gates.
    high_energy_report_rows = []
    high_energy_audit = df_te.copy()
    high_energy_audit['mode_b_predicted_kwh_per_batch'] = pred_b_kwh
    high_energy_audit['mode_b_predicted_kwh_per_ton'] = pred_b_kwh_ton
    high_energy_audit['branch_high_energy_batch_flag'] = specialist_labels['high_energy_batch_flag']
    high_energy_audit['branch_high_energy_ton_flag'] = specialist_labels['high_energy_ton_flag']
    high_energy_audit['high_energy_probability'] = high_energy_components['high_energy_probability']
    high_energy_audit['high_energy_specialist_active'] = high_energy_components['high_energy_specialist_active']
    high_energy_audit['high_energy_residual_correction_kwh'] = high_energy_components['high_energy_residual_correction_kwh']
    high_energy_audit['high_energy_adjusted_prediction_kwh'] = high_energy_components['high_energy_adjusted_prediction_kwh']
    high_energy_audit['high_energy_adjusted_prediction_kwh_per_ton'] = (
        high_energy_audit['high_energy_adjusted_prediction_kwh'] / high_energy_audit['batch_weight_ton'].clip(lower=0.05)
    )
    high_energy_audit['is_top10_kwh_per_batch'] = (
        high_energy_audit['total_kwh_per_batch'] >= batch_energy_threshold
    )
    high_energy_audit['is_top10_kwh_per_ton'] = (
        high_energy_audit['kwh_per_ton'] >= ton_energy_threshold
    )
    high_energy_audit['model_gap_kwh'] = (
        high_energy_audit['mode_b_predicted_kwh_per_batch'] - high_energy_audit['total_kwh_per_batch']
    )
    high_energy_audit['abs_model_gap_kwh'] = high_energy_audit['model_gap_kwh'].abs()
    high_energy_audit['model_gap_pct_of_actual'] = (
        high_energy_audit['model_gap_kwh'] / high_energy_audit['total_kwh_per_batch'].clip(lower=1.0) * 100.0
    )
    high_energy_audit['abs_model_gap_pct'] = high_energy_audit['model_gap_pct_of_actual'].abs()

    top_union_df = high_energy_audit.loc[
        high_energy_audit['is_top10_kwh_per_batch'] | high_energy_audit['is_top10_kwh_per_ton']
    ]
    for _, row in top_union_df.iterrows():
        high_energy_report_rows.append({
            'report_section': 'TOP10_SAMPLE',
            'subset_definition': 'TOP10_UNION',
            'OrderID': row.get('OrderID', ''),
            'BatchNumber': row.get('BatchNumber', ''),
            'CompoundName': row.get('CompoundName', ''),
            'MixerLine': row.get('MixerLine', ''),
            'material_system': row.get('material_system', ''),
            'phase_route': row.get('phase_route', ''),
            'system_route_branch': row.get('system_route_branch', ''),
            'energy_bucket': row['energy_bucket'],
            'high_energy_flag': row['high_energy_flag'],
            'is_top10_kwh_per_batch': row['is_top10_kwh_per_batch'],
            'is_top10_kwh_per_ton': row['is_top10_kwh_per_ton'],
            'branch_high_energy_batch_flag': row['branch_high_energy_batch_flag'],
            'branch_high_energy_ton_flag': row['branch_high_energy_ton_flag'],
            'high_energy_probability': row['high_energy_probability'],
            'high_energy_specialist_active': row['high_energy_specialist_active'],
            'actual_kwh_per_batch': row['total_kwh_per_batch'],
            'mode_b_predicted_kwh_per_batch': row['mode_b_predicted_kwh_per_batch'],
            'high_energy_adjusted_prediction_kwh_per_batch': row['high_energy_adjusted_prediction_kwh'],
            'high_energy_residual_correction_kwh': row['high_energy_residual_correction_kwh'],
            'batch_weight_ton': row['batch_weight_ton'],
            'actual_kwh_per_ton': row['kwh_per_ton'],
            'mode_b_predicted_kwh_per_ton': row['mode_b_predicted_kwh_per_ton'],
            'high_energy_adjusted_prediction_kwh_per_ton': row['high_energy_adjusted_prediction_kwh_per_ton'],
            'model_gap_kwh': row['model_gap_kwh'],
            'abs_model_gap_kwh': row['abs_model_gap_kwh'],
            'model_gap_pct_of_actual': row['model_gap_pct_of_actual'],
            'abs_model_gap_pct': row['abs_model_gap_pct'],
        })

    for subset_definition, subset_mask in [
        ('TOP_10PCT_KWH_PER_BATCH', high_energy_audit['is_top10_kwh_per_batch']),
        ('TOP_10PCT_KWH_PER_TON', high_energy_audit['is_top10_kwh_per_ton']),
    ]:
        for branch, branch_df in high_energy_audit.loc[subset_mask].groupby('system_route_branch'):
            branch_metrics = evaluate_energy_model_performance(
                branch_df['total_kwh_per_batch'].to_numpy(),
                branch_df['mode_b_predicted_kwh_per_batch'].to_numpy(),
                branch_df['batch_weight_ton'].to_numpy(),
                branch_df['kwh_per_ton'].to_numpy(),
            )
            high_energy_report_rows.append({
                'report_section': 'BRANCH_HIGH_ENERGY_METRICS',
                'subset_definition': subset_definition,
                'system_route_branch': branch,
                'material_system': branch_df['material_system'].iloc[0],
                'phase_route': branch_df['phase_route'].iloc[0],
                'sample_count': len(branch_df),
                **branch_metrics,
            })

    calibrator_features = [
        ('batch_weight_ton', 'Controls load-dependent baseline energy.'),
        ('Top_Fill_Factor', 'Captures fill-level operating regime before batching.'),
        ('MixerLine', 'Captures equipment-specific pre-batch context.'),
        ('material_system', 'Separates CarbonBlack and Silica process behavior.'),
        ('phase_route', 'Separates OilWet and NoOilDry route behavior.'),
        ('weight_pct_silica', 'Captures silica-related high-energy material loading.'),
        ('weight_pct_carbon_black', 'Captures carbon-black-related high-energy material loading.'),
        ('weight_pct_oil', 'Captures oil loading context before batching.'),
        ('supplier_rubber_viscosity_avg', 'Captures raw-material viscosity variation.'),
        ('supplier_silica_surface_area_avg', 'Captures silica surface-area variation.'),
        ('supplier_carbon_black_structure_avg', 'Captures carbon-black structure variation.'),
        ('Stage2_DryMixing_Duration', 'Controllable DryMix duration setpoint.'),
        ('Stage4_WetMixing_Duration', 'Controllable WetMix duration setpoint.'),
        ('Stage5_PID_Duration', 'Controllable PID duration setpoint when applicable.'),
        ('Stage6_BottomMixing_Duration', 'Controllable BottomMix duration setpoint.'),
        ('Target_Temperature', 'Controllable target temperature setpoint.'),
    ]
    for feature_name, rationale in calibrator_features:
        high_energy_report_rows.append({
            'report_section': 'CALIBRATOR_FEATURE_RECOMMENDATION',
            'subset_definition': 'FUTURE_HIGH_ENERGY_RESIDUAL_CALIBRATOR',
            'feature_name': feature_name,
            'feature_rationale': rationale,
            'feature_availability': 'PRE_BATCH_OR_CONTROLLABLE_SETPOINT',
            'implementation_status': 'DIAGNOSTIC_ONLY_NOT_USED_IN_CURRENT_MODEL',
        })

    pd.DataFrame(high_energy_report_rows).to_csv(
        os.path.join(out_dir, 'high_energy_model_gap_report.csv'),
        index=False,
        encoding='utf-8-sig',
    )

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
            'recipe_code': cmp,
            'material_system': material_system,
            'phase_route': route,
            'system_route_branch': f"{material_system}_{route}",
            'MixerID': mix,
            'n_batches': len(grp),
            'mean_kwh_per_ton': round(mean_ton, 2),
            'p10_kwh_per_ton': round(p10, 2),
            'p50_kwh_per_ton': round(p50, 2),
            'p90_kwh_per_ton': round(p90, 2),
            'best_10pct_energy_kwh_per_ton': round(p10, 2),
            'worst_10pct_energy_kwh_per_ton': round(p90, 2),
            'estimated_saving_potential': saving_pot,
            'saving_potential': saving_pot,
        })
    pd.DataFrame(bench_rows).to_csv(os.path.join(out_dir, 'recipe_energy_benchmark.csv'), index=False, encoding='utf-8-sig')

    # 4. Stage Energy Share Export
    stage_share_cols = {
        'drymix_kwh_share': 'Stage2_DryMixing_power_Integral',
        'wetmix_kwh_share': 'Stage4_WetMixing_power_Integral',
        'pid_kwh_share': 'Stage5_PID_power_Integral',
        'bottom_kwh_share': 'Stage6_BottomMixing_power_Integral',
    }
    stage_share_rows = []
    for branch, grp in df_clean.groupby('system_route_branch'):
        total_stage_kwh = sum(
            pd.to_numeric(grp[column], errors='coerce').fillna(0.0).sum() / 3600.0
            for column in [
                'Stage1_Loading_power_Integral',
                'Stage2_DryMixing_power_Integral',
                'Stage3_OilLoading_power_Integral',
                'Stage4_WetMixing_power_Integral',
                'Stage5_PID_power_Integral',
                'Stage6_BottomMixing_power_Integral',
            ]
            if column in grp.columns
        )
        stage_kwh = {
            share_col: pd.to_numeric(grp[source_col], errors='coerce').fillna(0.0).sum() / 3600.0
            if source_col in grp.columns else 0.0
            for share_col, source_col in stage_share_cols.items()
        }
        stage_share_rows.append({
            'material_system': grp['material_system'].iloc[0],
            'phase_route': grp['phase_route'].iloc[0],
            'system_route_branch': branch,
            'n_batches': len(grp),
            'total_stage_kwh': round(total_stage_kwh, 2),
            **{
                share_col: round(stage_kwh[share_col] / max(total_stage_kwh, 1e-6), 4)
                for share_col in stage_share_cols
            },
        })
    pd.DataFrame(stage_share_rows).to_csv(
        os.path.join(out_dir, 'stage_energy_share.csv'),
        index=False,
        encoding='utf-8-sig',
    )

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
    optimizer = EnergyMooneyOptimizer(model_mode_b, mooney_checker, historical_df=df_tr)

    # ------------------------------------------------------------------------
    # STEP 6: Execute Gated Shadow Mode Simulation
    # ------------------------------------------------------------------------
    print("\n[Task 6 & 7] Running Gated Shadow Mode Simulation on Test Set...", flush=True)

    standard_shadow_sample = df_te.groupby('CompoundName', group_keys=False).head(
        SHADOW_STANDARD_BATCHES_PER_RECIPE
    ).copy()
    standard_shadow_sample['_shadow_source_index'] = standard_shadow_sample.index
    standard_shadow_sample['shadow_cohort'] = 'TEST_HOLDOUT'
    priority_shadow_sample = (
        df_clean[
            (df_clean['system_route_branch'] == 'CarbonBlack_OilWet')
            & (df_clean['MixerLine'].astype(str) == SHADOW_PRIORITY_MIXER)
            & (df_clean['CompoundName'].astype(str).str.contains(
                r'(?<![A-Za-z0-9])(?:' + '|'.join(SHADOW_PRIORITY_RECIPES) + r')(?![A-Za-z0-9])',
                regex=True,
                na=False,
            ))
        ]
        .groupby('CompoundName', group_keys=False)
        .head(SHADOW_PRIORITY_BATCHES_PER_RECIPE)
        .copy()
    )
    priority_shadow_sample['_shadow_source_index'] = priority_shadow_sample.index
    priority_shadow_sample['shadow_cohort'] = 'PRIORITY_OPERATIONAL_HISTORY'
    sample_te = (
        pd.concat([priority_shadow_sample, standard_shadow_sample], ignore_index=False)
        .drop_duplicates(subset=['_shadow_source_index'])
        .reset_index(drop=True)
    )

    shadow_summary_rows = []
    rejection_rows = []
    tradeoff_rows = []
    shadow_audit_rows = []
    raw_shadow_rows = []
    validated_shadow_rows = []
    historical_template_rows = []
    normalized_prediction_rows = []
    operating_profile_rows = []
    uncertainty_rows = []
    recipe_history_counts = df_tr.groupby('CompoundName').size().to_dict()

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
        if len(b_sub) == 0 and material_system == 'Silica':
            b_sub = bounds_df[
                (bounds_df['CompoundName'] == 'ALL')
                & (bounds_df['material_system'] == material_system)
                & (bounds_df['phase_route'] == route)
                & (bounds_df['MixerLine'] == 'ALL')
                & (bounds_df['bounds_scope'] == 'SILICA_ROUTE_GLOBAL_FALLBACK')
            ]
            bounds_match_level = 'SILICA_ROUTE_GLOBAL_FALLBACK'
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
            historical_df=df_tr,
        )

        historical_templates = build_historical_best_reference_cohort(
            df_tr, r, spec_low, spec_high
        )
        for _, template in historical_templates.iterrows():
            profile = template.get('process_profile', {})
            template_row = {
                'CompoundName': cmp,
                'MixerLine': mix,
                'material_system': material_system,
                'phase_route': route,
                'template_id': template.get('template_id', ''),
                'historical_order_id': template.get('historical_order_id', ''),
                'historical_batch_number': template.get('historical_batch_number', ''),
                'historical_actual_kwh': template.get('historical_actual_kwh', np.nan),
                'historical_actual_mooney': template.get('historical_actual_mooney', np.nan),
                'historical_best_actual_saving_pct': template.get('historical_best_actual_saving_pct', np.nan),
                'candidate_role': 'HISTORICAL_TEMPLATE_AND_BENCHMARK_ONLY',
                **{f'profile_{key}': value for key, value in profile.items()},
                **{f'context_difference_{key}': value for key, value in template.items() if key.endswith('_difference') or key == 'historical_batch_age_days'},
            }
            historical_template_rows.append(template_row)

        recipe_history_n_batches = int(recipe_history_counts.get(cmp, 0))
        similar_recipe_count, similar_recipe_list, similarity_score = find_similar_recipes(
            r, df_tr, limit=5
        )
        has_exact_safe_bounds = bounds_match_level == 'EXACT_RECIPE_MIXER_ROUTE'
        base_gates_pass = (
            res['recommendation_status'] == 'RECOMMENDED'
            and res['confidence_label'] != 'LOW'
            and not res['ood_flag']
            and res['mooney_pred_lower'] >= spec_low
            and res['mooney_pred_upper'] <= spec_high
            and res['route_stage_status'] == 'VALID_ROUTE_STAGES'
            and res['safe_bound_status'] == 'WITHIN_SAFE_BOUNDS'
        )
        recipe_maturity_level = classify_recipe_maturity(
            recipe_history_n_batches, has_exact_safe_bounds, base_gates_pass
        )
        if recipe_maturity_level in {'L0_NEW_RECIPE_COLD_START', 'L1_SIMILAR_RECIPE_REFERENCE'}:
            cold_start_strategy = 'INTERSECT_BRANCH_SIMILAR_RECIPE_ENGINEERING_BOUNDS'
            recommendation_readiness = 'OBSERVATION_ONLY' if recipe_maturity_level == 'L0_NEW_RECIPE_COLD_START' else 'REFERENCE_ONLY'
        elif recipe_maturity_level == 'L2_PRELIMINARY_RECIPE_WINDOW':
            cold_start_strategy = 'SHADOW_REVIEW_ONLY_WITH_PRELIMINARY_WINDOW'
            recommendation_readiness = 'SHADOW_REVIEW_ONLY'
        elif recipe_maturity_level == 'L3_SHADOW_REVIEW_READY':
            cold_start_strategy = 'VALIDATED_SHADOW_REVIEW'
            recommendation_readiness = 'SHADOW_REVIEW_READY'
        else:
            cold_start_strategy = 'EXACT_RECIPE_MIXER_PILOT_GATE'
            recommendation_readiness = 'PILOT_REVIEW_READY'

        stage_mask = derive_route_stage_mask(r)

        duration_columns = {
            'DryMixing': ('Stage2_DryMixing_Duration', 'has_drymix_stage'),
            'WetMixing': ('Stage4_WetMixing_Duration', 'has_wetmix_stage'),
            'PID': ('Stage5_PID_Duration', 'has_pid_stage'),
            'BottomMixing': ('Stage6_BottomMixing_Duration', 'has_bottom_stage'),
        }

        def duration_fields(recommended_setpoints: dict) -> dict:
            fields = {}
            for display_name, (source_column, stage_flag) in duration_columns.items():
                current_value = float(r.get(source_column, 0.0)) if stage_mask[stage_flag] else 0.0
                recommended_value = float(recommended_setpoints.get(source_column, 0.0))
                fields[f'current_{display_name}_Duration'] = current_value
                fields[f'recommended_{display_name}_Duration'] = recommended_value
                fields[f'delta_{display_name}_Duration'] = round(recommended_value - current_value, 2)
            return fields

        common_item = {
            'OrderID': r.get('OrderID', f'ORD_{idx}'),
            'BatchNumber': r.get('BatchNumber', idx),
            'CompoundName': cmp,
            'MixerLine': mix,
            'material_system': material_system,
            'phase_route': route,
            'system_route_branch': f"{material_system}_{route}",
            'shadow_cohort': r.get('shadow_cohort', 'TEST_HOLDOUT'),
            'safe_bounds_match_level': bounds_match_level,
            'recipe_maturity_level': recipe_maturity_level,
            'recipe_history_n_batches': recipe_history_n_batches,
            'similar_recipe_count': similar_recipe_count,
            'similar_recipe_list': similar_recipe_list,
            'similarity_score': similarity_score,
            'cold_start_strategy': cold_start_strategy,
            'recommendation_readiness': recommendation_readiness,
            **stage_mask,
        }

        normalized_prediction_rows.append({
            **common_item,
            'selected_template_id': res.get('selected_template_id', ''),
            'model_predicted_current_kwh': res.get('mode_b_predicted_current_kwh_per_batch', np.nan),
            'model_predicted_normalized_candidate_kwh': res.get('mode_b_predicted_recommended_kwh_per_batch', np.nan),
            'model_based_saving_kwh': res.get('model_based_saving_kwh', np.nan),
            'model_based_saving_pct': res.get('model_based_saving_pct', np.nan),
            'historical_best_actual_saving_pct': res.get('historical_best_actual_saving_pct', np.nan),
            'confidence_label': res.get('confidence_label', 'LOW'),
            'ood_flag': res.get('ood_flag', True),
            'safe_bound_status': res.get('safe_bound_status', 'UNAVAILABLE'),
            'route_stage_status': res.get('route_stage_status', 'UNAVAILABLE'),
            'recommendation_status': res.get('recommendation_status', ''),
        })
        operating_profile_rows.append({
            **common_item,
            'profile_reference_status': res.get('operating_profile_reference_status', 'NONE'),
            'profile_reference_type': 'OPERATING_PROFILE_REFERENCE',
            **{f'{key}': value for key, value in res.get('operating_profile_reference', {}).items()},
        })
        uncertainty_rows.append({
            **common_item,
            'historical_best_actual_saving_pct': res.get('historical_best_actual_saving_pct', np.nan),
            'model_adjusted_saving_pct': res.get('model_adjusted_saving_pct', np.nan),
            'top_uncertainty_factor_1': res.get('top_uncertainty_factor_1', ''),
            'top_uncertainty_factor_2': res.get('top_uncertainty_factor_2', ''),
            'top_uncertainty_factor_3': res.get('top_uncertainty_factor_3', ''),
            'uncertainty_explanation_text': res.get('uncertainty_explanation_text', ''),
            'engineer_facing_explanation': (
                'Historical best performance suggests this candidate profile has been feasible. '
                'The model re-predicts it under current material, environment, batch, and machine conditions. '
                'The reported saving is model-adjusted, not raw historical saving. '
                f"Main uncertainty factors: {res.get('top_uncertainty_factor_1', '')}, "
                f"{res.get('top_uncertainty_factor_2', '')}, {res.get('top_uncertainty_factor_3', '')}. "
                'Rotor speed and ram averages are monitoring references, not setpoint recommendations unless confirmed controllable.'
            ),
        })

        raw_kwh = res['raw_recommended_kwh_per_batch']
        raw_saving_kwh = res['actual_kwh_per_batch'] - raw_kwh
        raw_item = {
            **common_item,
            'actual_current_kwh_per_batch': res['actual_current_kwh_per_batch'],
            'mode_b_predicted_current_kwh_per_batch': res['mode_b_predicted_current_kwh_per_batch'],
            'mode_b_predicted_recommended_kwh_per_batch': raw_kwh,
            'shadow_actual_saving_kwh': round(raw_saving_kwh, 2),
            'shadow_actual_saving_pct': round(raw_saving_kwh / max(res['actual_kwh_per_batch'], 1.0) * 100.0, 2),
            'model_based_saving_kwh': round(
                res['mode_b_predicted_current_kwh_per_batch'] - raw_kwh, 2
            ),
            'model_based_saving_pct': round(
                (res['mode_b_predicted_current_kwh_per_batch'] - raw_kwh)
                / max(res['mode_b_predicted_current_kwh_per_batch'], 1.0) * 100.0,
                2,
            ),
            'current_kwh_per_batch': res['actual_current_kwh_per_batch'],
            'recommended_kwh_per_batch': raw_kwh,
            'saving_kwh': round(raw_saving_kwh, 2),
            'saving_pct': round(raw_saving_kwh / max(res['actual_kwh_per_batch'], 1.0) * 100.0, 2),
            'current_mooney_pred': res['mooney_pred_current'],
            'recommended_mooney_pred': res['raw_mooney_pred'],
            'recommended_mooney_low': res['raw_mooney_low'],
            'recommended_mooney_high': res['raw_mooney_high'],
            'mooney_spec_low': res.get('spec_lower', spec_low),
            'mooney_spec_high': res.get('spec_upper', spec_high),
            'confidence_label': res['raw_confidence_label'],
            'ood_flag': res['raw_ood_flag'],
            'valid_candidate_count': res['valid_candidates_count'],
            'safe_bound_status': res['safe_bound_status'],
            'route_stage_status': res['route_stage_status'],
            'recommendation_status': 'RAW_BEST_CANDIDATE',
            'rejection_reason': res['rejection_reason'],
            **duration_fields(res['raw_candidate_setpoints']),
        }
        log_item = {
            **common_item,
            'actual_current_kwh_per_batch': res['actual_current_kwh_per_batch'],
            'mode_b_predicted_current_kwh_per_batch': res['mode_b_predicted_current_kwh_per_batch'],
            'mode_b_predicted_recommended_kwh_per_batch': res['mode_b_predicted_recommended_kwh_per_batch'],
            'shadow_actual_saving_kwh': res['shadow_actual_saving_kwh'],
            'shadow_actual_saving_pct': res['shadow_actual_saving_pct'],
            'model_based_saving_kwh': res['model_based_saving_kwh'],
            'model_based_saving_pct': res['model_based_saving_pct'],
            'current_kwh_per_batch': res['actual_current_kwh_per_batch'],
            'recommended_kwh_per_batch': res['mode_b_predicted_recommended_kwh_per_batch'],
            'saving_kwh': res['shadow_actual_saving_kwh'],
            'saving_pct': res['shadow_actual_saving_pct'],
            'current_mooney_pred': res['mooney_pred_current'],
            'recommended_mooney_pred': res['mooney_pred'],
            'recommended_mooney_low': res.get('mooney_pred_lower', np.nan),
            'recommended_mooney_high': res.get('mooney_pred_upper', np.nan),
            'mooney_spec_low': res.get('spec_lower', spec_low),
            'mooney_spec_high': res.get('spec_upper', spec_high),
            'confidence_label': res['confidence_label'],
            'ood_flag': res['ood_flag'],
            'valid_candidate_count': res['valid_candidates_count'],
            'safe_bound_status': res['safe_bound_status'],
            'route_stage_status': res['route_stage_status'],
            'recommendation_status': res['recommendation_status'],
            'rejection_reason': res['rejection_reason'],
            'selected_template_id': res.get('selected_template_id', ''),
            'historical_best_actual_saving_pct': res.get('historical_best_actual_saving_pct', np.nan),
            'model_adjusted_saving_pct': res.get('model_adjusted_saving_pct', np.nan),
            'top_uncertainty_factor_1': res.get('top_uncertainty_factor_1', ''),
            'top_uncertainty_factor_2': res.get('top_uncertainty_factor_2', ''),
            'top_uncertainty_factor_3': res.get('top_uncertainty_factor_3', ''),
            'uncertainty_explanation_text': res.get('uncertainty_explanation_text', ''),
            'operating_profile_reference_status': res.get('operating_profile_reference_status', 'NONE'),
            **duration_fields(res['recommended_setpoints']),
            # Existing column aliases retained for downstream compatibility.
            'actual_mny': round(mny_actual, 2),
            'actual_kwh_per_batch': res['actual_current_kwh_per_batch'],
            'actual_kwh': res['actual_current_kwh_per_batch'],
            'predicted_current_kwh_per_batch': res['predicted_baseline_kwh_per_batch'],
            'predicted_current_kwh': res['predicted_baseline_kwh_per_batch'],
            'predicted_recommended_kwh': res['mode_b_predicted_recommended_kwh_per_batch'],
            'actual_kwh_per_ton': res['actual_kwh_per_ton'],
            'predicted_current_kwh_per_ton': res['predicted_baseline_kwh_per_ton'],
            'recommended_kwh_per_ton': res['recommended_kwh_per_ton'],
            'predicted_baseline_kwh_per_batch': res['predicted_baseline_kwh_per_batch'],
            'predicted_baseline_kwh_per_ton': res['predicted_baseline_kwh_per_ton'],
            'estimated_saving_kwh': res['shadow_actual_saving_kwh'],
            'estimated_saving_pct': res['shadow_actual_saving_pct'],
            'mooney_pred_current': res['mooney_pred_current'],
            'mooney_pred_recommended': res['mooney_pred'],
            'mooney_interval_recommended_lower': res.get('mooney_pred_lower', np.nan),
            'mooney_interval_recommended_upper': res.get('mooney_pred_upper', np.nan),
            'confidence': res['confidence_label'],
            'valid_candidates_count': res['valid_candidates_count'],
        }

        if recipe_maturity_level in {'L0_NEW_RECIPE_COLD_START', 'L1_SIMILAR_RECIPE_REFERENCE'}:
            log_item['review_readiness'] = 'OBSERVATION_ONLY' if recipe_maturity_level == 'L0_NEW_RECIPE_COLD_START' else 'REFERENCE_ONLY'
            log_item['recommendation_readiness'] = log_item['review_readiness']
        elif recipe_maturity_level == 'L2_PRELIMINARY_RECIPE_WINDOW':
            log_item['review_readiness'] = 'SHADOW_REVIEW_ONLY'
        elif recipe_maturity_level == 'L3_SHADOW_REVIEW_READY':
            log_item['review_readiness'] = 'SHADOW_REVIEW_READY'
        else:
            log_item['review_readiness'] = 'PILOT_REVIEW_READY'

        if res['recommendation_status'] == 'RECOMMENDED':
            shadow_summary_rows.append(log_item)
        else:
            rejection_rows.append(log_item)
        shadow_audit_rows.append(log_item)
        raw_shadow_rows.append(raw_item)
        validated_shadow_rows.append(log_item)

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

    rejection_reason_top1 = (
        pd.Series([row['rejection_reason'] for row in rejection_rows]).mode().iloc[0]
        if rejection_rows else ''
    )
    for row in shadow_audit_rows:
        is_recommended = row['recommendation_status'] == 'RECOMMENDED'
        row['rejection_reason_top1'] = '' if is_recommended else rejection_reason_top1
        row['gate_pass_reason'] = 'GATE_PASS_OPTIMAL' if is_recommended else ''

    # Export Shadow Reports
    pd.DataFrame(raw_shadow_rows).to_csv(
        os.path.join(out_dir, 'raw_shadow_candidate_output.csv'), index=False, encoding='utf-8-sig'
    )
    pd.DataFrame(validated_shadow_rows).to_csv(
        os.path.join(out_dir, 'shadow_recommendation_validated.csv'), index=False, encoding='utf-8-sig'
    )
    pd.DataFrame(shadow_summary_rows).to_csv(os.path.join(out_dir, 'shadow_recommendation_summary.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(rejection_rows).to_csv(os.path.join(out_dir, 'shadow_recommendation_rejection_reasons.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(tradeoff_rows).to_csv(os.path.join(out_dir, 'energy_mooney_tradeoff_report.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(historical_template_rows).to_csv(os.path.join(out_dir, 'historical_best_candidate_templates.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(normalized_prediction_rows).to_csv(os.path.join(out_dir, 'model_normalized_candidate_predictions.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(operating_profile_rows).to_csv(os.path.join(out_dir, 'recipe_operating_profile_reference.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(uncertainty_rows).to_csv(os.path.join(out_dir, 'recommendation_uncertainty_explanation.csv'), index=False, encoding='utf-8-sig')

    shadow_audit_df = pd.DataFrame(shadow_audit_rows)
    shadow_summary_df = pd.DataFrame(shadow_summary_rows)
    fallback_bound_levels = {'SAME_RECIPE_ROUTE_ANY_MIXER', 'SILICA_ROUTE_GLOBAL_FALLBACK'}
    pilot_review_df = shadow_audit_df.loc[
        (shadow_audit_df['system_route_branch'] == 'CarbonBlack_OilWet')
        & (shadow_audit_df['safe_bounds_match_level'] == 'EXACT_RECIPE_MIXER_ROUTE')
        & (shadow_audit_df['recipe_maturity_level'] == 'L4_PILOT_READY')
        & (shadow_audit_df['recommendation_status'] == 'RECOMMENDED')
        & (shadow_audit_df['confidence_label'] == 'HIGH')
        & (shadow_audit_df['ood_flag'] == False)
        & (shadow_audit_df['shadow_actual_saving_pct'] >= 8.0)
        & (shadow_audit_df['model_based_saving_pct'] >= 5.0)
        & (shadow_audit_df['valid_candidate_count'] >= 20)
    ].copy()
    pilot_review_df['review_readiness'] = 'PILOT_REVIEW_CANDIDATE'
    pilot_review_df['engineer_decision'] = ''
    pilot_review_df['trial_priority'] = 'P1'
    pilot_review_df['trial_status'] = 'NOT_STARTED'
    pilot_review_df['trial_batch_id'] = ''
    pilot_review_df['trial_actual_kwh'] = np.nan
    pilot_review_df['trial_actual_mooney'] = np.nan
    pilot_review_df['trial_quality_ok'] = np.nan
    pilot_review_df['trial_comment'] = ''
    pilot_review_df['final_decision'] = 'PENDING_ENGINEERING_REVIEW'
    pilot_review_df.to_csv(
        os.path.join(out_dir, 'validated_shadow_review_candidates_cb_oilwet.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    window_group_columns = ['CompoundName', 'MixerLine', 'system_route_branch']
    window_parameters = [
        'recommended_DryMixing_Duration',
        'recommended_WetMixing_Duration',
        'recommended_PID_Duration',
        'recommended_BottomMixing_Duration',
        'delta_DryMixing_Duration',
        'delta_WetMixing_Duration',
        'delta_PID_Duration',
        'delta_BottomMixing_Duration',
    ]
    window_rows = []
    for group_key, group in pilot_review_df.groupby(window_group_columns, dropna=False):
        compound_name, mixer_line, branch = group_key
        window_row = {
            'CompoundName': compound_name,
            'MixerLine': mixer_line,
            'system_route_branch': branch,
            'n_recommended_batches': len(group),
            'avg_shadow_actual_saving_pct': round(group['shadow_actual_saving_pct'].mean(), 2),
            'avg_model_based_saving_pct': round(group['model_based_saving_pct'].mean(), 2),
            'avg_historical_best_actual_saving_pct': round(group['historical_best_actual_saving_pct'].mean(), 2),
            'avg_model_adjusted_saving_pct': round(group['model_adjusted_saving_pct'].mean(), 2),
            'uncertainty_factor_1': group['top_uncertainty_factor_1'].mode().iloc[0] if not group['top_uncertainty_factor_1'].mode().empty else '',
            'uncertainty_factor_2': group['top_uncertainty_factor_2'].mode().iloc[0] if not group['top_uncertainty_factor_2'].mode().empty else '',
            'uncertainty_factor_3': group['top_uncertainty_factor_3'].mode().iloc[0] if not group['top_uncertainty_factor_3'].mode().empty else '',
            'min_recommended_mooney_low': round(group['recommended_mooney_low'].min(), 3),
            'max_recommended_mooney_high': round(group['recommended_mooney_high'].max(), 3),
            'min_mooney_spec_low': round(group['mooney_spec_low'].min(), 3),
            'max_mooney_spec_high': round(group['mooney_spec_high'].max(), 3),
            'engineer_decision': '',
            'trial_priority': 'P1',
            'trial_status': 'NOT_STARTED',
            'trial_batch_id': '',
            'trial_actual_kwh': np.nan,
            'trial_actual_mooney': np.nan,
            'trial_quality_ok': np.nan,
            'trial_comment': '',
            'final_decision': 'PENDING_ENGINEERING_REVIEW',
        }
        for parameter in window_parameters:
            window_row[f'{parameter}_median'] = round(group[parameter].median(), 2)
            window_row[f'{parameter}_p25'] = round(group[parameter].quantile(0.25), 2)
            window_row[f'{parameter}_p75'] = round(group[parameter].quantile(0.75), 2)
        window_rows.append(window_row)
    pd.DataFrame(window_rows).to_csv(
        os.path.join(out_dir, 'recipe_level_pilot_recommendation_window.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame(window_rows).to_csv(
        os.path.join(out_dir, 'recipe_level_pilot_recommendation_window_with_uncertainty.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    silica_observation_df = shadow_audit_df.loc[
        (shadow_audit_df['material_system'] == 'Silica')
        & (shadow_audit_df['recommendation_status'] == 'RECOMMENDED')
        & (shadow_audit_df['confidence_label'] == 'HIGH')
        & (shadow_audit_df['safe_bounds_match_level'].isin(fallback_bound_levels))
    ].copy()
    silica_observation_df['review_readiness'] = 'OBSERVATION_ONLY_FALLBACK_BOUND'
    silica_observation_df['pilot_ready'] = False
    silica_observation_df['final_decision'] = 'OBSERVATION_ONLY'
    silica_observation_df.to_csv(
        os.path.join(out_dir, 'silica_shadow_observation_pool.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    branch_index = shadow_audit_df[['system_route_branch']].drop_duplicates()
    management_summary_df = branch_index.merge(
        shadow_summary_df.groupby('system_route_branch', dropna=False).agg(
            recommended_row_count=('recommendation_status', 'size'),
            average_saving_pct=('shadow_actual_saving_pct', 'mean'),
            total_estimated_saving_kwh=('shadow_actual_saving_kwh', 'sum'),
            exact_bound_recommendation_count=(
                'safe_bounds_match_level',
                lambda levels: (levels == 'EXACT_RECIPE_MIXER_ROUTE').sum(),
            ),
            fallback_bound_recommendation_count=(
                'safe_bounds_match_level',
                lambda levels: levels.isin(fallback_bound_levels).sum(),
            ),
        ).reset_index(),
        on='system_route_branch',
        how='left',
    )
    count_columns = [
        'recommended_row_count',
        'total_estimated_saving_kwh',
        'exact_bound_recommendation_count',
        'fallback_bound_recommendation_count',
    ]
    management_summary_df[count_columns] = management_summary_df[count_columns].fillna(0)
    management_summary_df['average_saving_pct'] = management_summary_df['average_saving_pct'].fillna(0.0)
    management_summary_df.to_csv(
        os.path.join(out_dir, 'shadow_recommendation_management_summary.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    maturity_columns = [
        'CompoundName',
        'MixerLine',
        'material_system',
        'phase_route',
        'system_route_branch',
        'recipe_maturity_level',
        'recipe_history_n_batches',
        'similar_recipe_count',
        'similar_recipe_list',
        'similarity_score',
        'cold_start_strategy',
        'recommendation_readiness',
        'safe_bounds_match_level',
        'safe_bound_status',
        'route_stage_status',
        'recommendation_status',
        'confidence_label',
        'ood_flag',
        'model_based_saving_pct',
        'shadow_actual_saving_pct',
        'recommended_DryMixing_Duration',
        'recommended_WetMixing_Duration',
        'recommended_PID_Duration',
        'recommended_BottomMixing_Duration',
        'delta_DryMixing_Duration',
        'delta_WetMixing_Duration',
        'delta_PID_Duration',
        'delta_BottomMixing_Duration',
    ]
    maturity_view = shadow_audit_df[maturity_columns].copy()
    cold_start_view = maturity_view.loc[
        maturity_view['recipe_maturity_level'].isin(
            ['L0_NEW_RECIPE_COLD_START', 'L1_SIMILAR_RECIPE_REFERENCE']
        )
    ].copy()
    cold_start_view['recommendation_readiness'] = cold_start_view['recipe_maturity_level'].map({
        'L0_NEW_RECIPE_COLD_START': 'OBSERVATION_ONLY',
        'L1_SIMILAR_RECIPE_REFERENCE': 'REFERENCE_ONLY',
    })
    cold_start_view.to_csv(
        os.path.join(out_dir, 'new_recipe_cold_start_report.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    similar_reference_view = cold_start_view.loc[
        cold_start_view['recipe_maturity_level'] == 'L1_SIMILAR_RECIPE_REFERENCE'
    ].copy()
    similar_reference_view['recommendation_readiness'] = 'REFERENCE_ONLY'
    similar_reference_view.to_csv(
        os.path.join(out_dir, 'similar_recipe_reference_window.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    preliminary_view = maturity_view.loc[
        maturity_view['recipe_maturity_level'] == 'L2_PRELIMINARY_RECIPE_WINDOW'
    ].copy()
    preliminary_view['recommendation_readiness'] = 'SHADOW_REVIEW_ONLY'
    preliminary_view.to_csv(
        os.path.join(out_dir, 'preliminary_recipe_recommendation_window.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    pilot_window_view = maturity_view.loc[
        (maturity_view['recipe_maturity_level'] == 'L4_PILOT_READY')
        & (maturity_view['recommendation_status'] == 'RECOMMENDED')
        & (maturity_view['safe_bounds_match_level'] == 'EXACT_RECIPE_MIXER_ROUTE')
        & (maturity_view['confidence_label'] == 'HIGH')
        & (maturity_view['ood_flag'] == False)
        & (maturity_view['safe_bound_status'] == 'WITHIN_SAFE_BOUNDS')
        & (maturity_view['route_stage_status'] == 'VALID_ROUTE_STAGES')
        & (maturity_view['model_based_saving_pct'] >= 5.0)
    ].copy()
    pilot_window_view['recommendation_readiness'] = 'PILOT_REVIEW_READY'
    pilot_window_view.to_csv(
        os.path.join(out_dir, 'pilot_ready_recipe_window.csv'),
        index=False,
        encoding='utf-8-sig',
    )

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
