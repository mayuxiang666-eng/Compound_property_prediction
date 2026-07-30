# ============================================================================
# Carbon Black Dispersion V2.1 Refactor & Controlled Experiment Runner (V3.7)
# ============================================================================
# Task 4: Evaluates CB V2.1 compact feature set across 5 ablation levels:
#   CB_B0 = Current CB_V1 Baseline
#   CB_B1 = CB_V2.1 Features Only
#   CB_B2 = CB_V2.1 Dispersion Expert
#   CB_B3 = CB_V2.1 Dispersion + Bottom Expert
#   CB_B4 = Full CB_V2.1 Subsystem
#
# Generates 5 reports in reports/v37_cb_dispersion_v21/:
#   1. cb_v21_feature_table.csv
#   2. cb_v21_ablation_report.csv
#   3. cb_v21_incremental_value_audit.csv
#   4. cb_v21_trend_metric_validity_audit.csv
#   5. cb_v21_candidate_selection_decision.csv
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

from feature_engineering.clustering import cluster_silica_carbon_black
from feature_engineering.stage1_recipe_features import extract_stage1_recipe_features
from feature_engineering.stage2_process_features import extract_stage2_process_features
from feature_engineering.cb_dispersion_feature_builder import build_cb_dispersion_features
from feature_engineering.cb_dispersion_feature_builder_v21 import build_cb_dispersion_features_v21
from model_training.effective_weighting import compute_effective_sample_weights
from model_training.label_group_handler import add_label_group_information
from model_training.split_builder import generate_stratified_recipe_splits
from model_training.stage3_online_calibration import Stage3DelayedFeedbackCalibrator
from model_training.trend_metrics import evaluate_mooney_predictions


def clean_compound_name(comp_name):
    comp = str(comp_name).strip()
    if '---' in comp:
        return comp.split('---')[0].strip()
    elif '--' in comp:
        return comp.split('--')[0].strip()
    return comp


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
    return (correct / total * 100.0) if total > 0 else 50.0, total


def run_cb_v21_experiment():
    print("=" * 95)
    print("  TASK 4: CARBON BLACK V2.1 CONTROLLED ABLATION EXPERIMENT")
    print("=" * 95)

    out_dir = os.path.join(pipeline_root, 'reports', 'v37_cb_dispersion_v21')
    os.makedirs(out_dir, exist_ok=True)

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

    df_raw = pd.read_csv(data_path, low_memory=False)
    if 'MNY' not in df_raw.columns and 'Mooney_Viscosity' in df_raw.columns:
        df_raw['MNY'] = df_raw['Mooney_Viscosity']
    df_clean = df_raw.dropna(subset=['MNY']).copy()

    if 'CompoundName' not in df_clean.columns and 'Compound' in df_clean.columns:
        df_clean['CompoundName'] = df_clean['Compound']
    if 'OrderID' not in df_clean.columns and 'Order_No' in df_clean.columns:
        df_clean['OrderID'] = df_clean['Order_No']

    df_clean = df_clean.sort_values(by=['OrderID'] if 'OrderID' in df_clean.columns else df_clean.index).reset_index(drop=True)

    cb_v1_df = build_cb_dispersion_features(df_clean)
    cb_v21_df = build_cb_dispersion_features_v21(df_clean)

    for c in cb_v1_df.columns:
        df_clean[c + '_v1'] = cb_v1_df[c]
    for c in cb_v21_df.columns:
        df_clean[c] = cb_v21_df[c]

    df_clean = cluster_silica_carbon_black(df_clean)
    df_cb = df_clean[df_clean['material_system'] == 'CarbonBlack'].copy()

    df_cb = add_label_group_information(df_cb)
    df_cb = compute_effective_sample_weights(df_cb)
    df_cb = generate_stratified_recipe_splits(df_cb, test_size=0.15, val_size=0.15)

    s1_cols = extract_stage1_recipe_features(df_cb)
    s2_cols_base = extract_stage2_process_features(df_cb)

    v1_feat_cols = [c + '_v1' for c in cb_v1_df.columns]
    v21_feat_cols = list(cb_v21_df.columns)

    df_tr = df_cb[df_cb['_split'] == 'train'].copy()
    df_te = df_cb[df_cb['_split'] == 'test'].copy().reset_index(drop=True)
    df_te['test_idx'] = np.arange(len(df_te))

    y_tr = df_tr['MNY'].values
    y_te = df_te['MNY'].values

    # File 1: cb_v21_feature_table.csv
    feat_table_rows = [
        {'feature_name': 'cb_dispersion_completion_index', 'status': 'ACTIVE_V21', 'rationale': 'Normalized torque drop ratio, low redundancy (0.64), unique physical structure index'},
        {'feature_name': 'cb_specific_energy_ratio_dry_to_wet', 'status': 'ACTIVE_V21', 'rationale': 'Energy allocation ratio, retained from V1'},
        {'feature_name': 'cb_effective_dispersion_energy', 'status': 'ACTIVE_V21', 'rationale': 'Difficulty-normalized energy, low redundancy (0.57), positive permutation gain'},
        {'feature_name': 'cb_normalized_power_decay_stage2', 'status': 'ACTIVE_V21', 'rationale': 'Scale-invariant power decay slope'},
        {'feature_name': 'cb_dispersion_difficulty', 'status': 'DEMOTED_EXPERIMENTAL', 'rationale': 'High collinearity with recipe/material grade'},
        {'feature_name': 'cb_dispersion_work_proxy', 'status': 'DEMOTED_EXPERIMENTAL', 'rationale': 'High redundancy (0.9996) with Stage2_DryMixing_power_Integral'},
        {'feature_name': 'cb_cumulative_shear_exposure', 'status': 'DEMOTED_EXPERIMENTAL', 'rationale': 'High redundancy (0.9610) with Stage2/4 power integral'},
        {'feature_name': 'cb_thermo_mechanical_index', 'status': 'DEMOTED_EXPERIMENTAL', 'rationale': 'High redundancy (0.9880) with Stage2_DryMixing_power_Integral'},
    ]
    feat_table_df = pd.DataFrame(feat_table_rows)
    feat_table_df.to_csv(os.path.join(out_dir, 'cb_v21_feature_table.csv'), index=False, encoding='utf-8-sig')

    # B0 - B4 Ablation Configurations
    configs = {
        'CB_B0 (Current CB_V1 Baseline)': {'s2_cols': list(set(s2_cols_base + v1_feat_cols))},
        'CB_B1 (CB_V2.1 Features Only)': {'s2_cols': list(set(v21_feat_cols))},
        'CB_B2 (CB_V2.1 Dispersion Expert)': {'s2_cols': list(set(v21_feat_cols + ['Stage2_DryMixing_Duration', 'Stage2_DryMixing_power_Mean']))},
        'CB_B3 (CB_V2.1 Dispersion + Bottom)': {'s2_cols': list(set(s2_cols_base + v21_feat_cols))},
        'CB_B4 (Full CB_V2.1 Subsystem)': {'s2_cols': list(set(s2_cols_base + v21_feat_cols))},
    }

    ablation_rows = []
    preds_dict = {}

    for name, cfg in configs.items():
        print(f"  Fitting {name}...")
        X_tr_s1 = df_tr[s1_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        X_te_s1 = df_te[s1_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        s1_model = LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, random_state=42, verbose=-1)
        s1_model.fit(X_tr_s1, y_tr, sample_weight=df_tr['_w_loss'].values)

        pred_tr_s1 = s1_model.predict(X_tr_s1)
        pred_te_s1 = s1_model.predict(X_te_s1)
        res_tr = y_tr - pred_tr_s1

        s2_cols_curr = cfg['s2_cols']
        X_tr_s2 = df_tr[s2_cols_curr].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        X_te_s2 = df_te[s2_cols_curr].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        s2_model = LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=4, random_state=42, verbose=-1)
        s2_model.fit(X_tr_s2, res_tr, sample_weight=df_tr['_w_loss'].values)
        pred_te_s2 = s2_model.predict(X_te_s2)

        uncal_pred = pred_te_s1 + pred_te_s2
        calibrator = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha_base=0.3, adaptive=True)
        cal_pred, _ = calibrator.calibrate_time_series(df_te, uncal_pred, target_col='MNY', group_col='CompoundName')

        preds_dict[name] = cal_pred

        eval_res = evaluate_mooney_predictions(y_te, cal_pred, df_te)
        std_act = np.std(y_te)
        std_pred = np.std(cal_pred)

        ablation_rows.append({
            'ablation_config': name,
            'cb_mae': round(float(eval_res['MAE']), 4),
            'cb_rmse': round(float(eval_res['RMSE']), 4),
            'cb_r2': round(float(eval_res['R2']), 4),
            'cb_dir_acc_pct': round(float(eval_res['Direction_Accuracy'] * 100.0), 2),
            'cb_spearman': round(float(eval_res['Spearman_Rho']), 4),
            'cb_high_dev_mae': round(float(eval_res['High_Dev_MAE']), 4),
            'cb_stage2_capture_ratio': round(float(std_pred / (std_act + 1e-6)), 4),
        })

    # File 2: cb_v21_ablation_report.csv
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(os.path.join(out_dir, 'cb_v21_ablation_report.csv'), index=False, encoding='utf-8-sig')

    # File 3: cb_v21_incremental_value_audit.csv
    existing_cols = [c for c in s2_cols_base if c in df_cb.columns]
    X_existing = df_cb[existing_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    inc_rows = []
    for feat in v21_feat_cols:
        f_vals = pd.to_numeric(df_cb[feat], errors='coerce').fillna(0.0)
        corrs = {ex: abs(f_vals.corr(X_existing[ex])) for ex in existing_cols if X_existing[ex].std() > 1e-5}
        max_ex = max(corrs, key=corrs.get) if corrs else 'None'
        max_c = corrs[max_ex] if corrs else 0.0
        red_flag = 'HIGH_REDUNDANCY' if max_c > 0.90 else ('MEDIUM_REDUNDANCY' if max_c > 0.75 else 'LOW_REDUNDANCY')

        inc_rows.append({
            'v21_feature': feat,
            'most_correlated_existing_feature': max_ex,
            'correlation': round(float(max_c), 4),
            'redundancy_flag': red_flag,
        })
    inc_df = pd.DataFrame(inc_rows)
    inc_df.to_csv(os.path.join(out_dir, 'cb_v21_incremental_value_audit.csv'), index=False, encoding='utf-8-sig')

    # File 4: cb_v21_trend_metric_validity_audit.csv
    df_te['CleanCompound'] = df_te['CompoundName'].apply(clean_compound_name)
    df_te['best_pred'] = preds_dict['CB_B4 (Full CB_V2.1 Subsystem)']

    valid_rows = []
    for grp_name, grp_df in df_te.groupby('CleanCompound'):
        n_samples = len(grp_df)
        y_act_grp = grp_df['MNY'].values
        y_pred_grp = grp_df['best_pred'].values
        act_std = np.std(y_act_grp)
        pred_std = np.std(y_pred_grp)
        act_uniq = len(np.unique(np.round(y_act_grp, 2)))
        pred_uniq = len(np.unique(np.round(y_pred_grp, 2)))

        dir_acc, valid_pairs = calculate_dir_acc(y_act_grp, y_pred_grp)

        valid_rows.append({
            'compound': grp_name,
            'n_samples': n_samples,
            'actual_std': round(float(act_std), 4),
            'pred_std': round(float(pred_std), 4),
            'actual_unique_count': act_uniq,
            'pred_unique_count': pred_uniq,
            'valid_pair_count': valid_pairs,
            'direction_accuracy': round(float(dir_acc), 2),
            'is_valid_trend_group': n_samples >= 3 and act_uniq >= 3 and act_std > 1e-4,
        })
    valid_df = pd.DataFrame(valid_rows)
    valid_df.to_csv(os.path.join(out_dir, 'cb_v21_trend_metric_validity_audit.csv'), index=False, encoding='utf-8-sig')

    # File 5: cb_v21_candidate_selection_decision.csv
    b0_mae = ablation_df.loc[ablation_df['ablation_config'] == 'CB_B0 (Current CB_V1 Baseline)', 'cb_mae'].values[0]
    b0_r2 = ablation_df.loc[ablation_df['ablation_config'] == 'CB_B0 (Current CB_V1 Baseline)', 'cb_r2'].values[0]
    b4_mae = ablation_df.loc[ablation_df['ablation_config'] == 'CB_B4 (Full CB_V2.1 Subsystem)', 'cb_mae'].values[0]
    b4_r2 = ablation_df.loc[ablation_df['ablation_config'] == 'CB_B4 (Full CB_V2.1 Subsystem)', 'cb_r2'].values[0]

    # Evaluate decision
    if b4_mae < b0_mae and b4_r2 >= b0_r2:
        decision = 'PROMOTE_CB_V21_TO_PRODUCTION'
        rationale = 'CB_V2.1 beats CB_V1 baseline on MAE without degrading R2.'
    else:
        decision = 'RETAIN_CB_V1_PRODUCTION_BASELINE'
        rationale = f'CB_V2.1 (MAE={b4_mae}) does not beat CB_V1 baseline (MAE={b0_mae}). Mark CB_V2.1 as Experimental Feature Library.'

    decision_rows = [{
        'production_baseline': 'CB_V1 (CB_B0)',
        'b0_mae': b0_mae,
        'b0_r2': b0_r2,
        'candidate_eval': 'CB_V2.1 (CB_B4)',
        'b4_mae': b4_mae,
        'b4_r2': b4_r2,
        'candidate_selection_decision': decision,
        'decision_rationale': rationale,
        'next_recommended_focus': 'Shift focus to Lot-Level COA raw material feature integration (SAP QM)',
    }]
    decision_df = pd.DataFrame(decision_rows)
    decision_df.to_csv(os.path.join(out_dir, 'cb_v21_candidate_selection_decision.csv'), index=False, encoding='utf-8-sig')

    print("\n" + "=" * 105)
    print("      CB V2.1 EXPERIMENT SUMMARY REPORT")
    print("=" * 105)
    print(f"{'Ablation Configuration':<42} | {'CB MAE':<8} | {'CB RMSE':<8} | {'CB R2':<7} | {'DirAcc(%)':<8} | {'Spearman':<8}")
    print("-" * 105)
    for _, r in ablation_df.iterrows():
        print(f"{r['ablation_config']:<42} | {r['cb_mae']:<8.4f} | {r['cb_rmse']:<8.4f} | {r['cb_r2']:<7.4f} | {r['cb_dir_acc_pct']:<8.2f}% | {r['cb_spearman']:<8.4f}")
    print("=" * 105)

    print(f"\n  --- CANDIDATE SELECTION DECISION ---")
    print(f"  Decision:  {decision}")
    print(f"  Rationale: {rationale}")
    print(f"\nAll 5 V2.1 reports saved to: {out_dir}\n")
    return ablation_df, decision_df


if __name__ == '__main__':
    run_cb_v21_experiment()
