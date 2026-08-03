# ============================================================================
# Carbon Black V2 Validity & Incremental Value Audit Engine (V3.7)
# ============================================================================
# Task 1: Audits CB trend metric validity (group-level tie ratios, sample sizes)
# Task 2: Audits CB V2 feature redundancy, permutation importance & ablation deltas
#
# Output:
# - reports/v37_cb_dispersion_v2/cb_trend_metric_validity_audit.csv
# - reports/v37_cb_dispersion_v2/cb_trend_metric_summary.csv
# - reports/v37_cb_dispersion_v2/cb_v2_incremental_value_audit.csv
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
from feature_engineering.cb_dispersion_feature_builder_v2 import build_cb_dispersion_features_v2
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


def run_validity_and_incremental_audit():
    print("=" * 95)
    print("  TASK 1 & 2: CB TREND METRIC VALIDITY & V2 INCREMENTAL VALUE AUDIT")
    print("=" * 95)

    out_dir = os.path.join(pipeline_root, 'reports', 'v37_cb_dispersion_v2')
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
    cb_v2_df = build_cb_dispersion_features_v2(df_clean)

    for c in cb_v1_df.columns:
        df_clean[c + '_v1'] = cb_v1_df[c]
    for c in cb_v2_df.columns:
        df_clean[c] = cb_v2_df[c]

    df_clean = cluster_silica_carbon_black(df_clean)
    df_cb = df_clean[df_clean['material_system'] == 'CarbonBlack'].copy()

    df_cb = add_label_group_information(df_cb)
    df_cb = compute_effective_sample_weights(df_cb)
    df_cb = generate_stratified_recipe_splits(df_cb, test_size=0.15, val_size=0.15)

    s1_cols = extract_stage1_recipe_features(df_cb)
    s2_cols_base = extract_stage2_process_features(df_cb)
    v2_feat_cols = list(cb_v2_df.columns)

    df_tr = df_cb[df_cb['_split'] == 'train'].copy()
    df_te = df_cb[df_cb['_split'] == 'test'].copy().reset_index(drop=True)

    X_tr_s1 = df_tr[s1_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    X_te_s1 = df_te[s1_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    y_tr = df_tr['MNY'].values
    y_te = df_te['MNY'].values

    # Train Baseline Model (A0) and Full V2 Model (A4)
    s1_model = LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, random_state=42, verbose=-1)
    s1_model.fit(X_tr_s1, y_tr, sample_weight=df_tr['_w_loss'].values)

    pred_tr_s1 = s1_model.predict(X_tr_s1)
    pred_te_s1 = s1_model.predict(X_te_s1)
    res_tr = y_tr - pred_tr_s1

    s2_cols_full = list(set(s2_cols_base + v2_feat_cols))
    X_tr_s2 = df_tr[s2_cols_full].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    X_te_s2 = df_te[s2_cols_full].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    s2_model = LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=4, random_state=42, verbose=-1)
    s2_model.fit(X_tr_s2, res_tr, sample_weight=df_tr['_w_loss'].values)
    pred_te_s2 = s2_model.predict(X_te_s2)

    uncal_pred = pred_te_s1 + pred_te_s2
    calibrator = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha_base=0.3, adaptive=True)
    cal_pred, _ = calibrator.calibrate_time_series(df_te, uncal_pred, target_col='MNY', group_col='CompoundName')

    df_te['pred'] = cal_pred
    df_te['CleanCompound'] = df_te['CompoundName'].apply(clean_compound_name)

    # --- TASK 1: TREND METRIC VALIDITY AUDIT ---
    audit_rows = []
    group_col = 'CleanCompound'

    for grp_name, grp_df in df_te.groupby(group_col):
        n_samples = len(grp_df)
        y_act_grp = grp_df['MNY'].values
        y_pred_grp = grp_df['pred'].values

        act_std = np.std(y_act_grp)
        pred_std = np.std(y_pred_grp)

        act_uniq = len(np.unique(np.round(y_act_grp, 2)))
        pred_uniq = len(np.unique(np.round(y_pred_grp, 2)))

        tie_act = 1.0 - (act_uniq / max(n_samples, 1))
        tie_pred = 1.0 - (pred_uniq / max(n_samples, 1))

        dir_acc, valid_pairs = calculate_dir_acc(y_act_grp, y_pred_grp)

        # Spearman
        if n_samples >= 3 and act_std > 1e-4 and pred_std > 1e-4 and act_uniq >= 3 and pred_uniq >= 3:
            spearman_val = pd.Series(y_act_grp).corr(pd.Series(y_pred_grp), method='spearman')
            spearman_valid = True
        else:
            spearman_val = np.nan
            spearman_valid = False

        # Reason Code
        if n_samples < 3:
            fail_reason = 'TOO_FEW_SAMPLES'
        elif act_uniq < 3:
            fail_reason = 'ACTUAL_TIES_TOO_HIGH'
        elif pred_uniq < 3:
            fail_reason = 'PREDICTION_TIES_TOO_HIGH'
        elif act_std < 1e-4:
            fail_reason = 'LOW_ACTUAL_VARIANCE'
        elif pred_std < 1e-4:
            fail_reason = 'LOW_PRED_VARIANCE'
        else:
            fail_reason = 'VALID'

        audit_rows.append({
            'compound': grp_name,
            'order_id': grp_df['OrderID'].iloc[0] if 'OrderID' in grp_df.columns else 'N/A',
            'n_samples': n_samples,
            'actual_std': round(float(act_std), 4),
            'pred_std': round(float(pred_std), 4),
            'actual_unique_count': act_uniq,
            'pred_unique_count': pred_uniq,
            'spearman_valid': spearman_valid,
            'spearman_value': round(float(spearman_val), 4) if not np.isnan(spearman_val) else np.nan,
            'direction_valid': valid_pairs > 0,
            'direction_accuracy': round(float(dir_acc), 2),
            'tie_ratio_actual': round(float(tie_act), 4),
            'tie_ratio_pred': round(float(tie_pred), 4),
            'valid_pair_count': valid_pairs,
            'failure_reason': fail_reason,
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(os.path.join(out_dir, 'cb_trend_metric_validity_audit.csv'), index=False, encoding='utf-8-sig')

    # Summary
    valid_groups = audit_df[audit_df['failure_reason'] == 'VALID']
    invalid_groups = audit_df[audit_df['failure_reason'] != 'VALID']

    mean_v_spear = valid_groups['spearman_value'].dropna().mean() if not valid_groups.empty else np.nan
    w_v_spear = np.average(valid_groups['spearman_value'].dropna(), weights=valid_groups.loc[valid_groups['spearman_value'].notna(), 'n_samples']) if not valid_groups.empty and valid_groups['spearman_value'].notna().sum() > 0 else np.nan

    mean_v_dir = valid_groups['direction_accuracy'].mean() if not valid_groups.empty else np.nan
    w_v_dir = np.average(valid_groups['direction_accuracy'], weights=valid_groups['n_samples']) if not valid_groups.empty else np.nan

    main_reason = invalid_groups['failure_reason'].value_counts().index[0] if not invalid_groups.empty else 'NONE'

    summary_row = [{
        'valid_group_count': len(valid_groups),
        'invalid_group_count': len(invalid_groups),
        'mean_valid_spearman': round(float(mean_v_spear), 4) if not np.isnan(mean_v_spear) else np.nan,
        'weighted_valid_spearman': round(float(w_v_spear), 4) if not np.isnan(w_v_spear) else np.nan,
        'mean_valid_direction_accuracy': round(float(mean_v_dir), 2) if not np.isnan(mean_v_dir) else np.nan,
        'weighted_valid_direction_accuracy': round(float(w_v_dir), 2) if not np.isnan(w_v_dir) else np.nan,
        'main_invalid_reason': main_reason,
    }]
    summary_df = pd.DataFrame(summary_row)
    summary_df.to_csv(os.path.join(out_dir, 'cb_trend_metric_summary.csv'), index=False, encoding='utf-8-sig')

    # --- TASK 2: V2 FEATURE INCREMENTAL VALUE & REDUNDANCY AUDIT ---
    existing_cols = [c for c in s2_cols_base if c in df_cb.columns]
    X_existing = df_cb[existing_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    corr_rows = []
    for feat in v2_feat_cols:
        f_vals = pd.to_numeric(df_cb[feat], errors='coerce').fillna(0.0)
        corrs = {ex: abs(f_vals.corr(X_existing[ex])) for ex in existing_cols if X_existing[ex].std() > 1e-5}
        if corrs:
            max_ex = max(corrs, key=corrs.get)
            max_c = corrs[max_ex]
        else:
            max_ex, max_c = 'None', 0.0

        if max_c > 0.90:
            red_flag = 'HIGH_REDUNDANCY'
        elif max_c > 0.75:
            red_flag = 'MEDIUM_REDUNDANCY'
        else:
            red_flag = 'LOW_REDUNDANCY'

        corr_rows.append({
            'cb_feature': feat,
            'most_correlated_existing_feature': max_ex,
            'correlation': round(float(max_c), 4),
            'redundancy_flag': red_flag,
        })
    corr_df = pd.DataFrame(corr_rows)

    # 2. Permutation Importance on Test Set
    split_imps = dict(zip(s2_cols_full, s2_model.feature_importances_))
    base_mae = evaluate_mooney_predictions(y_te, cal_pred, df_te)['MAE']
    base_r2 = evaluate_mooney_predictions(y_te, cal_pred, df_te)['R2']

    perm_rows = []
    for feat in v2_feat_cols:
        X_te_s2_perm = X_te_s2.copy()
        np.random.seed(42)
        X_te_s2_perm[feat] = np.random.permutation(X_te_s2_perm[feat].values)

        pred_perm_s2 = s2_model.predict(X_te_s2_perm)
        uncal_perm = pred_te_s1 + pred_perm_s2
        cal_perm, _ = calibrator.calibrate_time_series(df_te, uncal_perm, target_col='MNY', group_col='CompoundName')

        perm_mae = evaluate_mooney_predictions(y_te, cal_perm, df_te)['MAE']
        perm_r2 = evaluate_mooney_predictions(y_te, cal_perm, df_te)['R2']

        mae_delta = perm_mae - base_mae
        r2_delta = base_r2 - perm_r2
        split_imp = split_imps.get(feat, 0)

        if split_imp > 50 and (mae_delta <= 0 or r2_delta <= 0):
            flag = 'SPLIT_IMPORTANCE_ONLY_NO_GENERALIZATION_GAIN'
        else:
            flag = 'CONSISTENT_GAIN' if mae_delta > 0 else 'LOW_IMPORTANCE'

        perm_rows.append({
            'feature': feat,
            'split_importance': int(split_imp),
            'permutation_importance_mae_delta': round(float(mae_delta), 4),
            'permutation_importance_r2_delta': round(float(r2_delta), 4),
            'importance_consistency_flag': flag,
        })
    perm_df = pd.DataFrame(perm_rows)

    # Combine Incremental Value Audit File
    inc_df = corr_df.merge(perm_df, left_on='cb_feature', right_on='feature').drop(columns=['feature'])
    inc_df.to_csv(os.path.join(out_dir, 'cb_v2_incremental_value_audit.csv'), index=False, encoding='utf-8-sig')

    print("\n" + "=" * 95)
    print("      TASK 1 & 2 AUDIT COMPLETED. REPORTS GENERATED IN: reports/v37_cb_dispersion_v2/")
    print("=" * 95)
    return audit_df, summary_df, inc_df


if __name__ == '__main__':
    run_validity_and_incremental_audit()
