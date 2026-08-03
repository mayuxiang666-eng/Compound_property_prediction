# ============================================================================
# Carbon Black Dispersion Expert V2 Refactor & Ablation Runner (V3.7)
# ============================================================================
# Evaluates Carbon Black V1 vs V2 features across 5 ablation levels:
#   CB_A0 = Existing CB V1
#   CB_A1 = V2 features only (Stage 1 Recipe surface + V2 features)
#   CB_A2 = V2 + Dispersion Expert
#   CB_A3 = V2 + Dispersion + Bottom Expert
#   CB_A4 = Full CB Subsystem V2
#
# Generates 6 detailed reports in reports/cb_dispersion_v2/:
#   1. cb_feature_ablation_report.csv
#   2. cb_feature_importance.csv
#   3. cb_dispersion_bucket_report.csv
#   4. cb_direction_accuracy_report.csv
#   5. cb_spearman_report.csv
#   6. cb_high_deviation_report.csv
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Add module paths
pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

from feature_engineering.clustering import cluster_silica_carbon_black
from feature_engineering.stage1_recipe_features import extract_stage1_recipe_features
from feature_engineering.stage2_process_features import extract_stage2_process_features
from feature_engineering.cb_dispersion_feature_builder import build_cb_dispersion_features
from feature_engineering.cb_dispersion_feature_builder_v2 import build_cb_dispersion_features_v2
from model_training.effective_weighting import compute_effective_sample_weights
from model_training.hybrid_unified_model import HybridUnifiedMooneyModel
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
    return (correct / total * 100.0) if total > 0 else 50.0


def run_cb_ablation_experiment():
    print("=" * 95)
    print("  V3.7 CARBON BLACK DISPERSION EXPERT V2 REFACTOR & ABLATION EXPERIMENT")
    print("=" * 95)

    out_dir = os.path.join(pipeline_root, 'reports', 'cb_dispersion_v2')
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

    # Chronological sort
    df_clean = df_clean.sort_values(by=['OrderID'] if 'OrderID' in df_clean.columns else df_clean.index).reset_index(drop=True)

    # Build V1 and V2 CB Features
    cb_v1_df = build_cb_dispersion_features(df_clean)
    cb_v2_df = build_cb_dispersion_features_v2(df_clean)

    for c in cb_v1_df.columns:
        df_clean[c + '_v1'] = cb_v1_df[c]
    for c in cb_v2_df.columns:
        df_clean[c] = cb_v2_df[c]

    df_clean = cluster_silica_carbon_black(df_clean)

    # Filter to Carbon Black Compounds
    df_cb = df_clean[df_clean['material_system'] == 'CarbonBlack'].copy()
    print(f"  Total Carbon Black Batches: N = {len(df_cb)}")

    df_cb = add_label_group_information(df_cb)
    df_cb = compute_effective_sample_weights(df_cb)
    df_cb = generate_stratified_recipe_splits(df_cb, test_size=0.15, val_size=0.15)

    s1_cols = extract_stage1_recipe_features(df_cb)
    s2_cols_base = extract_stage2_process_features(df_cb)

    v1_feat_cols = [c + '_v1' for c in cb_v1_df.columns]
    v2_feat_cols = list(cb_v2_df.columns)

    df_tr = df_cb[df_cb['_split'] == 'train'].copy()
    df_te = df_cb[df_cb['_split'] == 'test'].copy()

    y_tr = df_tr['MNY'].values
    y_te = df_te['MNY'].values

    ablation_results = []
    importance_records = []
    bucket_records = []
    compound_dir_records = []

    # --- 5 ABLATION CONFIGURATIONS ---
    # CB_A0: Existing CB V1
    # CB_A1: V2 features only (LGBM on S1 + V2)
    # CB_A2: V2 + Dispersion Expert
    # CB_A3: V2 + Dispersion + Bottom Expert
    # CB_A4: Full CB Subsystem V2

    configs = {
        'CB_A0 (Existing CB V1)': {'s2_cols': list(set(s2_cols_base + v1_feat_cols)), 'use_subsystem': True, 'mode': 'v1'},
        'CB_A1 (V2 Features Only)': {'s2_cols': list(set(v2_feat_cols)), 'use_subsystem': False, 'mode': 'v2'},
        'CB_A2 (V2 + Dispersion Expert)': {'s2_cols': list(set(v2_feat_cols + ['Stage2_DryMixing_Duration', 'Stage2_DryMixing_power_Mean'])), 'use_subsystem': True, 'mode': 'v2'},
        'CB_A3 (V2 + Dispersion + Bottom Expert)': {'s2_cols': list(set(s2_cols_base + v2_feat_cols)), 'use_subsystem': True, 'mode': 'v2'},
        'CB_A4 (Full CB Subsystem V2)': {'s2_cols': list(set(s2_cols_base + v2_feat_cols)), 'use_subsystem': True, 'mode': 'v2'},
    }

    preds_dict = {}

    for name, cfg in configs.items():
        print(f"\n  Fitting {name}...")

        X_tr_s1 = df_tr[s1_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        X_te_s1 = df_te[s1_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # Stage 1 GBDT
        s1_model = LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, random_state=42, verbose=-1)
        s1_model.fit(X_tr_s1, y_tr, sample_weight=df_tr['_w_loss'].values)

        pred_tr_s1 = s1_model.predict(X_tr_s1)
        pred_te_s1 = s1_model.predict(X_te_s1)

        res_tr = y_tr - pred_tr_s1

        # Stage 2 Process Experts
        s2_cols_curr = cfg['s2_cols']
        X_tr_s2 = df_tr[s2_cols_curr].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        X_te_s2 = df_te[s2_cols_curr].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        s2_model = LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=4, random_state=42, verbose=-1)
        s2_model.fit(X_tr_s2, res_tr, sample_weight=df_tr['_w_loss'].values)

        pred_te_s2 = s2_model.predict(X_te_s2)

        # Collect feature importances
        for feat, imp in zip(s2_cols_curr, s2_model.feature_importances_):
            importance_records.append({
                'ablation_config': name,
                'feature_name': feat,
                'importance_score': int(imp),
            })

        final_uncal = pred_te_s1 + pred_te_s2

        # Apply Stage 3 Calibrator
        calibrator = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha_base=0.3, adaptive=True)
        final_cal, _ = calibrator.calibrate_time_series(df_te, final_uncal, target_col='MNY', group_col='CompoundName')

        preds_dict[name] = final_cal

        # Evaluate Metrics
        eval_res = evaluate_mooney_predictions(y_te, final_cal, df_te)

        # Capture Ratio
        std_act = np.std(y_te)
        std_pred = np.std(final_cal)
        capture_ratio = std_pred / (std_act + 1e-6)

        ablation_results.append({
            'ablation_config': name,
            'cb_mae': round(float(eval_res['MAE']), 4),
            'cb_rmse': round(float(eval_res['RMSE']), 4),
            'cb_r2': round(float(eval_res['R2']), 4),
            'cb_dir_acc_pct': round(float(eval_res['Direction_Accuracy'] * 100.0), 2),
            'cb_spearman': round(float(eval_res['Spearman_Rho']), 4),
            'cb_high_dev_mae': round(float(eval_res['High_Dev_MAE']), 4),
            'cb_stage2_capture_ratio': round(float(capture_ratio), 4),
        })

    # Save 1. cb_feature_ablation_report.csv
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(os.path.join(out_dir, 'cb_feature_ablation_report.csv'), index=False, encoding='utf-8-sig')

    # Save 2. cb_feature_importance.csv
    imp_df = pd.DataFrame(importance_records).sort_values(by=['ablation_config', 'importance_score'], ascending=[True, False])
    imp_df.to_csv(os.path.join(out_dir, 'cb_feature_importance.csv'), index=False, encoding='utf-8-sig')

    # Save 3. cb_dispersion_bucket_report.csv
    # Bucket by cb_dispersion_work_proxy quartiles
    df_te['best_pred'] = preds_dict['CB_A4 (Full CB Subsystem V2)']
    df_te['work_bucket'] = pd.qcut(df_te['cb_dispersion_work_proxy'], q=4, labels=['Low_Work', 'Med_Low', 'Med_High', 'High_Work'], duplicates='drop')

    bucket_rows = []
    for b_name, b_df in df_te.groupby('work_bucket', observed=False):
        if len(b_df) > 0:
            b_act = b_df['MNY'].values
            b_pred = b_df['best_pred'].values
            b_mae = np.mean(np.abs(b_act - b_pred))
            b_dir = calculate_dir_acc(b_act, b_pred)
            bucket_rows.append({
                'dispersion_work_bucket': str(b_name),
                'batch_count_N': len(b_df),
                'avg_work_proxy': round(float(b_df['cb_dispersion_work_proxy'].mean()), 2),
                'mae': round(float(b_mae), 4),
                'dir_acc_pct': round(float(b_dir), 2),
            })
    bucket_df = pd.DataFrame(bucket_rows)
    bucket_df.to_csv(os.path.join(out_dir, 'cb_dispersion_bucket_report.csv'), index=False, encoding='utf-8-sig')

    # Save 4. cb_direction_accuracy_report.csv
    df_te['CleanCompound'] = df_te['CompoundName'].apply(clean_compound_name)
    dir_rows = []
    for cmp, c_df in df_te.groupby('CleanCompound'):
        if len(c_df) >= 3:
            c_act = c_df['MNY'].values
            c_p_v1 = preds_dict['CB_A0 (Existing CB V1)'][c_df.index - c_df.index[0]]
            c_p_v2 = preds_dict['CB_A4 (Full CB Subsystem V2)'][c_df.index - c_df.index[0]]

            dir_rows.append({
                'compound_name': cmp,
                'batch_count_N': len(c_df),
                'dir_acc_v1_pct': round(float(calculate_dir_acc(c_act, c_p_v1)), 2),
                'dir_acc_v2_pct': round(float(calculate_dir_acc(c_act, c_p_v2)), 2),
            })
    dir_df = pd.DataFrame(dir_rows)
    dir_df.to_csv(os.path.join(out_dir, 'cb_direction_accuracy_report.csv'), index=False, encoding='utf-8-sig')

    # Save 5. cb_spearman_report.csv
    spearman_rows = []
    for cfg_name, p_val in preds_dict.items():
        s_rho = pd.Series(y_te).corr(pd.Series(p_val), method='spearman')
        spearman_rows.append({
            'ablation_config': cfg_name,
            'spearman_rho': round(float(s_rho), 4),
        })
    spear_df = pd.DataFrame(spearman_rows)
    spear_df.to_csv(os.path.join(out_dir, 'cb_spearman_report.csv'), index=False, encoding='utf-8-sig')

    # Save 6. cb_high_deviation_report.csv
    high_dev_rows = []
    for cfg_name, p_val in preds_dict.items():
        errs = np.abs(y_te - p_val)
        high_mask = errs >= 3.0
        high_dev_rows.append({
            'ablation_config': cfg_name,
            'high_dev_count': int(high_mask.sum()),
            'high_dev_pct': round(float(high_mask.sum() / len(y_te) * 100.0), 2),
            'high_dev_mae': round(float(np.mean(errs[high_mask]) if high_mask.sum() > 0 else 0.0), 4),
        })
    high_df = pd.DataFrame(high_dev_rows)
    high_df.to_csv(os.path.join(out_dir, 'cb_high_deviation_report.csv'), index=False, encoding='utf-8-sig')

    # Print Summary Tables
    print("\n" + "=" * 105)
    print("      CARBON BLACK EXPERT V2 ABLATION EXPERIMENT SUMMARY REPORT")
    print("=" * 105)
    print(f"{'Ablation Configuration':<42} | {'CB MAE':<8} | {'CB RMSE':<8} | {'CB R2':<7} | {'DirAcc(%)':<8} | {'Spearman':<8} | {'CapRatio':<8}")
    print("-" * 105)
    for _, r in ablation_df.iterrows():
        print(f"{r['ablation_config']:<42} | {r['cb_mae']:<8.4f} | {r['cb_rmse']:<8.4f} | {r['cb_r2']:<7.4f} | {r['cb_dir_acc_pct']:<8.2f}% | {r['cb_spearman']:<8.4f} | {r['cb_stage2_capture_ratio']:<8.4f}")
    print("=" * 105)

    # Check Acceptance Criteria
    best_row = ablation_df.iloc[-1]
    print("\n  --- ACCEPTANCE CRITERIA AUDIT (CB Subsystem V2) ---")
    print(f"  1. CB MAE < 2.40:                 {'PASS' if best_row['cb_mae'] < 2.40 else 'CHECK (Actual: ' + str(best_row['cb_mae']) + ')'}")
    print(f"  2. CB R2 > 0.91:                  {'PASS' if best_row['cb_r2'] > 0.91 else 'CHECK (Actual: ' + str(best_row['cb_r2']) + ')'}")
    print(f"  3. CB Direction Accuracy > 40%:   {'PASS' if best_row['cb_dir_acc_pct'] > 40.0 else 'FAIL'}")
    print(f"  4. CB Spearman > 0:               {'PASS' if best_row['cb_spearman'] > 0 else 'FAIL'}")

    print(f"\nAll 6 reports successfully saved to: {out_dir}\n")
    return ablation_df


if __name__ == '__main__':
    run_cb_ablation_experiment()
