# ============================================================================
# V3.7 Latest Feature Importance & Architecture Extractor
# ============================================================================
# Extracts feature importances across all stages and calculates normalized %
# for reporting against Continental's 4 physical levers.
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
from feature_engineering.silica_pid_feature_builder import build_silica_pid_features
from feature_engineering.cb_dispersion_feature_builder import build_cb_dispersion_features
from model_training.effective_weighting import compute_effective_sample_weights
from model_training.label_group_handler import add_label_group_information
from model_training.split_builder import generate_stratified_recipe_splits


def run_feature_importance_extract():
    print("=" * 95)
    print("  EXTRACTING V3.7 LATEST FEATURE IMPORTANCE & PHYSICAL LEVERS")
    print("=" * 95)

    out_dir = os.path.join(pipeline_root, 'reports', 'v37_architecture')
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

    pid_feats = build_silica_pid_features(df_clean)
    cb_feats = build_cb_dispersion_features(df_clean)

    for c in pid_feats.columns:
        df_clean[c] = pid_feats[c]
    for c in cb_feats.columns:
        df_clean[c] = cb_feats[c]

    df_clean = cluster_silica_carbon_black(df_clean)
    df_clean = add_label_group_information(df_clean)
    df_clean = compute_effective_sample_weights(df_clean)
    df_clean = generate_stratified_recipe_splits(df_clean, test_size=0.15, val_size=0.15)

    s1_cols = extract_stage1_recipe_features(df_clean)
    s2_cols_base = extract_stage2_process_features(df_clean)
    s2_cols = list(set(s2_cols_base + list(pid_feats.columns) + list(cb_feats.columns)))

    for col in set(s1_cols + s2_cols):
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)

    df_tr = df_clean[df_clean['_split'] == 'train'].copy()

    # Fit Stage 1 Recipe Surface Model
    X_tr_s1 = df_tr[s1_cols]
    y_tr = df_tr['MNY'].values
    weights = df_tr['_w_loss'].values

    s1_model = LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, random_state=42, verbose=-1)
    s1_model.fit(X_tr_s1, y_tr, sample_weight=weights)
    pred_s1 = s1_model.predict(X_tr_s1)
    res_s1 = y_tr - pred_s1

    # Fit Stage 2 Process Model
    X_tr_s2 = df_tr[s2_cols]
    s2_model = LGBMRegressor(n_estimators=250, learning_rate=0.03, max_depth=5, random_state=42, verbose=-1)
    s2_model.fit(X_tr_s2, res_s1, sample_weight=weights)

    raw_imps = s2_model.feature_importances_
    imp_df = pd.DataFrame({'feature': s2_cols, 'score': raw_imps}).sort_values(by='score', ascending=False).reset_index(drop=True)

    # Group into Continental 4 Physical Levers
    def categorize_lever(feat):
        f = feat.lower()
        if 'supplier' in f or 'visco' in f or 'raw_mat' in f or 'p0' in f or 'ph' in f or 'oan' in f or 'stsa' in f:
            return '1. Supplier Rubber Viscosity & Material Baseline'
        elif 'bottom' in f or 'stage6' in f or 'torque' in f:
            return '2. Bottom-Mix Power, Torque & Work Integral'
        elif 'temp' in f or 'thermal' in f or '100' in f or 'pid' in f or 'arrhenius' in f or 'discharge' in f:
            return '3. Thermal History & Silanization Temp (>100°C)'
        elif 'stage2' in f or 'dry' in f or 'dispersion' in f:
            return '4. Dry Mixing Power & Dispersion Kinetics'
        elif 'stage4' in f or 'wet' in f or 'oil' in f or 'duration' in f:
            return '5. Wet Mixing Power & Duration Window'
        elif 'env' in f or 'weather' in f or 'humidity' in f:
            return '6. Ambient Weather & Environment'
        else:
            return '7. Other Process & Machine Dynamics'

    imp_df['lever_category'] = imp_df['feature'].apply(categorize_lever)
    lever_summary = imp_df.groupby('lever_category')['score'].sum().reset_index()
    total_score = lever_summary['score'].sum()
    lever_summary['importance_pct'] = (lever_summary['score'] / total_score * 100.0).round(1)
    lever_summary = lever_summary.sort_values(by='importance_pct', ascending=False).reset_index(drop=True)

    imp_df['importance_pct'] = (imp_df['score'] / total_score * 100.0).round(2)

    imp_df.head(25).to_csv(os.path.join(out_dir, 'v37_top25_features_importance.csv'), index=False, encoding='utf-8-sig')
    lever_summary.to_csv(os.path.join(out_dir, 'v37_physical_levers_importance.csv'), index=False, encoding='utf-8-sig')

    print("\n" + "=" * 95)
    print("      V3.7 LATEST FEATURE IMPORTANCE & PHYSICAL LEVER BREAKDOWN")
    print("=" * 95)
    for _, r in lever_summary.iterrows():
        print(f"  {r['lever_category']:<52} | {r['importance_pct']:>5.1f}%")
    print("=" * 95)

    print("\n  TOP 10 INDIVIDUAL PROCESS FEATURES:")
    for _, r in imp_df.head(10).iterrows():
        print(f"    - {r['feature']:<45}: {r['importance_pct']:>5.2f}% (Score: {r['score']})")

    return imp_df, lever_summary


if __name__ == '__main__':
    run_feature_importance_extract()
