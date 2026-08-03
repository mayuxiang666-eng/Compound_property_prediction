# ============================================================================
# Raw Material Feature Correlation & Importance Analysis Script (V3.6)
# ============================================================================
# Analyzes raw material inspection metrics for Silica vs Carbon Black compounds
# separately. Computes Spearman/Pearson correlations and GBDT feature importance
# against actual Lab Mooney Viscosity (MNY) to guide SAP QM feature integration.
#
# Output:
# - reports/v36_explainable_production/raw_material_correlation_analysis.csv
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


def analyze_material_correlations():
    print("=" * 90)
    print("  RAW MATERIAL FEATURE CORRELATION & IMPORTANCE ANALYSIS (Silica vs Carbon Black)")
    print("=" * 90)

    out_dir = os.path.join(pipeline_root, 'reports', 'v36_explainable_production')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Enriched Full Dataset
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

    df = cluster_silica_carbon_black(df)

    # Candidate Raw Material & COA Feature List
    raw_material_cols = [
        'supplier_rubber_viscosity_avg',
        'supplier_silica_moisture_avg',
        'supplier_silica_surface_area_avg',
        'supplier_carbon_black_structure_avg',
        'supplier_carbon_black_surface_area_avg',
        'supplier_carbon_black_moisture_avg',
        'weight_pct_solid_elastomer',
        'weight_pct_natural_rubber',
        'weight_pct_silica',
        'weight_pct_carbon_black',
        'weight_pct_oil',
        'weight_pct_silian',
        'silica_phr',
        'ratio_nr_rubber',
        'ratio_filler_polymer',
        'ratio_oil_polymer',
        'ratio_oil_filler',
    ]

    avail_raw_cols = [c for c in raw_material_cols if c in df.columns]

    results = []

    # Separate Silica vs Carbon Black Analysis
    for system in ['Silica', 'CarbonBlack']:
        sub_df = df[df['material_system'] == system].copy()
        print(f"\n  Analyzing {system} Compounds (Total Batches N = {len(sub_df)})...")

        # Fit LGBM Regressor to get Feature Importance
        X = sub_df[avail_raw_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        y = sub_df['MNY'].values

        model = LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=5, random_state=42, verbose=-1)
        model.fit(X, y)
        importances = model.feature_importances_
        imp_dict = dict(zip(avail_raw_cols, importances))

        for col in avail_raw_cols:
            vals = pd.to_numeric(sub_df[col], errors='coerce')
            valid_mask = vals.notnull() & sub_df['MNY'].notnull()

            if valid_mask.sum() > 30:
                spearman_corr = vals[valid_mask].corr(sub_df.loc[valid_mask, 'MNY'], method='spearman')
                pearson_corr = vals[valid_mask].corr(sub_df.loc[valid_mask, 'MNY'], method='pearson')
                coverage = valid_mask.sum() / len(sub_df) * 100.0
            else:
                spearman_corr, pearson_corr, coverage = 0.0, 0.0, 0.0

            results.append({
                'material_system': system,
                'feature_name': col,
                'spearman_rho': round(float(spearman_corr), 4),
                'pearson_r': round(float(pearson_corr), 4),
                'abs_spearman': round(abs(float(spearman_corr)), 4),
                'lgbm_importance': int(imp_dict.get(col, 0)),
                'data_coverage_pct': round(float(coverage), 1),
                'batch_count': int(valid_mask.sum()),
            })

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by=['material_system', 'abs_spearman'], ascending=[True, False]).reset_index(drop=True)

    csv_path = os.path.join(out_dir, 'raw_material_correlation_analysis.csv')
    res_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 90)
    print("      SILICA SYSTEM RAW MATERIAL FEATURE CORRELATION & IMPORTANCE")
    print("=" * 90)
    silica_df = res_df[res_df['material_system'] == 'Silica'].copy()
    print(f"{'Feature Name':<38} | {'Spearman Rho':<12} | {'Pearson r':<10} | {'Importance':<10} | {'Coverage (%)':<10}")
    print("-" * 90)
    for _, r in silica_df.iterrows():
        print(f"{r['feature_name']:<38} | {r['spearman_rho']:<12.4f} | {r['pearson_r']:<10.4f} | {r['lgbm_importance']:<10} | {r['data_coverage_pct']:<10.1f}%")

    print("\n" + "=" * 90)
    print("      CARBON BLACK SYSTEM RAW MATERIAL FEATURE CORRELATION & IMPORTANCE")
    print("=" * 90)
    cb_df = res_df[res_df['material_system'] == 'CarbonBlack'].copy()
    print(f"{'Feature Name':<38} | {'Spearman Rho':<12} | {'Pearson r':<10} | {'Importance':<10} | {'Coverage (%)':<10}")
    print("-" * 90)
    for _, r in cb_df.iterrows():
        print(f"{r['feature_name']:<38} | {r['spearman_rho']:<12.4f} | {r['pearson_r']:<10.4f} | {r['lgbm_importance']:<10} | {r['data_coverage_pct']:<10.1f}%")

    print(f"\nReport saved to: {csv_path}\n")
    return res_df


if __name__ == '__main__':
    analyze_material_correlations()
