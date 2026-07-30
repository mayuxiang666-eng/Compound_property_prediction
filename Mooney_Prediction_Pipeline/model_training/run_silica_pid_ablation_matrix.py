# ============================================================================
# P1: Silica PID Ablation Matrix Runner (A0 - A8)
# ============================================================================
# Systematically evaluates 9 ablation variants to quantify individual and
# combined contributions of PID features, sub-experts, and OOF combiner:
#   A0: V3.4_E1 no PID (Control)
#   A1: PID features only, no PID expert
#   A2: PID Reaction Expert only
#   A3: PID + Wet Expert
#   A4: PID + Bottom Expert
#   A5: PID + Wet + Bottom
#   A6: PID + Wet + Bottom + Material
#   A7: Full 4 expert + OOF Combiner (non-negative constraint)
#   A8: Full 4 expert without non-negative constraint
#
# Generates pid_feature_ablation_report.csv with exact required schema.
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

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


class ConfigurableSilicaSubsystem:
    """Configurable Silica Subsystem for Ablation Matrix Experiments (A0 - A8)."""

    def __init__(self, active_experts=('pid', 'wet', 'bottom', 'material'), use_combiner=True, positive_constraint=True):
        self.active_experts = active_experts
        self.use_combiner = use_combiner
        self.positive_constraint = positive_constraint
        self.fitted_experts_ = {}
        self.combiner_ = None
        self.feature_sets_ = {}

    def fit(self, X_delta: pd.DataFrame, y_residual: np.ndarray, sample_weights: np.ndarray, pid_cols: list, s2_cols: list):
        n_samples = len(X_delta)
        all_cols = list(X_delta.columns)

        # Categorize features
        p_cols = [c for c in all_cols if 'pid' in c.lower() or 'stage5' in c.lower()] or all_cols[:5]
        w_cols = [c for c in all_cols if 'stage2' in c.lower() or 'stage3' in c.lower() or 'stage4' in c.lower()] or all_cols[:5]
        b_cols = [c for c in all_cols if 'stage6' in c.lower() or 'bottom' in c.lower()] or all_cols[:5]
        m_cols = [c for c in all_cols if 'phr' in c.lower() or 'coa' in c.lower() or 'silica' in c.lower()] or all_cols[:5]

        self.feature_sets_ = {'pid': p_cols, 'wet': w_cols, 'bottom': b_cols, 'material': m_cols}

        oof_matrix = np.zeros((n_samples, len(self.active_experts)))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for tr_idx, val_idx in kf.split(X_delta):
            X_tr, y_tr, w_tr = X_delta.iloc[tr_idx], y_residual[tr_idx], sample_weights[tr_idx]
            X_val = X_delta.iloc[val_idx]

            for idx, exp_name in enumerate(self.active_experts):
                cols = self.feature_sets_[exp_name]
                m = LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=3.0, random_state=42, verbose=-1)
                m.fit(X_tr[cols], y_tr, sample_weight=w_tr)
                oof_matrix[val_idx, idx] = m.predict(X_val[cols])

        # Combiner
        if self.use_combiner and len(self.active_experts) > 1:
            if self.positive_constraint:
                self.combiner_ = Ridge(alpha=1.0, fit_intercept=False, positive=True)
            else:
                self.combiner_ = LinearRegression(fit_intercept=False)
            self.combiner_.fit(oof_matrix, y_residual, sample_weight=sample_weights)

        # Fit final sub-experts
        for exp_name in self.active_experts:
            cols = self.feature_sets_[exp_name]
            m = LGBMRegressor(n_estimators=120, learning_rate=0.03, max_depth=4, num_leaves=15, reg_lambda=3.0, random_state=42, verbose=-1)
            m.fit(X_delta[cols], y_residual, sample_weight=sample_weights)
            self.fitted_experts_[exp_name] = m

        return self

    def predict(self, X_delta: pd.DataFrame) -> np.ndarray:
        preds = [self.fitted_experts_[e].predict(X_delta[self.feature_sets_[e]]) for e in self.active_experts]
        X_test_oof = np.column_stack(preds)
        if self.combiner_ is not None:
            return self.combiner_.predict(X_test_oof)
        return np.mean(X_test_oof, axis=1)


def run_ablation_matrix():
    print("=" * 80)
    print("      SILICA PID ABLATION MATRIX RUNNER (A0 - A8)")
    print("=" * 80)

    # Output directories
    pipeline_out = os.path.join(pipeline_root, 'reports', 'v35_silica_pid_expert_validation')
    v1_out = os.path.join(pipeline_root, 'reports', 'silica_pid_expert_v1')
    os.makedirs(pipeline_out, exist_ok=True)
    os.makedirs(v1_out, exist_ok=True)

    # Load Data
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

    # Define 9 Ablation Variants (A0 - A8)
    variants = [
        {'id': 'A0', 'desc': 'V3.4_E1 no PID', 'type': 'standard_lgbm', 'use_pid': False},
        {'id': 'A1', 'desc': 'PID features only, no PID expert', 'type': 'standard_lgbm', 'use_pid': True},
        {'id': 'A2', 'desc': 'PID Reaction Expert only', 'type': 'custom_subsystem', 'experts': ('pid',), 'combiner': False, 'pos': True},
        {'id': 'A3', 'desc': 'PID + Wet Expert', 'type': 'custom_subsystem', 'experts': ('pid', 'wet'), 'combiner': True, 'pos': True},
        {'id': 'A4', 'desc': 'PID + Bottom Expert', 'type': 'custom_subsystem', 'experts': ('pid', 'bottom'), 'combiner': True, 'pos': True},
        {'id': 'A5', 'desc': 'PID + Wet + Bottom', 'type': 'custom_subsystem', 'experts': ('pid', 'wet', 'bottom'), 'combiner': True, 'pos': True},
        {'id': 'A6', 'desc': 'PID + Wet + Bottom + Material', 'type': 'custom_subsystem', 'experts': ('pid', 'wet', 'bottom', 'material'), 'combiner': False, 'pos': True},
        {'id': 'A7', 'desc': 'Full 4 expert + OOF Combiner', 'type': 'custom_subsystem', 'experts': ('pid', 'wet', 'bottom', 'material'), 'combiner': True, 'pos': True},
        {'id': 'A8', 'desc': 'Full 4 expert without non-negative constraint', 'type': 'custom_subsystem', 'experts': ('pid', 'wet', 'bottom', 'material'), 'combiner': True, 'pos': False},
    ]

    report_rows = []

    # Cold start & OOT masks
    silica_mask = df_test['material_system'] == 'Silica'
    oil_mask = silica_mask & (pd.to_numeric(df_test.get('is_oil_loading_present', 0.0), errors='coerce').fillna(0.0) > 0)
    no_oil_mask = silica_mask & (~oil_mask)
    
    # Cold start recipes: recipes in test with <3 training instances
    train_recipe_counts = df_train['recipe_code'].value_counts() if 'recipe_code' in df_train.columns else pd.Series()
    cold_start_mask = df_test['recipe_code'].map(lambda r: train_recipe_counts.get(r, 0) < 3) if 'recipe_code' in df_test.columns else pd.Series(False, index=df_test.index)
    oot_mask = df_test.index >= (df_test.index.max() - len(df_test) // 3)

    for v in variants:
        var_id = v['id']
        print(f"Running Ablation Variant {var_id}: {v['desc']}...")

        s2_features = s2_cols_pid if v.get('use_pid', True) else s2_cols_base

        if v['type'] == 'standard_lgbm':
            model = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=False)
            model.fit(df_train, s1_cols, s2_features, target_col='MNY', cluster_col='material_system')
        else:
            # Custom subsystem
            model = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=False)
            model.fit(df_train, s1_cols, s2_features, target_col='MNY', cluster_col='material_system')
            
            # Compute Stage 1 + 1b residuals on training set
            pred_s1_tr = model.stage1_model_.predict(df_train[s1_cols])
            pred_s1b_tr = model.stage1b_bias_.predict_bias(df_train)
            res_s1b_tr = df_train['MNY'].values - (pred_s1_tr + pred_s1b_tr)

            X_s2_delta_tr, route_tr = model._transform_process_deltas(df_train, 'material_system')
            for route_key in model.stage2_experts_:
                is_silica = ('Silica' in route_key) if isinstance(route_key, tuple) else False
                if is_silica:
                    rmask = route_tr == route_key
                    custom_exp = ConfigurableSilicaSubsystem(
                        active_experts=v['experts'],
                        use_combiner=v['combiner'],
                        positive_constraint=v['pos'],
                    )
                    custom_exp.fit(
                        X_s2_delta_tr.loc[rmask],
                        res_s1b_tr[rmask],
                        df_train.loc[rmask, '_w_loss'].values,
                        list(pid_feats.columns),
                        s2_features,
                    )
                    model.stage2_experts_[route_key] = custom_exp

        final_preds, s1_preds, s1b_biases, s2_res_preds = model.predict(df_test, cluster_col='material_system')
        y_test = df_test['MNY'].values

        # Compute intra-order S2 capture
        orders = df_test['OrderID'].values if 'OrderID' in df_test.columns else np.zeros(len(df_test))
        res_for_s2 = y_test - (s1_preds + s1b_biases)
        var_res_s2 = float(np.mean([np.var(g['v'].values) for _, g in pd.DataFrame({'v': res_for_s2, 'o': orders}).groupby('o') if len(g) >= 3])) if len(orders) > 0 else 1.0
        var_s2_out = float(np.mean([np.var(g['v'].values) for _, g in pd.DataFrame({'v': s2_res_preds, 'o': orders}).groupby('o') if len(g) >= 3])) if len(orders) > 0 else 0.0
        s2_capture_pct = (var_s2_out / var_res_s2 * 100.0) if var_res_s2 > 1e-5 else 0.0

        # Metrics for subsets
        m_overall = evaluate_mooney_predictions(y_test, final_preds, df_test)
        
        mae_silica_oil = float(np.mean(np.abs(y_test[oil_mask] - final_preds[oil_mask]))) if oil_mask.sum() > 0 else np.nan
        mae_silica_nooil = float(np.mean(np.abs(y_test[no_oil_mask] - final_preds[no_oil_mask]))) if no_oil_mask.sum() > 0 else np.nan
        mae_oot = float(np.mean(np.abs(y_test[oot_mask] - final_preds[oot_mask]))) if oot_mask.sum() > 0 else np.nan
        mae_coldstart = float(np.mean(np.abs(y_test[cold_start_mask] - final_preds[cold_start_mask]))) if cold_start_mask.sum() > 0 else np.nan

        for subset_name, mask_curr in [('overall', slice(None)), ('silica_subset', silica_mask)]:
            y_sub = y_test[mask_curr]
            p_sub = final_preds[mask_curr]
            df_sub = df_test[mask_curr] if isinstance(mask_curr, pd.Series) else df_test
            m_sub = evaluate_mooney_predictions(y_sub, p_sub, df_sub)

            report_rows.append({
                'variant': f"{var_id}_{v['desc']}",
                'split_id': 'stratified_recipe_leak_free',
                'subset': subset_name,
                'MAE': m_sub['MAE'],
                'RMSE': m_sub['RMSE'],
                'R2': m_sub['R2'],
                'Spearman': m_sub['Spearman_Rho'],
                'Direction_Accuracy': m_sub['Direction_Accuracy'] * 100.0,
                'Variance_Ratio': m_sub['Variance_Ratio'],
                'High_Deviation_MAE': m_sub['High_Dev_MAE'],
                'Stage2_Capture': s2_capture_pct,
                'Silica_Oil_MAE': mae_silica_oil,
                'Silica_NoOil_MAE': mae_silica_nooil,
                'OOT_MAE': mae_oot,
                'ColdStart_MAE': mae_coldstart,
            })

    ablation_df = pd.DataFrame(report_rows)
    ablation_df.to_csv(os.path.join(pipeline_out, 'pid_feature_ablation_report.csv'), index=False, encoding='utf-8-sig')
    ablation_df.to_csv(os.path.join(v1_out, 'pid_feature_ablation_report.csv'), index=False, encoding='utf-8-sig')

    # Print Summary Table
    print("\n" + "=" * 90)
    print("            SILICA PID ABLATION MATRIX SUMMARY (A0 - A8, Silica Subset)")
    print("=" * 90)
    silica_ablation = ablation_df[ablation_df['subset'] == 'silica_subset']
    print(f"{'Variant':<35} | {'MAE':<7} | {'RMSE':<7} | {'R2':<7} | {'Spearman':<8} | {'DirAcc(%)':<8} | {'S2 Capture':<10}")
    print("-" * 90)
    for _, r in silica_ablation.iterrows():
        print(f"{r['variant']:<35} | {r['MAE']:<7.4f} | {r['RMSE']:<7.4f} | {r['R2']:<7.4f} | {r['Spearman']:<8.4f} | {r['Direction_Accuracy']:<8.2f}% | {r['Stage2_Capture']:>5.2f}%")
    print("=" * 90)

    print(f"\nAblation report saved to:")
    print(f"  1. {os.path.join(pipeline_out, 'pid_feature_ablation_report.csv')}")
    print(f"  2. {os.path.join(v1_out, 'pid_feature_ablation_report.csv')}\n")

    return ablation_df


if __name__ == '__main__':
    run_ablation_matrix()
