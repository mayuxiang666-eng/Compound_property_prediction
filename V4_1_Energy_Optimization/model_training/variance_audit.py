# ============================================================================
# Variance Collapse Root-Cause Audit (P0 Diagnostic)
# ============================================================================
# Traces variance at every pipeline layer to identify where prediction
# variance disappears.
#
# Output layers:
#   1. Actual y (ground truth)
#   2. Stage 1 pred (recipe GBDT)
#   3. Raw Bias (compound-level raw mean residual)
#   4. Shrunk Bias (after Empirical Bayes shrinkage)
#   5. Residuals after S1 + Shrunk Bias
#   6. Stage 2 output (phase-routed process-delta GBDT)
#   7. Final prediction (S1 + ShrunkBias + S2)
#
# Variance is reported:
#   - Globally (total sample variance)
#   - Intra-order (mean variance within OrderID groups, ≥3 batches)
#   - Per-compound (across compounds)
#   - Per-phase-route (oil_wet vs no_oil_dry)
#   - Per-material-system (Silica vs CarbonBlack)
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
from model_training.split_builder import generate_stratified_recipe_splits


# ── Helpers ──────────────────────────────────────────────────────────────────

def _global_variance(arr):
    """Total sample variance."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.var(arr)) if len(arr) > 1 else 0.0


def _global_std(arr):
    return float(np.sqrt(_global_variance(arr)))


def _intra_order_variance(series, order_ids):
    """Mean intra-order variance (orders with ≥3 batches)."""
    df = pd.DataFrame({'value': series, 'order': order_ids})
    order_vars = []
    for _, g in df.groupby('order'):
        vals = g['value'].dropna().values
        if len(vals) >= 3:
            order_vars.append(np.var(vals))
    return float(np.mean(order_vars)) if order_vars else 0.0


def _intra_order_std(series, order_ids):
    return float(np.sqrt(_intra_order_variance(series, order_ids)))


def _variance_ratio(pred_series, actual_series, order_ids):
    """Intra-order Var(pred) / Var(actual), averaged across qualifying orders."""
    df = pd.DataFrame({
        'pred': pred_series,
        'actual': actual_series,
        'order': order_ids,
    })
    ratios = []
    for _, g in df.groupby('order'):
        a = g['actual'].dropna().values
        p = g['pred'].dropna().values
        if len(a) >= 3:
            var_a = np.var(a)
            var_p = np.var(p)
            if var_a > 1e-5:
                ratios.append(var_p / var_a)
    return float(np.mean(ratios)) if ratios else np.nan


# ── Main Audit ───────────────────────────────────────────────────────────────

def run_variance_audit():
    print("=" * 80)
    print("     VARIANCE COLLAPSE ROOT-CAUSE AUDIT")
    print("=" * 80)

    # ── 1. Load data (same logic as run_architecture_evaluation) ─────────────
    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../../data/stage_statistics_enriched_all_features_weather_v4.csv',
    ))
    if not os.path.exists(data_path):
        data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '../../data/enriched_mny_all.csv',
        ))
    print(f"  Loading: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    if 'MNY' not in df.columns and 'Mooney_Viscosity' in df.columns:
        df['MNY'] = df['Mooney_Viscosity']
    df = df.dropna(subset=['MNY']).copy()

    if 'CompoundName' not in df.columns and 'Compound' in df.columns:
        df['CompoundName'] = df['Compound']
    if 'OrderID' not in df.columns and 'Order_No' in df.columns:
        df['OrderID'] = df['Order_No']

    df = cluster_silica_carbon_black(df)
    s1_cols = extract_stage1_recipe_features(df)
    s2_cols = extract_stage2_process_features(df)
    for col in s1_cols + s2_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    df = add_label_group_information(df)
    df = compute_effective_sample_weights(df)
    df = generate_stratified_recipe_splits(df, test_size=0.15, val_size=0.15)

    df_train = df[df['_split'] == 'train'].copy()
    df_test = df[df['_split'] == 'test'].copy()
    print(f"  Train: {len(df_train)} rows | Test: {len(df_test)} rows\n")

    # ── 2. Fit V3.2 and capture all intermediate signals ─────────────────────
    model = HybridUnifiedMooneyModel(shrinkage_k=5.0, use_cluster_experts=True)
    model.fit(
        df_train, s1_cols, s2_cols,
        target_col='MNY', cluster_col='material_system',
    )

    # --- Extract Stage-by-stage on TEST set ---
    y_actual = df_test['MNY'].values
    pred_s1 = model.stage1_model_.predict(df_test[model.feature_names_s1_])

    # Raw bias: un-shrunk compound-level mean residual from training
    bias_report = model.get_bias_report()
    raw_bias_map = dict(zip(bias_report['compound_name'], bias_report['raw_compound_bias']))
    shrunk_bias_map = dict(zip(bias_report['compound_name'], bias_report['shrunk_bias']))
    cluster_bias_map = model.stage1b_bias_.cluster_biases_
    global_bias = model.stage1b_bias_.global_bias_

    raw_bias_arr = np.array([
        raw_bias_map.get(c, cluster_bias_map.get(
            str(df_test.iloc[i].get('material_system', 'GLOBAL')),
            global_bias,
        ))
        for i, c in enumerate(df_test['CompoundName'])
    ])

    shrunk_bias_arr = model.stage1b_bias_.predict_bias(df_test)

    residual_after_s1 = y_actual - pred_s1
    residual_after_s1_bias = y_actual - (pred_s1 + shrunk_bias_arr)

    # Stage 2
    X_s2_delta, route = model._transform_process_deltas(df_test, 'material_system')
    pred_s2 = np.zeros(len(df_test))
    fallback_expert = (
        model.stage2_experts_.get('oil_wet')
        or next(iter(model.stage2_experts_.values()))
    )
    for route_name in route.unique():
        mask = route == route_name
        expert = model.stage2_experts_.get(route_name, fallback_expert)
        pred_s2[mask] = expert.predict(X_s2_delta.loc[mask])

    final_pred = pred_s1 + shrunk_bias_arr + pred_s2

    # ── 3. Gather order IDs ──────────────────────────────────────────────────
    orders = df_test['OrderID'].values if 'OrderID' in df_test.columns else np.zeros(len(df_test))
    phases = route.values
    mat_sys = df_test['material_system'].values if 'material_system' in df_test.columns else np.full(len(df_test), 'ALL')

    # ── 4. Build layer-by-layer audit table ──────────────────────────────────
    layers = [
        ('1_actual_y',          y_actual),
        ('2_stage1_pred',       pred_s1),
        ('3_raw_bias',          raw_bias_arr),
        ('4_shrunk_bias',       shrunk_bias_arr),
        ('5_s1_plus_bias_pred', pred_s1 + shrunk_bias_arr),
        ('6_residual_for_s2',   residual_after_s1_bias),
        ('7_stage2_output',     pred_s2),
        ('8_final_pred',        final_pred),
    ]

    rows = []
    for layer_name, values in layers:
        row = {
            'layer': layer_name,
            # Global
            'global_var': _global_variance(values),
            'global_std': _global_std(values),
            # Intra-order
            'intra_order_var': _intra_order_variance(values, orders),
            'intra_order_std': _intra_order_std(values, orders),
        }
        # Variance ratio vs actual (only for prediction-like layers)
        if layer_name not in ('1_actual_y', '3_raw_bias', '4_shrunk_bias', '6_residual_for_s2', '7_stage2_output'):
            row['intra_order_var_ratio_vs_actual'] = _variance_ratio(values, y_actual, orders)
        else:
            row['intra_order_var_ratio_vs_actual'] = np.nan

        rows.append(row)

    audit_df = pd.DataFrame(rows)

    # ── 5. Per-phase-route breakdown ─────────────────────────────────────────
    phase_rows = []
    for phase_name in sorted(np.unique(phases)):
        mask = phases == phase_name
        for layer_name, values in layers:
            vals = values[mask]
            actual_subset = y_actual[mask]
            orders_subset = orders[mask]
            r = {
                'phase_route': phase_name,
                'layer': layer_name,
                'n_samples': int(mask.sum()),
                'global_var': _global_variance(vals),
                'global_std': _global_std(vals),
                'intra_order_var': _intra_order_variance(vals, orders_subset),
                'intra_order_std': _intra_order_std(vals, orders_subset),
            }
            if layer_name not in ('1_actual_y', '3_raw_bias', '4_shrunk_bias', '6_residual_for_s2', '7_stage2_output'):
                r['intra_order_var_ratio'] = _variance_ratio(vals, actual_subset, orders_subset)
            else:
                r['intra_order_var_ratio'] = np.nan
            phase_rows.append(r)
    phase_df = pd.DataFrame(phase_rows)

    # ── 6. Per-material-system breakdown ─────────────────────────────────────
    matsys_rows = []
    for ms in sorted(np.unique(mat_sys)):
        mask = mat_sys == ms
        for layer_name, values in layers:
            vals = values[mask]
            actual_subset = y_actual[mask]
            orders_subset = orders[mask]
            r = {
                'material_system': ms,
                'layer': layer_name,
                'n_samples': int(mask.sum()),
                'global_var': _global_variance(vals),
                'global_std': _global_std(vals),
                'intra_order_var': _intra_order_variance(vals, orders_subset),
                'intra_order_std': _intra_order_std(vals, orders_subset),
            }
            if layer_name not in ('1_actual_y', '3_raw_bias', '4_shrunk_bias', '6_residual_for_s2', '7_stage2_output'):
                r['intra_order_var_ratio'] = _variance_ratio(vals, actual_subset, orders_subset)
            else:
                r['intra_order_var_ratio'] = np.nan
            matsys_rows.append(r)
    matsys_df = pd.DataFrame(matsys_rows)

    # ── 7. Per-compound bias variance analysis ───────────────────────────────
    compound_names = df_test['CompoundName'].values
    unique_compounds = np.unique(compound_names)
    bias_var_rows = []
    for comp in unique_compounds:
        comp_mask = compound_names == comp
        n = int(comp_mask.sum())
        raw_b = raw_bias_map.get(comp, np.nan)
        shrunk_b = shrunk_bias_map.get(comp, np.nan)
        bias_var_rows.append({
            'compound': comp,
            'n_test_rows': n,
            'raw_bias': raw_b,
            'shrunk_bias': shrunk_b,
            'bias_shrinkage_ratio': abs(shrunk_b / raw_b) if raw_b != 0 and np.isfinite(raw_b) else np.nan,
            'actual_std_within': float(np.std(y_actual[comp_mask])) if n >= 2 else 0.0,
            'pred_std_within': float(np.std(final_pred[comp_mask])) if n >= 2 else 0.0,
            's1_std_within': float(np.std(pred_s1[comp_mask])) if n >= 2 else 0.0,
            's2_std_within': float(np.std(pred_s2[comp_mask])) if n >= 2 else 0.0,
        })
    compound_df = pd.DataFrame(bias_var_rows).sort_values('n_test_rows', ascending=False).reset_index(drop=True)

    # ── 8. Stage 2 process-delta feature variance ────────────────────────────
    # Check if delta features themselves have variance or are all near-zero
    delta_var_rows = []
    for col in model.feature_names_s2_:
        raw_vals = pd.to_numeric(df_test[col], errors='coerce').fillna(0.0).values
        delta_vals = X_s2_delta[col].values if col in X_s2_delta.columns else np.zeros(len(df_test))
        delta_var_rows.append({
            'feature': col,
            'raw_var': float(np.var(raw_vals)),
            'raw_std': float(np.std(raw_vals)),
            'delta_var': float(np.var(delta_vals)),
            'delta_std': float(np.std(delta_vals)),
            'delta_vs_raw_ratio': float(np.var(delta_vals) / np.var(raw_vals)) if np.var(raw_vals) > 1e-10 else np.nan,
            'pct_zero_delta': float(np.mean(np.abs(delta_vals) < 1e-8) * 100),
        })
    delta_df = pd.DataFrame(delta_var_rows).sort_values('delta_var', ascending=False).reset_index(drop=True)

    # ── 9. Save outputs ─────────────────────────────────────────────────────
    output_dir = os.path.join(pipeline_root, 'reports', 'variance_audit')
    v33_baseline_dir = os.path.join(pipeline_root, 'reports', 'v33_phase_route_fix')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(v33_baseline_dir, exist_ok=True)

    # ── 10. Console Report ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  LAYER-BY-LAYER VARIANCE DECOMPOSITION (Test Set)")
    print("=" * 80)
    print(f"{'Layer':<25} | {'Global Var':>12} | {'Global Std':>10} | {'IntraOrd Var':>12} | {'IntraOrd Std':>12} | {'VarRatio':>10}")
    print("-" * 95)
    for _, r in audit_df.iterrows():
        vr = f"{r['intra_order_var_ratio_vs_actual']:.4f}" if np.isfinite(r['intra_order_var_ratio_vs_actual']) else "  -"
        print(f"{r['layer']:<25} | {r['global_var']:>12.4f} | {r['global_std']:>10.4f} | {r['intra_order_var']:>12.4f} | {r['intra_order_std']:>12.4f} | {vr:>10}")

    print("\n" + "=" * 80)
    print("  VARIANCE BY PHASE ROUTE (Prediction Layers Only)")
    print("=" * 80)
    pred_layers = ['2_stage1_pred', '5_s1_plus_bias_pred', '8_final_pred', '1_actual_y']
    for phase_name in sorted(phase_df['phase_route'].unique()):
        pf = phase_df[phase_df['phase_route'] == phase_name]
        n = pf['n_samples'].iloc[0]
        print(f"\n  -- {phase_name} (n={n}) --")
        for lyr in pred_layers:
            row = pf[pf['layer'] == lyr]
            if row.empty:
                continue
            row = row.iloc[0]
            vr = f"{row['intra_order_var_ratio']:.4f}" if np.isfinite(row['intra_order_var_ratio']) else "  -"
            print(f"    {lyr:<25} | GlobalStd={row['global_std']:>8.4f} | IntraOrdStd={row['intra_order_std']:>8.4f} | VarRatio={vr}")

    print("\n" + "=" * 80)
    print("  VARIANCE BY MATERIAL SYSTEM (Prediction Layers Only)")
    print("=" * 80)
    for ms in sorted(matsys_df['material_system'].unique()):
        mf = matsys_df[matsys_df['material_system'] == ms]
        n = mf['n_samples'].iloc[0]
        print(f"\n  -- {ms} (n={n}) --")
        for lyr in pred_layers:
            row = mf[mf['layer'] == lyr]
            if row.empty:
                continue
            row = row.iloc[0]
            vr = f"{row['intra_order_var_ratio']:.4f}" if np.isfinite(row['intra_order_var_ratio']) else "  -"
            print(f"    {lyr:<25} | GlobalStd={row['global_std']:>8.4f} | IntraOrdStd={row['intra_order_std']:>8.4f} | VarRatio={vr}")

    # ── 11. Key diagnostic summary ───────────────────────────────────────────
    actual_intra_var = audit_df.loc[audit_df['layer'] == '1_actual_y', 'intra_order_var'].values[0]
    s1_vr = audit_df.loc[audit_df['layer'] == '2_stage1_pred', 'intra_order_var_ratio_vs_actual'].values[0]
    s1b_vr = audit_df.loc[audit_df['layer'] == '5_s1_plus_bias_pred', 'intra_order_var_ratio_vs_actual'].values[0]
    final_vr = audit_df.loc[audit_df['layer'] == '8_final_pred', 'intra_order_var_ratio_vs_actual'].values[0]

    s2_intra_var = audit_df.loc[audit_df['layer'] == '7_stage2_output', 'intra_order_var'].values[0]
    residual_intra_var = audit_df.loc[audit_df['layer'] == '6_residual_for_s2', 'intra_order_var'].values[0]

    print("\n" + "=" * 80)
    print("  ** DIAGNOSTIC SUMMARY **")
    print("=" * 80)
    print(f"  Actual intra-order variance:                 {actual_intra_var:.4f}")
    print(f"  Stage 1 intra-order variance ratio:          {s1_vr:.4f}  {'!! LOW' if s1_vr < 0.50 else 'OK'}")
    print(f"  S1+Bias intra-order variance ratio:          {s1b_vr:.4f}  {'!! LOW' if s1b_vr < 0.50 else 'OK'}")
    print(f"  Final pred intra-order variance ratio:       {final_vr:.4f}  {'!! LOW' if final_vr < 0.50 else 'OK'}")
    print(f"  Residual-for-S2 intra-order variance:        {residual_intra_var:.4f}")
    print(f"  Stage 2 output intra-order variance:         {s2_intra_var:.4f}")
    s2_capture = s2_intra_var / residual_intra_var if residual_intra_var > 1e-5 else 0.0
    print(f"  S2 variance capture ratio (S2var/resVar):   {s2_capture:.4f}  {'!! LOW' if s2_capture < 0.10 else 'OK'}")
    print()

    # ── 12. Top-10 process delta features by variance ────────────────────────
    print("\n  Top-10 process delta features by delta variance:")
    for _, r in delta_df.head(10).iterrows():
        ratio_str = f"{r['delta_vs_raw_ratio']:.4f}" if np.isfinite(r['delta_vs_raw_ratio']) else "N/A"
        print(f"    {r['feature']:<45} | delta_std={r['delta_std']:>8.4f} | raw_std={r['raw_std']:>8.4f} | ratio={ratio_str} | %zero={r['pct_zero_delta']:.1f}%")

    # ── 13. Top-10 compounds with largest bias shrinkage ─────────────────────
    compound_df_valid = compound_df[compound_df['bias_shrinkage_ratio'].notna() & (compound_df['n_test_rows'] >= 3)].copy()
    if not compound_df_valid.empty:
        print(f"\n  Top-10 compounds by bias shrinkage impact (n_test >= 3):")
        for _, r in compound_df_valid.head(10).iterrows():
            print(f"    {r['compound']:<30} | n={r['n_test_rows']:>3} | raw_bias={r['raw_bias']:>+7.3f} | shrunk={r['shrunk_bias']:>+7.3f} | actual_std={r['actual_std_within']:>6.3f} | pred_std={r['pred_std_within']:>6.3f}")

    # ── 14. FORMAL PASS / FAIL GATES ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  VARIANCE AUDIT ACCEPTANCE GATES")
    print("=" * 80)

    # Retrieve per-subsystem variance ratios
    def _get_vr(df, col, val, layer):
        row = df[(df[col] == val) & (df['layer'] == layer)]
        if row.empty or 'intra_order_var_ratio' not in row.columns:
            return np.nan
        return row.iloc[0]['intra_order_var_ratio']

    cb_final_vr = _get_vr(matsys_df, 'material_system', 'CarbonBlack', '8_final_pred')
    si_final_vr = _get_vr(matsys_df, 'material_system', 'Silica', '8_final_pred')
    nod_final_vr = _get_vr(phase_df, 'phase_route', 'no_oil_dry', '8_final_pred')
    ow_final_vr = _get_vr(phase_df, 'phase_route', 'oil_wet', '8_final_pred')

    # Risk B Fix: Robust G7 Compound Standard Deviation Ratio
    valid_compounds = compound_df[(compound_df['actual_std_within'] > 0.2) & (compound_df['n_test_rows'] >= 5)].copy()
    if not valid_compounds.empty:
        raw_ratios = valid_compounds['pred_std_within'] / valid_compounds['actual_std_within']
        clipped_ratios = np.clip(raw_ratios, 0, 5.0)
        robust_median_std_ratio = float(np.median(clipped_ratios))
        weighted_std_ratio = float(np.average(clipped_ratios, weights=valid_compounds['n_test_rows']))
    else:
        robust_median_std_ratio = np.nan
        weighted_std_ratio = np.nan

    capture_status = (
        "Strong Pass (>= 35%)" if s2_capture >= 0.35 else
        ("Target Pass (>= 25%)" if s2_capture >= 0.25 else
         ("Minimum Pass (>= 10%)" if s2_capture >= 0.10 else "FAIL (< 10%)"))
    )

    gates = [
        ("G1 Overall Final VarRatio in [0.50, 2.00]",
         0.50 <= final_vr <= 2.00, f"{final_vr:.4f}"),
        ("G2 S2 Variance Capture Tier (Min >= 10%, Target >= 25%, Strong >= 35%)",
         s2_capture >= 0.10, f"{s2_capture*100:.2f}% [{capture_status}]"),
        ("G3 CarbonBlack Final VarRatio >= 0.20",
         np.isnan(cb_final_vr) or cb_final_vr >= 0.20, f"{cb_final_vr:.4f}" if np.isfinite(cb_final_vr) else "N/A"),
        ("G4 Silica Final VarRatio in [0.40, 2.50]",
         np.isnan(si_final_vr) or (0.40 <= si_final_vr <= 2.50), f"{si_final_vr:.4f}" if np.isfinite(si_final_vr) else "N/A"),
        ("G5 No-Oil/Dry Final VarRatio >= 0.20",
         np.isnan(nod_final_vr) or nod_final_vr >= 0.20, f"{nod_final_vr:.4f}" if np.isfinite(nod_final_vr) else "N/A"),
        ("G6 Oil/Wet Final VarRatio <= 3.00",
         np.isnan(ow_final_vr) or ow_final_vr <= 3.00, f"{ow_final_vr:.4f}" if np.isfinite(ow_final_vr) else "N/A"),
        ("G7 Robust Compound std ratio (valid n>=5, actual_std>0.2) in [0.30, 1.50]",
         np.isnan(robust_median_std_ratio) or (0.30 <= robust_median_std_ratio <= 1.50),
         f"median={robust_median_std_ratio:.4f}, weighted={weighted_std_ratio:.4f}" if np.isfinite(robust_median_std_ratio) else "N/A"),
    ]

    all_pass = True
    gate_summary_rows = []
    for name, passed, value in gates:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}  (value={value})")
        gate_summary_rows.append({'gate': name, 'status': status, 'value': value})

    gate_summary_df = pd.DataFrame(gate_summary_rows)

    stage2_capture_df = pd.DataFrame([{
        'residual_intra_order_var': residual_intra_var,
        'stage2_output_intra_order_var': s2_intra_var,
        'capture_ratio': s2_capture,
        'capture_pct': s2_capture * 100,
        'tier_status': capture_status
    }])

    # Write files to output_dir and v33_baseline_dir
    audit_df.to_csv(os.path.join(output_dir, 'layer_variance_summary.csv'), index=False, encoding='utf-8-sig')
    phase_df.to_csv(os.path.join(output_dir, 'phase_route_variance.csv'), index=False, encoding='utf-8-sig')
    matsys_df.to_csv(os.path.join(output_dir, 'material_system_variance.csv'), index=False, encoding='utf-8-sig')
    compound_df.to_csv(os.path.join(output_dir, 'compound_std_ratio.csv'), index=False, encoding='utf-8-sig')
    stage2_capture_df.to_csv(os.path.join(output_dir, 'stage2_capture_summary.csv'), index=False, encoding='utf-8-sig')
    gate_summary_df.to_csv(os.path.join(output_dir, 'gate_summary.csv'), index=False, encoding='utf-8-sig')

    audit_df.to_csv(os.path.join(v33_baseline_dir, 'layer_variance_summary.csv'), index=False, encoding='utf-8-sig')
    phase_df.to_csv(os.path.join(v33_baseline_dir, 'phase_route_variance.csv'), index=False, encoding='utf-8-sig')
    matsys_df.to_csv(os.path.join(v33_baseline_dir, 'material_system_variance.csv'), index=False, encoding='utf-8-sig')
    compound_df.to_csv(os.path.join(v33_baseline_dir, 'compound_std_ratio.csv'), index=False, encoding='utf-8-sig')
    stage2_capture_df.to_csv(os.path.join(v33_baseline_dir, 'stage2_capture_summary.csv'), index=False, encoding='utf-8-sig')
    gate_summary_df.to_csv(os.path.join(v33_baseline_dir, 'gate_summary.csv'), index=False, encoding='utf-8-sig')

    print("-" * 80)
    if all_pass:
        print("  >>> ALL GATES PASSED <<<")
    else:
        n_fail = sum(1 for _, p, _ in gates if not p)
        print(f"  >>> {n_fail} GATE(S) FAILED — further tuning required <<<")
    print("=" * 80)

    return {
        'audit_df': audit_df,
        'phase_df': phase_df,
        'matsys_df': matsys_df,
        'compound_df': compound_df,
        'delta_df': delta_df,
        'all_gates_passed': all_pass,
    }

    # ── 10. Console Report ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  LAYER-BY-LAYER VARIANCE DECOMPOSITION (Test Set)")
    print("=" * 80)
    print(f"{'Layer':<25} | {'Global Var':>12} | {'Global Std':>10} | {'IntraOrd Var':>12} | {'IntraOrd Std':>12} | {'VarRatio':>10}")
    print("-" * 95)
    for _, r in audit_df.iterrows():
        vr = f"{r['intra_order_var_ratio_vs_actual']:.4f}" if np.isfinite(r['intra_order_var_ratio_vs_actual']) else "  -"
        print(f"{r['layer']:<25} | {r['global_var']:>12.4f} | {r['global_std']:>10.4f} | {r['intra_order_var']:>12.4f} | {r['intra_order_std']:>12.4f} | {vr:>10}")

    print("\n" + "=" * 80)
    print("  VARIANCE BY PHASE ROUTE (Prediction Layers Only)")
    print("=" * 80)
    pred_layers = ['2_stage1_pred', '5_s1_plus_bias_pred', '8_final_pred', '1_actual_y']
    for phase_name in sorted(phase_df['phase_route'].unique()):
        pf = phase_df[phase_df['phase_route'] == phase_name]
        n = pf['n_samples'].iloc[0]
        print(f"\n  -- {phase_name} (n={n}) --")
        for lyr in pred_layers:
            row = pf[pf['layer'] == lyr]
            if row.empty:
                continue
            row = row.iloc[0]
            vr = f"{row['intra_order_var_ratio']:.4f}" if np.isfinite(row['intra_order_var_ratio']) else "  —"
            print(f"    {lyr:<25} | GlobalStd={row['global_std']:>8.4f} | IntraOrdStd={row['intra_order_std']:>8.4f} | VarRatio={vr}")

    print("\n" + "=" * 80)
    print("  VARIANCE BY MATERIAL SYSTEM (Prediction Layers Only)")
    print("=" * 80)
    for ms in sorted(matsys_df['material_system'].unique()):
        mf = matsys_df[matsys_df['material_system'] == ms]
        n = mf['n_samples'].iloc[0]
        print(f"\n  -- {ms} (n={n}) --")
        for lyr in pred_layers:
            row = mf[mf['layer'] == lyr]
            if row.empty:
                continue
            row = row.iloc[0]
            vr = f"{row['intra_order_var_ratio']:.4f}" if np.isfinite(row['intra_order_var_ratio']) else "  —"
            print(f"    {lyr:<25} | GlobalStd={row['global_std']:>8.4f} | IntraOrdStd={row['intra_order_std']:>8.4f} | VarRatio={vr}")

    # ── 11. Key diagnostic summary ───────────────────────────────────────────
    actual_intra_var = audit_df.loc[audit_df['layer'] == '1_actual_y', 'intra_order_var'].values[0]
    s1_vr = audit_df.loc[audit_df['layer'] == '2_stage1_pred', 'intra_order_var_ratio_vs_actual'].values[0]
    s1b_vr = audit_df.loc[audit_df['layer'] == '5_s1_plus_bias_pred', 'intra_order_var_ratio_vs_actual'].values[0]
    final_vr = audit_df.loc[audit_df['layer'] == '8_final_pred', 'intra_order_var_ratio_vs_actual'].values[0]

    s2_intra_var = audit_df.loc[audit_df['layer'] == '7_stage2_output', 'intra_order_var'].values[0]
    residual_intra_var = audit_df.loc[audit_df['layer'] == '6_residual_for_s2', 'intra_order_var'].values[0]

    print("\n" + "=" * 80)
    print("  ** DIAGNOSTIC SUMMARY **")
    print("=" * 80)
    print(f"  Actual intra-order variance:                 {actual_intra_var:.4f}")
    print(f"  Stage 1 intra-order variance ratio:          {s1_vr:.4f}  {'!! LOW' if s1_vr < 0.50 else 'OK'}")
    print(f"  S1+Bias intra-order variance ratio:          {s1b_vr:.4f}  {'!! LOW' if s1b_vr < 0.50 else 'OK'}")
    print(f"  Final pred intra-order variance ratio:       {final_vr:.4f}  {'!! LOW' if final_vr < 0.50 else 'OK'}")
    print(f"  Residual-for-S2 intra-order variance:        {residual_intra_var:.4f}")
    print(f"  Stage 2 output intra-order variance:         {s2_intra_var:.4f}")
    if residual_intra_var > 1e-5:
        s2_capture = s2_intra_var / residual_intra_var
        print(f"  S2 variance capture ratio (S2var/resVar):   {s2_capture:.4f}  {'!! LOW' if s2_capture < 0.10 else 'OK'}")
    print()

    # Identify the biggest drop
    vr_layers = [
        ('Stage1', s1_vr),
        ('S1+Bias', s1b_vr),
        ('Final', final_vr),
    ]
    print("  Variance ratio progression: actual=1.0000", end="")
    for name, vr in vr_layers:
        print(f" -> {name}={vr:.4f}", end="")
    print()

    if s1_vr < 0.30:
        print("\n  [VERDICT] Stage 1 (Recipe GBDT) is already severely collapsing variance.")
        print("     Root cause likely: Stage 1 recipe features do not capture intra-order process variation.")
    elif s1b_vr < s1_vr * 0.70:
        print(f"\n  [VERDICT] Bias Shrinkage causes major variance loss (S1:{s1_vr:.4f} -> S1+Bias:{s1b_vr:.4f}).")
        print("     Root cause likely: Shrinkage is over-aggressive or bias granularity too coarse.")
    elif final_vr < s1b_vr * 0.70:
        print(f"\n  [VERDICT] Stage 2 causes major variance loss (S1+Bias:{s1b_vr:.4f} -> Final:{final_vr:.4f}).")
        print("     Root cause likely: Stage 2 GBDT too regularized, process-delta features too flat, or delta subtraction removes signal.")
    elif final_vr < 0.50:
        print(f"\n  [VERDICT] Gradual variance erosion across all stages. No single culprit.")
    else:
        print(f"\n  [OK] Variance ratio {final_vr:.4f} is within acceptable range. No collapse detected.")

    # ── 12. Top-10 process delta features by variance ────────────────────────
    print("\n  Top-10 process delta features by delta variance:")
    for _, r in delta_df.head(10).iterrows():
        ratio_str = f"{r['delta_vs_raw_ratio']:.4f}" if np.isfinite(r['delta_vs_raw_ratio']) else "N/A"
        print(f"    {r['feature']:<45} | delta_std={r['delta_std']:>8.4f} | raw_std={r['raw_std']:>8.4f} | ratio={ratio_str} | %zero={r['pct_zero_delta']:.1f}%")

    # ── 13. Top-10 compounds with largest bias shrinkage ─────────────────────
    compound_df_valid = compound_df[compound_df['bias_shrinkage_ratio'].notna() & (compound_df['n_test_rows'] >= 3)].copy()
    if not compound_df_valid.empty:
        print(f"\n  Top-10 compounds by bias shrinkage impact (n_test >= 3):")
        for _, r in compound_df_valid.head(10).iterrows():
            print(f"    {r['compound']:<30} | n={r['n_test_rows']:>3} | raw_bias={r['raw_bias']:>+7.3f} | shrunk={r['shrunk_bias']:>+7.3f} | actual_std={r['actual_std_within']:>6.3f} | pred_std={r['pred_std_within']:>6.3f}")

    print(f"\n  Reports saved to: {output_dir}")

    # ── 14. FORMAL PASS / FAIL GATES ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  VARIANCE AUDIT ACCEPTANCE GATES")
    print("=" * 80)

    # Retrieve per-subsystem variance ratios
    def _get_vr(df, col, val, layer):
        row = df[(df[col] == val) & (df['layer'] == layer)]
        if row.empty or 'intra_order_var_ratio' not in row.columns:
            return np.nan
        return row.iloc[0]['intra_order_var_ratio']

    cb_final_vr = _get_vr(matsys_df, 'material_system', 'CarbonBlack', '8_final_pred')
    si_final_vr = _get_vr(matsys_df, 'material_system', 'Silica', '8_final_pred')
    nod_final_vr = _get_vr(phase_df, 'phase_route', 'no_oil_dry', '8_final_pred')
    ow_final_vr = _get_vr(phase_df, 'phase_route', 'oil_wet', '8_final_pred')

    s2_capture = s2_intra_var / residual_intra_var if residual_intra_var > 1e-5 else 0.0

    # Risk B Fix: Robust G7 Compound Standard Deviation Ratio
    # Filter to valid compounds with actual_std > 0.2 and n >= 5 to prevent division-by-near-zero explosion
    valid_compounds = compound_df[(compound_df['actual_std_within'] > 0.2) & (compound_df['n_test_rows'] >= 5)].copy()
    if not valid_compounds.empty:
        raw_ratios = valid_compounds['pred_std_within'] / valid_compounds['actual_std_within']
        clipped_ratios = np.clip(raw_ratios, 0, 5.0)
        robust_median_std_ratio = float(np.median(clipped_ratios))
        weighted_std_ratio = float(np.average(clipped_ratios, weights=valid_compounds['n_test_rows']))
    else:
        robust_median_std_ratio = np.nan
        weighted_std_ratio = np.nan

    # Risk A Fix: Multi-tier S2 Capture Gate
    capture_status = (
        "Strong Pass (>= 35%)" if s2_capture >= 0.35 else
        ("Target Pass (>= 25%)" if s2_capture >= 0.25 else
         ("Minimum Pass (>= 10%)" if s2_capture >= 0.10 else "FAIL (< 10%)"))
    )

    gates = [
        ("G1 Overall Final VarRatio in [0.50, 2.00]",
         0.50 <= final_vr <= 2.00, f"{final_vr:.4f}"),
        ("G2 S2 Variance Capture Tier (Min >= 10%, Target >= 25%, Strong >= 35%)",
         s2_capture >= 0.10, f"{s2_capture*100:.2f}% [{capture_status}]"),
        ("G3 CarbonBlack Final VarRatio >= 0.20",
         np.isnan(cb_final_vr) or cb_final_vr >= 0.20, f"{cb_final_vr:.4f}" if np.isfinite(cb_final_vr) else "N/A"),
        ("G4 Silica Final VarRatio in [0.40, 2.50]",
         np.isnan(si_final_vr) or (0.40 <= si_final_vr <= 2.50), f"{si_final_vr:.4f}" if np.isfinite(si_final_vr) else "N/A"),
        ("G5 No-Oil/Dry Final VarRatio >= 0.20",
         np.isnan(nod_final_vr) or nod_final_vr >= 0.20, f"{nod_final_vr:.4f}" if np.isfinite(nod_final_vr) else "N/A"),
        ("G6 Oil/Wet Final VarRatio <= 3.00",
         np.isnan(ow_final_vr) or ow_final_vr <= 3.00, f"{ow_final_vr:.4f}" if np.isfinite(ow_final_vr) else "N/A"),
        ("G7 Robust Compound std ratio (valid n>=5, actual_std>0.2) in [0.30, 1.50]",
         np.isnan(robust_median_std_ratio) or (0.30 <= robust_median_std_ratio <= 1.50),
         f"median={robust_median_std_ratio:.4f}, weighted={weighted_std_ratio:.4f}" if np.isfinite(robust_median_std_ratio) else "N/A"),
    ]

    all_pass = True
    for name, passed, value in gates:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}  (value={value})")

    print("-" * 80)
    if all_pass:
        print("  >>> ALL GATES PASSED <<<")
    else:
        n_fail = sum(1 for _, p, _ in gates if not p)
        print(f"  >>> {n_fail} GATE(S) FAILED — further tuning required <<<")
    print("=" * 80)

    return {
        'audit_df': audit_df,
        'phase_df': phase_df,
        'matsys_df': matsys_df,
        'compound_df': compound_df,
        'delta_df': delta_df,
        'all_gates_passed': all_pass,
    }


if __name__ == '__main__':
    run_variance_audit()
