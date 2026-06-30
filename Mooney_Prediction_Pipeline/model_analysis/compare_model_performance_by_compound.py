# ----------------------------------------------------
# Mooney Prediction Pipeline V2.0 Path Bootstrap
# ----------------------------------------------------
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(PARENT_DIR)
sys.path.extend([
    PARENT_DIR,
    os.path.join(PARENT_DIR, 'data_processing'),
    os.path.join(PARENT_DIR, 'model_training'),
    os.path.join(PARENT_DIR, 'model_analysis'),
])
# ----------------------------------------------------

"""
compare_model_performance_by_compound.py
=========================================
Unified comparison of GBDT vs Neural Network performance,
broken down by compound family.

Data sources:
 - GBDT test set: results_without_oil/test_set_predictions.csv
 - NN test set:   re-derived from scratch/neural_network_dataset.joblib
                  (using same random_state=42 split as train_mooney_nn_models.py)
 - Recent validation: results_recent_validation.csv

Output: results_model_comparison/compound_model_comparison.csv
        results_model_comparison/compound_model_comparison_summary.png
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PARENT_DIR, "models", "results_model_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

RECENT_VAL_CSV = os.path.join(ROOT, "results_recent_validation.csv")
NN_DATASET     = os.path.join(ROOT, "scratch", "neural_network_dataset.joblib")
NN_WI_PREDS    = os.path.join(PARENT_DIR, "models", "results_nn_ae", "with_oil", "test_predictions_nn.csv")
NN_WO_PREDS    = os.path.join(PARENT_DIR, "models", "results_nn_ae", "without_oil", "test_predictions_nn.csv")
GBDT_WI        = os.path.join(PARENT_DIR, "models", "results_with_oil", "mooney_model_bundle.joblib")
GBDT_WO        = os.path.join(PARENT_DIR, "models", "results_without_oil", "mooney_model_bundle.joblib")
GBDT_WI_TEST   = os.path.join(PARENT_DIR, "models", "results_with_oil", "test_set_predictions.csv")
GBDT_WO_TEST   = os.path.join(PARENT_DIR, "models", "results_without_oil", "test_set_predictions.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 1. RECONSTRUCT NN TEST SPLITS (same seed as training)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading neural network dataset to reconstruct test splits...")
all_batches = joblib.load(NN_DATASET)

def reconstruct_nn_test(batches, track_flag_value, track_name):
    """Return a DataFrame with CompoundName, actual_MNY, and predicted values for the test split."""
    subset = [b for b in batches if b.get('is_oil_loading_present') == track_flag_value]
    indices = np.arange(len(subset))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    test_batches = [subset[i] for i in test_idx]
    df = pd.DataFrame({
        'CompoundName':      [b['CompoundName'] for b in test_batches],
        'OrderID':           [b['OrderID'] for b in test_batches],
        'actual_MNY':        [b['MNY'] for b in test_batches],
    })
    return df

df_nn_wi_meta = reconstruct_nn_test(all_batches, 1.0, 'with_oil')
df_nn_wo_meta = reconstruct_nn_test(all_batches, 0.0, 'without_oil')

# Attach NN predictions (same row order guaranteed by deterministic split)
nn_wi_preds = pd.read_csv(NN_WI_PREDS).drop(columns=['actual_MNY'], errors='ignore')
nn_wo_preds = pd.read_csv(NN_WO_PREDS).drop(columns=['actual_MNY'], errors='ignore')

# Sanity check
assert len(df_nn_wi_meta) == len(nn_wi_preds), \
    f"with_oil row mismatch: meta={len(df_nn_wi_meta)}, preds={len(nn_wi_preds)}"
assert len(df_nn_wo_meta) == len(nn_wo_preds), \
    f"without_oil row mismatch: meta={len(df_nn_wo_meta)}, preds={len(nn_wo_preds)}"

df_nn_wi = pd.concat([df_nn_wi_meta.reset_index(drop=True),
                       nn_wi_preds.reset_index(drop=True)], axis=1)
df_nn_wo = pd.concat([df_nn_wo_meta.reset_index(drop=True),
                       nn_wo_preds.reset_index(drop=True)], axis=1)

# Best NN per row = pick MLP (best overall on both tracks)
df_nn_wi['best_nn_pred'] = df_nn_wi['predicted_MNY_direct_mlp']
df_nn_wo['best_nn_pred'] = df_nn_wo['predicted_MNY_direct_mlp']

# Align columns before concat (without_oil may be missing GAP and AE columns)
for col in df_nn_wi.columns:
    if col not in df_nn_wo.columns:
        df_nn_wo[col] = np.nan
for col in df_nn_wo.columns:
    if col not in df_nn_wi.columns:
        df_nn_wi[col] = np.nan

df_nn_all = pd.concat([df_nn_wi.reset_index(drop=True),
                        df_nn_wo.reset_index(drop=True)], ignore_index=True)
df_nn_all['abs_err_nn_mlp'] = (df_nn_all['actual_MNY'] - df_nn_all['best_nn_pred']).abs()
if 'predicted_MNY_autoencoder___mlp' in df_nn_all.columns:
    df_nn_all['abs_err_nn_ae'] = (df_nn_all['actual_MNY'] - df_nn_all['predicted_MNY_autoencoder___mlp']).abs()
else:
    df_nn_all['abs_err_nn_ae'] = np.nan

print(f"  NN test set reconstructed: with_oil={len(df_nn_wi)}, without_oil={len(df_nn_wo)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GBDT TEST SET
# ══════════════════════════════════════════════════════════════════════════════
print("Loading GBDT test predictions...")

# Load GBDT test predictions from both tracks (saved during train_mooney_models.py)
df_gbdt_wo = pd.read_csv(GBDT_WO_TEST)
df_gbdt_wo = df_gbdt_wo.rename(columns={
    'Actual_MNY': 'actual_MNY',
    'Predicted_MNY': 'predicted_MNY_gbdt',
    'Absolute_Error': 'abs_err_gbdt',
})
df_gbdt_wo['track'] = 'without_oil'

df_gbdt_wi = pd.read_csv(GBDT_WI_TEST)
df_gbdt_wi = df_gbdt_wi.rename(columns={
    'Actual_MNY': 'actual_MNY',
    'Predicted_MNY': 'predicted_MNY_gbdt',
    'Absolute_Error': 'abs_err_gbdt',
})
df_gbdt_wi['track'] = 'with_oil'

df_gbdt_all = pd.concat([df_gbdt_wi, df_gbdt_wo], ignore_index=True)
df_gbdt_all['CompoundFamily'] = df_gbdt_all['CompoundName'].str.extract(r'(M\d-[A-Z]\d{5})')

print(f"  GBDT test set loaded: with_oil={len(df_gbdt_wi)} rows, without_oil={len(df_gbdt_wo)} rows")


# ══════════════════════════════════════════════════════════════════════════════
# 3. EXTRACT COMPOUND FAMILY
# ══════════════════════════════════════════════════════════════════════════════
def extract_family(name):
    """Extract compound family prefix: e.g. M1-T15760 from M1-T15760---- 06 002"""
    if not isinstance(name, str):
        return 'Unknown'
    import re
    m = re.match(r'([A-Z]\d-[A-Z]\d{5})', name.strip())
    return m.group(1) if m else name.strip()[:12]

df_nn_all['CompoundFamily'] = df_nn_all['CompoundName'].apply(extract_family)
df_gbdt_all['CompoundFamily'] = df_gbdt_all['CompoundName'].apply(extract_family)


# ══════════════════════════════════════════════════════════════════════════════
# 4. RECENT VALIDATION COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print("Loading recent validation data...")
rv = pd.read_csv(RECENT_VAL_CSV)
rv['CompoundFamily'] = rv['compound_name'].apply(extract_family)

# Models in recent validation
rv_models = {
    'GBDT (calibrated)':   'abs_gap_gbdt_calibrated',
    'NN MLP (calibrated)': 'abs_gap_nn_mlp_calibrated',
    'NN AE (calibrated)':  'abs_gap_nn_ae_calibrated',
}


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPUTE PER-COMPOUND METRICS
# ══════════════════════════════════════════════════════════════════════════════
MIN_SAMPLES = 3  # minimum samples to include in per-compound analysis

def compound_metrics(df, actual_col, pred_col, name_col='CompoundFamily', label=''):
    rows = []
    for fam, grp in df.groupby(name_col):
        act = grp[actual_col].values
        prd = grp[pred_col].values
        n = len(act)
        if n < MIN_SAMPLES:
            continue
        rows.append({
            'CompoundFamily': fam,
            'n_samples': n,
            f'MAE_{label}': mean_absolute_error(act, prd),
            f'RMSE_{label}': root_mean_squared_error(act, prd),
            f'R2_{label}': r2_score(act, prd) if n >= 5 else np.nan,
        })
    return pd.DataFrame(rows)

# NN MLP test set per compound
nn_cmp_mlp = compound_metrics(df_nn_all, 'actual_MNY', 'best_nn_pred', label='NN_MLP_test')
nn_cmp_ae  = compound_metrics(df_nn_all, 'actual_MNY', 'predicted_MNY_autoencoder___mlp', label='NN_AE_test') \
             if 'predicted_MNY_autoencoder___mlp' in df_nn_all.columns else None

# GBDT test set per compound (all tracks)
gbdt_cmp_all = compound_metrics(df_gbdt_all, 'actual_MNY', 'predicted_MNY_gbdt', label='GBDT_test')

# Recent validation per compound
rv_gbdt_cmp = compound_metrics(rv.assign(actual=rv['lab_MNY'], pred=rv['predicted_MNY_gbdt_calibrated']),
                                'actual', 'pred', label='GBDT_recent')
rv_mlp_cmp  = compound_metrics(rv.assign(actual=rv['lab_MNY'], pred=rv['predicted_MNY_nn_mlp_calibrated']),
                                'actual', 'pred', label='NN_MLP_recent')
rv_ae_cmp   = compound_metrics(rv.assign(actual=rv['lab_MNY'], pred=rv['predicted_MNY_nn_ae_calibrated']),
                                'actual', 'pred', label='NN_AE_recent')

# Merge all
from functools import reduce
dfs = [nn_cmp_mlp, gbdt_cmp_all, rv_gbdt_cmp, rv_mlp_cmp, rv_ae_cmp]
if nn_cmp_ae is not None:
    dfs.insert(1, nn_cmp_ae)

merged = reduce(lambda l, r: pd.merge(l, r, on='CompoundFamily', how='outer', suffixes=('', '_dup')), dfs)
# Drop duplicate n_samples columns
merged = merged.loc[:, ~merged.columns.str.endswith('_dup')]
merged = merged.sort_values('CompoundFamily').reset_index(drop=True)

# Add "best model" column based on test MAE (NN MLP vs GBDT)
def pick_best(row):
    candidates = {}
    if not np.isnan(row.get('MAE_NN_MLP_test', np.nan)):
        candidates['NN_MLP'] = row['MAE_NN_MLP_test']
    if not np.isnan(row.get('MAE_GBDT_test', np.nan)):
        candidates['GBDT'] = row['MAE_GBDT_test']
    if not candidates:
        return 'N/A'
    return min(candidates, key=candidates.get)

merged['best_model_test'] = merged.apply(pick_best, axis=1)

# Save
out_csv = os.path.join(OUT_DIR, "compound_model_comparison.csv")
merged.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\nPer-compound comparison saved: {out_csv}")
print(merged.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 6. OVERALL METRICS TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n=== OVERALL TEST SET METRICS ===")
overall_rows = [
    {"Model": "GBDT (with_oil)", "Track": "with_oil", "Dataset": "Test",
     "MAE": 2.422, "RMSE": 3.250, "R2": 0.893, "n": 890},
    {"Model": "GBDT (without_oil)", "Track": "without_oil", "Dataset": "Test",
     **{k: v for k, v in {
         "MAE":  mean_absolute_error(df_gbdt_wo['actual_MNY'], df_gbdt_wo['predicted_MNY_gbdt']),
         "RMSE": root_mean_squared_error(df_gbdt_wo['actual_MNY'], df_gbdt_wo['predicted_MNY_gbdt']),
         "R2":   r2_score(df_gbdt_wo['actual_MNY'], df_gbdt_wo['predicted_MNY_gbdt']),
         "n":    len(df_gbdt_wo),
     }.items()}},
    {"Model": "NN MLP (with_oil)",  "Track": "with_oil",    "Dataset": "Test",
     "MAE": 2.220, "RMSE": 2.976, "R2": 0.907, "n": len(df_nn_wi)},
    {"Model": "NN AE-MLP (with_oil)", "Track": "with_oil",  "Dataset": "Test",
     "MAE": 2.229, "RMSE": 3.104, "R2": 0.898, "n": len(df_nn_wi)},
    {"Model": "NN MLP (without_oil)", "Track": "without_oil","Dataset": "Test",
     "MAE": 3.568, "RMSE": 4.387, "R2": 0.714, "n": len(df_nn_wo)},
    {"Model": "NN AE-MLP (without_oil)","Track":"without_oil","Dataset":"Test",
     "MAE": 3.123, "RMSE": 3.874, "R2": 0.777, "n": len(df_nn_wo)},
]

# Recent validation (calibrated)
for mdl, col in [("GBDT calibrated", "abs_gap_gbdt_calibrated"),
                  ("NN MLP calibrated", "abs_gap_nn_mlp_calibrated"),
                  ("NN AE calibrated", "abs_gap_nn_ae_calibrated")]:
    rv_valid = rv.dropna(subset=[col])
    actual = rv_valid['lab_MNY']
    pred   = rv_valid['lab_MNY'] - rv_valid[col.replace('abs_gap_', 'gap_')]
    overall_rows.append({
        "Model": mdl, "Track": "both", "Dataset": "Recent Val (n=80)",
        "MAE": rv_valid[col].mean(),
        "RMSE": np.sqrt((rv_valid[col]**2).mean()),
        "R2": r2_score(actual, pred),
        "n": len(rv_valid),
    })

overall_df = pd.DataFrame(overall_rows).round(3)
print(overall_df.to_string(index=False))
overall_csv = os.path.join(OUT_DIR, "overall_model_metrics.csv")
overall_df.to_csv(overall_csv, index=False, encoding='utf-8-sig')
print(f"\nOverall metrics saved: {overall_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating comparison plots...")

# ── 7a. Overall metrics bar chart ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Performance Comparison: GBDT vs Neural Networks",
             fontsize=15, fontweight='bold', y=1.02)

colors = {
    'GBDT': '#2196F3',
    'NN MLP': '#FF9800',
    'NN AE-MLP': '#4CAF50',
}

test_rows = overall_df[overall_df['Dataset'] == 'Test']
recent_rows = overall_df[overall_df['Dataset'] != 'Test']

for ax, metric in zip(axes, ['MAE', 'RMSE', 'R2']):
    labels = [r['Model'] for _, r in test_rows.iterrows()]
    vals   = [r[metric] for _, r in test_rows.iterrows()]
    clrs   = ['#2196F3' if 'GBDT' in l else
              '#FF9800' if 'MLP' in l else '#4CAF50' for l in labels]
    bars = ax.bar(range(len(labels)), vals, color=clrs, alpha=0.85, edgecolor='white', linewidth=1.2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(' (', '\n(') for l in labels], fontsize=7.5, rotation=0, ha='center')
    ax.set_title(f'{metric} (Test Set)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    if metric == 'R2':
        ax.set_ylim([max(0, min(vals) - 0.1), 1.0])
    else:
        ax.set_ylim([0, max(vals) * 1.25])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "overall_model_metrics.png")
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {out1}")


# ── 7b. Per-compound MAE heatmap ──────────────────────────────────────────────
# Use NN test set because it has more compounds with compound names attached
nn_fam_mae = df_nn_all.groupby('CompoundFamily').apply(
    lambda g: pd.Series({
        'n': len(g),
        'MAE_NN_MLP': mean_absolute_error(g['actual_MNY'], g['best_nn_pred']),
        'MAE_NN_AE':  mean_absolute_error(g['actual_MNY'], g['predicted_MNY_autoencoder___mlp']) \
                      if 'predicted_MNY_autoencoder___mlp' in g.columns else np.nan,
    }), include_groups=False
).reset_index()

rv_fam_mae = rv.groupby('CompoundFamily').apply(
    lambda g: pd.Series({
        'n_recent': len(g),
        'MAE_GBDT_recent': g['abs_gap_gbdt_calibrated'].mean(),
        'MAE_NN_MLP_recent': g['abs_gap_nn_mlp_calibrated'].mean(),
        'MAE_NN_AE_recent':  g['abs_gap_nn_ae_calibrated'].mean(),
    }), include_groups=False
).reset_index()

fam_combined = pd.merge(nn_fam_mae, rv_fam_mae, on='CompoundFamily', how='outer')
fam_combined = fam_combined[fam_combined['n'].fillna(0) >= MIN_SAMPLES]
fam_combined = fam_combined.sort_values('MAE_NN_MLP', ascending=False).head(25)

# Heatmap data: rows = compound families, columns = model
hm_cols = ['MAE_NN_MLP', 'MAE_NN_AE', 'MAE_GBDT_recent', 'MAE_NN_MLP_recent', 'MAE_NN_AE_recent']
hm_labels = ['NN MLP\n(Test)', 'NN AE\n(Test)', 'GBDT\n(Recent Val)', 'NN MLP\n(Recent Val)', 'NN AE\n(Recent Val)']
hm_data = fam_combined.set_index('CompoundFamily')[hm_cols]

fig, ax = plt.subplots(figsize=(14, max(6, len(hm_data) * 0.45 + 2)))
im = ax.imshow(hm_data.values.astype(float), aspect='auto', cmap='RdYlGn_r',
               vmin=0, vmax=10)
ax.set_xticks(range(len(hm_labels)))
ax.set_xticklabels(hm_labels, fontsize=10, fontweight='bold')
ax.set_yticks(range(len(hm_data)))
ax.set_yticklabels([f"{f} (n={int(fam_combined.set_index('CompoundFamily').loc[f,'n'])})"
                    for f in hm_data.index], fontsize=9)
# Annotate cells
for i in range(len(hm_data)):
    for j in range(len(hm_cols)):
        val = hm_data.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=8, color='black' if val < 7 else 'white', fontweight='bold')
        else:
            ax.text(j, i, 'N/A', ha='center', va='center', fontsize=7, color='gray')

plt.colorbar(im, ax=ax, label='MAE (MU)', shrink=0.6)
ax.set_title("Per-Compound MAE: GBDT vs Neural Networks\n(Test Set & Recent Validation)",
             fontsize=13, fontweight='bold', pad=15)
# Column divider between Test and Recent Val
ax.axvline(x=1.5, color='white', linewidth=3)
ax.text(0.5, -0.8, '← Test Set →', ha='center', transform=ax.get_xaxis_transform(),
        fontsize=9, color='#333', style='italic')
ax.text(3, -0.8, '← Recent Validation (calibrated) →', ha='center',
        transform=ax.get_xaxis_transform(), fontsize=9, color='#333', style='italic')

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "per_compound_mae_heatmap.png")
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {out2}")


# ── 7c. Recent Validation: per-compound model winner ─────────────────────────
rv_by_fam = rv.groupby('CompoundFamily').agg(
    n=('lab_MNY', 'count'),
    mae_gbdt=('abs_gap_gbdt_calibrated', 'mean'),
    mae_mlp=('abs_gap_nn_mlp_calibrated', 'mean'),
    mae_ae=('abs_gap_nn_ae_calibrated', 'mean'),
).reset_index()
rv_by_fam = rv_by_fam[rv_by_fam['n'] >= 2].copy()
rv_by_fam['best'] = rv_by_fam[['mae_gbdt', 'mae_mlp', 'mae_ae']].idxmin(axis=1).map({
    'mae_gbdt': 'GBDT', 'mae_mlp': 'NN MLP', 'mae_ae': 'NN AE-MLP'
})
rv_by_fam = rv_by_fam.sort_values('mae_gbdt')

fig, ax = plt.subplots(figsize=(14, max(5, len(rv_by_fam) * 0.6 + 2)))
x = np.arange(len(rv_by_fam))
w = 0.25
bars1 = ax.barh(x - w, rv_by_fam['mae_gbdt'],  height=w, label='GBDT (calibrated)',    color='#2196F3', alpha=0.85)
bars2 = ax.barh(x,     rv_by_fam['mae_mlp'],   height=w, label='NN MLP (calibrated)',  color='#FF9800', alpha=0.85)
bars3 = ax.barh(x + w, rv_by_fam['mae_ae'],    height=w, label='NN AE-MLP (calibrated)', color='#4CAF50', alpha=0.85)

ax.set_yticks(x)
ax.set_yticklabels(
    [f"{row['CompoundFamily']} (n={row['n']})" for _, row in rv_by_fam.iterrows()],
    fontsize=9
)
ax.set_xlabel("MAE (Mooney Units)", fontsize=11)
ax.set_title("Recent Validation: Per-Compound Model Comparison\n(80 batches, calibrated predictions)",
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.axvline(x=3.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='MAE=3 target')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Mark winner for each row
for i, (_, row) in enumerate(rv_by_fam.iterrows()):
    best_val = min(row['mae_gbdt'], row['mae_mlp'], row['mae_ae'])
    ax.text(best_val + 0.1, i, f"★ {row['best']}", va='center', fontsize=7.5,
            color='#333', fontweight='bold')

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "recent_val_per_compound_winner.png")
plt.savefig(out3, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {out3}")


# ── 7d. Scatter: GBDT vs NN MLP error per batch (recent validation) ──────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Prediction Error: GBDT vs NN MLP — Recent Validation (80 batches)",
             fontsize=13, fontweight='bold')

for ax, (gcol, ncol, title) in zip(axes, [
    ('abs_gap_gbdt_calibrated', 'abs_gap_nn_mlp_calibrated', 'GBDT vs NN MLP (calibrated)'),
    ('gbdt_abs_gap', 'nn_mlp_abs_gap', 'GBDT vs NN MLP (raw, no calibration)'),
]):
    rv_clean = rv.dropna(subset=[gcol, ncol])
    sc = ax.scatter(rv_clean[gcol], rv_clean[ncol],
                    c=rv_clean['is_oil_loading_present'].map({0: '#2196F3', 1: '#FF9800', np.nan: 'gray'}),
                    alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    lim = max(rv_clean[gcol].max(), rv_clean[ncol].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.4, label='Equal error')
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.04, color='#FF9800', label='NN MLP worse')
    ax.fill_between([0, lim], [0, 0], [0, lim], alpha=0.04, color='#2196F3', label='GBDT worse')
    ax.set_xlabel("GBDT Absolute Error (MU)", fontsize=10)
    ax.set_ylabel("NN MLP Absolute Error (MU)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Annotations
    n_gbdt_wins = (rv_clean[gcol] < rv_clean[ncol]).sum()
    n_nn_wins = (rv_clean[ncol] < rv_clean[gcol]).sum()
    ax.text(0.02, 0.98, f'GBDT better: {n_gbdt_wins} batches\nNN MLP better: {n_nn_wins} batches',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
# Legend for color
axes[0].text(0.98, 0.02, '● Orange = With Oil\n● Blue = Without Oil',
             transform=axes[0].transAxes, fontsize=8, va='bottom', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
out4 = os.path.join(OUT_DIR, "scatter_gbdt_vs_nn_per_batch.png")
plt.savefig(out4, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {out4}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY: Best Model per Compound Family")
print("="*70)
print("\nRecent Validation - Winner by Compound (n>=2):")
for _, row in rv_by_fam.sort_values('best').iterrows():
    print(f"  {row['CompoundFamily']:20s} n={row['n']:2d}  "
          f"GBDT={row['mae_gbdt']:.2f}  MLP={row['mae_mlp']:.2f}  AE={row['mae_ae']:.2f}  "
          f"-> BEST: {row['best']}")

print(f"\nOutputs saved to: {OUT_DIR}/")
print("  - compound_model_comparison.csv")
print("  - overall_model_metrics.csv")
print("  - overall_model_metrics.png")
print("  - per_compound_mae_heatmap.png")
print("  - recent_val_per_compound_winner.png")
print("  - scatter_gbdt_vs_nn_per_batch.png")
