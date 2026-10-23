"""
Scheme 3 Execution Script:
catch22 (catch24) Feature Extraction + FOCI Conditional Correlation Selection.
Strictly following: d:\\AntigravityIDE\\Project\\方案3-catch22与FOCI.md
"""
import sys
import os
import json
import importlib.metadata
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import pycatch22
import xicorpy

def run_scheme3():
    npz_path = r"d:\AntigravityIDE\Project\data\scheme1_aligned_input.npz"
    data = np.load(npz_path)
    
    n = data['mooney'].shape[0]
    T = 390
    try:
        catch22_ver = importlib.metadata.version('pycatch22')
    except Exception:
        catch22_ver = getattr(pycatch22, '__version__', 'unknown')
        
    print(f"Data loaded: n={n}, T={T}, pycatch22 version: {catch22_ver}")
    
    channels = [
        ('温度', data['temperature']),
        ('功率', data['power']),
        ('压栓', data['ram']),
        ('转速', data['speed'])
    ]
    
    # 1. Extract catch24 features for each batch and each channel
    print("Extracting catch24 features for all channels...")
    all_features = {}
    
    for ch_name, ch_arr in channels:
        print(f"Extracting for channel: {ch_name}...")
        # ch_arr shape: (n, T)
        first_res = pycatch22.catch22_all(ch_arr[0].tolist(), catch24=True, short_names=True, mask_nans=False)
        feature_names = [f"{ch_name}__{s}" for s in first_res['short_names']]
        
        ch_feats = np.zeros((n, len(feature_names)), dtype=np.float64)
        for i in range(n):
            res = pycatch22.catch22_all(ch_arr[i].tolist(), catch24=True, short_names=True, mask_nans=False)
            ch_feats[i, :] = res['values']
            
        for f_idx, f_name in enumerate(feature_names):
            all_features[f_name] = ch_feats[:, f_idx]
            
    # Add aging hours
    all_features['停留时间'] = data['aging_hours'].flatten().astype(np.float64)
    y = data['mooney'].flatten().astype(np.float64)
    
    df_raw = pd.DataFrame(all_features)
    print(f"Initial raw feature matrix shape: {df_raw.shape}")
    
    # Clean step per spec:
    # 1. Drop any column containing NaN in any row
    # 2. Drop any column with constant values across all rows
    valid_cols = []
    for col in df_raw.columns:
        vals = df_raw[col].values
        if np.any(np.isnan(vals)):
            print(f"Dropping {col}: contains NaN")
            continue
        if np.all(vals == vals[0]) or np.std(vals) == 0:
            print(f"Dropping {col}: all values constant")
            continue
        valid_cols.append(col)
        
    df_clean = df_raw[valid_cols].copy()
    print(f"Cleaned feature matrix shape: {df_clean.shape} (retained {len(valid_cols)} columns)")
    
    # Standardize on training set
    df_norm = (df_clean - df_clean.mean(axis=0)) / df_clean.std(axis=0)
    
    # 2. FOCI Forward Selection
    print("\nStarting FOCI forward selection...")
    selected_cols = []
    selected_records = []
    candidates = list(valid_cols)
    round_idx = 1
    
    while len(candidates) > 0:
        best_col = None
        best_val = -1e9
        
        given = df_norm[selected_cols].to_numpy() if len(selected_cols) > 0 else None
        
        for cand in candidates:
            z = df_norm[cand].to_numpy()
            val = xicorpy.compute_conditional_dependence(y, z, given)
            if val > best_val:
                best_val = val
                best_col = cand
                
        print(f"Round {round_idx}: Best candidate is '{best_col}' with cond_dep = {best_val:.6f}")
        
        # Stopping condition: coefficient <= 0
        if best_val <= 0:
            print(f"Stopping condition reached (best_val = {best_val:.6f} <= 0). FOCI terminated.")
            break
            
        selected_cols.append(best_col)
        candidates.remove(best_col)
        selected_records.append({
            'name': best_col,
            'order': round_idx,
            'cond_dep': float(best_val),
            'pycatch22_version': catch22_ver
        })
        round_idx += 1
        
    print(f"\nFOCI Selection completed. Total selected: {len(selected_records)}")
    for r in selected_records:
        print(f"Rank {r['order']}: {r['name']} | Cond Dep: {r['cond_dep']:.6f}")
        
    out_dir = r"C:\Users\uih83490\.gemini\antigravity-ide\brain\0f968684-3a6b-429f-82f1-62de38081140"
    out_file = os.path.join(out_dir, "scheme3_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "selected": selected_records,
            "pycatch22_version": catch22_ver,
            "total_candidates_evaluated": len(valid_cols)
        }, f, ensure_ascii=False, indent=2)
        
    print(f"Scheme 3 results saved to {out_file}")

if __name__ == '__main__':
    run_scheme3()
