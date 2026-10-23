"""
New Scheme 1: Variable-length Path Signature + CODEC (Conditional Dependence)
Strictly following: d:\\AntigravityIDE\\Project\\方案1-变长路径签名与CODEC.md
"""
import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
import xicorpy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def compute_path_signature_order2(path):
    '''
    Exact path signature up to order m=2 for piecewise linear path X in R^(T x d).
    Returns flat vector of length d + d^2 matching iisignature.sig(path, 2).
    Order of elements:
    - First d elements: Level 1 increments: X_T - X_0
    - Next d^2 elements: Level 2 iterated integrals in row-major order:
      S^{(2)}_{i, j} = int_0^T (X_i(t) - X_i(0)) dX_j(t)
    '''
    T, d = path.shape
    deltas = np.diff(path, axis=0)
    level1 = path[-1] - path[0]
    displacements = path[:-1] - path[0]
    level2 = np.einsum('ki,kj->ij', displacements, deltas) + 0.5 * np.einsum('ki,kj->ij', deltas, deltas)
    return np.concatenate([level1, level2.flatten()])

def run():
    pkl_path = r"C:\Users\uih83490\.gemini\antigravity-ide\brain\0f968684-3a6b-429f-82f1-62de38081140\scratch\variable_length_dataset.pkl"
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
        
    n = len(records)
    print(f"Loaded {n} variable-length batch records.")
    
    # Check for NaN / Inf
    for idx, r in enumerate(records):
        for k in ['temp', 'power', 'ram', 'speed']:
            if np.isnan(r[k]).any() or np.isinf(r[k]).any():
                raise ValueError(f"Batch {idx} contains NaN/Inf in channel {k}. Execution stopped.")
                
    d = 5 # temp, power, ram, speed, time_idx
    m = 2
    sig_len = d + d**2 # 30
    print(f"Path dimension d={d} (温度, 功率, 压栓, 转速, 采样序号), truncation order m={m}, signature length={sig_len}")
    
    sig_matrix = np.zeros((n, sig_len), dtype=np.float64)
    aging_arr = np.zeros(n, dtype=np.float64)
    mooney_arr = np.zeros(n, dtype=np.float64)
    
    for i, r in enumerate(records):
        T_i = r['T_i']
        time_idx = np.arange(T_i, dtype=np.float32) # 0 to T_i - 1
        path = np.column_stack([r['temp'], r['power'], r['ram'], r['speed'], time_idx])
        sig_matrix[i, :] = compute_path_signature_order2(path)
        aging_arr[i] = r['AgingHours']
        mooney_arr[i] = r['FM_Mooney']
        
    col_names = [f"签名_{j:03d}" for j in range(sig_len)]
    df = pd.DataFrame(sig_matrix, columns=col_names)
    df['停留时间'] = aging_arr
    y = mooney_arr
    
    # Drop any column that is constant across all batches
    valid_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1 or np.std(df[col]) == 0:
            print(f"Dropping constant column: {col}")
        else:
            valid_cols.append(col)
            
    df_valid = df[valid_cols].copy()
    print(f"Valid feature columns: {len(valid_cols)}")
    
    # Standardize
    df_norm = (df_valid - df_valid.mean()) / df_valid.std()
    
    # FOCI forward selection
    selected_cols = []
    selected_records = []
    candidates = list(valid_cols)
    round_idx = 1
    
    print("\nStarting FOCI forward selection on signatures...")
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
        
        if best_val <= 0:
            print(f"Stopping condition reached (best_val = {best_val:.6f} <= 0). FOCI terminated.")
            break
            
        selected_cols.append(best_col)
        candidates.remove(best_col)
        selected_records.append({
            'name': best_col,
            'order': round_idx,
            'cond_dep': float(best_val),
            'd': d,
            'm': m
        })
        round_idx += 1
        
    out_dir = r"C:\Users\uih83490\.gemini\antigravity-ide\brain\0f968684-3a6b-429f-82f1-62de38081140"
    out_file = os.path.join(out_dir, "new_scheme1_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            'selected': selected_records,
            'd': d,
            'm': m,
            'total_candidates': len(valid_cols)
        }, f, ensure_ascii=False, indent=2)
        
    print(f"\nScheme 1 completed. Saved to {out_file}")

if __name__ == '__main__':
    run()
