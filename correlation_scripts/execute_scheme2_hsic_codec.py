"""
Scheme 2 Execution Script:
Phase point HSIC Lasso dedup + Conditional Dependence Calculation.
Strictly following: d:\\AntigravityIDE\\Project\\方案2-相位点HSIC与CODEC.md
"""
import sys
import os
import json
import numpy as np
import numpy.linalg

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Fix NumPy 2.x compatibility for pyHSICLasso
numpy.linalg.linalg = numpy.linalg
import pandas as pd
from pyHSICLasso import HSICLasso
import xicorpy

def run_scheme2():
    npz_path = r"d:\AntigravityIDE\Project\data\scheme1_aligned_input.npz"
    data = np.load(npz_path)
    
    n = data['mooney'].shape[0]
    T = 390
    print(f"Data loaded: n={n}, T={T}")
    
    # 1. Prepare X matrix and column names
    cols = []
    X_list = []
    ch_map = [('温度', 'temperature'), ('功率', 'power'), ('压栓', 'ram'), ('转速', 'speed')]
    for prefix, ch in ch_map:
        arr = data[ch]
        for t in range(T):
            cols.append(f"{prefix}_t{t:03d}")
        X_list.append(arr)
        
    cols.append('停留时间')
    X_list.append(data['aging_hours'])
    
    X_raw = np.hstack(X_list).astype(np.float64)
    y = data['mooney'].astype(np.float64) # (n,) 1D float
    
    # Standardize X using training set (n rows)
    mean = np.mean(X_raw, axis=0)
    std = np.std(X_raw, axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    X_norm = (X_raw - mean) / std_safe
    
    df_X = pd.DataFrame(X_norm, columns=cols)
    
    # Determine B: largest divisor of n <= 20
    divisors = [i for i in range(1, 21) if n % i == 0]
    B = max(divisors)
    print(f"B chosen: {B} (divisors of {n} <= 20: {divisors})")
    
    # Step 1: HSIC Lasso
    print("Running HSIC Lasso feature selection...")
    model = HSICLasso()
    model.input(X_norm, y, featname=cols)
    model.regression(num_feat=X_norm.shape[1], B=B, M=3, discrete_x=False)
    kept = model.get_features()
    print(f"HSIC Lasso completed. Total kept features: {len(kept)}")
    
    # Step 2: Calculate conditional dependence for each kept feature
    print("Calculating conditional dependence for kept features...")
    results = []
    
    for idx, name in enumerate(kept):
        others = [c for c in kept if c != name]
        if len(others) == 0:
            given = None
        else:
            given = df_X[others].to_numpy()
            
        z = df_X[name].to_numpy()
        val = xicorpy.compute_conditional_dependence(y, z, given)
        
        results.append({
            'name': name,
            'hsic_order': idx + 1,
            'cond_dep': float(val)
        })
        print(f"[{idx+1}/{len(kept)}] {name}: cond_dep = {val:.6f}")
        
    # Filter: columns with cond_dep <= 0 are removed from final table per spec
    final_kept = [r for r in results if r['cond_dep'] > 0]
    print(f"\nFinal retained features (cond_dep > 0): {len(final_kept)} out of {len(kept)}")
    
    # Sort by cond_dep descending or hsic_order
    out_dir = r"C:\Users\uih83490\.gemini\antigravity-ide\brain\0f968684-3a6b-429f-82f1-62de38081140"
    out_file = os.path.join(out_dir, "scheme2_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "all_kept_hsic": results,
            "final_retained": final_kept
        }, f, ensure_ascii=False, indent=2)
        
    print(f"Results saved to {out_file}")

if __name__ == '__main__':
    run_scheme2()
