"""
New Scheme 2: Variable-length GAK + HSIC
Strictly following: d:\\AntigravityIDE\\Project\\方案2-变长GAK与HSIC.md
"""
import sys
import os
import json
import pickle
import time
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.metrics import cdist_gak, sigma_gak
import xicorpy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def compute_hsic_and_pvalue(K, y, reps=1000, random_state=0):
    n = len(y)
    # Check median for gamma
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append((y[i] - y[j]) ** 2)
    diffs = np.array(diffs)
    med = np.median(diffs)
    if med == 0:
        raise ValueError("Median of squared differences is 0. Execution stopped.")
    gamma = 1.0 / med
    
    # Target kernel L
    y_col = y[:, None]
    L = np.exp(-gamma * (y_col - y_col.T) ** 2)
    
    # Center K and L
    H = np.eye(n) - (1.0 / n) * np.ones((n, n))
    Kc = H @ K @ H
    Lc = H @ L @ H
    
    hsic_stat = float(np.sum(Kc * Lc) / ((n - 1) ** 2))
    
    # Permutation test
    rng = np.random.RandomState(random_state)
    greater_count = 0
    for _ in range(reps):
        y_perm = rng.permutation(y)
        y_perm_col = y_perm[:, None]
        L_perm = np.exp(-gamma * (y_perm_col - y_perm_col.T) ** 2)
        Lc_perm = H @ L_perm @ H
        hsic_perm = float(np.sum(Kc * Lc_perm) / ((n - 1) ** 2))
        if hsic_perm >= hsic_stat:
            greater_count += 1
            
    pvalue = float(greater_count / reps)
    return hsic_stat, pvalue

def run():
    pkl_path = r"C:\Users\uih83490\.gemini\antigravity-ide\brain\0f968684-3a6b-429f-82f1-62de38081140\scratch\variable_length_dataset.pkl"
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
        
    n = len(records)
    print(f"Loaded {n} variable-length records.")
    y = np.array([r['FM_Mooney'] for r in records], dtype=np.float64)
    dwell = np.array([r['AgingHours'] for r in records], dtype=np.float64)
    
    # 1. Define curve configurations
    configs = [
        ("多元曲线", [np.column_stack([r['temp'], r['power'], r['ram'], r['speed']]) for r in records]),
        ("温度", [r['temp'].reshape(-1, 1) for r in records]),
        ("功率", [r['power'].reshape(-1, 1) for r in records]),
        ("压栓", [r['ram'].reshape(-1, 1) for r in records]),
        ("转速", [r['speed'].reshape(-1, 1) for r in records])
    ]
    
    results = []
    
    for name, curves in configs:
        print(f"\nComputing GAK and HSIC for: {name}...")
        t0 = time.time()
        X = to_time_series_dataset(curves)
        sigma = float(sigma_gak(X))
        print(f"  sigma_gak = {sigma:.4f}")
        K = cdist_gak(X, sigma=sigma, n_jobs=-1)
        print(f"  cdist_gak completed in {time.time()-t0:.2f}s. Running permutation test (1000 reps)...")
        hsic_stat, pvalue = compute_hsic_and_pvalue(K, y, reps=1000, random_state=0)
        print(f"  Result: HSIC = {hsic_stat:.6e}, p-value = {pvalue:.4f}")
        results.append({
            'name': name,
            'hsic': hsic_stat,
            'pvalue': pvalue,
            'stat_type': 'HSIC'
        })
        
    # 2. 停留时间 (dwell)
    print("\nComputing conditional dependence for: 停留时间...")
    val_dwell = float(xicorpy.compute_conditional_dependence(y, dwell))
    print(f"  Result: 停留时间条件依赖系数 = {val_dwell:.6f}")
    results.append({
        'name': '停留时间',
        'hsic': val_dwell,
        'pvalue': None,
        'stat_type': 'Conditional Dependence'
    })
    
    out_dir = r"C:\Users\uih83490\.gemini\antigravity-ide\brain\0f968684-3a6b-429f-82f1-62de38081140"
    out_file = os.path.join(out_dir, "new_scheme2_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\nScheme 2 completed. Saved to {out_file}")

if __name__ == '__main__':
    run()
