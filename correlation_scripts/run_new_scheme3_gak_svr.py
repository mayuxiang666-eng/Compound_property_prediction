"""
New Scheme 3: Variable-length GAK SVR + Channel Permutation Error
Strictly following: d:\\AntigravityIDE\\Project\\方案3-变长GAK回归与通道误差.md
"""
import sys
import os
import json
import pickle
import time
import numpy as np
import pandas as pd
from tslearn.svm import TimeSeriesSVR
from tslearn.utils import to_time_series_dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def run():
    pkl_path = r"C:\Users\uih83490\.gemini\antigravity-ide\brain\0f968684-3a6b-429f-82f1-62de38081140\scratch\variable_length_dataset.pkl"
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
        
    n = len(records)
    print(f"Loaded {n} variable-length records.")
    
    # Group by FM_PalletID (all batches sharing the same FM_PalletID go to same set)
    df_meta = pd.DataFrame([
        {'idx': i, 'pallet': r['FM_PalletID'], 'mooney': r['FM_Mooney']}
        for i, r in enumerate(records)
    ])
    
    pallets = df_meta['pallet'].unique()
    rng = np.random.RandomState(0)
    shuffled_pallets = rng.permutation(pallets)
    
    # 75% train pallets, 25% holdout pallets
    n_train_pallets = int(len(pallets) * 0.75)
    train_pallets = set(shuffled_pallets[:n_train_pallets])
    holdout_pallets = set(shuffled_pallets[n_train_pallets:])
    
    train_indices = df_meta[df_meta['pallet'].isin(train_pallets)]['idx'].values
    holdout_indices = df_meta[df_meta['pallet'].isin(holdout_pallets)]['idx'].values
    
    print(f"Total pallets: {len(pallets)}. Train pallets: {len(train_pallets)} ({len(train_indices)} batches), Holdout pallets: {len(holdout_pallets)} ({len(holdout_indices)} batches)")
    
    # Build multivariate curves (T_i, d)
    # Order: 温度, 功率, 压栓, 转速
    channel_names = ["温度", "功率", "压栓", "转速"]
    d = len(channel_names)
    
    curves_train = [
        np.column_stack([records[i]['temp'], records[i]['power'], records[i]['ram'], records[i]['speed']])
        for i in train_indices
    ]
    y_train = np.array([records[i]['FM_Mooney'] for i in train_indices], dtype=np.float64)
    
    curves_holdout = [
        np.column_stack([records[i]['temp'], records[i]['power'], records[i]['ram'], records[i]['speed']])
        for i in holdout_indices
    ]
    y_holdout = np.array([records[i]['FM_Mooney'] for i in holdout_indices], dtype=np.float64)
    
    X_train = to_time_series_dataset(curves_train)
    X_holdout = to_time_series_dataset(curves_holdout)
    
    print(f"Train dataset shape: {X_train.shape}, Holdout dataset shape: {X_holdout.shape}")
    
    # 1. Fit TimeSeriesSVR with C=1.0 and kernel='gak'
    print("\nFitting TimeSeriesSVR(C=1.0, kernel='gak')...")
    t0 = time.time()
    svr = TimeSeriesSVR(C=1.0, kernel="gak", n_jobs=-1)
    svr.fit(X_train, y_train)
    print(f"SVR fit completed in {time.time()-t0:.2f}s.")
    
    # 2. Compute mse_full on holdout set
    y_pred_full = svr.predict(X_holdout)
    mse_full = float(np.mean((y_holdout - y_pred_full) ** 2))
    print(f"Full model holdout MSE (mse_full) = {mse_full:.6f}")
    
    # 3. Channel permutation error
    results = []
    
    for ch_idx, ch_name in enumerate(channel_names):
        print(f"\nPermuting channel {ch_idx}: {ch_name} on holdout set...")
        # Deep copy holdout curves list
        shuffled_curves = [c.copy() for c in curves_holdout]
        n_h = len(shuffled_curves)
        
        # Shuffle this column across holdout batches, keeping each curve's length with original batch
        perm_idx = rng.permutation(n_h)
        for i in range(n_h):
            donor_i = perm_idx[i]
            # Column ch_idx taken from donor_i
            donor_col = curves_holdout[donor_i][:, ch_idx]
            target_len = len(shuffled_curves[i])
            
            # If lengths differ, handle assignment
            if len(donor_col) >= target_len:
                shuffled_curves[i][:, ch_idx] = donor_col[:target_len]
            else:
                # pad donor col with its last value or mean if donor is shorter
                padded = np.pad(donor_col, (0, target_len - len(donor_col)), mode='edge')
                shuffled_curves[i][:, ch_idx] = padded
                
        X_shuffled = to_time_series_dataset(shuffled_curves)
        y_pred_shuffled = svr.predict(X_shuffled)
        mse_without = float(np.mean((y_holdout - y_pred_shuffled) ** 2))
        diff = float(mse_without - mse_full)
        retained = (diff > 0)
        
        print(f"  mse_without: {mse_without:.6f}, diff (mse_without - mse_full): {diff:+.6f}, retained: {retained}")
        results.append({
            'channel': ch_name,
            'mse_full': mse_full,
            'mse_without': mse_without,
            'diff': diff,
            'retained': retained
        })
        
    out_dir = r"C:\Users\uih83490\.gemini\antigravity-ide\brain\0f968684-3a6b-429f-82f1-62de38081140"
    out_file = os.path.join(out_dir, "new_scheme3_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            'mse_full': mse_full,
            'channels': results,
            'n_train': len(train_indices),
            'n_holdout': len(holdout_indices)
        }, f, ensure_ascii=False, indent=2)
        
    print(f"\nScheme 3 completed. Saved to {out_file}")

if __name__ == '__main__':
    run()
