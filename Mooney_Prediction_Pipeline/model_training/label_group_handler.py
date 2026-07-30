# ============================================================================
# V3.2 Label Group Handler (Phase 0.2)
# ============================================================================
# Mandated Rules:
# 1. Do NOT include BatchNumber in the label group ID.
# 2. label_group_id is defined by OrderID + PalletID (or LabSampleID if available).
# 3. Compute _label_group_size and _sample_weight = 1.0 / size.
# 4. Detect and audit LABEL_GROUP_CONFLICT (conflicting MNY for same group).
# ============================================================================

import numpy as np
import pandas as pd


def get_label_group_columns(df):
    """Determine best available columns for defining label groups."""
    candidate_lab_ids = ["MNY_SampleID", "LabSampleID", "SampleID", "TestSampleID"]
    for col in candidate_lab_ids:
        if col in df.columns and df[col].notna().sum() > 0:
            return [col]
    
    # Standard industrial grouping: OrderID + PalletID
    cols = []
    if "OrderID" in df.columns:
        cols.append("OrderID")
    if "PalletID" in df.columns:
        cols.append("PalletID")
    
    return cols if cols else ["OrderID"]


def add_label_group_information(df, label_group_cols=None):
    """
    Adds '_label_group_id', '_label_group_size', and '_sample_weight' to DataFrame.
    Guarantees that batches sharing the same physical lab measurement share 
    a single effective sample weight sum of 1.0.
    """
    df = df.copy()
    if label_group_cols is None:
        label_group_cols = get_label_group_columns(df)
        
    valid_cols = [c for c in label_group_cols if c in df.columns]
    
    if not valid_cols:
        # Emergency fallback (should not happen if OrderID exists)
        df['_label_group_id'] = df['OrderID'].astype(str)
    else:
        work = df[valid_cols].copy()
        for col in valid_cols:
            work[col] = work[col].astype(str).str.strip().str.upper()
            
        # Check invalid/missing
        invalid = work.isna().any(axis=1)
        for col in valid_cols:
            invalid |= work[col].isin(['', 'NAN', 'NONE', 'NULL', '<NA>'])
            
        df['_label_group_id'] = work.agg('|'.join, axis=1)
        
        # If OrderID + PalletID was used, fallback for rows missing PalletID is OrderID
        if invalid.any():
            fallback = df['OrderID'].astype(str)
            df.loc[invalid, '_label_group_id'] = fallback[invalid]
            
    # Calculate group sizes and sample weights
    df['_label_group_size'] = df.groupby('_label_group_id')['_label_group_id'].transform('size')
    df['_sample_weight'] = 1.0 / df['_label_group_size'].clip(lower=1)
    
    return df


def audit_label_group_conflicts(df, mny_col="MNY", max_std_threshold=3.0):
    """
    Identifies groups where multiple conflicting MNY lab values are assigned to 
    the same label_group_id. Returns audit report DataFrame and flagged IDs.
    """
    if "_label_group_id" not in df.columns:
        df = add_label_group_information(df)
        
    group_stats = df.groupby("_label_group_id")[mny_col].agg(
        n_batches="count",
        mny_mean="mean",
        mny_std="std",
        mny_min="min",
        mny_max="max"
    ).reset_index()
    
    group_stats["mny_range"] = group_stats["mny_max"] - group_stats["mny_min"]
    
    # Conflicts: std > threshold or range > 2 * threshold
    conflicts = group_stats[
        (group_stats["n_batches"] > 1) & 
        ((group_stats["mny_std"] > max_std_threshold) | (group_stats["mny_range"] > 2.0 * max_std_threshold))
    ].copy()
    
    conflicts["conflict_reason"] = "HIGH_MNY_VARIANCE_IN_SAME_PALLET_GROUP"
    conflict_ids = set(conflicts["_label_group_id"])
    
    return group_stats, conflicts, conflict_ids


if __name__ == '__main__':
    print("Label Group Handler Module ready.")
