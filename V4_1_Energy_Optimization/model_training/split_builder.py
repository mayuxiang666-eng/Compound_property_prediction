# ============================================================================
# V3.2 Strict Leak-Free Split Builder (Phase 0.1 & Phase 0.4)
# ============================================================================
# Core Rules:
# 1. Primary splitting unit: recipe_code (or connected component of recipe_code + label_group_id).
# 2. Zero Leakage: No shared recipe_code or label_group_id between Train/Val/Test.
# 3. Stratified Multi-level Group Split (Silica, Wet-mix, MNY quantiles).
# 4. Supports Out-Of-Time (OOT) Holdout splitting by production timestamp.
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def build_recipe_code(df):
    """
    Constructs a deterministic recipe_code if not directly present in dataset.
    Combines CompoundName with key PHR formulation fingerprints.
    """
    df = df.copy()
    if 'recipe_code' in df.columns and df['recipe_code'].notna().sum() > 0:
        return df['recipe_code'].astype(str).str.strip().str.upper()
        
    comp = df['CompoundName'].astype(str).str.strip().str.upper()
    
    # Key formula features
    phr_cols = ['silica_phr', 'weight_pct_oil', 'weight_pct_natural_rubber', 'weight_pct_carbon_black', 'Top_Fill_Factor']
    valid_phrs = [c for c in phr_cols if c in df.columns]
    
    if valid_phrs:
        phr_fingerprint = df[valid_phrs].fillna(0.0).round(1).astype(str).agg('-'.join, axis=1)
        return comp + '_REC_' + phr_fingerprint
    else:
        return comp


def resolve_connected_recipe_groups(df):
    """
    Builds connected components between recipe_codes and _label_group_ids.
    If a single _label_group_id spans multiple recipe_codes, merges them into 
    a single '_super_recipe_group' so no label group is fragmented.
    """
    df = df.copy()
    if '_label_group_id' not in df.columns:
        raise ValueError("_label_group_id missing. Run label_group_handler first.")
        
    df['_recipe_code'] = build_recipe_code(df)
    
    # Simple union-find / graph connectivity via pandas
    # Map each label_group_id to all recipe_codes it touches
    label_to_recipes = df.groupby('_label_group_id')['_recipe_code'].unique().to_dict()
    recipe_to_super = {}
    
    super_id_counter = 0
    for l_id, recipes in label_to_recipes.items():
        existing_supers = {recipe_to_super[r] for r in recipes if r in recipe_to_super}
        if existing_supers:
            target_super = min(existing_supers)
            # Reassign all existing supers to target_super
            for r, s in list(recipe_to_super.items()):
                if s in existing_supers:
                    recipe_to_super[r] = target_super
            for r in recipes:
                recipe_to_super[r] = target_super
        else:
            super_id_counter += 1
            for r in recipes:
                recipe_to_super[r] = super_id_counter
                
    df['_super_recipe_group'] = df['_recipe_code'].map(recipe_to_super)
    return df


def _safe_stratify_labels(labels, test_size):
    """Return labels only when every stratum can support a two-way split."""
    labels = pd.Series(labels)
    counts = labels.value_counts(dropna=False)
    n_test = int(np.ceil(len(labels) * test_size))
    n_train = len(labels) - n_test
    if len(counts) < 2 or counts.min() < 2 or n_test < len(counts) or n_train < len(counts):
        return None
    return labels


def generate_stratified_recipe_splits(df, n_splits=5, test_size=0.15, val_size=0.15, random_state=42):
    """
    Generates strict recipe_code group splits with multi-level stratification.
    
    Stratification criteria:
    - is_silica_system
    - has_wet_mix (Stage4_WetMixing_Duration > 0)
    - MNY quantile bin
    """
    df = resolve_connected_recipe_groups(df)
    
    # Define strata
    is_silica = (df['is_silica_system'] if 'is_silica_system' in df.columns else (df.get('silica_phr', 0) > 10.0)).astype(int)
    has_wet = (df['Stage4_WetMixing_Duration'] > 0).astype(int) if 'Stage4_WetMixing_Duration' in df.columns else np.zeros(len(df), dtype=int)
    
    mny_bins = pd.qcut(df['MNY'].fillna(df['MNY'].median()), q=4, labels=False, duplicates='drop')
    
    strata = is_silica.astype(str) + '_' + has_wet.astype(str) + '_' + mny_bins.astype(str)
    df['_strata'] = strata
    
    # Perform Group-based split by _super_recipe_group
    unique_groups = df.groupby('_super_recipe_group').agg({
        '_strata': lambda x: x.mode()[0] if len(x) > 0 else '0_0_0',
        'MNY': 'count'
    }).reset_index()
    
    test_strata = _safe_stratify_labels(unique_groups['_strata'], test_size)
    if test_strata is None:
        print("  [Warning] Sparse recipe strata; using unstratified recipe split for test holdout.")
    train_val_groups, test_groups = train_test_split(
        unique_groups['_super_recipe_group'].values,
        test_size=test_size,
        stratify=test_strata,
        random_state=random_state
    )
    
    # Sub-split train vs val
    val_ratio = val_size / (1.0 - test_size)
    unique_train_val = unique_groups[unique_groups['_super_recipe_group'].isin(train_val_groups)]
    
    val_strata = _safe_stratify_labels(unique_train_val['_strata'], val_ratio)
    if val_strata is None:
        print("  [Warning] Sparse recipe strata; using unstratified recipe split for validation holdout.")
    train_groups, val_groups = train_test_split(
        unique_train_val['_super_recipe_group'].values,
        test_size=val_ratio,
        stratify=val_strata,
        random_state=random_state
    )
    
    df['_split'] = 'train'
    df.loc[df['_super_recipe_group'].isin(val_groups), '_split'] = 'val'
    df.loc[df['_super_recipe_group'].isin(test_groups), '_split'] = 'test'
    
    # Audit verification
    audit_splits(df)
    return df


def generate_time_holdout_split(df, time_col='OrderStartTime', holdout_ratio=0.15):
    """
    Splits the dataset into Train and Out-Of-Time (OOT) Holdout set by production timestamp.
    """
    df = df.copy()
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be between 0 and 1.")
    if "_recipe_code" not in df.columns:
        df["_recipe_code"] = build_recipe_code(df)
    if time_col not in df.columns or df[time_col].isna().all():
        time_col = 'test_result_start_time' if 'test_result_start_time' in df.columns else None
        
    if time_col is None or df[time_col].isna().all():
        print("  [Warning] No timestamp column found for OOT split; falling back to recipe order.")
        recipe_times = pd.DataFrame({"_recipe_code": sorted(df["_recipe_code"].unique())})
    else:
        df['_parsed_time'] = pd.to_datetime(df[time_col], errors='coerce')
        recipe_times = (
            df.groupby("_recipe_code", as_index=False)["_parsed_time"]
            .median()
            .sort_values(["_parsed_time", "_recipe_code"], na_position="first")
            .reset_index(drop=True)
        )

    cutoff_idx = int(len(recipe_times) * (1.0 - holdout_ratio))
    oot_recipes = set(recipe_times.iloc[cutoff_idx:]["_recipe_code"])
    df['_time_split'] = np.where(df['_recipe_code'].isin(oot_recipes), 'oot_test', 'train')
        
    print(f"  Time Holdout Split: {len(df[df['_time_split']=='train'])} train batches, {len(df[df['_time_split']=='oot_test'])} OOT test batches.")
    return df


def audit_splits(df, split_col='_split'):
    """
    Verifies 0% leakage between train, val, and test splits.
    """
    train_recipes = set(df[df[split_col]=='train']['_recipe_code'])
    val_recipes = set(df[df[split_col]=='val']['_recipe_code'])
    test_recipes = set(df[df[split_col]=='test']['_recipe_code'])
    
    train_labels = set(df[df[split_col]=='train']['_label_group_id'])
    val_labels = set(df[df[split_col]=='val']['_label_group_id'])
    test_labels = set(df[df[split_col]=='test']['_label_group_id'])
    
    recipe_leak_val = train_recipes.intersection(val_recipes)
    recipe_leak_test = train_recipes.intersection(test_recipes)
    label_leak_val = train_labels.intersection(val_labels)
    label_leak_test = train_labels.intersection(test_labels)
    
    print("\n  --- SPLIT LEAKAGE AUDIT REPORT ---")
    print(f"  Train: {len(train_recipes)} recipes, {len(train_labels)} label groups, {sum(df[split_col]=='train')} batches")
    print(f"  Val:   {len(val_recipes)} recipes, {len(val_labels)} label groups, {sum(df[split_col]=='val')} batches")
    print(f"  Test:  {len(test_recipes)} recipes, {len(test_labels)} label groups, {sum(df[split_col]=='test')} batches")
    print(f"  Recipe Code Leakage Train-Val:  {len(recipe_leak_val)} (Expected: 0)")
    print(f"  Recipe Code Leakage Train-Test: {len(recipe_leak_test)} (Expected: 0)")
    print(f"  Label Group Leakage Train-Val:  {len(label_leak_val)} (Expected: 0)")
    print(f"  Label Group Leakage Train-Test: {len(label_leak_test)} (Expected: 0)")
    
    assert len(recipe_leak_val) == 0 and len(recipe_leak_test) == 0, "CRITICAL: Recipe code leakage detected!"
    assert len(label_leak_val) == 0 and len(label_leak_test) == 0, "CRITICAL: Label group leakage detected!"
    print("  Status: 100% LEAK-FREE SPLIT CONFIRMED.\n")


if __name__ == '__main__':
    print("Split Builder Module ready.")
