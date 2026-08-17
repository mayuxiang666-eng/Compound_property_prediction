"""Build the batch-level Phase frame for group-atomic Mooney training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ALIASES = {"weight_pct_silian": "weight_pct_silan"}
GROUP_INVARIANTS = (
    "RecipeID", "ProcessVersion", "MixerLine", "phase_route", "TemplateKey"
)


def build_batch_phase_training_frame(
    batch_features: pd.DataFrame,
    group_labels: pd.DataFrame,
    source_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    """Create a no-leak batch frame with one effective vote per Lab group."""

    required_batch = {
        "OrderID", "BatchNumber", "PalletID", "LabelGroupID", "CompoundName",
        "RecipeID", "ProcessVersion", "MixerLine", "material_system", "phase_route",
        "TemplateKey",
    }
    required_group = {"LabelGroupID", "MNY", "lab_time", "split"}
    missing_batch = sorted(required_batch.difference(batch_features.columns))
    missing_group = sorted(required_group.difference(group_labels.columns))
    if missing_batch or missing_group:
        raise ValueError(
            f"Missing batch/group contract columns: batch={missing_batch}, group={missing_group}"
        )
    if batch_features.duplicated(["OrderID", "BatchNumber"]).any():
        raise ValueError("DUPLICATE_ORDERID_BATCHNUMBER_IN_BATCH_PHASE_SNAPSHOT")
    if group_labels["LabelGroupID"].duplicated().any():
        raise ValueError("DUPLICATE_LAB_GROUP_ROWS_IN_LABEL_SNAPSHOT")

    labels = group_labels[[
        column for column in ("LabelGroupID", "MNY", "lab_time", "split", "batch_count")
        if column in group_labels.columns
    ]].copy()
    labels["LabelGroupID"] = labels["LabelGroupID"].astype(str)
    work = batch_features.copy()
    work["LabelGroupID"] = work["LabelGroupID"].astype(str)
    work = work.loc[work["LabelGroupID"].isin(set(labels["LabelGroupID"]))].copy()
    work = work.merge(labels, on="LabelGroupID", how="left", validate="many_to_one")
    missing_groups = set(labels["LabelGroupID"]).difference(work["LabelGroupID"])
    if missing_groups:
        raise ValueError(f"BATCH_PHASE_MISSING_LAB_GROUPS: {len(missing_groups)}")
    invariant_conflicts = {
        column: int((work.groupby("LabelGroupID")[column].nunique(dropna=False) > 1).sum())
        for column in GROUP_INVARIANTS
    }
    if any(invariant_conflicts.values()):
        raise ValueError(f"INCONSISTENT_TEMPLATE_WITHIN_LAB_GROUP: {invariant_conflicts}")

    stage1_features = list(source_metadata["s1_features"])
    stage2_features = [
        column for column in source_metadata["s2_features"] if "__group_" not in column
    ]
    original_missing: dict[str, pd.Series] = {}
    for feature in list(dict.fromkeys(stage1_features + stage2_features)):
        if feature.endswith("__missing"):
            continue
        source = ALIASES.get(feature, feature)
        if source not in work.columns:
            raise ValueError(f"BATCH_PHASE_MISSING_FEATURE_SOURCE: {feature} <- {source}")
        if feature not in work.columns:
            work[feature] = work[source]
        work[feature] = pd.to_numeric(work[feature], errors="coerce")
        original_missing[feature] = work[feature].isna()

    for feature_list in (stage1_features, stage2_features):
        for feature in list(feature_list):
            if not feature.endswith("__missing"):
                continue
            base = feature[: -len("__missing")]
            source = ALIASES.get(base, base)
            if source not in work.columns:
                raise ValueError(f"BATCH_PHASE_MISSING_FLAG_SOURCE: {feature} <- {source}")
            work[feature] = pd.to_numeric(work[source], errors="coerce").isna().astype(float)

    added_missing_flags = []
    for feature_list in (stage1_features, stage2_features):
        for feature in list(feature_list):
            if feature.endswith("__missing"):
                continue
            flag = f"{feature}__missing"
            if original_missing[feature].any() and flag not in feature_list:
                work[flag] = original_missing[feature].astype(float)
                feature_list.append(flag)
                added_missing_flags.append(flag)

    train_mask = work["split"].eq("train")
    if not train_mask.any():
        raise ValueError("Batch Phase contract has no training groups")
    imputation_values: dict[str, float] = {}
    removed_all_missing = []
    for feature_list in (stage1_features, stage2_features):
        for feature in list(feature_list):
            work[feature] = pd.to_numeric(work[feature], errors="coerce")
            if feature.endswith("__missing"):
                work[feature] = work[feature].fillna(1.0)
                imputation_values[feature] = 1.0
                continue
            training = work.loc[train_mask, feature]
            if not training.notna().any():
                feature_list.remove(feature)
                removed_all_missing.append(feature)
                continue
            median = float(training.median())
            imputation_values[feature] = median
            work[feature] = work[feature].fillna(median)

    work["_label_group_id"] = work["LabelGroupID"].astype(str)
    work["_label_group_size"] = work.groupby("_label_group_id")["_label_group_id"].transform("size")
    work["_sample_weight"] = 1.0 / work["_label_group_size"]
    work["_w_label_raw"] = work["_sample_weight"]
    work["_w_loss"] = work["_sample_weight"]
    work["_w_metric"] = work["_sample_weight"]
    group_weight = work.groupby("_label_group_id")["_w_loss"].sum()
    if not np.allclose(group_weight.to_numpy(dtype=float), 1.0):
        raise RuntimeError("Batch weights do not sum to one per Lab group")

    audit = {
        "contract": "GROUP_ATOMIC_BATCH_MODEL_THEN_GROUP_MEAN_METRIC",
        "batch_rows": int(len(work)),
        "lab_groups": int(work["LabelGroupID"].nunique()),
        "duplicate_physical_batches": 0,
        "group_invariant_conflicts": invariant_conflicts,
        "stage1_features": stage1_features,
        "stage2_features": stage2_features,
        "excluded_group_aggregate_feature_count": int(
            len(source_metadata["s2_features"]) - len([
                column for column in source_metadata["s2_features"] if "__group_" not in column
            ])
        ),
        "added_missing_flags": sorted(set(added_missing_flags)),
        "removed_all_missing_features": sorted(set(removed_all_missing)),
        "imputation_values": imputation_values,
        "group_weight_min": float(group_weight.min()),
        "group_weight_max": float(group_weight.max()),
    }
    return work, stage1_features, stage2_features, audit


__all__ = ["build_batch_phase_training_frame"]
