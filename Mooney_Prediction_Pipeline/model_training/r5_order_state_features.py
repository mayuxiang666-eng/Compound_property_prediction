"""Feature contract for the R5 order-state level and delta experts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


def parse_target_vector(raw: object) -> dict[str, float]:
    """Parse the canonical TargetVectorRaw string without inventing values."""

    if raw is None or pd.isna(raw) or not str(raw).strip():
        raise ValueError("BLOCKED_MISSING_TARGET_VECTOR")
    result: dict[str, float] = {}
    for token in str(raw).split("|"):
        key_value = token.rsplit(":", 1)
        if len(key_value) != 2 or not key_value[0].strip():
            raise ValueError(f"Invalid target vector token: {token}")
        key = key_value[0].strip()
        try:
            value = float(key_value[1])
        except ValueError as error:
            raise ValueError(f"Non-numeric target vector value: {token}") from error
        if key in result and not np.isclose(result[key], value, equal_nan=False):
            raise ValueError(f"Conflicting target vector key: {key}")
        result[key] = value
    return result


def target_feature_name(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"target__{digest}"


@dataclass(frozen=True)
class TargetFeatureFrame:
    frame: pd.DataFrame
    feature_map: dict[str, str]


def build_group_target_features(phase_batches: pd.DataFrame) -> TargetFeatureFrame:
    """Build one sparse numeric Target vector per Lab group."""

    required = {
        "LabelGroupID", "TargetVectorRaw", "TargetVectorHash", "StepPatternHash",
        "TemplateKey", "BatchStartTime", "BatchEndTime",
    }
    missing = sorted(required.difference(phase_batches.columns))
    if missing:
        raise ValueError(f"R5 phase target source missing columns: {missing}")
    source = phase_batches.copy()
    for column in ("BatchStartTime", "BatchEndTime"):
        source[column] = pd.to_datetime(source[column], errors="coerce", utc=True)
    if source[["BatchStartTime", "BatchEndTime"]].isna().any().any():
        raise ValueError("R5 phase target source contains invalid batch timestamps")
    consistency = source.groupby("LabelGroupID").agg(
        target_vectors=("TargetVectorRaw", "nunique"),
        target_hashes=("TargetVectorHash", "nunique"),
        step_hashes=("StepPatternHash", "nunique"),
        templates=("TemplateKey", "nunique"),
    )
    if (consistency > 1).any().any():
        raise ValueError("BLOCKED_INCONSISTENT_GROUP_TARGET_OR_TEMPLATE")

    one = (
        source.sort_values(["LabelGroupID", "BatchStartTime"], kind="stable")
        .groupby("LabelGroupID", as_index=False)
        .agg(
            TargetVectorRaw=("TargetVectorRaw", "first"),
            TargetVectorHash=("TargetVectorHash", "first"),
            StepPatternHash=("StepPatternHash", "first"),
            TemplateKey=("TemplateKey", "first"),
            GroupProductionStart=("BatchStartTime", "min"),
            GroupProductionComplete=("BatchEndTime", "max"),
            PhaseBatchCountR5=("BatchStartTime", "size"),
        )
    )
    parsed = [parse_target_vector(raw) for raw in one["TargetVectorRaw"]]
    all_keys = sorted({key for values in parsed for key in values})
    key_to_feature = {key: target_feature_name(key) for key in all_keys}
    if len(set(key_to_feature.values())) != len(key_to_feature):
        raise RuntimeError("Target feature hash collision")
    numeric = pd.DataFrame(
        [
            {key_to_feature[key]: value for key, value in values.items()}
            for values in parsed
        ],
        index=one.index,
    )
    output = pd.concat([one.drop(columns=["TargetVectorRaw"]), numeric], axis=1)
    feature_map = {feature: key for key, feature in key_to_feature.items()}
    return TargetFeatureFrame(output, feature_map)


def add_prior_order_state(
    groups: pd.DataFrame,
    *,
    state_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Add previous completed Lab-group process state without using its Lab."""

    required = {
        "LabelGroupID", "OrderID", "GroupProductionStart", "GroupProductionComplete",
        *state_features,
    }
    missing = sorted(required.difference(groups.columns))
    if missing:
        raise ValueError(f"R5 prior-state source missing columns: {missing}")
    frame = groups.copy()
    frame["GroupProductionStart"] = pd.to_datetime(
        frame["GroupProductionStart"], errors="coerce", utc=True
    )
    frame["GroupProductionComplete"] = pd.to_datetime(
        frame["GroupProductionComplete"], errors="coerce", utc=True
    )
    if frame[["GroupProductionStart", "GroupProductionComplete"]].isna().any().any():
        raise ValueError("R5 prior-state timestamps contain invalid values")
    frame["_original_order"] = np.arange(len(frame), dtype=int)
    frame = frame.sort_values(
        ["OrderID", "GroupProductionStart", "LabelGroupID"], kind="stable"
    ).reset_index(drop=True)
    order_group = frame.groupby("OrderID", sort=False)
    previous_complete = order_group["GroupProductionComplete"].shift(1)
    valid_prior = previous_complete.le(frame["GroupProductionStart"])
    frame["prior_order_state_available"] = valid_prior.fillna(False).astype(int)
    frame["prior_group_gap_hours"] = (
        frame["GroupProductionStart"] - previous_complete
    ).dt.total_seconds() / 3600.0
    frame["order_group_index"] = order_group.cumcount()
    added = [
        "prior_order_state_available", "prior_group_gap_hours", "order_group_index"
    ]
    for feature in state_features:
        current = pd.to_numeric(frame[feature], errors="coerce")
        prior = order_group[feature].shift(1)
        prior = pd.to_numeric(prior, errors="coerce").where(valid_prior)
        prior_name = f"prior__{feature}"
        delta_name = f"delta_prior__{feature}"
        frame[prior_name] = prior
        frame[delta_name] = current - prior
        added.extend([prior_name, delta_name])
    frame = frame.sort_values("_original_order", kind="stable").drop(
        columns="_original_order"
    ).reset_index(drop=True)
    return frame, added


def build_adjacent_order_pairs(
    groups: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_col: str = "MNY",
    prediction_col: str = "LevelPrediction",
) -> pd.DataFrame:
    """Build current-minus-prior pairs inside complete, unsplit orders."""

    required = {
        "LabelGroupID", "OrderID", "split", "GroupProductionStart",
        target_col, prediction_col, *feature_columns,
    }
    missing = sorted(required.difference(groups.columns))
    if missing:
        raise ValueError(f"R5 delta-pair source missing columns: {missing}")
    if groups.groupby("OrderID")["split"].nunique().max() != 1:
        raise ValueError("R5 delta pairs require an order-atomic outer split")
    ordered = groups.sort_values(
        ["OrderID", "GroupProductionStart", "LabelGroupID"], kind="stable"
    ).copy()
    previous = ordered.groupby("OrderID", sort=False).shift(1)
    has_prior = previous["LabelGroupID"].notna()
    current = ordered.loc[has_prior].copy()
    previous = previous.loc[has_prior].copy()
    output = current[
        ["LabelGroupID", "OrderID", "split", "GroupProductionStart"]
    ].rename(columns={"LabelGroupID": "CurrentLabelGroupID"})
    output["PriorLabelGroupID"] = previous["LabelGroupID"].astype(str).to_numpy()
    output["ActualDelta"] = (
        pd.to_numeric(current[target_col], errors="coerce").to_numpy()
        - pd.to_numeric(previous[target_col], errors="coerce").to_numpy()
    )
    output["LevelDelta"] = (
        pd.to_numeric(current[prediction_col], errors="coerce").to_numpy()
        - pd.to_numeric(previous[prediction_col], errors="coerce").to_numpy()
    )
    output["PriorLevelPrediction"] = pd.to_numeric(
        previous[prediction_col], errors="coerce"
    ).to_numpy()
    for feature in feature_columns:
        output[f"pair_delta__{feature}"] = (
            pd.to_numeric(current[feature], errors="coerce").to_numpy()
            - pd.to_numeric(previous[feature], errors="coerce").to_numpy()
        )
    return output.reset_index(drop=True)


__all__ = [
    "TargetFeatureFrame",
    "add_prior_order_state",
    "build_adjacent_order_pairs",
    "build_group_target_features",
    "parse_target_vector",
    "target_feature_name",
]
