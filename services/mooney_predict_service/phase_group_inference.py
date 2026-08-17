"""Strict Phase preparation and canonical V3.7 pallet-group inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .phase_group_contract import (
    FEATURE_BUILDER_VERSION,
    WEATHER_COLUMNS,
    PhaseFeatureContractError,
    PhasePredictionResult,
    PreparedPhaseBatch,
    _add_v37_derived_features,
    _normalise_identifiers,
    _route_fill_allowed,
    feature_columns_from_schema,
    remove_lab_fields,
)


def prepare_phase_batches(frame: pd.DataFrame, schema: dict[str, Any]) -> PreparedPhaseBatch:
    clean, removed_lab_columns = remove_lab_fields(frame)
    work = _add_v37_derived_features(_normalise_identifiers(clean))
    feature_columns = feature_columns_from_schema(schema)
    missing_columns = sorted(set(feature_columns) - set(work.columns))
    if missing_columns:
        raise PhaseFeatureContractError(
            "Phase builder is missing V3.7 columns: " + ", ".join(missing_columns)
        )
    missing_weather = sorted(set(WEATHER_COLUMNS) - set(work.columns))
    if missing_weather:
        raise PhaseFeatureContractError("Phase builder is missing weather: " + ", ".join(missing_weather))

    quality_rows: list[dict[str, Any]] = []
    valid_indexes: list[Any] = []
    for index, row in work.iterrows():
        hard_missing: list[str] = []
        route_filled: list[str] = []
        for column in feature_columns:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.isna(value):
                if _route_fill_allowed(row, column):
                    work.at[index, column] = 0.0
                    route_filled.append(column)
                else:
                    hard_missing.append(column)
            else:
                work.at[index, column] = float(value)
        if not hard_missing:
            valid_indexes.append(index)
        quality_rows.append({
            "OrderID": str(row["OrderID"]),
            "BatchNumber": int(row["BatchNumber"]),
            "PalletID": str(row["PalletID"]),
            "LabelGroupID": str(row["LabelGroupID"]),
            "FeatureQualityStatus": (
                "MODEL_INPUT_OK" if not hard_missing else "BLOCKED_FEATURE_DATA_MISSING"
            ),
            "MissingModelColumnCount": 0,
            "MissingFeatureCellCount": len(hard_missing),
            "MissingFeatureCells": "|".join(sorted(set(hard_missing))),
            "RouteCompatibleZeroFillCount": len(route_filled),
            "RouteCompatibleZeroFillCells": "|".join(sorted(route_filled)),
            "LabColumnsRemoved": "|".join(sorted(removed_lab_columns)),
            "FeatureBuilderVersion": FEATURE_BUILDER_VERSION,
        })
    return PreparedPhaseBatch(work.loc[valid_indexes].copy(), pd.DataFrame(quality_rows), feature_columns)


def predict_batches_and_groups(
    prepared: PreparedPhaseBatch,
    model: Any,
    model_version: str,
) -> PhasePredictionResult:
    if prepared.features.empty:
        return PhasePredictionResult(pd.DataFrame(), pd.DataFrame())
    features = prepared.features.copy()
    output = model.predict(features, cluster_col="material_system")
    if not isinstance(output, tuple) or len(output) != 4:
        raise PhaseFeatureContractError("V3.7 model.predict must return four Stage outputs")
    final, stage1, stage1b, stage2 = [np.asarray(value, dtype=float).reshape(-1) for value in output]
    if not all(len(value) == len(features) for value in (final, stage1, stage1b, stage2)):
        raise PhaseFeatureContractError("Prediction output row count differs from Phase input")
    if not np.isfinite(final).all():
        raise PhaseFeatureContractError("V3.7 returned non-finite predictions")

    identity = [
        "OrderID", "BatchNumber", "PalletID", "LabelGroupID", "CompoundName",
        "OrderStartTime", "material_system",
    ]
    optional = [column for column in ("CompoundDescription", "Mixer", "PhaseRoute") if column in features]
    batch = features[identity + optional].copy()
    batch["ModelVersion"] = str(model_version)
    batch["Stage1Pred"], batch["Stage1bBias"] = stage1, stage1b
    batch["Stage2Delta"], batch["RawPrediction"] = stage2, final
    batch["FeatureBuilderVersion"] = FEATURE_BUILDER_VERSION
    group = batch.groupby(
        ["OrderID", "PalletID", "LabelGroupID", "CompoundName"], as_index=False
    ).agg(
        BatchCount=("BatchNumber", "size"),
        FirstBatchNumber=("BatchNumber", "min"),
        LastBatchNumber=("BatchNumber", "max"),
        GroupRawPrediction=("RawPrediction", "mean"),
        BatchPredictionStd=("RawPrediction", "std"),
        ProductTime=("OrderStartTime", "max"),
    )
    group["BatchPredictionStd"] = group["BatchPredictionStd"].fillna(0.0)
    group["ModelVersion"] = str(model_version)
    group["FeatureBuilderVersion"] = FEATURE_BUILDER_VERSION
    group["GroupPredictionStatus"] = "PREDICTED_GROUP_MEAN"
    group["AutomaticControlStatus"] = "DISABLED"
    return PhasePredictionResult(batch, group)


def quality_summary(quality: pd.DataFrame, batch_predictions: pd.DataFrame) -> dict[str, Any]:
    counts = quality["FeatureQualityStatus"].value_counts(dropna=False).to_dict() if not quality.empty else {}
    rejected = int((quality["FeatureQualityStatus"] != "MODEL_INPUT_OK").sum()) if not quality.empty else 0
    return {
        "source_rows": int(len(quality)),
        "accepted_batch_rows": int(len(batch_predictions)),
        "rejected_batch_rows": rejected,
        "quality_status_counts": {str(key): int(value) for key, value in counts.items()},
        "missing_model_columns": 0,
        "weather_contract": "ORDER_START_DATE_PHASE_WEATHER_REQUIRED",
        "label_contract": "BATCH_PREDICT_THEN_ORDERID_PALLETID_MEAN",
        "feature_builder_version": FEATURE_BUILDER_VERSION,
        "automatic_control": "DISABLED",
    }
