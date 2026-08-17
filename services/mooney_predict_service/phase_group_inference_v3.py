"""V3 strict Phase preparation with explicit Stage6 ram non-applicability."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .phase_group_contract import (
    FEATURE_BUILDER_VERSION,
    WEATHER_COLUMNS,
    PhaseFeatureContractError,
    PreparedPhaseBatch,
    _add_v37_derived_features,
    _normalise_identifiers,
    _route_fill_allowed,
    feature_columns_from_schema,
    remove_lab_fields,
)
from .phase_group_inference import predict_batches_and_groups, quality_summary


FEATURE_BUILDER_VERSION_V3 = "V3.7-phase-group-v3-stage6-ram-physical-zero"
STAGE6_RAM_COLUMNS = {
    "Stage6_BottomMixing_WayofRam_Mean",
    "Stage6_BottomMixing_WayofRam_Std",
}


def _route_fill_allowed_v3(row: pd.Series, column: str) -> bool:
    if column in STAGE6_RAM_COLUMNS:
        return True
    return _route_fill_allowed(row, column)


def prepare_phase_batches_v3(frame: pd.DataFrame, schema: dict[str, Any]) -> PreparedPhaseBatch:
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
                if _route_fill_allowed_v3(row, column):
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
            "FeatureQualityStatus": "MODEL_INPUT_OK" if not hard_missing else "BLOCKED_FEATURE_DATA_MISSING",
            "MissingModelColumnCount": 0,
            "MissingFeatureCellCount": len(hard_missing),
            "MissingFeatureCells": "|".join(sorted(set(hard_missing))),
            "RouteCompatibleZeroFillCount": len(route_filled),
            "RouteCompatibleZeroFillCells": "|".join(sorted(route_filled)),
            "LabColumnsRemoved": "|".join(sorted(removed_lab_columns)),
            "FeatureBuilderVersion": FEATURE_BUILDER_VERSION_V3,
        })
    return PreparedPhaseBatch(
        work.loc[valid_indexes].copy(), pd.DataFrame(quality_rows), feature_columns
    )


# Keep an unsurprising name for version-specific consumers.
prepare_phase_batches = prepare_phase_batches_v3


__all__ = [
    "FEATURE_BUILDER_VERSION_V3",
    "STAGE6_RAM_COLUMNS",
    "prepare_phase_batches",
    "prepare_phase_batches_v3",
    "predict_batches_and_groups",
    "quality_summary",
]
