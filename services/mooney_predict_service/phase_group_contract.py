"""Fail-closed V3.7 Phase input and pallet-label-group prediction contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.feature_engineering.cb_dispersion_feature_builder import build_cb_dispersion_features
from Mooney_Prediction_Pipeline.feature_engineering.clustering import cluster_silica_carbon_black
from Mooney_Prediction_Pipeline.feature_engineering.silica_pid_feature_builder import build_silica_pid_features


FEATURE_BUILDER_VERSION = "V3.7-phase-group-v2"
WEATHER_COLUMNS = (
    "env_temp_mean", "env_temp_min", "env_temp_max",
    "env_humidity_mean", "env_humidity_min", "env_humidity_max",
)
IDENTIFIER_COLUMNS = ("OrderID", "BatchNumber", "PalletID", "CompoundName", "OrderStartTime")
LAB_ONLY_COLUMNS = {
    "mny", "mooney_viscosity", "actual_mny", "current_batch_mny", "labactualmny",
}


class PhaseFeatureContractError(ValueError):
    """The Phase row cannot safely enter V3.7 inference."""


@dataclass(frozen=True)
class PreparedPhaseBatch:
    features: pd.DataFrame
    quality: pd.DataFrame
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class PhasePredictionResult:
    batch_predictions: pd.DataFrame
    group_predictions: pd.DataFrame


def feature_columns_from_schema(schema: dict[str, Any]) -> tuple[str, ...]:
    s1, s2 = schema.get("s1_features"), schema.get("s2_features")
    if not isinstance(s1, list) or not isinstance(s2, list):
        raise PhaseFeatureContractError("feature_metadata must define list-valued s1_features and s2_features")
    columns = tuple(dict.fromkeys(str(value) for value in [*s1, *s2]))
    if not columns:
        raise PhaseFeatureContractError("V3.7 feature contract is empty")
    return columns


def remove_lab_fields(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    removed = [column for column in frame.columns if str(column).strip().lower() in LAB_ONLY_COLUMNS]
    return frame.drop(columns=removed, errors="ignore").copy(), removed


def _normalise_identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    missing = sorted(set(IDENTIFIER_COLUMNS) - set(work.columns))
    if missing:
        raise PhaseFeatureContractError(f"Missing Phase identity columns: {', '.join(missing)}")
    for column in ("OrderID", "PalletID", "CompoundName"):
        work[column] = work[column].astype("string").str.strip()
    work["BatchNumber"] = pd.to_numeric(work["BatchNumber"], errors="coerce")
    work["OrderStartTime"] = pd.to_datetime(work["OrderStartTime"], errors="coerce")
    invalid = (
        work[["OrderID", "PalletID", "CompoundName"]].isna().any(axis=1)
        | work[["OrderID", "PalletID", "CompoundName"]].eq("").any(axis=1)
        | work["BatchNumber"].isna()
        | work["OrderStartTime"].isna()
    )
    if invalid.any():
        raise PhaseFeatureContractError(f"{int(invalid.sum())} rows have invalid batch/group identity")
    duplicated = work.duplicated(["OrderID", "BatchNumber"], keep=False)
    if duplicated.any():
        examples = work.loc[duplicated, ["OrderID", "BatchNumber"]].head(5).to_dict("records")
        raise PhaseFeatureContractError(f"Duplicate OrderID+BatchNumber rows: {examples}")
    work["BatchNumber"] = work["BatchNumber"].astype(int)
    work["LabelGroupID"] = work["OrderID"].astype(str) + "|" + work["PalletID"].astype(str)
    return work


def _add_v37_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for derived in (build_silica_pid_features(work), build_cb_dispersion_features(work)):
        for column in derived.columns:
            work[column] = derived[column]
    return cluster_silica_carbon_black(work)


def _route_fill_allowed(row: pd.Series, column: str) -> bool:
    """Allow zero only where null explicitly means an absent route/material."""
    material = str(row.get("material_system", "")).strip().lower()
    oil = pd.to_numeric(pd.Series([row.get("is_oil_loading_present")]), errors="coerce").iloc[0]
    no_oil = pd.notna(oil) and float(oil) <= 0.0
    if column.startswith(("Stage3_OilLoading_", "Stage4_WetMixing_")) or column == "phys_eta_app_wetmix_mean":
        return no_oil
    if column == "weight_pct_oil":
        return no_oil
    if material == "carbonblack" and column in {
        "weight_pct_silica", "weight_pct_silian",
        "supplier_silica_moisture_avg", "supplier_silica_surface_area_avg",
    }:
        return True
    if material == "silica" and column in {
        "weight_pct_carbon_black", "supplier_carbon_black_structure_avg",
        "supplier_carbon_black_surface_area_avg", "supplier_carbon_black_moisture_avg",
    }:
        return True
    return False
