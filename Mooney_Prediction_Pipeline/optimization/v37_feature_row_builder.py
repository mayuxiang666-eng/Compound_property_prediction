"""Build a no-Lab V3.7 baseline feature row for one real OrderID/BatchNumber."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from Mooney_Prediction_Pipeline.feature_engineering.cb_dispersion_feature_builder import build_cb_dispersion_features
from Mooney_Prediction_Pipeline.feature_engineering.clustering import cluster_silica_carbon_black
from Mooney_Prediction_Pipeline.feature_engineering.silica_pid_feature_builder import build_silica_pid_features
from Mooney_Prediction_Pipeline.model_analysis.predict_latest_sfe_order import extract_features_for_row
from Mooney_Prediction_Pipeline.model_analysis.predict_order_v3 import query_order_rows
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor


LAB_ONLY_COLUMNS = {'actual_mny', 'current_batch_mny', 'MNY', 'Mooney_Viscosity'}


@dataclass(frozen=True)
class V37FeatureBuildResult:
    """One artifact-aligned feature row plus transparent audit metadata."""

    feature_row: pd.Series
    source_row: pd.Series
    missing_feature_count: int
    missing_feature_names: tuple[str, ...]


def build_v37_baseline_feature_row(
    order_id: str,
    batch_number: int,
    raw_predictor: V37RawPredictor,
) -> V37FeatureBuildResult:
    """Build a single no-Lab baseline row aligned to the loaded V3.7 artifact.

    The source path is exactly the existing V3 real-order feature extractor:
    query SFE order/batch rows -> decode BatchCurve -> derive stage features ->
    add Silica/CB proxy features -> classify material system -> align to artifact.
    """
    order_rows = query_order_rows(str(order_id))
    batch_rows = order_rows[order_rows['BatchNumber'] == int(batch_number)].copy()
    if batch_rows.empty:
        raise ValueError(f'No feature source row found for OrderID={order_id}, BatchNumber={batch_number}.')

    source_row = batch_rows.iloc[0].copy()
    extracted = extract_features_for_row(source_row)
    feature_df = pd.DataFrame([extracted])
    feature_df = cluster_silica_carbon_black(feature_df)

    pid_features = build_silica_pid_features(feature_df)
    cb_features = build_cb_dispersion_features(feature_df)
    for column in pid_features.columns:
        feature_df[column] = pid_features[column]
    for column in cb_features.columns:
        feature_df[column] = cb_features[column]

    present_lab_columns = LAB_ONLY_COLUMNS & set(feature_df.columns)
    if present_lab_columns:
        raise ValueError(f'Baseline feature row must not contain lab fields: {sorted(present_lab_columns)}')

    missing_features = [column for column in raw_predictor.feature_columns if column not in feature_df.columns]
    for column in missing_features:
        feature_df[column] = 0.0

    for column in raw_predictor.feature_columns:
        feature_df[column] = pd.to_numeric(feature_df[column], errors='coerce').fillna(0.0)

    if 'CompoundName' not in feature_df.columns:
        feature_df['CompoundName'] = source_row['CompoundName']
    if 'material_system' not in feature_df.columns:
        feature_df['material_system'] = feature_df['compound_cluster'] if 'compound_cluster' in feature_df.columns else 'CarbonBlack'
    if 'BatchNumber' not in feature_df.columns:
        feature_df['BatchNumber'] = int(batch_number)
    if 'OrderID' not in feature_df.columns:
        feature_df['OrderID'] = str(order_id)

    return V37FeatureBuildResult(
        feature_row=feature_df.iloc[0].copy(),
        source_row=source_row,
        missing_feature_count=len(missing_features),
        missing_feature_names=tuple(missing_features),
    )