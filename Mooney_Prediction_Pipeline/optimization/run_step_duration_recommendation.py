"""Single-batch shadow recommendation runner for transparent multi-Step duration search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.mooney_step_bridge import recommend_step_durations
from Mooney_Prediction_Pipeline.optimization.v37_feature_row_builder import build_v37_baseline_feature_row
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor
from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags
from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping
from V4_2_Step_Phase_Architecture.optimization.step_duration_bounds import attach_historical_step_duration_bounds


@dataclass(frozen=True)
class BatchRecommendationResult:
    """End-to-end shadow recommendation payload for one batch."""

    baseline_prediction: float
    recommended_steps: pd.DataFrame | None
    audit_table: pd.DataFrame
    status: str
    reason_codes: tuple[str, ...]
    feature_build_missing_count: int


def run_single_batch_duration_recommendation(
    order_id: str,
    batch_number: int,
    *,
    target_phase: str,
    expected_time_direction: int,
    spec_center: float,
    spec_low: float,
    spec_high: float,
    duration_increment_s: float = 1.0,
    max_candidates: int = 100,
) -> BatchRecommendationResult:
    """Generate a transparent, manual-review-only duration recommendation.

    This function uses the production V3.7 artifact, real BatchData Step SPs, real
    BatchCurve-derived baseline features, and historical step-duration bounds. It
    never reads current-batch lab Mooney.
    """
    package_dir = Path('Mooney_Prediction_Pipeline/models/v37_production_model_package')
    raw_predictor = V37RawPredictor(package_dir).load()
    feature_build = build_v37_baseline_feature_row(order_id, batch_number, raw_predictor)

    with get_sfe_connection() as conn:
        df_wide = pivot_batchdata_to_step_wide(conn, order_ids=[order_id], batch_numbers=[batch_number])
        if df_wide.empty:
            raise ValueError(f'No BatchData Step rows found for OrderID={order_id}, BatchNumber={batch_number}.')
        df_flags = build_settable_flags(df_wide)
        df_flags['CompoundName'] = feature_build.feature_row['CompoundName']
        df_mapped = build_step_phase_mapping(
            df_flags,
            target_temp=float(feature_build.feature_row.get('Target_Temperature', spec_center)),
        )
        df_bounded = attach_historical_step_duration_bounds(
            df_mapped,
            str(feature_build.feature_row['CompoundName']),
            conn,
        )

    recommendation = recommend_step_durations(
        df_bounded,
        feature_build.feature_row,
        raw_predictor,
        target_phase=target_phase,
        expected_time_direction=expected_time_direction,
        spec_center=spec_center,
        spec_low=spec_low,
        spec_high=spec_high,
        duration_increment_s=duration_increment_s,
        max_candidates=max_candidates,
    )
    return BatchRecommendationResult(
        baseline_prediction=recommendation.baseline_prediction,
        recommended_steps=recommendation.recommended_steps,
        audit_table=recommendation.audit_table,
        status=recommendation.status,
        reason_codes=recommendation.reason_codes,
        feature_build_missing_count=feature_build.missing_feature_count,
    )