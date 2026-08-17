"""Canonical V3.7 shadow-only temperature recommendation entry point."""

from __future__ import annotations

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.counterfactual_temperature_search_v3 import (
    search_counterfactual_temperature_candidates,
)
from Mooney_Prediction_Pipeline.optimization.step_candidate_search import CandidateSearchResult
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor


def recommend_step_temperatures_counterfactual(
    df_step_mapped: pd.DataFrame,
    baseline_feature_row: pd.Series,
    raw_predictor: V37RawPredictor,
    *,
    target_phase: str,
    expected_temp_direction: int,
    spec_center: float,
    spec_low: float,
    spec_high: float,
    temperature_increment_c: float = 1.0,
    max_candidates: int = 100,
) -> CandidateSearchResult:
    """Return a recommendation only when every Target action reaches V3.7."""
    return search_counterfactual_temperature_candidates(
        df_step_mapped,
        baseline_feature_row,
        raw_predictor,
        target_phase=target_phase,
        expected_temp_direction=expected_temp_direction,
        target_center=spec_center,
        spec_lower=spec_low,
        spec_upper=spec_high,
        temperature_increment_c=temperature_increment_c,
        max_candidates=max_candidates,
    )
