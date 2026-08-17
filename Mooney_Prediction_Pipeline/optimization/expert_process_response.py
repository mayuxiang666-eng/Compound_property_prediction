"""Time-split process-response models for expert shadow recommendations.

These models translate Step Target vectors to predicted *actual* process
endpoints before a Mooney model is allowed to score a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


PROCESS_OUTPUTS = (
    "dry_end_temp_actual",
    "post_oil_temp_actual",
    "wet_end_temp_actual",
    "wet_duration_actual",
    "discharge_temp_actual",
)
TEMPERATURE_OUTPUTS = {
    "dry_end_temp_actual",
    "post_oil_temp_actual",
    "wet_end_temp_actual",
    "discharge_temp_actual",
}


@dataclass(frozen=True)
class ResponseMetric:
    output: str
    mae: float
    interval_coverage: float
    sample_count: int
    passed: bool


@dataclass
class ProcessResponseModel:
    """Three quantile models per endpoint with deterministic time-split validation."""

    feature_columns: tuple[str, ...]
    models: dict[str, tuple[GradientBoostingRegressor, GradientBoostingRegressor, GradientBoostingRegressor]] = field(default_factory=dict)
    metrics: dict[str, ResponseMetric] = field(default_factory=dict)

    @staticmethod
    def _frame(history: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        frame = history.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        return frame.replace([np.inf, -np.inf], np.nan)

    def fit_validate(
        self,
        history: pd.DataFrame,
        *,
        time_column: str,
        minimum_samples: int = 150,
        validation_fraction: float = 0.2,
    ) -> dict[str, ResponseMetric]:
        """Fit on the oldest window and validate on the newest window only."""
        if time_column not in history.columns:
            raise ValueError(f"Missing time split column: {time_column}")
        ordered = history.sort_values(time_column, kind="stable").copy()
        for output in PROCESS_OUTPUTS:
            columns = [*self.feature_columns, output]
            available = ordered.dropna(subset=columns)
            if len(available) < minimum_samples:
                self.metrics[output] = ResponseMetric(output, np.nan, np.nan, len(available), False)
                continue
            split = max(1, int(len(available) * (1.0 - validation_fraction)))
            train, valid = available.iloc[:split], available.iloc[split:]
            if valid.empty:
                self.metrics[output] = ResponseMetric(output, np.nan, np.nan, len(available), False)
                continue
            x_train, x_valid = self._frame(train, self.feature_columns), self._frame(valid, self.feature_columns)
            y_train = pd.to_numeric(train[output], errors="coerce")
            y_valid = pd.to_numeric(valid[output], errors="coerce")
            low = GradientBoostingRegressor(loss="quantile", alpha=0.1, random_state=17)
            mid = GradientBoostingRegressor(loss="squared_error", random_state=17)
            high = GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=17)
            low.fit(x_train, y_train)
            mid.fit(x_train, y_train)
            high.fit(x_train, y_train)
            low_p, mid_p, high_p = low.predict(x_valid), mid.predict(x_valid), high.predict(x_valid)
            mae = float(mean_absolute_error(y_valid, mid_p))
            coverage = float(((y_valid >= low_p) & (y_valid <= high_p)).mean())
            threshold = 2.0 if output in TEMPERATURE_OUTPUTS else 5.0
            self.metrics[output] = ResponseMetric(output, mae, coverage, len(available), mae <= threshold and coverage >= 0.8)
            # Refit validated artifacts using all data, preserving the same hyperparameters.
            x_all, y_all = self._frame(available, self.feature_columns), pd.to_numeric(available[output], errors="coerce")
            for model in (low, mid, high):
                model.fit(x_all, y_all)
            self.models[output] = (low, mid, high)
        return self.metrics

    @property
    def approved(self) -> bool:
        return bool(self.metrics) and all(self.metrics.get(output, ResponseMetric(output, np.nan, np.nan, 0, False)).passed for output in PROCESS_OUTPUTS)

    def predict(self, vector_features: pd.DataFrame) -> pd.DataFrame:
        """Return P10/P50/P90 actual-process predictions for one or more vectors."""
        if not self.approved:
            raise RuntimeError("CONTROL_TRANSFER_UNAVAILABLE")
        x = self._frame(vector_features, self.feature_columns)
        if x.isna().any().any():
            raise ValueError("Process-response feature vector contains missing values.")
        result = pd.DataFrame(index=vector_features.index)
        for output in PROCESS_OUTPUTS:
            low, mid, high = self.models[output]
            result[f"{output}_p10"] = low.predict(x)
            result[f"{output}_p50"] = mid.predict(x)
            result[f"{output}_p90"] = high.predict(x)
        return result


def flatten_step_targets(steps: pd.DataFrame) -> pd.DataFrame:
    """Flatten one complete Step vector into stable process-model feature names."""
    required = {"GroupName", "StepNo", "recommended_dur_SP", "recommended_rotor_SP", "recommended_temp_SP"}
    missing = required - set(steps.columns)
    if missing:
        raise ValueError(f"Missing process-response vector fields: {sorted(missing)}")
    values: dict[str, float] = {}
    for _, row in steps.sort_values(["GroupName", "StepNo"], kind="stable").iterrows():
        prefix = f"{row['GroupName']}_S{int(row['StepNo'])}"
        for parameter, column in (("duration", "recommended_dur_SP"), ("rotor", "recommended_rotor_SP"), ("temp", "recommended_temp_SP")):
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                values[f"{prefix}_{parameter}"] = float(value)
    return pd.DataFrame([values])


def process_predictions_to_v37_features(base_features: pd.DataFrame, process_prediction: pd.DataFrame) -> pd.DataFrame:
    """Overlay predicted actual endpoints onto the V3.7 Phase feature contract."""
    if len(base_features) != 1 or len(process_prediction) != 1:
        raise ValueError("Exactly one baseline feature row and one process prediction are required.")
    out = base_features.copy()
    predicted = process_prediction.iloc[0]
    mapping = {
        "Stage2_DryMixing_temp_Mean": "dry_end_temp_actual_p50",
        "Stage3_OilLoading_temp_Mean": "post_oil_temp_actual_p50",
        "Stage4_WetMixing_temp_Mean": "wet_end_temp_actual_p50",
        "Stage4_WetMixing_Duration": "wet_duration_actual_p50",
        "Stage6_BottomMixing_temp_Mean": "discharge_temp_actual_p50",
    }
    for feature, response_column in mapping.items():
        if feature in out.columns and response_column in predicted.index and np.isfinite(predicted[response_column]):
            out.at[out.index[0], feature] = float(predicted[response_column])
    return out
