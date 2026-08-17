"""Conformally calibrated route-aware process-response intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from Mooney_Prediction_Pipeline.optimization.expert_process_response import PROCESS_OUTPUTS, TEMPERATURE_OUTPUTS, ResponseMetric
from Mooney_Prediction_Pipeline.optimization.expert_process_response_v2 import RouteAwareProcessResponseModel


class CalibratedRouteAwareProcessResponseModel(RouteAwareProcessResponseModel):
    """Uses chronological residual calibration to make interval coverage auditable."""

    def __init__(self, feature_columns: tuple[str, ...], *, required_outputs: tuple[str, ...] = PROCESS_OUTPUTS):
        super().__init__(feature_columns, required_outputs=required_outputs)
        self.conformal_radius: dict[str, float] = {}

    def fit_validate(self, history: pd.DataFrame, *, time_column: str, minimum_samples: int = 150, validation_fraction: float = 0.2) -> dict[str, ResponseMetric]:
        super().fit_validate(history, time_column=time_column, minimum_samples=minimum_samples, validation_fraction=validation_fraction)
        ordered = history.sort_values(time_column, kind="stable").copy()
        for output in self.required_outputs:
            available = ordered.dropna(subset=[*self.feature_columns, output])
            if len(available) < minimum_samples:
                continue
            split = max(1, int(len(available) * (1.0 - validation_fraction)))
            train, valid = available.iloc[:split], available.iloc[split:]
            if valid.empty:
                continue
            model = GradientBoostingRegressor(loss="squared_error", random_state=17)
            model.fit(self._frame(train, self.feature_columns), pd.to_numeric(train[output], errors="coerce"))
            observed = pd.to_numeric(valid[output], errors="coerce").to_numpy(dtype=float)
            predicted = model.predict(self._frame(valid, self.feature_columns))
            radius = float(np.quantile(np.abs(observed - predicted), 0.90))
            coverage = float((np.abs(observed - predicted) <= radius).mean())
            mae = float(mean_absolute_error(observed, predicted))
            threshold = 2.0 if output in TEMPERATURE_OUTPUTS else 5.0
            self.conformal_radius[output] = radius
            self.metrics[output] = ResponseMetric(output, mae, coverage, len(available), mae <= threshold and coverage >= 0.8)
        return self.metrics

    def predict(self, vector_features: pd.DataFrame) -> pd.DataFrame:
        result = super().predict(vector_features)
        for output, radius in self.conformal_radius.items():
            center = result[f"{output}_p50"]
            result[f"{output}_p10"] = center - radius
            result[f"{output}_p90"] = center + radius
        return result
