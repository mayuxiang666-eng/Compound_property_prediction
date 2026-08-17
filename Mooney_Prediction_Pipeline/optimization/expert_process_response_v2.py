"""Route-aware process-response model entry point for expert shadow search."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.optimization.expert_process_response import PROCESS_OUTPUTS, ProcessResponseModel, ResponseMetric


class RouteAwareProcessResponseModel(ProcessResponseModel):
    """Allows oil-free templates to omit the oil-only thermal-budget endpoint."""

    def __init__(self, feature_columns: tuple[str, ...], *, required_outputs: tuple[str, ...] = PROCESS_OUTPUTS):
        super().__init__(feature_columns)
        unknown = set(required_outputs) - set(PROCESS_OUTPUTS)
        if unknown:
            raise ValueError(f"Unknown process outputs: {sorted(unknown)}")
        self.required_outputs = required_outputs

    @property
    def approved(self) -> bool:
        return bool(self.metrics) and all(self.metrics.get(output, ResponseMetric(output, np.nan, np.nan, 0, False)).passed for output in self.required_outputs)

    def predict(self, vector_features: pd.DataFrame) -> pd.DataFrame:
        if not self.approved:
            raise RuntimeError("CONTROL_TRANSFER_UNAVAILABLE")
        x = self._frame(vector_features, self.feature_columns)
        if x.isna().any().any():
            raise ValueError("Process-response feature vector contains missing values.")
        result = pd.DataFrame(index=vector_features.index)
        for output in PROCESS_OUTPUTS:
            if output not in self.models:
                for suffix in ("p10", "p50", "p90"):
                    result[f"{output}_{suffix}"] = np.nan
                continue
            low, mid, high = self.models[output]
            result[f"{output}_p10"] = low.predict(x)
            result[f"{output}_p50"] = mid.predict(x)
            result[f"{output}_p90"] = high.predict(x)
        return result
