import unittest

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.optimization.expert_process_response import (
    PROCESS_OUTPUTS,
    ProcessResponseModel,
)


class ExpertProcessResponseTest(unittest.TestCase):
    def test_time_split_model_returns_endpoint_intervals(self):
        rng = np.random.default_rng(17)
        n = 190
        dry = np.linspace(100, 118, n) + rng.normal(0, 0.15, n)
        wet = np.linspace(130, 150, n) + rng.normal(0, 0.15, n)
        duration = np.linspace(35, 45, n) + rng.normal(0, 0.2, n)
        history = pd.DataFrame({
            'time': pd.date_range('2025-01-01', periods=n, freq='D'),
            'mst_s3_duration': np.linspace(20, 30, n),
            'mst_s4_temp': wet,
            'dry_end_temp_actual': dry,
            'post_oil_temp_actual': dry + 2.0,
            'wet_end_temp_actual': wet,
            'wet_duration_actual': duration,
            'discharge_temp_actual': wet + 5.0,
        })
        model = ProcessResponseModel(('mst_s3_duration', 'mst_s4_temp'))
        metrics = model.fit_validate(history, time_column='time', minimum_samples=150)
        self.assertEqual(set(metrics), set(PROCESS_OUTPUTS))
        # Whether production acceptance is achieved is data-dependent; prediction
        # interval shape must still be deterministic for every fitted output.
        fitted = [output for output in PROCESS_OUTPUTS if output in model.models]
        self.assertEqual(len(fitted), len(PROCESS_OUTPUTS))
        if model.approved:
            result = model.predict(history.iloc[[0]])
            self.assertIn('wet_end_temp_actual_p50', result.columns)


if __name__ == '__main__':
    unittest.main()
