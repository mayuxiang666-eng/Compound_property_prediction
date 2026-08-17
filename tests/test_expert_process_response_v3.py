import unittest

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.optimization.expert_process_response_v3 import CalibratedRouteAwareProcessResponseModel


class ExpertProcessResponseV3Test(unittest.TestCase):
    def test_calibrated_model_exposes_chronological_interval_radius(self):
        rng = np.random.default_rng(9)
        n = 180
        x = np.linspace(0., 20., n)
        history = pd.DataFrame({'ts': pd.date_range('2025-01-01', periods=n, freq='D'), 'x': x})
        for output, offset in [('dry_end_temp_actual',100.),('post_oil_temp_actual',105.),('wet_end_temp_actual',140.),('wet_duration_actual',30.),('discharge_temp_actual',160.)]:
            history[output] = offset + .5 * x + rng.normal(0, .2, n)
        model = CalibratedRouteAwareProcessResponseModel(('x',))
        model.fit_validate(history, time_column='ts', minimum_samples=150)
        self.assertEqual(set(model.conformal_radius), set(model.required_outputs))
        if model.approved:
            output = model.predict(pd.DataFrame({'x':[10.]}))
            self.assertGreaterEqual(output['wet_end_temp_actual_p90'].iloc[0], output['wet_end_temp_actual_p50'].iloc[0])


if __name__ == '__main__':
    unittest.main()
