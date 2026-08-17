import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.optimization.counterfactual_temperature_search_v3 import (
    search_counterfactual_temperature_candidates,
)


class _Predictor:
    feature_columns = ['Stage4_WetMixing_temp_Mean']

    def predict_raw(self, features):
        return SimpleNamespace(final_prediction=np.array([
            float(features.iloc[0]['Stage4_WetMixing_temp_Mean']) / 10.0,
        ]))


class CounterfactualTemperatureSearchV3Test(unittest.TestCase):
    def test_temperature_candidate_reaches_v37_and_is_scored(self):
        steps = pd.DataFrame([{
            'OrderID': 'O1', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 7,
            'time_SP': 10.0, 'temp_SP': 148.0, 'rotor_SP': 40.0, 'ram_SP': 100.0,
            'mapped_phase': 'Stage4_WetMixing_Ramp',
            'is_time_settable': True, 'is_temp_settable': True,
            'is_rotor_settable': True, 'is_ram_settable': True,
            'step_time_min': 5.0, 'step_time_max': 15.0,
            'step_temp_min': 140.0, 'step_temp_max': 155.0,
        }])
        result = search_counterfactual_temperature_candidates(
            steps, pd.Series({'CompoundName': 'C'}), _Predictor(),
            target_phase='Stage4_WetMixing_Ramp', expected_temp_direction=1,
            target_center=15.0, spec_lower=0.0, spec_upper=20.0,
            temperature_increment_c=1.0,
        )
        self.assertFalse(result.audit_table.empty)
        self.assertTrue((result.audit_table['model_path_status'] == 'MODEL_PATH_OK').all())
        self.assertIn('Stage4_WetMixing_temp_Mean', result.audit_table['changed_v37_input_columns'].iloc[0])
        self.assertNotEqual(result.audit_table['model_response_delta'].iloc[0], 0.0)


if __name__ == '__main__':
    unittest.main()
