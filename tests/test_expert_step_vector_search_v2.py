import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.optimization.expert_process_response import flatten_step_targets
from Mooney_Prediction_Pipeline.optimization.expert_shadow_policy import EndpointBand
from Mooney_Prediction_Pipeline.optimization.expert_step_vector_search_v2 import search_expert_step_vectors


class _Response:
    approved = True

    def predict(self, vector):
        temp = float(vector.iloc[0]['AVR_MST_S4_temp'])
        return pd.DataFrame([{
            'dry_end_temp_actual_p10': 100., 'dry_end_temp_actual_p50': 100., 'dry_end_temp_actual_p90': 100.,
            'post_oil_temp_actual_p10': 100., 'post_oil_temp_actual_p50': 100., 'post_oil_temp_actual_p90': 100.,
            'wet_end_temp_actual_p10': temp, 'wet_end_temp_actual_p50': temp, 'wet_end_temp_actual_p90': temp,
            'wet_duration_actual_p10': 20., 'wet_duration_actual_p50': 20., 'wet_duration_actual_p90': 20.,
            'discharge_temp_actual_p10': 150., 'discharge_temp_actual_p50': 150., 'discharge_temp_actual_p90': 150.,
        }])


class _Raw:
    feature_columns = ['Stage4_WetMixing_temp_Mean']

    def predict_raw(self, features):
        return SimpleNamespace(final_prediction=np.array([200. - float(features.iloc[0]['Stage4_WetMixing_temp_Mean'])]))


class ExpertStepVectorSearchV2Test(unittest.TestCase):
    def test_wet_temperature_vector_respects_process_cap_and_reaches_v37(self):
        steps = pd.DataFrame([{
            'OrderID': 'O', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 4,
            'time_SP': 20., 'rotor_SP': 40., 'ram_SP': 0., 'temp_SP': 148.,
            'mapped_phase': 'Stage4_WetMixing_Ramp',
            'is_time_settable': True, 'is_rotor_settable': True, 'is_ram_settable': False, 'is_temp_settable': True,
            'step_time_min': 10., 'step_time_max': 40., 'step_rotor_min': 5., 'step_rotor_max': 50.,
            'step_temp_min': 140., 'step_temp_max': 150.,
        }])
        bands = {
            'dry_end_temp_actual': EndpointBand(100., 90., 110., 80., 120., 200),
            'post_oil_temp_actual': EndpointBand(100., 90., 110., 80., 115., 200),
            'wet_end_temp_actual': EndpointBand(149., 140., 150., 130., 150., 200),
            'wet_duration_actual': EndpointBand(20., 10., 30., 5., 35., 200),
        }
        result = search_expert_step_vectors(
            steps, pd.Series({'CompoundName': 'C', 'material_system': 'CarbonBlack'}), _Raw(), _Response(),
            material_system='CarbonBlack', process_target_temperature_cap=150., endpoint_bands=bands,
            spec_center=50., spec_low=45., spec_high=55.,
        )
        self.assertEqual(result.status, 'MANUAL_REVIEW_ONLY')
        self.assertTrue((result.audit_table['model_path_status'] == 'MODEL_PATH_OK').any())
        self.assertTrue((result.audit_table['wet_temperature_cap_status'] == 'PASSED').all())


if __name__ == '__main__':
    unittest.main()
