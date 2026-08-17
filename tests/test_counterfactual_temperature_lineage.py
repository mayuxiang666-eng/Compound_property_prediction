import unittest

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.optimization.action_feature_lineage_v2 import build_action_feature_lineage_v2
from Mooney_Prediction_Pipeline.optimization.counterfactual_phase_features import merge_counterfactual_features


class CounterfactualTemperatureLineageTest(unittest.TestCase):
    def test_temperature_target_changes_stage4_v37_input(self):
        base = pd.DataFrame([{
            'OrderID': 'O1', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 7,
            'time_SP': 10.0, 'temp_SP': 148.0, 'rotor_SP': 40.0, 'ram_SP': 100.0,
            'is_temp_settable': True, 'is_rotor_settable': True, 'is_ram_settable': True,
            'mapped_phase': 'Stage4_WetMixing_Ramp', 'dur_SP': 10.0,
            'recommended_time_SP': 10.0, 'recommended_dur_SP': 10.0,
            'recommended_temp_SP': 148.0, 'recommended_rotor_SP': 40.0,
            'recommended_ram_SP': 100.0,
        }])
        candidate = base.copy()
        candidate.loc[0, 'recommended_temp_SP'] = 149.0
        baseline_row = pd.Series({
            'CompoundName': 'C', 'material_system': 'CarbonBlack',
            'Stage4_WetMixing_temp_Mean': 0.0,
        })
        before = merge_counterfactual_features(base, baseline_row)
        after = merge_counterfactual_features(candidate, baseline_row)
        records = build_action_feature_lineage_v2(
            base, candidate, before, after, ['Stage4_WetMixing_temp_Mean'],
        )
        self.assertEqual(records[0]['model_path_status'], 'MODEL_PATH_OK')
        self.assertEqual(records[0]['changed_v37_input_columns'], ['Stage4_WetMixing_temp_Mean'])
        self.assertEqual(after.iloc[0]['Stage4_WetMixing_temp_Mean'], 149.0)

    def test_unmapped_temperature_target_is_no_model_path(self):
        base = pd.DataFrame([{
            'OrderID': 'O1', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 99,
            'time_SP': 10.0, 'temp_SP': 148.0, 'rotor_SP': 40.0, 'ram_SP': 100.0,
            'is_temp_settable': True, 'is_rotor_settable': True, 'is_ram_settable': True,
            'mapped_phase': 'Other', 'dur_SP': 10.0,
            'recommended_time_SP': 10.0, 'recommended_dur_SP': 10.0,
            'recommended_temp_SP': 148.0, 'recommended_rotor_SP': 40.0,
            'recommended_ram_SP': 100.0,
        }])
        candidate = base.copy()
        candidate.loc[0, 'recommended_temp_SP'] = 149.0
        row = pd.Series({'CompoundName': 'C', 'material_system': 'CarbonBlack'})
        records = build_action_feature_lineage_v2(
            base, candidate, merge_counterfactual_features(base, row),
            merge_counterfactual_features(candidate, row), [],
        )
        self.assertEqual(records[0]['model_path_status'], 'NO_MODEL_PATH')


if __name__ == '__main__':
    unittest.main()
