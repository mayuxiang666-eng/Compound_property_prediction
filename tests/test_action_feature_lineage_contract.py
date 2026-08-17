import unittest

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.action_feature_lineage_contract import (
    build_action_feature_lineage,
)
from Mooney_Prediction_Pipeline.optimization.counterfactual_phase_features import (
    merge_counterfactual_features,
)


class ActionFeatureLineageContractTest(unittest.TestCase):
    def test_each_control_is_attributed_only_to_its_own_v37_columns(self):
        before = pd.DataFrame([{
            'OrderID': 'O1', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 9,
            'time_SP': 10.0, 'temp_SP': 150.0, 'rotor_SP': 40.0, 'ram_SP': 100.0,
            'mapped_phase': 'Stage5_MST_PID_Silanization',
            'is_temp_settable': True, 'is_rotor_settable': True, 'is_ram_settable': True,
            'dur_SP': 10.0, 'recommended_time_SP': 10.0, 'recommended_dur_SP': 10.0,
            'recommended_temp_SP': 150.0, 'recommended_rotor_SP': 40.0, 'recommended_ram_SP': 100.0,
        }])
        candidate = before.copy()
        candidate.loc[0, ['recommended_time_SP', 'recommended_dur_SP']] = 11.0
        candidate.loc[0, 'recommended_temp_SP'] = 151.0
        candidate.loc[0, 'recommended_rotor_SP'] = 41.0
        candidate.loc[0, 'recommended_ram_SP'] = 99.0
        baseline = pd.Series({'CompoundName': 'C', 'material_system': 'Silica'})
        feature_columns = [
            'Stage5_PID_Duration', 'Stage5_PID_temp_Mean', 'Target_Temperature',
            'Stage5_PID_RotorSpeed_Mean', 'Stage5_PID_WayofRam_Mean',
        ]
        records = build_action_feature_lineage(
            before, candidate,
            merge_counterfactual_features(before, baseline),
            merge_counterfactual_features(candidate, baseline),
            feature_columns,
        )
        by_parameter = {row['action_parameter']: row for row in records}
        self.assertEqual(by_parameter['duration']['changed_v37_input_columns'], ['Stage5_PID_Duration'])
        self.assertEqual(by_parameter['temperature']['changed_v37_input_columns'], ['Stage5_PID_temp_Mean', 'Target_Temperature'])
        self.assertEqual(by_parameter['rotor']['changed_v37_input_columns'], ['Stage5_PID_RotorSpeed_Mean'])
        self.assertEqual(by_parameter['ram']['changed_v37_input_columns'], ['Stage5_PID_WayofRam_Mean'])

    def test_unknown_phase_is_fail_closed(self):
        before = pd.DataFrame([{
            'OrderID': 'O1', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 9,
            'time_SP': 10.0, 'temp_SP': 150.0, 'rotor_SP': 40.0, 'ram_SP': 100.0,
            'mapped_phase': 'Unknown', 'dur_SP': 10.0,
            'recommended_time_SP': 10.0, 'recommended_dur_SP': 10.0,
            'recommended_temp_SP': 150.0, 'recommended_rotor_SP': 40.0, 'recommended_ram_SP': 100.0,
        }])
        candidate = before.copy()
        candidate.loc[0, 'recommended_temp_SP'] = 151.0
        baseline = pd.Series({'CompoundName': 'C'})
        records = build_action_feature_lineage(
            before, candidate,
            merge_counterfactual_features(before, baseline),
            merge_counterfactual_features(candidate, baseline),
            [],
        )
        self.assertEqual(records[0]['model_path_status'], 'NO_MODEL_PATH')


if __name__ == '__main__':
    unittest.main()
