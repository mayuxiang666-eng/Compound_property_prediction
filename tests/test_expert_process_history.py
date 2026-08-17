import unittest

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.expert_process_history import actual_process_labels, build_process_response_history


class ExpertProcessHistoryTest(unittest.TestCase):
    def _steps(self):
        rows = []
        for batch, wet_actual in [(1, 28.), (2, 30.)]:
            rows.extend([
                {'OrderID': 'O', 'BatchNumber': batch, 'GroupName': 'AVR_MST', 'StepNo': 3, 'time_SP': 10., 'time_AV': 12., 'rotor_SP': 40., 'ram_SP': 0., 'temp_SP': 110., 'temp_AV': 108., 'mapped_phase': 'Stage2_DryMixing', 'ts': batch},
                {'OrderID': 'O', 'BatchNumber': batch, 'GroupName': 'AVR_MST', 'StepNo': 4, 'time_SP': 30., 'time_AV': wet_actual, 'rotor_SP': 35., 'ram_SP': 0., 'temp_SP': 148., 'temp_AV': 145., 'mapped_phase': 'Stage4_WetMixing_Ramp', 'ts': batch},
                {'OrderID': 'O', 'BatchNumber': batch, 'GroupName': 'AVR_MSB', 'StepNo': 2, 'time_SP': 20., 'time_AV': 21., 'rotor_SP': 5., 'ram_SP': 0., 'temp_SP': 160., 'temp_AV': 159., 'mapped_phase': 'Stage6_MSB_PID_Step2', 'ts': batch},
            ])
        return pd.DataFrame(rows)

    def test_wet_actual_duration_uses_time_av_not_target(self):
        labels = actual_process_labels(self._steps().query('BatchNumber == 1'))
        self.assertEqual(labels['wet_duration_actual'], 16.)
        history, features = build_process_response_history(self._steps(), time_column='ts')
        self.assertEqual(len(history), 2)
        self.assertIn('AVR_MST_S4_temp', features)


if __name__ == '__main__':
    unittest.main()
