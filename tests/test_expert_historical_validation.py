import unittest

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.expert_historical_validation import (
    discharge_temperature_tracking_audit,
    select_silica_pilot,
)


class ExpertHistoricalValidationTest(unittest.TestCase):
    def test_discharge_tracking_is_cross_step_and_fail_closed(self):
        audit = discharge_temperature_tracking_audit(pd.DataFrame({
            'discharge_temp_target': [160., 161., 160., 162.],
            'discharge_temp_actual': [160., 160., 159., 161.],
        }))
        self.assertEqual(audit['sample_count'], 4)
        self.assertEqual(audit['control_tracking'], 'PASS')

    def test_silica_pilot_selector_requires_all_readiness_gates(self):
        base = {
            'material_system': 'Silica', 'EquipmentID': 'E1', 'route_key': 'R', 'template_key': 'T',
            'n_labeled_batches': 180, 'template_consistency': .95, 'stage6_duration_levels': 3,
            'discharge_temperature_levels': 3, 'temperature_tracking_pass': True, 'endpoint_complete_rate': .99,
        }
        rows = [dict(base, CompoundName='GOOD'), dict(base, CompoundName='BAD', temperature_tracking_pass=False)]
        selected = select_silica_pilot(pd.DataFrame(rows))
        self.assertEqual(selected['CompoundName'].tolist(), ['GOOD'])


if __name__ == '__main__':
    unittest.main()
