import unittest

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.expert_shadow_policy_v2 import (
    DRY_TEMPERATURE_CAP_C,
    endpoint_band,
    policy_reasons,
    prepare_recommendation_vector,
    refresh_recommended_time,
)


class ExpertShadowPolicyV2Test(unittest.TestCase):
    def _steps(self):
        rows = pd.DataFrame([
            {'OrderID': 'O', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 3, 'time_SP': 20., 'rotor_SP': 40., 'ram_SP': 0., 'temp_SP': 115., 'mapped_phase': 'Stage2_DryMixing'},
            {'OrderID': 'O', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 4, 'time_SP': 30., 'rotor_SP': 40., 'ram_SP': 0., 'temp_SP': 148., 'mapped_phase': 'Stage4_WetMixing_Ramp'},
        ])
        for flag, source in [('time', 'time_SP'), ('rotor', 'rotor_SP'), ('ram', 'ram_SP'), ('temp', 'temp_SP')]:
            rows[f'is_{flag}_settable'] = rows[source].gt(0)
        return prepare_recommendation_vector(rows)

    def test_indexed_vector_enforces_both_temperature_caps(self):
        before = self._steps()
        candidate = before.copy()
        candidate.loc[candidate['mapped_phase'].eq('Stage2_DryMixing'), 'recommended_temp_SP'] = DRY_TEMPERATURE_CAP_C + 1.
        candidate.loc[candidate['mapped_phase'].eq('Stage4_WetMixing_Ramp'), 'recommended_temp_SP'] += 1.
        reasons = policy_reasons(before, refresh_recommended_time(candidate), material_system='CarbonBlack', process_target_temperature_cap=None)
        self.assertIn('REJECTED_DRY_TEMPERATURE_CAP', reasons)
        self.assertIn('BLOCKED_MISSING_PROCESS_TARGET_TEMPERATURE', reasons)

    def test_endpoint_band_has_feasible_capped_range(self):
        band = endpoint_band(range(50, 210), cap=120., minimum_samples=150)
        self.assertIsNotNone(band)
        self.assertLessEqual(band.recommended_high, 120.)
        self.assertLessEqual(band.safe_high, 120.)


if __name__ == '__main__':
    unittest.main()
