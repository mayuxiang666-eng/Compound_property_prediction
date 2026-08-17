import unittest

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.expert_shadow_policy import (
    DRY_TEMPERATURE_CAP_C,
    endpoint_band,
    identify_discharge_temperature_link,
    policy_reasons,
    prepare_recommendation_vector,
    refresh_recommended_time,
)


def _steps():
    raw = pd.DataFrame([
        {'OrderID': 'O', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 1, 'time_SP': 10., 'rotor_SP': 40., 'ram_SP': 0., 'temp_SP': 0., 'temp_AV': 50., 'mapped_phase': 'Stage1_Loading'},
        {'OrderID': 'O', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 3, 'time_SP': 30., 'rotor_SP': 40., 'ram_SP': 0., 'temp_SP': 115., 'temp_AV': 110., 'mapped_phase': 'Stage2_DryMixing'},
        {'OrderID': 'O', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 4, 'time_SP': 40., 'rotor_SP': 40., 'ram_SP': 0., 'temp_SP': 148., 'temp_AV': 145., 'mapped_phase': 'Stage4_WetMixing_Ramp'},
        {'OrderID': 'O', 'BatchNumber': 1, 'GroupName': 'AVR_MSB', 'StepNo': 2, 'time_SP': 20., 'rotor_SP': 5., 'ram_SP': 0., 'temp_SP': 160., 'temp_AV': 161., 'mapped_phase': 'Stage6_MSB_PID_Step2'},
        {'OrderID': 'O', 'BatchNumber': 1, 'GroupName': 'AVR_MSB', 'StepNo': 3, 'time_SP': 25., 'rotor_SP': 5., 'ram_SP': 0., 'temp_SP': 0., 'temp_AV': 159., 'mapped_phase': 'Stage6_MSB_PID_Step3'},
    ])
    for parameter, source in [('time', 'time_SP'), ('rotor', 'rotor_SP'), ('ram', 'ram_SP'), ('temp', 'temp_SP')]:
        raw[f'is_{parameter}_settable'] = raw[source].gt(0)
    return prepare_recommendation_vector(raw)


class ExpertShadowPolicyTest(unittest.TestCase):
    def test_discharge_control_uses_last_target_not_last_step(self):
        link = identify_discharge_temperature_link(_steps())
        self.assertIsNotNone(link)
        self.assertEqual(link.control_step_no, 2)
        self.assertEqual(link.final_step_no, 3)
        self.assertEqual(link.target, 160.)
        self.assertEqual(link.actual, 159.)

    def test_dry_cap_and_wet_missing_cap_are_fail_closed(self):
        before = _steps()
        candidate = before.copy()
        dry_idx = candidate.index[candidate['mapped_phase'].eq('Stage2_DryMixing')][0]
        wet_idx = candidate.index[candidate['mapped_phase'].eq('Stage4_WetMixing_Ramp')][0]
        candidate.loc[dry_idx, 'recommended_temp_SP'] = DRY_TEMPERATURE_CAP_C + 1.
        candidate.loc[wet_idx, 'recommended_temp_SP'] += 1.
        reasons = policy_reasons(before, refresh_recommended_time(candidate), material_system='CarbonBlack', process_target_temperature_cap=None)
        self.assertIn('REJECTED_DRY_TEMPERATURE_CAP', reasons)
        self.assertIn('BLOCKED_MISSING_PROCESS_TARGET_TEMPERATURE', reasons)

    def test_endpoint_band_clips_to_process_cap(self):
        band = endpoint_band(range(100, 260), cap=120., minimum_samples=150)
        self.assertIsNotNone(band)
        self.assertLessEqual(band.recommended_high, 120.)
        self.assertLessEqual(band.safe_high, 120.)


if __name__ == '__main__':
    unittest.main()
