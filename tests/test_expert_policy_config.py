import unittest

from Mooney_Prediction_Pipeline.optimization.expert_policy_config import load_expert_policy


class ExpertPolicyConfigTest(unittest.TestCase):
    def test_reviewed_shadow_policy_loads(self):
        policy = load_expert_policy()
        self.assertEqual(policy['mode'], 'SHADOW_ONLY')
        self.assertEqual(policy['dry_temperature_cap_c'], 120.0)


if __name__ == '__main__':
    unittest.main()
