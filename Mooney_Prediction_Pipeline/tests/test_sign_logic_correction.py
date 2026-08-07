"""
Unit Test for Sign Logic: test_wet_mixing_positive_contrib_reduces_mooney
Verifies that when mny_pred > spec_center (mny_gap < 0) and WetMixing contrib > 0,
the recommendation MUST increase wet mixing energy/time and REDUCE predicted Mooney (rec_final < base_final).
"""
import os, sys
import pandas as pd
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Mooney_Prediction_Pipeline.optimization.mooney_phase_correction import generate_mooney_corrections
from Mooney_Prediction_Pipeline.optimization.mooney_step_bridge import evaluate_mooney_step_bridge_dual_arm
from Mooney_Prediction_Pipeline.model_training.hybrid_unified_model import HybridUnifiedMooneyModel

def test_wet_mixing_positive_contrib_reduces_mooney():
    spec_center = 50.0
    mny_pred = 53.8 # Base Mooney is HIGH (53.8 > 50.0)
    contribs = {'WetMixing': 3.8}
    top_devs = {'WetMixing_power_integral': -1245.0}

    # 1. Verify target_dur_delta is POSITIVE (+dur) to extend wet mixing
    corrections = generate_mooney_corrections(contribs, top_devs, mny_pred=mny_pred, spec_center=spec_center)
    assert 'WetMixing' in corrections
    assert corrections['WetMixing']['target_dur_delta'] > 0, f"Expected positive duration delta to extend wet mixing, got: {corrections['WetMixing']['target_dur_delta']}"

    # 2. Dual-Arm Evaluation check: rec_final MUST be lower than base_final!
    dummy_step_mapped = pd.DataFrame([
        {'OrderID': '1', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 5, 'mapped_phase': 'Stage4_WetMixing_Ramp', 'time_SP': 80.0, 'rotor_SP': 25.0, 'ram_SP': 100.0, 'temp_SP': 100.0, 'temp_AV': 100.0, 'is_time_settable': True, 'is_rotor_settable': True, 'is_ram_settable': True, 'is_temp_settable': False}
    ])
    mooney_model = HybridUnifiedMooneyModel()
    context = pd.Series({'CompoundName': 'M1-T15760', 'material_system': 'Silica'})

    df_rec_steps, summary = evaluate_mooney_step_bridge_dual_arm(
        dummy_step_mapped, corrections, mooney_model, context, spec_center=spec_center
    )

    print(f"Baseline MNY: {summary['base_final']} | Recommended MNY: {summary['rec_final']} | Pull: {summary['mny_pull']}")
    assert summary['rec_final'] < summary['base_final'], f"Recommendation failed to pull Mooney down! base={summary['base_final']}, rec={summary['rec_final']}"
    assert summary['pull_effective'] == True, "pull_effective should be TRUE when Mooney is pulled toward target center!"
    print("[PASS] test_wet_mixing_positive_contrib_reduces_mooney passed successfully!")

if __name__ == '__main__':
    test_wet_mixing_positive_contrib_reduces_mooney()
