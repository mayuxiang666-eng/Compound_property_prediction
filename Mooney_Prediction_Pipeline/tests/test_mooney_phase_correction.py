"""
Unit Tests for Module A: mooney_phase_correction.py
Validates zero leakage contract, rule mapping, and deadband handling.
"""
import os, sys
import pandas as pd
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Mooney_Prediction_Pipeline.optimization.mooney_phase_correction import (
    load_phase_correction_rules,
    generate_mooney_corrections
)

def test_zero_leakage_contract():
    """Verify function signature has NO actual_mny parameter."""
    import inspect
    sig = inspect.signature(generate_mooney_corrections)
    params = list(sig.parameters.keys())
    assert 'actual_mny' not in params
    assert 'current_batch_mny' not in params

def test_rule_mapping_wet_mixing():
    """Verify WetMixing high contribution triggers increase_wet_energy rule."""
    contribs = {'WetMixing': 3.0}
    top_devs = {'WetMixing_power_integral': -1000.0}
    corrections = generate_mooney_corrections(
        phase_contributions=contribs,
        top_deviations=top_devs,
        mny_pred=62.0,
        spec_center=50.0,
        deadband_mny=1.0
    )
    assert 'WetMixing' in corrections
    assert corrections['WetMixing']['target_dur_delta'] > 0
    assert '湿混剪切分散不足' in corrections['WetMixing']['mechanism_note']

def test_normal_batch_small_correction():
    """Verify batch near target center within deadband outputs zero correction."""
    contribs = {'WetMixing': 0.2, 'PID': 0.1}
    top_devs = {}
    corrections = generate_mooney_corrections(
        phase_contributions=contribs,
        top_deviations=top_devs,
        mny_pred=50.4,
        spec_center=50.0,
        deadband_mny=1.0
    )
    assert len(corrections) == 0

if __name__ == '__main__':
    test_zero_leakage_contract()
    test_rule_mapping_wet_mixing()
    test_normal_batch_small_correction()
    print("[PASS] All Unit Tests for Module A passed successfully!")
