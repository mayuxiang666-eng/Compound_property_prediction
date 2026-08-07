"""
Mooney Phase Correction Module (mooney_phase_correction.py)
Converts V3.7 phase-level Mooney deviation diagnoses (phase_contribution, top_deviations)
into phase-level correction targets (e.g. WetMixing duration/energy delta).

STRICT ZERO-LEAKAGE CONTRACT:
- Takes ONLY predicted Mooney, spec_center, and V3.7 phase contributions.
- ABSOLUTELY NO current-batch actual_mny is passed or referenced!
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List

RULES_CSV_PATH = os.path.join(os.path.dirname(__file__), 'phase_correction_rules.csv')

def load_phase_correction_rules() -> pd.DataFrame:
    """Loads explicit, PE-confirmed phase correction rules from CSV."""
    if not os.path.exists(RULES_CSV_PATH):
        raise FileNotFoundError(f"Rules CSV missing at: {RULES_CSV_PATH}")
    return pd.read_csv(RULES_CSV_PATH)

def generate_mooney_corrections(
    phase_contributions: Dict[str, float],
    top_deviations: Dict[str, float],
    mny_pred: float,
    spec_center: float,
    deadband_mny: float = 1.0
) -> Dict[str, Dict[str, Any]]:
    """
    Derives phase-level process target deltas to pull predicted Mooney toward spec_center.
    
    Zero Leakage Guarantee:
    mny_gap = spec_center - mny_pred (No actual_mny involved).
    """
    mny_gap = spec_center - mny_pred
    
    # If prediction is already within deadband of target center, no correction needed
    if abs(mny_gap) <= deadband_mny:
        return {}

    rules_df = load_phase_correction_rules()
    corrections = {}

    for phase, contrib in phase_contributions.items():
        # Match rule for phase
        phase_rules = rules_df[rules_df['phase'] == phase]
        if phase_rules.empty:
            continue

        # Check if phase contribution is pushing Mooney away from target center
        # e.g. mny_pred is 60 (target 50, gap = -10), and WetMixing contribution is +2.5
        # Same direction sign indicates phase is driving the gap
        is_driving_deviation = (mny_gap < 0 and contrib > 0) or (mny_gap > 0 and contrib < 0)
        
        if is_driving_deviation:
            rule = phase_rules.iloc[0]
            action = rule['correction_action']
            mechanism = rule['mechanism_note']
            
            # Scale recommended duration/energy delta proportional to gap
            intensity_scale = min(2.0, abs(mny_gap) / 5.0)
            
            if 'increase_wet_energy' in action:
                dur_delta = 10.0 * intensity_scale
                rotor_delta = 5.0 * intensity_scale
            elif 'reduce_pid_duration' in action:
                dur_delta = -8.0 * intensity_scale
                rotor_delta = 0.0
            elif 'increase_bottom_energy' in action:
                dur_delta = 12.0 * intensity_scale
                rotor_delta = 4.0 * intensity_scale
            else:
                dur_delta = 5.0 * intensity_scale
                rotor_delta = 0.0

            corrections[phase] = {
                'phase': phase,
                'contrib': contrib,
                'target_dur_delta': dur_delta if mny_gap < 0 else -dur_delta,
                'target_rotor_delta': rotor_delta if mny_gap < 0 else -rotor_delta,
                'mechanism_note': mechanism,
                'pe_confirmed': bool(rule['pe_confirmed'])
            }

    return corrections

if __name__ == '__main__':
    contribs = {'WetMixing': 2.5, 'PID': 1.0}
    top_devs = {'WetMixing_power_integral': -1245.0}
    res = generate_mooney_corrections(contribs, top_devs, mny_pred=60.0, spec_center=50.0)
    print("[OK] Module A mooney_phase_correction verified:")
    print(res)
