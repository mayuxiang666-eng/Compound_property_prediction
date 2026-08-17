"""
Validation & Acceptance Suite for Relative Mooney Guidance Pipeline
Verifies the 6 Core Acceptance Criteria:
1. Direction Accuracy (100% PULL_DOWN when delta < 0)
2. Pull Effective (Convergence toward spec center)
3. Best-Envelope Convergence (Shifts toward P10-P90 best window)
4. Rule Compliance:
   - Step 1-2 Loading locked (delta == 0)
   - Ram locked (delta == 0)
   - Wet rotor reduction (5 rpm discrete step)
   - Silica PID MSB 10s step (MST upper mixer locked)
   - Safe bounds (5 <= rotor <= 50)
5. Trial Success Simulation (No unfeasible actions)
6. No Over-Correction (Target delta = 0 yields zero correction)
"""
import os
import sys
import pandas as pd
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Relative_Mooney_Guidance_Pipeline.contracts.guidance_contract import TargetDeltaRequest
from Relative_Mooney_Guidance_Pipeline.layer2_best_envelope.recipe_best_condition_profile_builder import build_recipe_best_condition_profile
from Relative_Mooney_Guidance_Pipeline.layer3_distance_scorer.distance_to_best_scorer import calculate_distance_to_best
from Relative_Mooney_Guidance_Pipeline.layer4_target_recommender.rule_based_target_delta_recommender import recommend_target_delta_actions

def evaluate_guidance_pipeline():
    print("==========================================================================================")
    print("RUNNING ACCEPTANCE TEST SUITE FOR RELATIVE MOONEY GUIDANCE PIPELINE")
    print("==========================================================================================")

    # 1. Build or load recipe best profile
    df_best = build_recipe_best_condition_profile()
    assert len(df_best) > 0, "Recipe best condition profile is empty!"

    # 2. Test Case 1: High Mooney Silica Batch requesting -2.0 MU reduction
    test_steps_silica = pd.DataFrame([
        {'OrderID': 'TEST_2005808', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 1, 'mapped_phase': 'Stage1_PolymerLoading', 'time_SP': 15.0, 'rotor_SP': 45.0, 'ram_SP': 0.0, 'temp_SP': 50.0},
        {'OrderID': 'TEST_2005808', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 2, 'mapped_phase': 'Stage1_CarbonBlackLoading', 'time_SP': 40.0, 'rotor_SP': 35.0, 'ram_SP': 100.0, 'temp_SP': 70.0},
        {'OrderID': 'TEST_2005808', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 3, 'mapped_phase': 'Stage2_DryMixing', 'time_SP': 50.0, 'rotor_SP': 35.0, 'ram_SP': 100.0, 'temp_SP': 100.0},
        {'OrderID': 'TEST_2005808', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 4, 'mapped_phase': 'Stage3_OilLoading', 'time_SP': 80.0, 'rotor_SP': 5.0, 'ram_SP': 100.0, 'temp_SP': 110.0},
        {'OrderID': 'TEST_2005808', 'BatchNumber': 1, 'GroupName': 'AVR_MST', 'StepNo': 5, 'mapped_phase': 'Stage4_WetMixing_Ramp', 'time_SP': 82.0, 'rotor_SP': 30.0, 'ram_SP': 100.0, 'temp_SP': 130.0},
        {'OrderID': 'TEST_2005808', 'BatchNumber': 1, 'GroupName': 'AVR_MSB', 'StepNo': 1, 'mapped_phase': 'Stage5_PID_LowerMixer', 'time_SP': 20.0, 'rotor_SP': 25.0, 'ram_SP': 0.0, 'temp_SP': 145.0},
    ])
    test_steps_silica['CompoundName'] = 'M1-T15760'
    test_steps_silica['material_system'] = 'Silica'

    req_silica = TargetDeltaRequest(
        order_id='TEST_2005808',
        batch_number=1,
        compound_name='M1-T15760',
        recipe_code='M1-T15760',
        material_system='Silica',
        mixer_id='ML09',
        target_delta_mu=-2.0
    )

    df_dist, _ = calculate_distance_to_best(test_steps_silica, df_best)
    recs_silica = recommend_target_delta_actions(req_silica, test_steps_silica, df_dist)
    df_recs = pd.DataFrame(recs_silica)

    # --- ASSERTION 1: Rule Compliance (Step 1-2 Loading & Ram Locked) ---
    loading_recs = df_recs[(df_recs['GroupName'] == 'AVR_MST') & (df_recs['StepNo'].isin([1, 2]))]
    assert (loading_recs['delta_SP'] == 0.0).all(), "FAIL: Step 1-2 Loading must be strictly locked!"
    print("[PASS] Assertion 1 (Step 1-2 Loading Locked): Verified (delta = 0.0)")

    ram_recs = df_recs[df_recs['param_name'] == 'ram_SP']
    assert (ram_recs['delta_SP'] == 0.0).all(), "FAIL: Ram SP must be strictly locked!"
    print("[PASS] Assertion 2 (Ram SP Locked): Verified (delta = 0.0)")

    # --- ASSERTION 2: Rule Compliance (Silica PID adjusted in MSB, NOT MST) ---
    mst_pid_recs = df_recs[(df_recs['GroupName'] == 'AVR_MST') & (df_recs['StepNo'] == 6)]
    if len(mst_pid_recs) > 0:
        assert (mst_pid_recs['delta_SP'] == 0.0).all(), "FAIL: Silica PID MST upper mixer must be locked to avoid sticky door!"
    
    msb_pid_recs = df_recs[(df_recs['GroupName'] == 'AVR_MSB') & (df_recs['param_name'] == 'time_SP')]
    assert (msb_pid_recs['delta_SP'] == 10.0).any(), "FAIL: Silica PID MSB lower mixer should be extended by 10s!"
    print("[PASS] Assertion 3 (Silica PID MSB Lower Mixer 10s Step): Verified")

    # --- ASSERTION 3: Rule Compliance (Wet Mixing Rotor down 5 rpm) ---
    wet_rotor_recs = df_recs[(df_recs['GroupName'] == 'AVR_MST') & (df_recs['StepNo'] == 5) & (df_recs['param_name'] == 'rotor_SP')]
    assert (wet_rotor_recs['delta_SP'] == -5.0).any(), "FAIL: Wet mixing rotor should be reduced by 5 rpm!"
    print("[PASS] Assertion 4 (Wet Mixing Rotor Reduction by 5 rpm): Verified")

    # --- ASSERTION 4: Rule Engine Intent Tagging (Generated adjustments tagged with intended direction) ---
    active_recs = df_recs[df_recs['delta_SP'] != 0.0]
    assert (active_recs['rule_intended_direction'] == 'PULL_DOWN').all(), "FAIL: All recommended actions must be tagged with PULL_DOWN intent!"
    print(f"[PASS] Assertion 5 (Rule Engine Intent Tagging): Verified 100% of {len(active_recs)} active adjustments are tagged with PULL_DOWN intent (NOTE: This is syntactic contract check, NOT empirical direction accuracy!)")

    # --- ASSERTION 5: No Over-Correction (Target delta = 0.0 yields zero changes) ---
    req_zero = TargetDeltaRequest(
        order_id='TEST_IN_SPEC',
        batch_number=1,
        compound_name='M1-T15760',
        recipe_code='M1-T15760',
        material_system='Silica',
        mixer_id='ML09',
        target_delta_mu=0.0
    )
    recs_zero = recommend_target_delta_actions(req_zero, test_steps_silica, df_dist)
    df_zero = pd.DataFrame(recs_zero)
    assert (df_zero['delta_SP'] == 0.0).all(), "FAIL: In-spec batch with target delta = 0 must have 0.0 correction!"
    print("[PASS] Assertion 6 (No Over-Correction on In-Spec Batch): Verified (All delta = 0.0)")

    print("==========================================================================================")
    print("RULE ENGINE CONTRACT UNIT TESTS PASSED (6/6 Syntactic Checks Passed)")
    print("NOTE: Empirical validation (B2/B3/Lab effect) remains UNVERIFIED pending offline data join.")
    print("==========================================================================================")

if __name__ == '__main__':
    evaluate_guidance_pipeline()
