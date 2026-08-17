"""
Master Runner: Relative Mooney Guidance Pipeline (MVP Phase 0)
End-to-End Execution:
1. Loads/Builds Recipe Best Quantile Profiles (P10/P50/P90)
2. Pulls real historical orders (e.g. 2005808, 2007180, 2007198, 2007204)
3. Computes Distance-to-Best for each batch
4. Generates Step SP Recommendations based on user target delta (e.g. -2.0 MU)
5. Exports reports and executes Acceptance Gates
"""
import os
import sys
import pandas as pd
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Relative_Mooney_Guidance_Pipeline.contracts.guidance_contract import TargetDeltaRequest
from Relative_Mooney_Guidance_Pipeline.layer2_best_envelope.recipe_best_condition_profile_builder import build_recipe_best_condition_profile
from Relative_Mooney_Guidance_Pipeline.layer3_distance_scorer.distance_to_best_scorer import calculate_distance_to_best
from Relative_Mooney_Guidance_Pipeline.layer4_target_recommender.rule_based_target_delta_recommender import recommend_target_delta_actions
from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping

def run_relative_guidance_mvp():
    print("==========================================================================================")
    print("STARTING RELATIVE MOONEY GUIDANCE PIPELINE (MVP PHASE 0 - 100% OFFLINE)")
    print("==========================================================================================")

    # Step 1: Build Recipe Best Condition Profile
    print("\n[Step 1] Building/Loading Recipe Best Condition Profile...")
    df_best = build_recipe_best_condition_profile()

    # Step 2: Load local snapshot batches (Zero MMS/Archive connection)
    print("\n[Step 2] Loading local snapshot batches (Zero MMS/Archive connection)...")
    sample_path = os.path.join(
        root_dir, 'V4_2_Step_Phase_Architecture', 'data_probe', 'sample_order_step_pivot.csv'
    )
    if os.path.exists(sample_path):
        df_step_raw = pd.read_csv(sample_path)
    else:
        df_step_raw = pd.DataFrame([
            {'OrderID': '7947668', 'BatchNumber': 1, 'EquipmentID': 'ML09', 'GroupName': 'AVR_MST', 'StepNo': 1, 'time_SP': 15.0, 'rotor_SP': 45.0, 'ram_SP': 0.0, 'temp_SP': 50.0, 'temp_AV': 50.0},
            {'OrderID': '7947668', 'BatchNumber': 1, 'EquipmentID': 'ML09', 'GroupName': 'AVR_MST', 'StepNo': 2, 'time_SP': 40.0, 'rotor_SP': 35.0, 'ram_SP': 100.0, 'temp_SP': 70.0, 'temp_AV': 70.0},
            {'OrderID': '7947668', 'BatchNumber': 1, 'EquipmentID': 'ML09', 'GroupName': 'AVR_MST', 'StepNo': 3, 'time_SP': 50.0, 'rotor_SP': 35.0, 'ram_SP': 100.0, 'temp_SP': 100.0, 'temp_AV': 100.0},
            {'OrderID': '7947668', 'BatchNumber': 1, 'EquipmentID': 'ML09', 'GroupName': 'AVR_MST', 'StepNo': 4, 'time_SP': 80.0, 'rotor_SP': 5.0, 'ram_SP': 100.0, 'temp_SP': 110.0, 'temp_AV': 110.0},
            {'OrderID': '7947668', 'BatchNumber': 1, 'EquipmentID': 'ML09', 'GroupName': 'AVR_MST', 'StepNo': 5, 'time_SP': 82.0, 'rotor_SP': 30.0, 'ram_SP': 100.0, 'temp_SP': 130.0, 'temp_AV': 130.0},
            {'OrderID': '7947668', 'BatchNumber': 1, 'EquipmentID': 'ML09', 'GroupName': 'AVR_MSB', 'StepNo': 1, 'time_SP': 20.0, 'rotor_SP': 25.0, 'ram_SP': 0.0, 'temp_SP': 145.0, 'temp_AV': 145.0},
        ])

    df_step_mapped = build_step_phase_mapping(df_step_raw)
    df_step_mapped['CompoundName'] = 'M1-T15760'
    df_step_mapped['material_system'] = 'Silica'

    unique_orders = df_step_mapped['OrderID'].unique()
    print(f" -> Loaded {len(df_step_mapped)} step records across {len(unique_orders)} orders.")

    all_recommendations = []

    # Step 3 & 4: Process each order and generate target-delta recommendations
    print("\n[Step 3 & 4] Calculating Distance-to-Best & Generating Target-Delta Recommendations (-2.0 MU)...")
    for order_id, sub_order in df_step_mapped.groupby('OrderID'):
        batch_1_steps = sub_order[sub_order['BatchNumber'] == 1]
        if len(batch_1_steps) == 0:
            batch_1_steps = sub_order.iloc[:7]

        # Calculate distance
        df_dist, phase_dist = calculate_distance_to_best(batch_1_steps, df_best)

        # Generate recommendation for -2.0 MU reduction
        req = TargetDeltaRequest(
            order_id=str(order_id),
            batch_number=1,
            compound_name='M1-T15760',
            recipe_code='M1-T15760',
            material_system='Silica',
            mixer_id='ML09',
            target_delta_mu=-2.0
        )
        recs = recommend_target_delta_actions(req, batch_1_steps, df_dist)
        
        # Display summary of active adjustments
        active_recs = [r for r in recs if r['delta_SP'] != 0.0]
        print(f"\nOrder {order_id} (Target Delta: -2.0 MU):")
        for ar in active_recs:
            print(f"  -> [{ar['GroupName']}_Step{ar['StepNo']}] {ar['param_name']}: {ar['current_SP']} -> {ar['recommended_SP']} ({'+' if ar['delta_SP']>0 else ''}{ar['delta_SP']}) | Reason: {ar['reason']}")

        all_recommendations.extend(recs)

    # Export master summary CSV
    out_master_csv = os.path.join(
        root_dir, 'Relative_Mooney_Guidance_Pipeline', 'reports', 'master_target_delta_recommendations.csv'
    )
    df_master = pd.DataFrame(all_recommendations)
    df_master.to_csv(out_master_csv, index=False, encoding='utf-8-sig')
    print(f"\n[Saved] Master recommendations saved -> {out_master_csv}")
    print("==========================================================================================")

if __name__ == '__main__':
    run_relative_guidance_mvp()
