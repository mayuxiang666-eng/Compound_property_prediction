"""
V4.2 Master Integration Runner
Executes the end-to-end V4.2 Step-Phase Architecture Pipeline.
"""
import os
import sys
import pandas as pd
import numpy as np

# Path bootstrap
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags
from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping
from V4_2_Step_Phase_Architecture.feature_engineering.aggregate_sp_to_phase import aggregate_steps_to_phase_features
from V4_2_Step_Phase_Architecture.model_training.phase_model_adapter import PhaseModelAdapter
from V4_2_Step_Phase_Architecture.optimization.phase_to_step_decomposer import decompose_phase_to_steps
from V4_2_Step_Phase_Architecture.optimization.roundtrip_validator import validate_step_candidates
from V4_2_Step_Phase_Architecture.output.output_builder import build_step_recommendation_output

from V4_1_Energy_Optimization.model_training.energy_model import MixingEnergyPredictionModel
from Mooney_Prediction_Pipeline.model_training.hybrid_unified_model import HybridUnifiedMooneyModel

def run_v42_master_pipeline(order_id: str, batch_number: int = 1) -> pd.DataFrame:
    """
    Runs end-to-end V4.2 pipeline for a given OrderID and BatchNumber.
    """
    print(f"================================================================================")
    print(f"RUNNING V4.2 MASTER PIPELINE | Order: {order_id} | Batch: {batch_number}")
    print(f"================================================================================")

    # 1. SQL Connect
    conn = get_sfe_connection()

    # 2. Pivot BatchData
    print("[1/7] Pivoting BatchData EAV to step-wide table...")
    df_wide = pivot_batchdata_to_step_wide(conn, order_ids=[order_id], batch_numbers=[batch_number])
    if df_wide.empty:
        raise ValueError(f"No BatchData found for OrderID={order_id}, BatchNumber={batch_number}")

    # 3. Settable Flags
    print("[2/7] Building settable flags & SP-AV tracking metrics...")
    df_flags = build_settable_flags(df_wide)

    # 4. Expert Step-to-Phase Mapping
    print("[3/7] Applying expert step-to-phase mapping...")
    df_mapped = build_step_phase_mapping(df_flags, target_temp=165.0)

    # 5. Phase Feature Aggregation
    print("[4/7] Aggregating step SP values into phase features...")
    df_phase = aggregate_steps_to_phase_features(df_mapped)

    # 6. Load V4.1 Models & Adapter
    print("[5/7] Initializing V4.1 Phase Model Adapter...")
    energy_model = MixingEnergyPredictionModel(mode='mode_b')
    mooney_model = HybridUnifiedMooneyModel()
    adapter = PhaseModelAdapter(energy_model, mooney_model)

    # 7. Phase-to-Step Decomposition & Validation
    print("[6/7] Decomposing phase targets to step candidates & validating...")
    target_durations = {'Stage5_PID_Duration': df_phase['Stage5_PID_Duration'].iloc[0]}
    df_decomp = decompose_phase_to_steps(target_durations, df_mapped)
    
    recipe_context = pd.Series({'CompoundName': 'M1-T15760', 'material_system': 'Silica'})
    df_validated = validate_step_candidates(df_decomp, target_durations, adapter, None, recipe_context)

    # 8. Build Final Recommendation Output
    print("[7/7] Building engineer-executable recommendation table...")
    df_output = build_step_recommendation_output(df_validated)

    conn.close()
    
    out_file = os.path.join(root_dir, 'V4_2_Step_Phase_Architecture', 'recipe_step_level_recommendation.csv')
    df_output.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"\n[SUCCESS] Pipeline Execution Complete! Saved output to:")
    print(f" -> {out_file}\n")

    return df_output

if __name__ == '__main__':
    df_rec = run_v42_master_pipeline(order_id='7951751', batch_number=75)
    print(df_rec[['GroupName', 'StepNo', 'mapped_phase', 'current_time_SP', 'recommended_time_SP', 'recommendation_status']].to_string(index=False))
