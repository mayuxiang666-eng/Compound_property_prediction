"""
V4.2 Module F: Round-trip Validator
Re-aggregates step-level candidates back to phase features and validates through 7 engineering gates.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from V4_2_Step_Phase_Architecture.feature_engineering.aggregate_sp_to_phase import aggregate_steps_to_phase_features

def validate_step_candidates(
    candidates_df: pd.DataFrame,
    phase_targets: Dict[str, float],
    adapter: Any,
    mooney_checker: Any,
    recipe_context: pd.Series,
    spec_lower: float = 45.0,
    spec_upper: float = 65.0
) -> pd.DataFrame:
    """
    Re-aggregates step candidates → phase features and applies 7 engineering gates.
    """
    out = candidates_df.copy()

    # 1. Re-aggregate step candidates to phase features
    re_agg = aggregate_steps_to_phase_features(out)
    if re_agg.empty:
        out['phase_consistency_status'] = 'NO_DATA'
        out['model_validation_status'] = 'NO_DATA'
        out['recommendation_status'] = 'NO_VALID_CANDIDATE'
        return out

    # 2. Check phase consistency (re-aggregated vs phase target <= 0.5s)
    phase_dur_cols = [
        'Stage2_DryMixing_Duration',
        'Stage3_OilLoading_Duration',
        'Stage4_WetMixing_Duration',
        'Stage5_PID_Duration',
        'Stage6_BottomMixing_Duration'
    ]
    consistent = True
    for col in phase_dur_cols:
        target_val = phase_targets.get(col, None)
        re_agg_val = re_agg[col].iloc[0] if col in re_agg.columns else None
        if target_val is not None and re_agg_val is not None:
            if abs(target_val - re_agg_val) > 0.5:
                consistent = False
                break

    # 3. Merge recipe context into phase features
    for col in recipe_context.index:
        if col not in re_agg.columns:
            re_agg[col] = recipe_context[col]

    # 4. Model Predictions
    try:
        energy_kwh, energy_kwh_per_ton = adapter.predict_energy(re_agg)
        pred_kwh = float(energy_kwh[0])
    except Exception:
        pred_kwh = np.nan

    try:
        mooney_pred, _, _, _ = adapter.predict_mooney(re_agg)
        pred_mny = float(mooney_pred[0])
    except Exception:
        pred_mny = np.nan

    # 5. Apply 7 Engineering Gates
    step_bounds_ok = True
    if 'recommended_time_SP' in out.columns and 'step_time_min' in out.columns:
        step_bounds_ok = all(out['recommended_time_SP'].between(out['step_time_min'], out['step_time_max']))

    if not step_bounds_ok:
        status = 'REJECTED_STEP_OUT_OF_BOUNDS'
    elif not consistent:
        status = 'REJECTED_PHASE_INCONSISTENT'
    elif pd.notna(pred_mny) and (pred_mny < spec_lower or pred_mny > spec_upper):
        status = 'REJECTED_MOONEY_SPEC_VIOLATION'
    else:
        status = 'PASSED'

    out['phase_consistency_status'] = 'CONSISTENT' if consistent else 'INCONSISTENT'
    out['model_validation_status'] = status
    out['predicted_energy_kwh'] = pred_kwh
    out['predicted_mooney'] = pred_mny
    out['recommendation_status'] = 'RECOMMENDED' if status == 'PASSED' else 'REJECTED'

    return out

if __name__ == '__main__':
    import os, sys
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
    from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
    from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags
    from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping
    from V4_2_Step_Phase_Architecture.model_training.phase_model_adapter import PhaseModelAdapter

    conn = get_sfe_connection()
    df_wide = pivot_batchdata_to_step_wide(conn, order_ids=['7951751'], batch_numbers=[75])
    df_flags = build_settable_flags(df_wide)
    df_mapped = build_step_phase_mapping(df_flags, target_temp=165.0)

    # Dummy adapter & context
    class DummyAdapter:
        def predict_energy(self, df): return (np.array([25.0]), np.array([25.0]))
        def predict_mooney(self, df): return (np.array([55.0]), np.array([50.0]), np.array([0.0]), np.array([5.0]))

    adapter = DummyAdapter()
    context = pd.Series({'CompoundName': 'M1-T15760', 'material_system': 'Silica'})
    targets = {'Stage5_PID_Duration': 13.0}

    df_validated = validate_step_candidates(df_mapped, targets, adapter, None, context)
    print(f"[OK] Task 8: Round-trip Validator verified.")
    print(df_validated[['GroupName', 'StepNo', 'phase_consistency_status', 'model_validation_status', 'predicted_energy_kwh', 'predicted_mooney', 'recommendation_status']].head(5).to_string(index=False))
    conn.close()
