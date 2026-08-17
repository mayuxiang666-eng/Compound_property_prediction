"""Corrected fail-closed Target-action lineage for counterfactual scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PREFIX = {
    'Stage1_Loading': 'Stage1_Loading',
    'Stage2_DryMixing': 'Stage2_DryMixing',
    'Stage3_OilLoading': 'Stage3_OilLoading',
    'Stage4_WetMixing_Ramp': 'Stage4_WetMixing',
    'Stage5_MST_PID_Silanization': 'Stage5_PID',
}


def _phase_columns(phase: str) -> list[str]:
    if phase.startswith('Stage6_MSB_PID'):
        return ['Stage6_BottomMixing_Duration', 'Stage6_BottomMixing_RotorSpeed_Mean', 'Stage6_BottomMixing_WayofRam_Mean', 'Stage6_BottomMixing_temp_Mean']
    prefix = PREFIX.get(phase)
    if prefix is None:
        return []
    columns = [f'{prefix}_Duration', f'{prefix}_RotorSpeed_Mean', f'{prefix}_WayofRam_Mean', f'{prefix}_temp_Mean']
    return columns + (['Target_Temperature'] if prefix == 'Stage5_PID' else [])


def build_action_feature_lineage_v2(before_steps: pd.DataFrame, candidate_steps: pd.DataFrame, before_features: pd.DataFrame, candidate_features: pd.DataFrame, model_feature_columns: list[str], *, tolerance: float = 1e-9) -> list[dict[str, Any]]:
    """Show each changed Target and whether it changed a valid V3.7 input."""
    before = before_steps.set_index(['GroupName', 'StepNo'])
    after = candidate_steps.set_index(['GroupName', 'StepNo'])
    controls = {
        'duration': ('dur_SP', 'recommended_dur_SP'),
        'temperature': ('temp_SP', 'recommended_temp_SP'),
        'rotor': ('rotor_SP', 'recommended_rotor_SP'),
        'ram': ('ram_SP', 'recommended_ram_SP'),
    }
    model_columns = set(model_feature_columns)
    result: list[dict[str, Any]] = []
    for parameter, (source, recommended) in controls.items():
        if source not in before.columns or recommended not in after.columns:
            continue
        delta = pd.to_numeric(after[recommended], errors='coerce') - pd.to_numeric(before[source], errors='coerce')
        for key, value in delta[delta.abs() > tolerance].items():
            group_name, step_no = key
            phase = str(after.at[key, 'mapped_phase'])
            candidates = _phase_columns(phase)
            v37_inputs = [column for column in candidates if column in model_columns]
            changed = []
            for column in v37_inputs:
                before_value = float(pd.to_numeric(before_features.iloc[0][column], errors='coerce'))
                candidate_value = float(pd.to_numeric(candidate_features.iloc[0][column], errors='coerce'))
                if np.isfinite(before_value) and np.isfinite(candidate_value) and abs(candidate_value - before_value) > tolerance:
                    changed.append(column)
            result.append({
                'GroupName': str(group_name), 'StepNo': int(step_no),
                'action_parameter': parameter, 'target_source_column': source,
                'target_candidate_column': recommended, 'target_delta': float(value),
                'mapped_phase': phase, 'phase_feature_candidates': candidates,
                'v37_phase_input_columns': v37_inputs,
                'changed_v37_input_columns': changed,
                'model_path_status': 'MODEL_PATH_OK' if changed else 'NO_MODEL_PATH',
            })
    return result


def candidate_has_model_path_v2(records: list[dict[str, Any]]) -> bool:
    return bool(records) and all(row['model_path_status'] == 'MODEL_PATH_OK' for row in records)
