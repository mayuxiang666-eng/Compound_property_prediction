"""Auditable Step Target -> Phase feature -> V3.7 model-input lineage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PHASE_PREFIX = {
    'Stage1_Loading': 'Stage1_Loading',
    'Stage2_DryMixing': 'Stage2_DryMixing',
    'Stage3_OilLoading': 'Stage3_OilLoading',
    'Stage4_WetMixing_Ramp': 'Stage4_WetMixing',
    'Stage5_MST_PID_Silanization': 'Stage5_PID',
}


def _phase_features(mapped_phase: str) -> list[str]:
    if mapped_phase.startswith('Stage6_MSB_PID'):
        return ['Stage6_BottomMixing_Duration', 'Stage6_BottomMixing_RotorSpeed_Mean', 'Stage6_BottomMixing_WayofRam_Mean', 'Stage6_BottomMixing_temp_Mean']
    prefix = PHASE_PREFIX.get(mapped_phase)
    if prefix is None:
        return []
    values = [f'{prefix}_Duration', f'{prefix}_RotorSpeed_Mean', f'{prefix}_WayofRam_Mean', f'{prefix}_temp_Mean']
    return values + (['Target_Temperature'] if prefix == 'Stage5_PID' else [])


def build_action_feature_lineage(before_steps: pd.DataFrame, candidate_steps: pd.DataFrame, before_features: pd.DataFrame, candidate_features: pd.DataFrame, model_feature_columns: list[str], *, tolerance: float = 1e-9) -> list[dict[str, Any]]:
    """Return one fail-closed lineage record per changed Target action."""
    before = before_steps.set_index(['GroupName', 'StepNo'])
    after = candidate_steps.set_index(['GroupName', 'StepNo'])
    model_columns = set(model_feature_columns)
    controls = {
        'duration': ('dur_SP', 'recommended_dur_SP'),
        'temperature': ('temp_SP', 'recommended_temp_SP'),
        'rotor': ('rotor_SP', 'recommended_rotor_SP'),
        'ram': ('ram_SP', 'recommended_ram_SP'),
    }
    records: list[dict[str, Any]] = []
    for parameter, (baseline_column, candidate_column) in controls.items():
        if candidate_column not in after.columns or baseline_column not in before.columns:
            continue
        delta = pd.to_numeric(after[candidate_column], errors='coerce') - pd.to_numeric(before[baseline_column], errors='coerce')
        for group_name, step_no in delta[delta.abs() > tolerance].items():
            mapped_phase = str(after.loc[(group_name, step_no), 'mapped_phase'])
            phase_columns = _phase_features(mapped_phase)
            v37_columns = [column for column in phase_columns if column in model_columns]
            changed_columns = []
            for column in v37_columns:
                before_value = float(pd.to_numeric(before_features.iloc[0][column], errors='coerce'))
                after_value = float(pd.to_numeric(candidate_features.iloc[0][column], errors='coerce'))
                if np.isfinite(before_value) and np.isfinite(after_value) and abs(after_value - before_value) > tolerance:
                    changed_columns.append(column)
            records.append({
                'GroupName': str(group_name), 'StepNo': int(step_no),
                'action_parameter': parameter, 'target_source_column': baseline_column,
                'target_candidate_column': candidate_column, 'target_delta': float(delta.loc[(group_name, step_no)]),
                'mapped_phase': mapped_phase, 'phase_feature_candidates': phase_columns,
                'v37_phase_input_columns': v37_columns, 'changed_v37_input_columns': changed_columns,
                'model_path_status': 'MODEL_PATH_OK' if changed_columns else 'NO_MODEL_PATH',
            })
    return records


def candidate_has_model_path(records: list[dict[str, Any]]) -> bool:
    return bool(records) and all(record['model_path_status'] == 'MODEL_PATH_OK' for record in records)
