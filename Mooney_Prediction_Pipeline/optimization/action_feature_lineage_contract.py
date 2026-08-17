"""Parameter-specific action-to-feature lineage contract for V3.7 shadow scoring."""

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


def phase_columns_for_action(phase: str, parameter: str) -> list[str]:
    """Return only the Phase features causally updated by one Target parameter."""
    if phase.startswith('Stage6_MSB_PID'):
        suffix = {
            'duration': 'Duration',
            'temperature': 'temp_Mean',
            'rotor': 'RotorSpeed_Mean',
            'ram': 'WayofRam_Mean',
        }.get(parameter)
        return [f'Stage6_BottomMixing_{suffix}'] if suffix else []
    prefix = PREFIX.get(phase)
    suffix = {
        'duration': 'Duration',
        'temperature': 'temp_Mean',
        'rotor': 'RotorSpeed_Mean',
        'ram': 'WayofRam_Mean',
    }.get(parameter)
    if prefix is None or suffix is None:
        return []
    columns = [f'{prefix}_{suffix}']
    if parameter == 'temperature' and prefix == 'Stage5_PID':
        columns.append('Target_Temperature')
    return columns


def build_action_feature_lineage(
    before_steps: pd.DataFrame,
    candidate_steps: pd.DataFrame,
    before_features: pd.DataFrame,
    candidate_features: pd.DataFrame,
    model_feature_columns: list[str],
    *,
    tolerance: float = 1e-9,
) -> list[dict[str, Any]]:
    """Return one fail-closed record per changed Target.

    A record is ``MODEL_PATH_OK`` only when the parameter's own declared Phase
    feature is a V3.7 feature and its numerical value changes.  Changes from
    unrelated actions in a combined candidate are intentionally excluded.
    """
    before = before_steps.set_index(['GroupName', 'StepNo'])
    after = candidate_steps.set_index(['GroupName', 'StepNo'])
    controls = {
        'duration': ('dur_SP', 'recommended_dur_SP'),
        'temperature': ('temp_SP', 'recommended_temp_SP'),
        'rotor': ('rotor_SP', 'recommended_rotor_SP'),
        'ram': ('ram_SP', 'recommended_ram_SP'),
    }
    model_columns = set(model_feature_columns)
    records: list[dict[str, Any]] = []
    for parameter, (source_column, candidate_column) in controls.items():
        if source_column not in before.columns or candidate_column not in after.columns:
            continue
        deltas = pd.to_numeric(after[candidate_column], errors='coerce') - pd.to_numeric(before[source_column], errors='coerce')
        for key, target_delta in deltas[deltas.abs() > tolerance].items():
            phase = str(after.at[key, 'mapped_phase'])
            phase_columns = phase_columns_for_action(phase, parameter)
            v37_columns = [column for column in phase_columns if column in model_columns]
            changed = []
            for column in v37_columns:
                before_value = float(pd.to_numeric(before_features.iloc[0][column], errors='coerce'))
                candidate_value = float(pd.to_numeric(candidate_features.iloc[0][column], errors='coerce'))
                if np.isfinite(before_value) and np.isfinite(candidate_value) and abs(candidate_value - before_value) > tolerance:
                    changed.append(column)
            records.append({
                'GroupName': str(key[0]),
                'StepNo': int(key[1]),
                'action_parameter': parameter,
                'target_source_column': source_column,
                'target_candidate_column': candidate_column,
                'target_delta': float(target_delta),
                'mapped_phase': phase,
                'phase_feature_candidates': phase_columns,
                'v37_phase_input_columns': v37_columns,
                'changed_v37_input_columns': changed,
                'model_path_status': 'MODEL_PATH_OK' if changed else 'NO_MODEL_PATH',
            })
    return records


def candidate_has_model_path(records: list[dict[str, Any]]) -> bool:
    return bool(records) and all(row['model_path_status'] == 'MODEL_PATH_OK' for row in records)
