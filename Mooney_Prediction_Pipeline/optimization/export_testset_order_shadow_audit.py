"""Export transparent shadow recommendation audit CSVs for one OOT test-set order."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.mooney_step_bridge import (
    recommend_step_durations,
    recommend_step_temperatures,
)
from Mooney_Prediction_Pipeline.optimization.v37_feature_row_builder import build_v37_baseline_feature_row
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor
from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags
from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping
from V4_2_Step_Phase_Architecture.optimization.step_duration_bounds import attach_historical_step_duration_bounds


@dataclass(frozen=True)
class ExportArtifacts:
    bounds_csv: str
    duration_summary_csv: str
    duration_audit_csv: str
    temperature_summary_csv: str
    temperature_audit_csv: str


def _prepare_bounded_steps(order_id: str, batch_number: int, feature_row: pd.Series) -> pd.DataFrame:
    with get_sfe_connection() as conn:
        df_wide = pivot_batchdata_to_step_wide(conn, order_ids=[order_id], batch_numbers=[batch_number])
        if df_wide.empty:
            raise ValueError(f'No BatchData Step rows found for OrderID={order_id}, BatchNumber={batch_number}.')
        df_flags = build_settable_flags(df_wide)
        df_flags['CompoundName'] = feature_row['CompoundName']
        df_mapped = build_step_phase_mapping(
            df_flags,
            target_temp=float(feature_row.get('Target_Temperature', 165.0)),
        )
        return attach_historical_step_duration_bounds(
            df_mapped,
            str(feature_row['CompoundName']),
            conn,
        )


def _duration_capacity(df_phase: pd.DataFrame) -> tuple[float, float, int]:
    current_duration = df_phase['time_SP'].astype(float) - df_phase.groupby('GroupName')['time_SP'].shift(1, fill_value=0.0)
    lower_room = (current_duration - df_phase['step_time_min'].astype(float)).clip(lower=0.0)
    upper_room = (df_phase['step_time_max'].astype(float) - current_duration).clip(lower=0.0)
    movable = int(((lower_room > 0) | (upper_room > 0)).sum())
    return float(lower_room.sum()), float(upper_room.sum()), movable


def export_testset_order_shadow_audit(
    order_id: str,
    batch_number: int,
    *,
    spec_center: float,
    spec_low: float,
    spec_high: float,
    output_dir: str,
    duration_increment_s: float = 1.0,
    temperature_increment_c: float = 1.0,
    max_candidates: int = 100,
) -> ExportArtifacts:
    package_dir = Path('Mooney_Prediction_Pipeline/models/v37_production_model_package')
    raw_predictor = V37RawPredictor(package_dir).load()
    feature_build = build_v37_baseline_feature_row(order_id, batch_number, raw_predictor)
    bounded = _prepare_bounded_steps(order_id, batch_number, feature_build.feature_row)
    bounded['dur_SP'] = bounded['time_SP'].astype(float) - bounded.groupby(['OrderID', 'BatchNumber', 'GroupName'])['time_SP'].shift(1, fill_value=0.0)
    bounded['step_duration_movable'] = (
        (bounded['dur_SP'] > bounded['step_time_min'].astype(float))
        | (bounded['dur_SP'] < bounded['step_time_max'].astype(float))
    )

    duration_phase_rows = []
    duration_audit_frames = []
    temperature_phase_rows = []
    temperature_audit_frames = []
    for phase_name in sorted(bounded['mapped_phase'].dropna().unique()):
        phase_steps = bounded[bounded['mapped_phase'] == phase_name].copy()
        total_lower_room, total_upper_room, movable_step_count = _duration_capacity(phase_steps)
        for direction, direction_name, available_room in ((-1, 'DECREASE_DURATION', total_lower_room), (1, 'INCREASE_DURATION', total_upper_room)):
            if movable_step_count == 0 or available_room < duration_increment_s:
                duration_phase_rows.append({
                    'OrderID': order_id,
                    'BatchNumber': batch_number,
                    'CompoundName': feature_build.feature_row['CompoundName'],
                    'mapped_phase': phase_name,
                    'direction': direction_name,
                    'movable_step_count': movable_step_count,
                    'available_duration_room_s': available_room,
                    'baseline_prediction': None,
                    'status': 'NO_VALID_CANDIDATE',
                    'reason_codes': json.dumps(['REJECTED_NO_DURATION_ROOM']),
                    'candidate_count': 0,
                    'accepted_count': 0,
                    'best_gap_after': None,
                    'best_model_response_delta': None,
                })
                continue

            result = recommend_step_durations(
                bounded,
                feature_build.feature_row,
                raw_predictor,
                target_phase=phase_name,
                expected_time_direction=direction,
                spec_center=spec_center,
                spec_low=spec_low,
                spec_high=spec_high,
                duration_increment_s=duration_increment_s,
                max_candidates=max_candidates,
            )
            audit = result.audit_table.copy()
            if not audit.empty:
                audit.insert(0, 'direction', direction_name)
                audit.insert(0, 'mapped_phase', phase_name)
                audit.insert(0, 'CompoundName', feature_build.feature_row['CompoundName'])
                audit.insert(0, 'BatchNumber', batch_number)
                audit.insert(0, 'OrderID', order_id)
                duration_audit_frames.append(audit)

            accepted_count = int(audit['accepted'].sum()) if not audit.empty else 0
            best_gap_after = float(audit['gap_after'].min()) if not audit.empty else None
            best_response_delta = float(audit['model_response_delta'].max()) if not audit.empty else None
            duration_phase_rows.append({
                'OrderID': order_id,
                'BatchNumber': batch_number,
                'CompoundName': feature_build.feature_row['CompoundName'],
                'mapped_phase': phase_name,
                'direction': direction_name,
                'movable_step_count': movable_step_count,
                'available_duration_room_s': available_room,
                'baseline_prediction': result.baseline_prediction,
                'status': result.status,
                'reason_codes': json.dumps(result.reason_codes),
                'candidate_count': len(audit),
                'accepted_count': accepted_count,
                'best_gap_after': best_gap_after,
                'best_model_response_delta': best_response_delta,
            })

        temp_phase_steps = phase_steps[
            phase_steps['is_temp_settable'].fillna(False).astype(bool)
            & pd.to_numeric(phase_steps['temp_SP'], errors='coerce').fillna(0.0).gt(0)
            & phase_steps['step_temp_min'].notna()
            & phase_steps['step_temp_max'].notna()
        ].copy()
        if temp_phase_steps.empty:
            for direction_name in ('DECREASE_TEMPERATURE', 'INCREASE_TEMPERATURE'):
                temperature_phase_rows.append({
                    'OrderID': order_id,
                    'BatchNumber': batch_number,
                    'CompoundName': feature_build.feature_row['CompoundName'],
                    'mapped_phase': phase_name,
                    'direction': direction_name,
                    'movable_step_count': 0,
                    'available_temperature_room_c': 0.0,
                    'baseline_prediction': None,
                    'status': 'NO_VALID_CANDIDATE',
                    'reason_codes': json.dumps(['REJECTED_NO_TEMPERATURE_ROOM']),
                    'candidate_count': 0,
                    'accepted_count': 0,
                    'best_gap_after': None,
                    'best_model_response_delta': None,
                })
            continue

        temp_current = pd.to_numeric(temp_phase_steps['temp_SP'], errors='coerce')
        temp_lower_room = (temp_current - pd.to_numeric(temp_phase_steps['step_temp_min'], errors='coerce')).clip(lower=0.0)
        temp_upper_room = (pd.to_numeric(temp_phase_steps['step_temp_max'], errors='coerce') - temp_current).clip(lower=0.0)
        temp_movable = int(((temp_lower_room > 0) | (temp_upper_room > 0)).sum())
        for direction, direction_name, available_room in ((-1, 'DECREASE_TEMPERATURE', float(temp_lower_room.sum())), (1, 'INCREASE_TEMPERATURE', float(temp_upper_room.sum()))):
            if temp_movable == 0 or available_room < temperature_increment_c:
                temperature_phase_rows.append({
                    'OrderID': order_id,
                    'BatchNumber': batch_number,
                    'CompoundName': feature_build.feature_row['CompoundName'],
                    'mapped_phase': phase_name,
                    'direction': direction_name,
                    'movable_step_count': temp_movable,
                    'available_temperature_room_c': available_room,
                    'baseline_prediction': None,
                    'status': 'NO_VALID_CANDIDATE',
                    'reason_codes': json.dumps(['REJECTED_NO_TEMPERATURE_ROOM']),
                    'candidate_count': 0,
                    'accepted_count': 0,
                    'best_gap_after': None,
                    'best_model_response_delta': None,
                })
                continue

            temp_result = recommend_step_temperatures(
                bounded,
                feature_build.feature_row,
                raw_predictor,
                target_phase=phase_name,
                expected_temp_direction=direction,
                spec_center=spec_center,
                spec_low=spec_low,
                spec_high=spec_high,
                temperature_increment_c=temperature_increment_c,
                max_candidates=max_candidates,
            )
            temp_audit = temp_result.audit_table.copy()
            if not temp_audit.empty:
                temp_audit.insert(0, 'direction', direction_name)
                temp_audit.insert(0, 'mapped_phase', phase_name)
                temp_audit.insert(0, 'CompoundName', feature_build.feature_row['CompoundName'])
                temp_audit.insert(0, 'BatchNumber', batch_number)
                temp_audit.insert(0, 'OrderID', order_id)
                temperature_audit_frames.append(temp_audit)

            accepted_count = int(temp_audit['accepted'].sum()) if not temp_audit.empty else 0
            best_gap_after = float(temp_audit['gap_after'].min()) if not temp_audit.empty else None
            best_response_delta = float(temp_audit['model_response_delta'].max()) if not temp_audit.empty else None
            temperature_phase_rows.append({
                'OrderID': order_id,
                'BatchNumber': batch_number,
                'CompoundName': feature_build.feature_row['CompoundName'],
                'mapped_phase': phase_name,
                'direction': direction_name,
                'movable_step_count': temp_movable,
                'available_temperature_room_c': available_room,
                'baseline_prediction': temp_result.baseline_prediction,
                'status': temp_result.status,
                'reason_codes': json.dumps(temp_result.reason_codes),
                'candidate_count': len(temp_audit),
                'accepted_count': accepted_count,
                'best_gap_after': best_gap_after,
                'best_model_response_delta': best_response_delta,
            })

    os.makedirs(output_dir, exist_ok=True)
    stem = f'{order_id}_b{batch_number}'
    bounds_csv = os.path.join(output_dir, f'{stem}_step_bounds.csv')
    duration_summary_csv = os.path.join(output_dir, f'{stem}_duration_phase_sweep_summary.csv')
    duration_audit_csv = os.path.join(output_dir, f'{stem}_duration_phase_sweep_audit.csv')
    temperature_summary_csv = os.path.join(output_dir, f'{stem}_temperature_phase_sweep_summary.csv')
    temperature_audit_csv = os.path.join(output_dir, f'{stem}_temperature_phase_sweep_audit.csv')

    bounded.to_csv(bounds_csv, index=False, encoding='utf-8-sig')
    pd.DataFrame(duration_phase_rows).to_csv(duration_summary_csv, index=False, encoding='utf-8-sig')
    pd.concat(duration_audit_frames, ignore_index=True).to_csv(duration_audit_csv, index=False, encoding='utf-8-sig') if duration_audit_frames else pd.DataFrame().to_csv(duration_audit_csv, index=False, encoding='utf-8-sig')
    pd.DataFrame(temperature_phase_rows).to_csv(temperature_summary_csv, index=False, encoding='utf-8-sig')
    pd.concat(temperature_audit_frames, ignore_index=True).to_csv(temperature_audit_csv, index=False, encoding='utf-8-sig') if temperature_audit_frames else pd.DataFrame().to_csv(temperature_audit_csv, index=False, encoding='utf-8-sig')
    return ExportArtifacts(
        bounds_csv=bounds_csv,
        duration_summary_csv=duration_summary_csv,
        duration_audit_csv=duration_audit_csv,
        temperature_summary_csv=temperature_summary_csv,
        temperature_audit_csv=temperature_audit_csv,
    )