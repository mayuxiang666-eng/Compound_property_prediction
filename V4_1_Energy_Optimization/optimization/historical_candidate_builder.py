"""Historical best-batch templates for model-normalized energy recommendations."""

import re
import numpy as np
import pandas as pd


STAGE_DURATION_COLUMNS = [
    'Stage2_DryMixing_Duration',
    'Stage3_OilLoading_Duration',
    'Stage4_WetMixing_Duration',
    'Stage5_PID_Duration',
    'Stage6_BottomMixing_Duration',
]
PROFILE_COLUMNS = [
    'Stage1_Loading_RotorSpeed_Mean',
    'Stage2_DryMixing_RotorSpeed_Mean',
    'Stage3_OilLoading_RotorSpeed_Mean',
    'Stage4_WetMixing_RotorSpeed_Mean',
    'Stage5_PID_RotorSpeed_Mean',
    'Stage6_BottomMixing_RotorSpeed_Mean',
    'Stage1_Loading_WayofRam_Mean',
    'Stage2_DryMixing_WayofRam_Mean',
    'Stage3_OilLoading_WayofRam_Mean',
    'Stage4_WetMixing_WayofRam_Mean',
    'Stage5_PID_WayofRam_Mean',
    'Stage6_BottomMixing_WayofRam_Mean',
]
CONTEXT_NUMERIC_COLUMNS = [
    'supplier_rubber_viscosity_avg',
    'supplier_silica_moisture_avg',
    'supplier_silica_surface_area_avg',
    'supplier_carbon_black_structure_avg',
    'supplier_carbon_black_surface_area_avg',
    'supplier_carbon_black_moisture_avg',
    'env_temp_mean', 'env_humidity_mean', 'phys_init_temp',
    'batch_weight_ton', 'Top_Fill_Factor', 'Bot_Fill_Factor',
]
CONTEXT_ALIASES = {
    'ambient_temperature_difference': ['env_temp_mean', 'ambient_temperature'],
    'ambient_humidity_difference': ['env_humidity_mean', 'ambient_humidity'],
    'material_initial_temperature_difference': ['phys_init_temp', 'material_initial_temperature'],
    'batch_weight_difference': ['batch_weight_ton', 'BatchWeight'],
    'fill_factor_difference': ['Top_Fill_Factor', 'fill_factor_top'],
}


def _number(value):
    return pd.to_numeric(value, errors='coerce')


def _first_existing(row: pd.Series, columns: list[str]):
    for column in columns:
        if column in row.index and pd.notna(_number(row.get(column))):
            return _number(row.get(column))
    return np.nan


def _quality_ok(row: pd.Series) -> bool:
    for column in ('quality_rejection_flag', 'quality_rejected', 'is_quality_rejected'):
        if column in row.index and pd.notna(row.get(column)):
            return not bool(row.get(column))
    for column in ('quality_status', 'disposition_of_compound', 'quality_state'):
        if column in row.index and pd.notna(row.get(column)):
            return str(row.get(column)).strip().upper() not in {'REJECT', 'REJECTED', 'FAIL', 'NOK'}
    return True


def _valid_stage_data(row: pd.Series) -> bool:
    observed = [_number(row.get(column)) for column in STAGE_DURATION_COLUMNS if column in row.index]
    return bool(observed) and all(pd.notna(value) and float(value) >= 0.0 for value in observed)


def _valid_batch_context(row: pd.Series) -> bool:
    weight = _first_existing(row, ['batch_weight_ton', 'BatchWeight'])
    top_fill = _first_existing(row, ['Top_Fill_Factor', 'fill_factor_top'])
    return bool(pd.notna(weight) and 0.05 <= float(weight) <= 1.0 and pd.notna(top_fill) and 0.0 < float(top_fill) <= 100.0)


def _context_difference(current: pd.Series, historical: pd.Series) -> dict:
    differences = {}
    for output_name, aliases in CONTEXT_ALIASES.items():
        left = _first_existing(current, aliases)
        right = _first_existing(historical, aliases)
        differences[output_name] = round(abs(float(left) - float(right)), 4) if pd.notna(left) and pd.notna(right) else np.nan

    for output_name, column in (
        ('supplier_COA_difference', 'supplier_rubber_viscosity_avg'),
    ):
        left = _number(current.get(column, np.nan))
        right = _number(historical.get(column, np.nan))
        differences[output_name] = round(abs(float(left) - float(right)), 4) if pd.notna(left) and pd.notna(right) else np.nan

    current_lot = str(current.get('MaterialLot', current.get('material_lot', ''))) or 'UNKNOWN'
    historical_lot = str(historical.get('MaterialLot', historical.get('material_lot', ''))) or 'UNKNOWN'
    differences['raw_material_lot_difference'] = int(current_lot != historical_lot)
    differences['mixer_line_difference'] = int(str(current.get('MixerLine', '')) != str(historical.get('MixerLine', '')))
    differences['route_difference'] = int(str(current.get('phase_route', '')) != str(historical.get('phase_route', '')))
    if 'OrderStartTime' in current.index and 'OrderStartTime' in historical.index:
        current_time = pd.to_datetime(current.get('OrderStartTime'), errors='coerce')
        historical_time = pd.to_datetime(historical.get('OrderStartTime'), errors='coerce')
        differences['historical_batch_age_days'] = (current_time - historical_time).days if pd.notna(current_time) and pd.notna(historical_time) else np.nan
    else:
        differences['historical_batch_age_days'] = np.nan
    return differences


def build_historical_best_reference_cohort(
    history_df: pd.DataFrame,
    current_row: pd.Series,
    spec_lower: float,
    spec_upper: float,
    low_percentile: float = 0.10,
    high_percentile: float = 0.30,
    max_templates: int = 20,
) -> pd.DataFrame:
    """Return eligible historical templates for one recipe/mixer/route context."""
    required = {'CompoundName', 'MixerLine', 'material_system', 'phase_route', 'total_kwh_per_batch', 'MNY'}
    if not required.issubset(history_df.columns):
        return pd.DataFrame()
    mask = (
        (history_df['CompoundName'] == current_row.get('CompoundName'))
        & (history_df['MixerLine'] == current_row.get('MixerLine'))
        & (history_df['material_system'] == current_row.get('material_system'))
        & (history_df['phase_route'] == current_row.get('phase_route'))
    )
    cohort = history_df.loc[mask].copy()
    if cohort.empty:
        return pd.DataFrame()
    energy = pd.to_numeric(cohort['total_kwh_per_batch'], errors='coerce')
    low, high = energy.quantile([low_percentile, high_percentile])
    cohort = cohort[
        energy.between(low, high, inclusive='both')
        & pd.to_numeric(cohort['MNY'], errors='coerce').between(spec_lower, spec_upper, inclusive='both')
    ]
    cohort = cohort[cohort.apply(_quality_ok, axis=1) & cohort.apply(_valid_stage_data, axis=1) & cohort.apply(_valid_batch_context, axis=1)]
    if cohort.empty:
        return pd.DataFrame()
    cohort = cohort.sort_values('total_kwh_per_batch').head(max_templates)
    rows = []
    for _, historical in cohort.iterrows():
        process_profile = {
            column: float(_number(historical.get(column)))
            for column in STAGE_DURATION_COLUMNS + ['Target_Temperature']
            if column in historical.index and pd.notna(_number(historical.get(column)))
        }
        differences = _context_difference(current_row, historical)
        rows.append({
            'template_id': f"{historical.get('OrderID', historical.name)}",
            'historical_order_id': historical.get('OrderID', historical.name),
            'historical_batch_number': historical.get('BatchNumber', ''),
            'historical_actual_kwh': float(energy.loc[historical.name]),
            'historical_actual_mooney': float(_number(historical.get('MNY'))),
            'historical_best_actual_saving_pct': round((float(_number(current_row.get('total_kwh_per_batch'))) - float(energy.loc[historical.name])) / max(float(_number(current_row.get('total_kwh_per_batch'))), 1.0) * 100.0, 2),
            'process_profile': process_profile,
            'profile_reference': {column: historical.get(column, np.nan) for column in PROFILE_COLUMNS if column in historical.index},
            **differences,
        })
    return pd.DataFrame(rows)


def normalize_candidate_context(current_row: pd.Series, template: pd.Series) -> pd.Series:
    """Keep template process controls while replacing all context with current values."""
    normalized = current_row.copy()
    for key, value in template.get('process_profile', {}).items():
        normalized[key] = value
    return normalized


def uncertainty_explanation(row: pd.Series) -> str:
    labels = {
        'raw_material_lot_difference': 'current material lot',
        'supplier_COA_difference': 'supplier COA',
        'ambient_temperature_difference': 'ambient temperature',
        'ambient_humidity_difference': 'ambient humidity',
        'material_initial_temperature_difference': 'material initial temperature',
        'batch_weight_difference': 'batch weight',
        'fill_factor_difference': 'fill factor',
        'mixer_line_difference': 'mixer line',
        'route_difference': 'route',
        'historical_batch_age_days': 'historical batch age',
    }
    scored = []
    for key, label in labels.items():
        value = row.get(key, np.nan)
        if pd.notna(value) and float(value) > 0:
            scored.append((float(value), label))
    factors = [label for _, label in sorted(scored, reverse=True)[:3]]
    while len(factors) < 3:
        factors.append('none identified')
    return (
        'Historical best energy was lower, but the model re-predicts the template under current '
        f'conditions and adjusts saving because {factors[0]}, {factors[1]}, and {factors[2]} differ.'
    )