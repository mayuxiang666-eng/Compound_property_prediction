# ============================================================================
# Step 5 & 6: Energy Optimization Engine & Gated Recommendation Module (V4.1)
# ============================================================================
# Evaluates valid candidate parameter setpoints passing Mooney Quality Constraints.
# Applies Task 6 Recommendation Gates:
# - predicted_saving_kwh >= 3.0 OR predicted_saving_pct >= 8.0%
# - Mooney lower/upper prediction interval inside spec
# - confidence != LOW
# - valid_candidate_count >= 20
# - candidate is not OOD
# Writes shadow recommendation logs and rejection reason logs.
# ============================================================================

import os
import sys
import hashlib
import numpy as np
import pandas as pd

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

from model_training.energy_model import MixingEnergyPredictionModel
from optimization.candidate_generator import (
    STAGE_DURATION_PARAMS,
    SUPPORTED_ROUTE_BRANCHES,
    derive_route_stage_mask,
    generate_route_masked_candidates,
)
from optimization.mooney_constraint_checker import MooneyQualityConstraintChecker
from optimization.historical_candidate_builder import (
    build_historical_best_reference_cohort,
    normalize_candidate_context,
    uncertainty_explanation,
)


class EnergyMooneyOptimizer:
    def __init__(self, energy_model: MixingEnergyPredictionModel, mooney_checker: MooneyQualityConstraintChecker, historical_df: pd.DataFrame | None = None):
        self.energy_model = energy_model
        self.mooney_checker = mooney_checker
        self.historical_df = historical_df if historical_df is not None else pd.DataFrame()

    @staticmethod
    def _current_setpoints(base_df_row: pd.Series, stage_mask: dict[str, bool]) -> dict:
        setpoints = {
            parameter: float(pd.to_numeric(base_df_row.get(parameter, 0.0), errors='coerce') or 0.0)
            for parameter in [
                'Stage2_DryMixing_Duration',
                'Stage4_WetMixing_Duration',
                'Stage5_PID_Duration',
                'Stage6_BottomMixing_Duration',
                'Target_Temperature',
            ]
        }
        for stage_flag, duration_parameter in STAGE_DURATION_PARAMS.items():
            if duration_parameter in setpoints and not stage_mask[stage_flag]:
                setpoints[duration_parameter] = 0.0
        return setpoints

    @staticmethod
    def _candidate_route_and_bounds_status(
        candidate: dict,
        bounds_row: pd.Series,
        stage_mask: dict[str, bool],
    ) -> tuple[str, str, str]:
        invalid_stages = [
            duration_parameter
            for stage_flag, duration_parameter in STAGE_DURATION_PARAMS.items()
            if duration_parameter in candidate
            and not stage_mask[stage_flag]
            and abs(float(candidate[duration_parameter])) > 1e-6
        ]
        if invalid_stages:
            return 'INVALID_STAGE_FOR_ROUTE', 'NOT_CHECKED', 'REJECTED_INVALID_STAGE_FOR_ROUTE'

        effective_durations = [
            float(candidate.get(duration_parameter, 0.0))
            for stage_flag, duration_parameter in STAGE_DURATION_PARAMS.items()
            if stage_mask[stage_flag] and duration_parameter in candidate
        ]
        if not effective_durations or all(abs(value) <= 1e-6 for value in effective_durations):
            return 'ZERO_EFFECTIVE_DURATIONS', 'NOT_CHECKED', 'REJECTED_ZERO_DURATION_CANDIDATE'

        out_of_bounds = []
        for parameter, value in candidate.items():
            stage_flag = next(
                (
                    flag
                    for flag, duration_parameter in STAGE_DURATION_PARAMS.items()
                    if duration_parameter == parameter
                ),
                None,
            )
            if stage_flag is not None and not stage_mask[stage_flag]:
                continue
            lower = pd.to_numeric(bounds_row.get(f'{parameter}_min', np.nan), errors='coerce')
            upper = pd.to_numeric(bounds_row.get(f'{parameter}_max', np.nan), errors='coerce')
            if pd.notna(lower) and pd.notna(upper) and float(upper) > float(lower):
                if float(value) < float(lower) - 1e-6 or float(value) > float(upper) + 1e-6:
                    out_of_bounds.append(parameter)
        if out_of_bounds:
            return 'VALID_ROUTE_STAGES', 'OUT_OF_SAFE_BOUNDS', 'REJECTED_OUT_OF_SAFE_BOUNDS'

        return 'VALID_ROUTE_STAGES', 'WITHIN_SAFE_BOUNDS', ''

    def optimize_batch(
        self,
        base_df_row: pd.Series,
        bounds_row: pd.Series,
        spec_lower: float,
        spec_upper: float,
        n_candidates=60,
        bounds_rejection_reason='SAFE_BOUNDS_UNAVAILABLE',
        historical_df: pd.DataFrame | None = None,
    ) -> dict:
        """
        Optimizes a single batch with Task 6 Recommendation Gates.
        """
        material_system = base_df_row.get('material_system', 'Silica')
        phase_route = base_df_row.get('phase_route', 'OilWet')
        branch = f'{material_system}_{phase_route}'
        if branch not in SUPPORTED_ROUTE_BRANCHES or bounds_row.empty:
            base_df = pd.DataFrame([base_df_row])
            base_kwh, base_kwh_per_ton = self.energy_model.predict(base_df)
            actual_kwh = float(base_df_row.get('total_kwh_per_batch', base_kwh[0]))
            actual_kwh_per_ton = float(base_df_row.get('kwh_per_ton', base_kwh_per_ton[0]))
            mode_b_current_kwh = float(base_kwh[0])
            return {
                'recommendation_status': 'REJECTED',
                'rejection_reason': (
                    'UNSUPPORTED_MATERIAL_ROUTE'
                    if branch not in SUPPORTED_ROUTE_BRANCHES
                    else bounds_rejection_reason
                ),
                'actual_kwh_per_batch': round(actual_kwh, 2),
                'recommended_kwh_per_batch': round(actual_kwh, 2),
                'actual_kwh_per_ton': round(actual_kwh_per_ton, 2),
                'recommended_kwh_per_ton': round(actual_kwh_per_ton, 2),
                'predicted_baseline_kwh_per_batch': round(mode_b_current_kwh, 2),
                'predicted_baseline_kwh_per_ton': round(float(base_kwh_per_ton[0]), 2),
                'estimated_saving_kwh': 0.0,
                'estimated_saving_pct': 0.0,
                'actual_current_kwh_per_batch': round(actual_kwh, 2),
                'mode_b_predicted_current_kwh_per_batch': round(mode_b_current_kwh, 2),
                'mode_b_predicted_recommended_kwh_per_batch': round(mode_b_current_kwh, 2),
                'shadow_actual_saving_kwh': 0.0,
                'shadow_actual_saving_pct': 0.0,
                'model_based_saving_kwh': 0.0,
                'model_based_saving_pct': 0.0,
                'mooney_pred': np.nan,
                'mooney_pred_current': np.nan,
                'mooney_pred_lower': np.nan,
                'mooney_pred_upper': np.nan,
                'confidence_label': 'LOW',
                'ood_flag': True,
                'valid_candidates_count': 0,
                'total_candidates_evaluated': 0,
                'recommended_setpoints': {},
                'raw_candidate_setpoints': {},
                'raw_recommended_kwh_per_batch': round(actual_kwh, 2),
                'raw_mooney_pred': np.nan,
                'raw_mooney_low': np.nan,
                'raw_mooney_high': np.nan,
                'raw_confidence_label': 'LOW',
                'raw_ood_flag': True,
                'safe_bound_status': 'UNAVAILABLE',
                'route_stage_status': 'UNAVAILABLE',
            }

        # 1. Generate Candidates
        seed_context = '|'.join(str(base_df_row.get(column, '')) for column in [
            'OrderID', 'BatchNumber', 'CompoundName', 'MixerLine', 'material_system', 'phase_route'
        ])
        random_state = int.from_bytes(hashlib.sha256(seed_context.encode('utf-8')).digest()[:8], 'little')
        stage_mask = derive_route_stage_mask(base_df_row)
        current_setpoints = self._current_setpoints(base_df_row, stage_mask)
        history = historical_df if historical_df is not None else self.historical_df
        templates = build_historical_best_reference_cohort(
            history, base_df_row, spec_lower, spec_upper
        ) if not history.empty else pd.DataFrame()
        if not templates.empty:
            candidates = []
            for _, row in templates.iterrows():
                profile = dict(row['process_profile'])
                for stage_flag, duration_parameter in STAGE_DURATION_PARAMS.items():
                    if duration_parameter in profile and not stage_mask[stage_flag]:
                        profile[duration_parameter] = 0.0
                candidates.append(profile)
            candidate_templates = [row for _, row in templates.iterrows()]
        else:
            candidates = generate_route_masked_candidates(
                base_df_row,
                bounds_row,
                n_candidates=n_candidates,
                random_state=random_state,
            )
            candidate_templates = [None] * len(candidates)

        # 2. Check Mooney Constraints for current settings and all raw candidates.
        current_eval = self.mooney_checker.evaluate_candidates(
            base_df_row, [current_setpoints], spec_lower, spec_upper
        )[0]
        eval_results = self.mooney_checker.evaluate_candidates(base_df_row, candidates, spec_lower, spec_upper)
        cand_rows = []
        for evaluation in eval_results:
            candidate_row = base_df_row.copy()
            for key, value in evaluation['candidate_setpoints'].items():
                candidate_row[key] = value
            cand_rows.append(candidate_row)
        all_candidate_kwh, all_candidate_kwh_per_ton = self.energy_model.predict(pd.DataFrame(cand_rows))
        raw_best_index = int(np.argmin(all_candidate_kwh))
        raw_best_eval = eval_results[raw_best_index]

        valid_indices = []
        validation_statuses = []
        for candidate_index, evaluation in enumerate(eval_results):
            route_status, safe_bound_status, validation_reason = self._candidate_route_and_bounds_status(
                evaluation['candidate_setpoints'], bounds_row, stage_mask
            )
            validation_statuses.append((route_status, safe_bound_status, validation_reason))
            if not validation_reason and evaluation['is_valid']:
                valid_indices.append(candidate_index)

        actual_kwh = float(base_df_row.get('total_kwh_per_batch', 0.0))
        actual_kwh_per_ton = float(base_df_row.get('kwh_per_ton', 0.0))
        if actual_kwh <= 0.0:
            actual_kwh = float(self.energy_model.predict(pd.DataFrame([base_df_row]))[0][0])
        if actual_kwh_per_ton <= 0.0:
            actual_kwh_per_ton = actual_kwh / max(float(base_df_row.get('batch_weight_ton', 0.25)), 0.05)

        current_model_row = base_df_row.copy()
        for key, value in current_setpoints.items():
            current_model_row[key] = value
        mode_b_current_kwh = float(self.energy_model.predict(pd.DataFrame([current_model_row]))[0][0])

        def energy_saving_fields(mode_b_recommended_kwh: float) -> dict:
            shadow_actual_saving_kwh = actual_kwh - mode_b_recommended_kwh
            model_based_saving_kwh = mode_b_current_kwh - mode_b_recommended_kwh
            return {
                'actual_current_kwh_per_batch': round(actual_kwh, 2),
                'mode_b_predicted_current_kwh_per_batch': round(mode_b_current_kwh, 2),
                'mode_b_predicted_recommended_kwh_per_batch': round(mode_b_recommended_kwh, 2),
                'shadow_actual_saving_kwh': round(shadow_actual_saving_kwh, 2),
                'shadow_actual_saving_pct': round(
                    shadow_actual_saving_kwh / max(actual_kwh, 1.0) * 100.0, 2
                ),
                'model_based_saving_kwh': round(model_based_saving_kwh, 2),
                'model_based_saving_pct': round(
                    model_based_saving_kwh / max(mode_b_current_kwh, 1.0) * 100.0, 2
                ),
            }

        def no_valid_candidate(
            reason: str,
            route_status: str = 'VALID_ROUTE_STAGES',
            safe_status: str = 'WITHIN_SAFE_BOUNDS',
            status: str = 'NO_VALID_CANDIDATE',
        ) -> dict:
            return {
                'recommendation_status': status,
                'rejection_reason': reason,
                'actual_kwh_per_batch': round(actual_kwh, 2),
                'recommended_kwh_per_batch': round(actual_kwh, 2),
                'actual_kwh_per_ton': round(actual_kwh_per_ton, 2),
                'recommended_kwh_per_ton': round(actual_kwh_per_ton, 2),
                'predicted_baseline_kwh_per_batch': round(mode_b_current_kwh, 2),
                'predicted_baseline_kwh_per_ton': round(actual_kwh_per_ton, 2),
                'estimated_saving_kwh': 0.0,
                'estimated_saving_pct': 0.0,
                **energy_saving_fields(mode_b_current_kwh),
                'mooney_pred': current_eval['mooney_pred'],
                'mooney_pred_current': current_eval['mooney_pred'],
                'mooney_pred_lower': current_eval['mooney_pred_lower'],
                'mooney_pred_upper': current_eval['mooney_pred_upper'],
                'spec_lower': spec_lower,
                'spec_upper': spec_upper,
                'confidence_label': current_eval['confidence_label'],
                'ood_flag': current_eval['is_ood'],
                'valid_candidates_count': len(valid_indices),
                'total_candidates_evaluated': len(candidates),
                'recommended_setpoints': current_setpoints,
                'raw_candidate_setpoints': raw_best_eval['candidate_setpoints'],
                'raw_recommended_kwh_per_batch': round(float(all_candidate_kwh[raw_best_index]), 2),
                'raw_mooney_pred': raw_best_eval['mooney_pred'],
                'raw_mooney_low': raw_best_eval['mooney_pred_lower'],
                'raw_mooney_high': raw_best_eval['mooney_pred_upper'],
                'raw_confidence_label': raw_best_eval['confidence_label'],
                'raw_ood_flag': raw_best_eval['is_ood'],
                'safe_bound_status': safe_status,
                'route_stage_status': route_status,
                'historical_template_count': len(templates),
                'historical_best_actual_saving_pct': np.nan,
                'model_adjusted_saving_pct': 0.0,
                'top_uncertainty_factor_1': '',
                'top_uncertainty_factor_2': '',
                'top_uncertainty_factor_3': '',
                'uncertainty_explanation_text': '',
                'operating_profile_reference_status': 'NONE',
                'operating_profile_reference': {},
            }

        if len(valid_indices) < 20:
            raw_route_status, raw_safe_status, raw_reason = validation_statuses[raw_best_index]
            return no_valid_candidate(
                raw_reason or 'INSUFFICIENT_VALID_CANDIDATES_COUNT', raw_route_status, raw_safe_status
            )

        # Predict Energy across all candidates that cleared route/bounds/Mooney gates.
        cand_rows = []
        for candidate_index in valid_indices:
            evaluation = eval_results[candidate_index]
            r = base_df_row.copy()
            for k, v in evaluation['candidate_setpoints'].items():
                r[k] = v
            cand_rows.append(r)

        df_cands = pd.DataFrame(cand_rows)
        cand_kwhs, cand_kwh_per_tons = self.energy_model.predict(df_cands)

        # Select Minimum kWh/ton Candidate
        best_idx = int(np.argmin(cand_kwh_per_tons))
        best_candidate_index = valid_indices[best_idx]
        best_eval = eval_results[best_candidate_index]
        best_kwh = float(cand_kwhs[best_idx])
        best_kwh_per_ton = float(cand_kwh_per_tons[best_idx])
        best_template = candidate_templates[best_candidate_index]
        uncertainty = uncertainty_explanation(best_template) if best_template is not None else ''
        uncertainty_factors = []
        if best_template is not None:
            factor_labels = {
                'raw_material_lot_difference': 'raw_material_lot_difference',
                'supplier_COA_difference': 'supplier_COA_difference',
                'ambient_temperature_difference': 'ambient_temperature_difference',
                'ambient_humidity_difference': 'ambient_humidity_difference',
                'material_initial_temperature_difference': 'material_initial_temperature_difference',
                'batch_weight_difference': 'batch_weight_difference',
                'fill_factor_difference': 'fill_factor_difference',
                'mixer_line_difference': 'mixer_line_difference',
                'route_difference': 'route_difference',
                'historical_batch_age_days': 'historical_batch_age_or_recency',
            }
            uncertainty_factors = sorted(
                [(float(best_template.get(key)), label) for key, label in factor_labels.items() if pd.notna(best_template.get(key)) and float(best_template.get(key)) > 0],
                reverse=True,
            )[:3]

        saving_kwh = actual_kwh - best_kwh
        saving_pct = (saving_kwh / max(actual_kwh, 1.0)) * 100.0

        # Check Gate: Saving Threshold & Quality Constraints
        saving_pass = saving_kwh > 0.0 and ((saving_kwh >= 3.0) or (saving_pct >= 8.0))
        confidence_pass = (best_eval['confidence_label'] != 'LOW')
        ood_pass = (not best_eval['is_ood'])
        mooney_spec_pass = best_eval['inside_spec']

        if saving_kwh <= 0.0:
            return no_valid_candidate(
                'REJECTED_NEGATIVE_OR_ZERO_SAVING',
                status='REJECTED_NEGATIVE_OR_ZERO_SAVING',
            )
        if not saving_pass:
            return no_valid_candidate('SAVING_BELOW_THRESHOLD')
        elif not confidence_pass:
            return no_valid_candidate('MOONEY_CONFIDENCE_LOW')
        elif not ood_pass:
            return no_valid_candidate('CANDIDATE_OUT_OF_DISTRIBUTION')
        elif not mooney_spec_pass:
            return no_valid_candidate('MOONEY_INTERVAL_OUTSIDE_SPEC')

        return {
            'recommendation_status': 'RECOMMENDED',
            'rejection_reason': 'GATE_PASS_OPTIMAL',
            'actual_kwh_per_batch': round(actual_kwh, 2),
            'recommended_kwh_per_batch': round(best_kwh, 2),
            'actual_kwh_per_ton': round(actual_kwh_per_ton, 2),
            'recommended_kwh_per_ton': round(best_kwh_per_ton, 2),
            'predicted_baseline_kwh_per_batch': round(mode_b_current_kwh, 2),
            'predicted_baseline_kwh_per_ton': round(actual_kwh_per_ton, 2),
            'estimated_saving_kwh': round(saving_kwh, 2),
            'estimated_saving_pct': round(saving_pct, 2),
            **energy_saving_fields(best_kwh),
            'mooney_pred': best_eval['mooney_pred'],
            'mooney_pred_current': current_eval['mooney_pred'],
            'mooney_pred_lower': best_eval['mooney_pred_lower'],
            'mooney_pred_upper': best_eval['mooney_pred_upper'],
            'spec_lower': spec_lower,
            'spec_upper': spec_upper,
            'confidence_label': best_eval['confidence_label'],
            'ood_flag': best_eval['is_ood'],
            'valid_candidates_count': len(valid_indices),
            'total_candidates_evaluated': len(candidates),
            'recommended_setpoints': best_eval['candidate_setpoints'],
            'raw_candidate_setpoints': raw_best_eval['candidate_setpoints'],
            'raw_recommended_kwh_per_batch': round(float(all_candidate_kwh[raw_best_index]), 2),
            'raw_mooney_pred': raw_best_eval['mooney_pred'],
            'raw_mooney_low': raw_best_eval['mooney_pred_lower'],
            'raw_mooney_high': raw_best_eval['mooney_pred_upper'],
            'raw_confidence_label': raw_best_eval['confidence_label'],
            'raw_ood_flag': raw_best_eval['is_ood'],
            'safe_bound_status': 'WITHIN_SAFE_BOUNDS',
            'route_stage_status': 'VALID_ROUTE_STAGES',
            'historical_template_count': len(templates),
            'historical_best_actual_saving_pct': best_template.get('historical_best_actual_saving_pct', np.nan) if best_template is not None else np.nan,
            'model_adjusted_saving_pct': round(saving_pct, 2),
            'top_uncertainty_factor_1': uncertainty_factors[0][1] if len(uncertainty_factors) > 0 else '',
            'top_uncertainty_factor_2': uncertainty_factors[1][1] if len(uncertainty_factors) > 1 else '',
            'top_uncertainty_factor_3': uncertainty_factors[2][1] if len(uncertainty_factors) > 2 else '',
            'uncertainty_explanation_text': uncertainty,
            'operating_profile_reference_status': 'OPERATING_PROFILE_REFERENCE' if best_template is not None else 'NONE',
            'operating_profile_reference': best_template.get('profile_reference', {}) if best_template is not None else {},
            'selected_template_id': best_template.get('template_id', '') if best_template is not None else '',
        }
