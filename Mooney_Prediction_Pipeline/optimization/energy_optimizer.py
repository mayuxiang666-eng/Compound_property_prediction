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
from optimization.candidate_generator import SUPPORTED_ROUTE_BRANCHES, generate_route_masked_candidates
from optimization.mooney_constraint_checker import MooneyQualityConstraintChecker


class EnergyMooneyOptimizer:
    def __init__(self, energy_model: MixingEnergyPredictionModel, mooney_checker: MooneyQualityConstraintChecker):
        self.energy_model = energy_model
        self.mooney_checker = mooney_checker

    def optimize_batch(
        self,
        base_df_row: pd.Series,
        bounds_row: pd.Series,
        spec_lower: float,
        spec_upper: float,
        n_candidates=60,
        bounds_rejection_reason='SAFE_BOUNDS_UNAVAILABLE',
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
                'predicted_baseline_kwh_per_batch': round(float(base_kwh[0]), 2),
                'predicted_baseline_kwh_per_ton': round(float(base_kwh_per_ton[0]), 2),
                'estimated_saving_kwh': 0.0,
                'estimated_saving_pct': 0.0,
                'mooney_pred': np.nan,
                'confidence_label': 'LOW',
                'valid_candidates_count': 0,
                'total_candidates_evaluated': 0,
                'recommended_setpoints': {},
            }

        # 1. Generate Candidates
        seed_context = '|'.join(str(base_df_row.get(column, '')) for column in [
            'OrderID', 'BatchNumber', 'CompoundName', 'MixerLine', 'material_system', 'phase_route'
        ])
        random_state = int.from_bytes(hashlib.sha256(seed_context.encode('utf-8')).digest()[:8], 'little')
        candidates = generate_route_masked_candidates(
            base_df_row,
            bounds_row,
            n_candidates=n_candidates,
            random_state=random_state,
        )

        # 2. Check Mooney Constraints
        eval_results = self.mooney_checker.evaluate_candidates(base_df_row, candidates, spec_lower, spec_upper)

        # 3. Filter Valid Candidates
        valid_evals = [e for e in eval_results if e['is_valid']]

        base_cand_eval = eval_results[0]
        base_cand_row = base_df_row.copy()
        for k, v in base_cand_eval['candidate_setpoints'].items():
            base_cand_row[k] = v

        base_df = pd.DataFrame([base_cand_row])
        base_kwh, base_kwh_per_ton = self.energy_model.predict(base_df)
        base_kwh = float(base_kwh[0])
        base_kwh_per_ton = float(base_kwh_per_ton[0])

        actual_kwh = float(base_df_row.get('total_kwh_per_batch', base_kwh))
        actual_kwh_per_ton = float(base_df_row.get('kwh_per_ton', base_kwh_per_ton))

        # Check Gate: Minimum Valid Candidates Threshold
        if len(valid_evals) < 20:
            return {
                'recommendation_status': 'REJECTED',
                'rejection_reason': 'INSUFFICIENT_VALID_CANDIDATES_COUNT',
                'actual_kwh_per_batch': round(actual_kwh, 2),
                'recommended_kwh_per_batch': round(base_kwh, 2),
                'actual_kwh_per_ton': round(actual_kwh_per_ton, 2),
                'recommended_kwh_per_ton': round(base_kwh_per_ton, 2),
                'predicted_baseline_kwh_per_batch': round(base_kwh, 2),
                'predicted_baseline_kwh_per_ton': round(base_kwh_per_ton, 2),
                'estimated_saving_kwh': 0.0,
                'estimated_saving_pct': 0.0,
                'mooney_pred': base_cand_eval['mooney_pred'],
                'confidence_label': base_cand_eval['confidence_label'],
                'valid_candidates_count': len(valid_evals),
                'total_candidates_evaluated': len(candidates),
                'recommended_setpoints': base_cand_eval['candidate_setpoints'],
            }

        # Predict Energy across all valid candidates
        cand_rows = []
        for e in valid_evals:
            r = base_df_row.copy()
            for k, v in e['candidate_setpoints'].items():
                r[k] = v
            cand_rows.append(r)

        df_cands = pd.DataFrame(cand_rows)
        cand_kwhs, cand_kwh_per_tons = self.energy_model.predict(df_cands)

        # Select Minimum kWh/ton Candidate
        best_idx = int(np.argmin(cand_kwh_per_tons))
        best_eval = valid_evals[best_idx]
        best_kwh = float(cand_kwhs[best_idx])
        best_kwh_per_ton = float(cand_kwh_per_tons[best_idx])

        saving_kwh = base_kwh - best_kwh
        saving_pct = (saving_kwh / max(base_kwh, 1.0)) * 100.0

        # Check Gate: Saving Threshold & Quality Constraints
        saving_pass = (saving_kwh >= 3.0) or (saving_pct >= 8.0)
        confidence_pass = (best_eval['confidence_label'] != 'LOW')
        ood_pass = (not best_eval['is_ood'])
        mooney_spec_pass = best_eval['inside_spec']

        if not saving_pass:
            status = 'REJECTED'
            reason = 'SAVING_BELOW_THRESHOLD'
        elif not confidence_pass:
            status = 'REJECTED'
            reason = 'MOONEY_CONFIDENCE_LOW'
        elif not ood_pass:
            status = 'REJECTED'
            reason = 'CANDIDATE_OUT_OF_DISTRIBUTION'
        elif not mooney_spec_pass:
            status = 'REJECTED'
            reason = 'MOONEY_INTERVAL_OUTSIDE_SPEC'
        else:
            status = 'RECOMMENDED'
            reason = 'GATE_PASS_OPTIMAL'

        return {
            'recommendation_status': status,
            'rejection_reason': reason,
            'actual_kwh_per_batch': round(actual_kwh, 2),
            'recommended_kwh_per_batch': round(best_kwh, 2) if status == 'RECOMMENDED' else round(actual_kwh, 2),
            'actual_kwh_per_ton': round(actual_kwh_per_ton, 2),
            'recommended_kwh_per_ton': round(best_kwh_per_ton, 2) if status == 'RECOMMENDED' else round(actual_kwh_per_ton, 2),
            'predicted_baseline_kwh_per_batch': round(base_kwh, 2),
            'predicted_baseline_kwh_per_ton': round(base_kwh_per_ton, 2),
            'estimated_saving_kwh': round(saving_kwh, 2) if status == 'RECOMMENDED' else 0.0,
            'estimated_saving_pct': round(saving_pct, 2) if status == 'RECOMMENDED' else 0.0,
            'mooney_pred': best_eval['mooney_pred'],
            'mooney_pred_lower': best_eval['mooney_pred_lower'],
            'mooney_pred_upper': best_eval['mooney_pred_upper'],
            'spec_lower': spec_lower,
            'spec_upper': spec_upper,
            'confidence_label': best_eval['confidence_label'],
            'valid_candidates_count': len(valid_evals),
            'total_candidates_evaluated': len(candidates),
            'recommended_setpoints': best_eval['candidate_setpoints'],
        }
