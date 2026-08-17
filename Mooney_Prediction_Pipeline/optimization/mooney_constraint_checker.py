# ============================================================================
# Step 4: Fast Vectorized Mooney Quality Constraint Checker (V4.1)
# ============================================================================
# Evaluates all candidate setpoint parameters in 1 single matrix prediction call:
# - Predicts Mooney Viscosity \hat{y} for all candidates in parallel
# - Computes 95% Prediction Interval [\hat{y} - 1.96\sigma, \hat{y} + 1.96\sigma]
# - Verifies [\hat{y}_{lower}, \hat{y}_{upper}] \subseteq [Spec_lower, Spec_upper]
# - Rejects candidates if confidence is 'LOW' or if OOD.
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

from model_training.hybrid_unified_model import HybridUnifiedMooneyModel


class MooneyQualityConstraintChecker:
    def __init__(self, mooney_model: HybridUnifiedMooneyModel, default_sigma=1.1):
        self.mooney_model = mooney_model
        self.default_sigma = default_sigma

    def evaluate_candidates(
        self,
        base_df_row: pd.Series,
        candidates: list[dict],
        spec_lower: float,
        spec_upper: float
    ) -> list[dict]:
        """
        Evaluates a list of candidate setpoints in a single batch call.
        """
        if not candidates:
            return []

        # 1. Build DataFrame for all candidates at once
        cand_rows = []
        for cand in candidates:
            r = base_df_row.copy()
            for k, v in cand.items():
                r[k] = v
            cand_rows.append(r)

        df_candidates = pd.DataFrame(cand_rows)

        # 2. Predict Mooney Viscosity for ALL candidates in 1 single call
        uncal_preds, _, _, _ = self.mooney_model.predict(df_candidates, cluster_col='material_system')

        # 3. Determine Sigma & Margin
        mat_sys = base_df_row.get('material_system', 'Silica')
        sigma = 1.2 if mat_sys == 'Silica' else 0.9
        margin = 1.96 * sigma

        results = []
        spec_center = (spec_lower + spec_upper) / 2.0
        spec_margin = (spec_upper - spec_lower) / 2.0

        for idx, cand in enumerate(candidates):
            mny_pred = float(uncal_preds[idx])
            pred_lower = mny_pred - margin
            pred_upper = mny_pred + margin

            inside_spec = (pred_lower >= spec_lower) and (pred_upper <= spec_upper)
            spec_risk_score = abs(mny_pred - spec_center) / max(spec_margin, 1.0)

            confidence_label = 'HIGH' if spec_risk_score <= 0.6 else ('MEDIUM' if spec_risk_score <= 0.9 else 'LOW')
            is_ood = spec_risk_score > 1.2
            is_valid = inside_spec and (confidence_label != 'LOW') and (not is_ood)

            results.append({
                'candidate_setpoints': cand,
                'mooney_pred': round(mny_pred, 2),
                'mooney_pred_lower': round(pred_lower, 2),
                'mooney_pred_upper': round(pred_upper, 2),
                'spec_lower': spec_lower,
                'spec_upper': spec_upper,
                'spec_risk_score': round(spec_risk_score, 3),
                'confidence_label': confidence_label,
                'is_ood': is_ood,
                'inside_spec': inside_spec,
                'is_valid': is_valid,
            })

        return results
