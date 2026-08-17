"""
Layer 4: Rule-Based Target-Delta Recommender (MVP Scheme A)
Maps user's relative requirement (e.g. "I want to reduce 2 MU") to executable Step SP recommendations
aiming to shift current batch toward historical best window with PE rules enforcement.
Outputs: target_delta_recommendation.csv
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Relative_Mooney_Guidance_Pipeline.contracts.guidance_contract import TargetDeltaRequest, StepRecommendation

def recommend_target_delta_actions(
    request: TargetDeltaRequest,
    df_current_batch_steps: pd.DataFrame,
    df_distance_to_best: pd.DataFrame,
    out_csv_path: Optional[str] = None
) -> List[Dict]:
    """
    Generates step SP recommendations to satisfy user's target_delta_mu request.
    """
    recommendations = []
    target_reduction = abs(request.target_delta_mu) if request.target_delta_mu < 0 else 0.0

    # Sort adjustable steps by distance score (highest priority first)
    adj_dist = df_distance_to_best[df_distance_to_best['adjustable_flag'] == True].copy()
    adj_dist = adj_dist.sort_values(by='distance_score', ascending=False)

    # Determine adjustment intensity scale based on user's target delta
    # e.g., 2 MU target -> 40% shift towards best; 4 MU target -> 80% shift towards best
    shift_factor = min(1.0, max(0.2, target_reduction / 5.0))

    for _, row in df_current_batch_steps.iterrows():
        grp = row['GroupName']
        step_no = int(row['StepNo'])
        phase = str(row.get('mapped_phase', 'Unknown'))
        is_loading = (step_no in [1, 2] and grp == 'AVR_MST') or 'Loading' in phase

        for param in ['time_SP', 'rotor_SP', 'temp_SP', 'ram_SP']:
            if param not in row or pd.isna(row[param]):
                continue

            curr_sp = float(row[param])
            rec_sp = curr_sp
            delta_sp = 0.0
            direction = 'STABLE'
            reason = 'Parameter in normal range'
            gate_status = 'PASS'
            conf = 'HIGH'

            # Lookup distance record
            dist_match = df_distance_to_best[
                (df_distance_to_best['GroupName'] == grp) &
                (df_distance_to_best['StepNo'] == step_no) &
                (df_distance_to_best['param_name'] == param)
            ]

            if len(dist_match) > 0 and not is_loading and param != 'ram_SP' and target_reduction > 0:
                dist_row = dist_match.iloc[0]
                best_p50 = float(dist_row['best_p50'])
                dist_score = float(dist_row['distance_score'])

                # PE Rule Enforcement
                if 'WetMixing' in phase and param == 'rotor_SP':
                    # PE Rule: Lowering wet rotor extends mixing duration and improves dispersion -> lowers Mooney
                    if curr_sp > 20.0:
                        delta_sp = -5.0  # discrete step: 5 rpm
                        rec_sp = max(15.0, curr_sp + delta_sp)
                        direction = 'PULL_DOWN'
                        reason = f'WetMixing rotor reduced by 5 rpm (toward best {best_p50} rpm) to extend shear duration and improve silica dispersion'
                        conf = 'HIGH'
                
                elif 'WetMixing' in phase and param == 'time_SP':
                    # WetMixing duration extension
                    if curr_sp < best_p50:
                        delta_sp = round((best_p50 - curr_sp) * shift_factor, 1)
                        # Quantize to 5s discrete step
                        delta_sp = max(5.0, round(delta_sp / 5.0) * 5.0)
                        rec_sp = curr_sp + delta_sp
                        direction = 'PULL_DOWN'
                        reason = f'WetMixing duration extended by +{delta_sp}s to increase mechanical energy input'
                        conf = 'HIGH'

                elif 'DryMixing' in phase and param == 'time_SP':
                    # DryMixing duration extension
                    if curr_sp < best_p50:
                        delta_sp = 5.0  # discrete step: 5s
                        rec_sp = curr_sp + delta_sp
                        direction = 'PULL_DOWN'
                        reason = f'DryMixing duration extended by +5s (toward best {best_p50}s) to ensure polymer breakdown'
                        conf = 'MEDIUM'

                elif 'PID' in phase and request.material_system == 'Silica' and grp == 'AVR_MSB' and param == 'time_SP':
                    # PE Rule: For Silica, adjust lower mixer MSB duration (10s step), NOT upper mixer MST (prevents sticky door)
                    delta_sp = 10.0  # discrete step: 10s
                    rec_sp = curr_sp + delta_sp
                    direction = 'PULL_DOWN'
                    reason = 'Silica PID reaction extended in MSB lower mixer by +10s (MST upper mixer locked to avoid door sticking)'
                    conf = 'HIGH'

                elif 'PID' in phase and request.material_system == 'CarbonBlack' and grp == 'AVR_MST' and param == 'time_SP':
                    # PE Rule: For CB, increase MST PID duration
                    delta_sp = 5.0
                    rec_sp = curr_sp + delta_sp
                    direction = 'PULL_DOWN'
                    reason = f'Carbon Black PID duration extended by +5s to complete carbon black incorporation'
                    conf = 'HIGH'

            # Safety Gate Verification
            if param == 'rotor_SP' and (rec_sp < 5.0 or rec_sp > 50.0):
                gate_status = 'BLOCKED_ROTOR_LIMIT'
            if is_loading and delta_sp != 0.0:
                gate_status = 'BLOCKED_LOADING_LOCKED'
            if param == 'ram_SP' and delta_sp != 0.0:
                gate_status = 'BLOCKED_RAM_LOCKED'

            # Heuristic Target Intent (NOT empirical model prediction or causal effect!)
            # Marked explicitly as UNCALIBRATED_HEURISTIC_INTENT to avoid circular claims
            if direction == 'PULL_DOWN':
                heuristic_intent = f"INTENT_PULL_DOWN_TARGET_{target_reduction}MU"
                conf = 'PROTOTYPE_HEURISTIC_UNVERIFIED'
            else:
                heuristic_intent = "INTENT_MAINTAIN_CURRENT"
                conf = 'N/A'

            recommendations.append({
                'user_target_delta_mu': request.target_delta_mu,
                'recommended_action': f"{grp}_Step{step_no}_{param}" if delta_sp != 0 else 'KEEP_CURRENT',
                'GroupName': grp,
                'StepNo': step_no,
                'param_name': param,
                'current_SP': round(curr_sp, 2),
                'recommended_SP': round(rec_sp, 2),
                'delta_SP': round(delta_sp, 2),
                'rule_intended_direction': direction,
                'heuristic_target_intent': heuristic_intent,
                'confidence': conf,
                'reason': reason,
                'gate_status': gate_status,
                'empirical_validation_status': 'UNVERIFIED_PENDING_B3_RESPONSE_AND_LAB'
            })

    df_rec = pd.DataFrame(recommendations)

    if out_csv_path is None:
        out_csv_path = os.path.join(
            root_dir, 'Relative_Mooney_Guidance_Pipeline', 'reports', 'target_delta_recommendation.csv'
        )
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_rec.to_csv(out_csv_path, index=False, encoding='utf-8-sig')

    return recommendations

if __name__ == '__main__':
    print("[OK] Rule-Based Target-Delta Recommender compiled.")
