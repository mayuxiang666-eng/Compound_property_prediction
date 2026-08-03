# ============================================================================
# Stage 3 Delayed-Feedback Online Calibration Engine V1.1 (Adaptive Change-Point)
# ============================================================================
# Implements online exponentially weighted moving average (EWMA) calibration
# with adaptive change-point detection for raw material lot-to-lot baseline jumps:
#
#   y_pred_s3(t) = y_pred_base(t) + EWMA(e_{t-k}, alpha_t)
#   where alpha_t = 0.65 if CUSUM/residual jump detected (lot shift)
#         alpha_t = 0.20 during steady lot production (noise suppression)
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd


class Stage3DelayedFeedbackCalibrator:
    """Stage 3 Online Delayed-Feedback Calibrator using Adaptive EWMA residual tracking."""

    def __init__(self, lag_k: int = 3, alpha_base: float = 0.3, max_offset_abs: float = 4.0, adaptive: bool = True):
        self.lag_k = lag_k
        self.alpha_base = alpha_base
        self.max_offset_abs = max_offset_abs
        self.adaptive = adaptive
        self.state_offsets_ = {}

    def calibrate_time_series(self, df_series: pd.DataFrame, base_preds: np.ndarray, target_col: str = 'MNY', group_col: str = 'CompoundName') -> tuple[np.ndarray, np.ndarray]:
        """Calculates Stage 3 online calibrated predictions and offsets on a time-ordered series."""
        n = len(df_series)
        calibrated_preds = np.copy(base_preds)
        offsets = np.zeros(n)

        buffers = {}

        actuals = pd.to_numeric(df_series[target_col], errors='coerce').values
        groups = df_series[group_col].values if group_col in df_series.columns else np.zeros(n)

        for i in range(n):
            grp = groups[i]
            if grp not in buffers:
                buffers[grp] = {'history_residuals': [], 'ewma': 0.0}

            buf = buffers[grp]
            offset_curr = buf['ewma']
            offset_curr = float(np.clip(offset_curr, -self.max_offset_abs, self.max_offset_abs))
            offsets[i] = offset_curr
            calibrated_preds[i] = base_preds[i] + offset_curr

            if not np.isnan(actuals[i]):
                res_i = actuals[i] - base_preds[i]
                buf['history_residuals'].append(res_i)

                if len(buf['history_residuals']) >= self.lag_k:
                    feedback_res = buf['history_residuals'][-self.lag_k]

                    if self.adaptive and len(buf['history_residuals']) > self.lag_k:
                        prev_res = buf['history_residuals'][-self.lag_k - 1]
                        delta_res = abs(feedback_res - prev_res)

                        # Detect Change-Point (Lot Shift or Step Jump)
                        if delta_res >= 2.0 or (np.sign(feedback_res) == np.sign(prev_res) and abs(feedback_res) >= 2.5):
                            alpha_eff = 0.65  # Fast response during lot shift
                        else:
                            alpha_eff = 0.20  # Smooth tracking during steady production
                    else:
                        alpha_eff = self.alpha_base

                    buf['ewma'] = (1.0 - alpha_eff) * buf['ewma'] + alpha_eff * feedback_res

        return calibrated_preds, offsets
