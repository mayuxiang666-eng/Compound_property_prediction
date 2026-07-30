# ============================================================================
# Stage 3 Delayed-Feedback Online Calibration Engine
# ============================================================================
# Implements online exponentially weighted moving average (EWMA) calibration
# to capture lot-to-lot raw material drift, environmental shift, and time decay:
#
#   y_pred_s3(t) = y_pred_s1_s1b_s2(t) + EWMA(e_{t-k}, lambda)
#   where e_{t-k} = y_actual(t-k) - y_pred_s1_s1b_s2(t-k)
#   k = Lab measurement delay (in batches).
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd

# Add module paths
pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

from feature_engineering.clustering import cluster_silica_carbon_black
from feature_engineering.stage1_recipe_features import extract_stage1_recipe_features
from feature_engineering.stage2_process_features import extract_stage2_process_features
from feature_engineering.silica_pid_feature_builder import build_silica_pid_features
from model_training.effective_weighting import compute_effective_sample_weights
from model_training.hybrid_unified_model import HybridUnifiedMooneyModel
from model_training.label_group_handler import add_label_group_information
from model_training.split_builder import generate_stratified_recipe_splits
from model_training.trend_metrics import evaluate_mooney_predictions


class Stage3DelayedFeedbackCalibrator:
    """Stage 3 Online Delayed-Feedback Calibrator using EWMA residual tracking."""

    def __init__(self, lag_k: int = 3, alpha: float = 0.3, max_offset_abs: float = 4.0):
        self.lag_k = lag_k
        self.alpha = alpha
        self.max_offset_abs = max_offset_abs
        self.state_offsets_ = {}

    def calibrate_time_series(self, df_series: pd.DataFrame, base_preds: np.ndarray, target_col: str = 'MNY', group_col: str = 'CompoundName') -> tuple[np.ndarray, np.ndarray]:
        """Calculates Stage 3 online calibrated predictions and offsets on a time-ordered series.

        Parameters
        ----------
        df_series : pd.DataFrame
            Time-ordered batch dataframe.
        base_preds : np.ndarray
            Uncalibrated predictions from Stage 1 + 1b + 2.
        target_col : str
            Target column name (MNY).
        group_col : str
            Grouping column for tracking compound-level drift.

        Returns
        -------
        calibrated_preds : np.ndarray
        offsets : np.ndarray
        """
        n = len(df_series)
        calibrated_preds = np.copy(base_preds)
        offsets = np.zeros(n)

        # Track rolling residual buffer per compound
        buffers = {}

        actuals = pd.to_numeric(df_series[target_col], errors='coerce').values
        groups = df_series[group_col].values if group_col in df_series.columns else np.zeros(n)

        for i in range(n):
            grp = groups[i]
            if grp not in buffers:
                buffers[grp] = {'history_residuals': [], 'ewma': 0.0}

            buf = buffers[grp]
            offset_curr = buf['ewma']
            # Clip offset to safe boundary
            offset_curr = float(np.clip(offset_curr, -self.max_offset_abs, self.max_offset_abs))
            offsets[i] = offset_curr
            calibrated_preds[i] = base_preds[i] + offset_curr

            # Simulate delayed Lab feedback arrival after k batches
            if not np.isnan(actuals[i]):
                res_i = actuals[i] - base_preds[i]
                buf['history_residuals'].append(res_i)

                # Update EWMA once lab feedback is available
                if len(buf['history_residuals']) >= self.lag_k:
                    feedback_res = buf['history_residuals'][-self.lag_k]
                    buf['ewma'] = (1.0 - self.alpha) * buf['ewma'] + self.alpha * feedback_res

        return calibrated_preds, offsets


def run_stage3_calibration_benchmark():
    print("=" * 80)
    print("  STAGE 3 DELAYED-FEEDBACK ONLINE CALIBRATION BENCHMARK")
    print("=" * 80)

    out_dir = os.path.join(pipeline_root, 'reports', 'v36_explainable_production')
    os.makedirs(out_dir, exist_ok=True)

    # Load Data
    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../../data/stage_statistics_enriched_all_features_weather_v4.csv',
    ))
    if not os.path.exists(data_path):
        data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '../../data/enriched_mny_all.csv',
        ))

    df = pd.read_csv(data_path, low_memory=False)
    if 'MNY' not in df.columns and 'Mooney_Viscosity' in df.columns:
        df['MNY'] = df['Mooney_Viscosity']
    df = df.dropna(subset=['MNY']).copy()

    if 'CompoundName' not in df.columns and 'Compound' in df.columns:
        df['CompoundName'] = df['Compound']
    if 'OrderID' not in df.columns and 'Order_No' in df.columns:
        df['OrderID'] = df['Order_No']

    # Sort chronologically by OrderID / index for time-series simulation
    df = df.sort_values(by=['OrderID'] if 'OrderID' in df.columns else df.index).reset_index(drop=True)

    # PID features
    pid_feats = build_silica_pid_features(df)
    for col in pid_feats.columns:
        df[col] = pid_feats[col]

    df = cluster_silica_carbon_black(df)
    s1_cols = extract_stage1_recipe_features(df)
    s2_cols_base = extract_stage2_process_features(df)
    s2_cols_pid = list(set(s2_cols_base + list(pid_feats.columns)))

    for col in set(s1_cols + s2_cols_pid):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    df = add_label_group_information(df)
    df = compute_effective_sample_weights(df)
    df = generate_stratified_recipe_splits(df, test_size=0.15, val_size=0.15)

    df_train = df[df['_split'] == 'train'].copy()
    df_test = df[df['_split'] == 'test'].copy()

    # Fit Full V3.6 Candidate Model (Silica + CB Subsystems)
    model = HybridUnifiedMooneyModel(use_material_route_matrix=True, use_silica_subsystem=True, use_cb_subsystem=True)
    model.fit(df_train, s1_cols, s2_cols_pid, target_col='MNY', cluster_col='material_system')

    # Uncalibrated predictions (S1 + S1b + S2)
    final_preds_uncal, s1_preds, s1b_biases, s2_res_preds = model.predict(df_test, cluster_col='material_system')
    m_uncal = evaluate_mooney_predictions(df_test['MNY'].values, final_preds_uncal, df_test)

    # Stage 3 Calibration with different Lab Lags (k=1, k=3, k=5)
    calibrator_k3 = Stage3DelayedFeedbackCalibrator(lag_k=3, alpha=0.3)
    calibrated_preds_k3, offsets_k3 = calibrator_k3.calibrate_time_series(df_test, final_preds_uncal, target_col='MNY', group_col='CompoundName')
    m_cal_k3 = evaluate_mooney_predictions(df_test['MNY'].values, calibrated_preds_k3, df_test)

    calibrator_k5 = Stage3DelayedFeedbackCalibrator(lag_k=5, alpha=0.3)
    calibrated_preds_k5, offsets_k5 = calibrator_k5.calibrate_time_series(df_test, final_preds_uncal, target_col='MNY', group_col='CompoundName')
    m_cal_k5 = evaluate_mooney_predictions(df_test['MNY'].values, calibrated_preds_k5, df_test)

    # Record Benchmark Table
    bench_rows = [
        {
            'stage_level': 'Stage 1 + 1b + 2 (Uncalibrated Base)',
            'lab_lag_batches': 'None',
            'overall_MAE': m_uncal['MAE'],
            'overall_RMSE': m_uncal['RMSE'],
            'overall_R2': m_uncal['R2'],
            'overall_Spearman': m_uncal['Spearman_Rho'],
            'overall_DirAcc_pct': m_uncal['Direction_Accuracy'] * 100.0,
            'high_dev_MAE': m_uncal['High_Dev_MAE'],
        },
        {
            'stage_level': 'Stage 3 Online Calibrated (k=3 Lag)',
            'lab_lag_batches': 3,
            'overall_MAE': m_cal_k3['MAE'],
            'overall_RMSE': m_cal_k3['RMSE'],
            'overall_R2': m_cal_k3['R2'],
            'overall_Spearman': m_cal_k3['Spearman_Rho'],
            'overall_DirAcc_pct': m_cal_k3['Direction_Accuracy'] * 100.0,
            'high_dev_MAE': m_cal_k3['High_Dev_MAE'],
        },
        {
            'stage_level': 'Stage 3 Online Calibrated (k=5 Lag)',
            'lab_lag_batches': 5,
            'overall_MAE': m_cal_k5['MAE'],
            'overall_RMSE': m_cal_k5['RMSE'],
            'overall_R2': m_cal_k5['R2'],
            'overall_Spearman': m_cal_k5['Spearman_Rho'],
            'overall_DirAcc_pct': m_cal_k5['Direction_Accuracy'] * 100.0,
            'high_dev_MAE': m_cal_k5['High_Dev_MAE'],
        },
    ]

    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_csv(os.path.join(out_dir, 'stage3_calibration_benchmark.csv'), index=False, encoding='utf-8-sig')

    # Export prediction time series dataset for real-time dashboard visualization
    df_test['stage1_pred'] = s1_preds
    df_test['stage1b_bias'] = s1b_biases
    df_test['stage2_residual_pred'] = s2_res_preds
    df_test['uncalibrated_pred'] = final_preds_uncal
    df_test['stage3_offset_k3'] = offsets_k3
    df_test['calibrated_pred_k3'] = calibrated_preds_k3

    export_cols = [
        'OrderID', 'CompoundName', 'Recipe_Code', 'material_system', 'MNY',
        'stage1_pred', 'stage1b_bias', 'stage2_residual_pred',
        'uncalibrated_pred', 'stage3_offset_k3', 'calibrated_pred_k3'
    ]
    avail_cols = [c for c in export_cols if c in df_test.columns]
    df_test[avail_cols].to_csv(os.path.join(out_dir, 'time_series_calibration_stream.csv'), index=False, encoding='utf-8-sig')

    # Print Summary Table
    print("\n" + "=" * 90)
    print("        STAGE 3 ONLINE CALIBRATION BENCHMARK (Uncalibrated vs Stage 3)")
    print("=" * 90)
    print(f"{'Stage Level':<35} | {'Lag (k)':<8} | {'MAE':<7} | {'RMSE':<7} | {'R2':<7} | {'Spearman':<8} | {'DirAcc(%)':<8}")
    print("-" * 90)
    for _, r in bench_df.iterrows():
        print(f"{r['stage_level']:<35} | {str(r['lab_lag_batches']):<8} | {r['overall_MAE']:<7.4f} | {r['overall_RMSE']:<7.4f} | {r['overall_R2']:<7.4f} | {r['overall_Spearman']:<8.4f} | {r['overall_DirAcc_pct']:<8.2f}%")
    print("=" * 90)

    print(f"\nStage 3 calibration benchmark saved to: {os.path.join(out_dir, 'stage3_calibration_benchmark.csv')}\n")
    return bench_df


if __name__ == '__main__':
    run_stage3_calibration_benchmark()
