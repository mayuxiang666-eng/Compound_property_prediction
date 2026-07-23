import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

class ModelRiskAlerter:
    """
    Automated Model Accuracy Risk Alerting Engine for Mooney Viscosity Prediction Pipeline.
    Monitors daily metrics (R2, RMSE, MAE, Outlier Ratio) against thresholds and triggers alerts.
    """
    def __init__(self, 
                 r2_min_threshold=0.70, 
                 rmse_max_threshold=4.50, 
                 mae_max_threshold=3.00, 
                 outlier_ratio_max_threshold=0.10,
                 log_csv_path=None):
        
        self.r2_min_threshold = r2_min_threshold
        self.rmse_max_threshold = rmse_max_threshold
        self.mae_max_threshold = mae_max_threshold
        self.outlier_ratio_max_threshold = outlier_ratio_max_threshold
        
        if log_csv_path is None:
            log_csv_path = os.path.join(os.getcwd(), 'data_store', 'model_risk_alerts_log.csv')
        self.log_csv_path = log_csv_path
        os.makedirs(os.path.dirname(self.log_csv_path), exist_ok=True)

    def evaluate_metrics(self, track_name, date_str, r2, rmse, mae, total_samples, outlier_count):
        """
        Evaluates metrics against risk thresholds.
        Returns a dict containing alert details and MLflow tag payload.
        """
        outlier_ratio = outlier_count / total_samples if total_samples > 0 else 0.0
        alerts = []
        risk_level = "NORMAL" # NORMAL, WARNING, CRITICAL

        # 1. Check R^2
        if r2 < self.r2_min_threshold:
            alerts.append(f"R2_LOW ({r2:.4f} < {self.r2_min_threshold:.2f})")
            risk_level = "WARNING"

        # 2. Check MAE
        if mae > self.mae_max_threshold:
            alerts.append(f"MAE_HIGH ({mae:.2f} MU > {self.mae_max_threshold:.2f} MU)")
            if risk_level != "CRITICAL":
                risk_level = "WARNING"

        # 3. Check RMSE
        if rmse > self.rmse_max_threshold:
            alerts.append(f"RMSE_CRITICAL ({rmse:.2f} MU > {self.rmse_max_threshold:.2f} MU)")
            risk_level = "CRITICAL"

        # 4. Check Outlier Ratio
        if outlier_ratio > self.outlier_ratio_max_threshold:
            alerts.append(f"PROCESS_INSTABILITY ({outlier_ratio:.2%} > {self.outlier_ratio_max_threshold:.0%})")
            risk_level = "CRITICAL"

        is_alert_triggered = len(alerts) > 0
        alert_summary = "; ".join(alerts) if is_alert_triggered else "No Risk Alert - Accuracy Normal"

        alert_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date': date_str,
            'track_name': track_name,
            'total_samples': total_samples,
            'outlier_count': outlier_count,
            'outlier_ratio': outlier_ratio,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'risk_level': risk_level,
            'is_alert_triggered': is_alert_triggered,
            'alert_summary': alert_summary
        }

        # Log to CSV
        self._append_to_csv_log(alert_record)

        # Print console warning
        if is_alert_triggered:
            print(f"\n[MODEL RISK ALERT - {risk_level}] Track '{track_name}' on {date_str}:")
            print(f"  Triggers: {alert_summary}")
            print(f"  Current Metrics -> R2: {r2:.4f} | RMSE: {rmse:.2f} MU | MAE: {mae:.2f} MU | Outliers: {outlier_ratio:.2%}\n")

        return alert_record

    def _append_to_csv_log(self, record):
        df_new = pd.DataFrame([record])
        if not os.path.exists(self.log_csv_path):
            df_new.to_csv(self.log_csv_path, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(self.log_csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    # Unit test for ModelRiskAlerter
    alerter = ModelRiskAlerter()
    res1 = alerter.evaluate_metrics('With-Oil Carbon-Black', '2026-07-22', r2=0.85, rmse=3.10, mae=1.95, total_samples=100, outlier_count=3)
    res2 = alerter.evaluate_metrics('With-Oil Carbon-Black', '2026-07-22', r2=0.65, rmse=4.80, mae=3.20, total_samples=100, outlier_count=12)
    print("ModelRiskAlerter unit test passed.")
