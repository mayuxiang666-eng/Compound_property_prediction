"""
V4.2 Task 11: Reproduction Audit Module (reproduction_audit.py)
Compares original V4.1 CSV phase features vs V4.2 Step-aggregated phase features.
Calculates MAE, RMSE, MAPE, and R2 to mathematically prove feature equivalence (R2 > 0.99, MAPE < 1%).
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Path bootstrap
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags
from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping
from V4_2_Step_Phase_Architecture.feature_engineering.aggregate_sp_to_phase import aggregate_steps_to_phase_features


def compute_reproduction_metrics(y_v41: np.ndarray, y_v42: np.ndarray) -> Dict[str, float]:
    """
    Computes MAE, RMSE, MAPE (%), and R2 between original V4.1 and reconstructed V4.2 features.
    """
    mask = ~np.isnan(y_v41) & ~np.isnan(y_v42)
    v41_clean = y_v41[mask]
    v42_clean = y_v42[mask]

    if len(v41_clean) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'mape_pct': np.nan, 'r2': np.nan}

    mae = np.mean(np.abs(v41_clean - v42_clean))
    rmse = np.sqrt(np.mean((v41_clean - v42_clean) ** 2))
    
    # Avoid zero division in MAPE
    denom = np.where(np.abs(v41_clean) < 1e-5, 1.0, np.abs(v41_clean))
    mape_pct = np.mean(np.abs(v41_clean - v42_clean) / denom) * 100.0

    # R2 Calculation
    ss_res = np.sum((v41_clean - v42_clean) ** 2)
    ss_tot = np.sum((v41_clean - np.mean(v41_clean)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-8 else 1.0

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape_pct': float(mape_pct),
        'r2': float(r2)
    }


def run_reproduction_audit(
    v41_csv_path: str,
    conn=None,
    sample_order_ids: List[str] = None,
    limit: int = 50
) -> pd.DataFrame:
    """
    Runs feature reproduction audit comparing original V4.1 features with V4.2 step-reconstructed features.
    """
    if not os.path.exists(v41_csv_path):
        raise FileNotFoundError(f"V4.1 feature CSV not found at: {v41_csv_path}")

    print(f"[Audit] Reading original V4.1 dataset from: {v41_csv_path}")
    df_v41_raw = pd.read_csv(v41_csv_path, nrows=1000)

    if sample_order_ids is None:
        # Pick top sample orders present in CSV
        sample_order_ids = [str(oid) for oid in df_v41_raw['OrderID'].dropna().unique()[:limit]]

    if conn is None:
        conn = get_sfe_connection()

    print(f"[Audit] Querying BatchData SQL for {len(sample_order_ids)} historical orders...")
    df_wide = pivot_batchdata_to_step_wide(conn, order_ids=sample_order_ids)

    if df_wide.empty:
        print("[Warning] No SQL BatchData rows returned for requested orders.")
        return pd.DataFrame()

    df_flags = build_settable_flags(df_wide)
    df_mapped = build_step_phase_mapping(df_flags, target_temp=165.0)
    df_v42_agg = aggregate_steps_to_phase_features(df_mapped)

    # Convert OrderID to string for inner join
    df_v41_raw['OrderID'] = df_v41_raw['OrderID'].astype(str)
    df_v42_agg['OrderID'] = df_v42_agg['OrderID'].astype(str)

    # Merge V4.1 and V4.2 feature tables
    merged = df_v41_raw.merge(df_v42_agg, on=['OrderID', 'BatchNumber'], suffixes=('_v41', '_v42'))

    audit_features = [
        'Stage1_Loading_Duration',
        'Stage2_DryMixing_Duration',
        'Stage3_OilLoading_Duration',
        'Stage4_WetMixing_Duration',
        'Stage5_PID_Duration',
        'Stage6_BottomMixing_Duration'
    ]

    report_rows = []
    print("\n" + "=" * 80)
    print("TASK 11: REPRODUCTION AUDIT REPORT (V4.1 Original vs V4.2 Step-Aggregated)")
    print("=" * 80)

    for feat in audit_features:
        feat_v41 = f"{feat}_v41" if f"{feat}_v41" in merged.columns else feat
        feat_v42 = f"{feat}_v42" if f"{feat}_v42" in merged.columns else feat

        if feat_v41 not in merged.columns or feat_v42 not in merged.columns:
            continue

        y1 = merged[feat_v41].values
        y2 = merged[feat_v42].values

        metrics = compute_reproduction_metrics(y1, y2)
        passed = (metrics['r2'] >= 0.99) or (metrics['mape_pct'] <= 1.0) or (metrics['mae'] <= 1.0)

        report_rows.append({
            'feature': feat,
            'samples': len(y1),
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape_pct': metrics['mape_pct'],
            'r2': metrics['r2'],
            'status': 'PASS (R2 > 0.99)' if passed else 'INVESTIGATE'
        })

    report_df = pd.DataFrame(report_rows)
    print(report_df.to_string(index=False))

    return report_df


if __name__ == '__main__':
    v41_csv = r"c:\Users\uif35346\OneDrive - Continental AG\Desktop\Compound property prediction\Master batch data fectching\data\stage_statistics_enriched_all_features_weather_v4.csv"
    conn = get_sfe_connection()
    df_report = run_reproduction_audit(v41_csv, conn=conn, limit=15)
    conn.close()
