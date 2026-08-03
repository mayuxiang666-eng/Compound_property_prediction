# ============================================================================
# Inspect Database Batch Mapping for Order 2325066
# ============================================================================
# Lists all raw database rows for Order 2325066:
# - Batch Number
# - Lab Mooney Viscosity (MNY)
# - SampleID / PalletID / OrderID
# - Detailed PLC stage features (Stage 2 Dry Mixing Power, Stage 6 Bottom Torque)
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if pipeline_root not in sys.path:
    sys.path.insert(0, pipeline_root)

def inspect_order_2325066():
    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../../data/stage_statistics_enriched_all_features_weather_v4.csv',
    ))
    if not os.path.exists(data_path):
        data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '../../data/enriched_mny_all.csv',
        ))

    df_raw = pd.read_csv(data_path, low_memory=False)
    if 'MNY' not in df_raw.columns and 'Mooney_Viscosity' in df_raw.columns:
        df_raw['MNY'] = df_raw['Mooney_Viscosity']
    if 'CompoundName' not in df_raw.columns and 'Compound' in df_raw.columns:
        df_raw['CompoundName'] = df_raw['Compound']
    if 'OrderID' not in df_raw.columns and 'Order_No' in df_raw.columns:
        df_raw['OrderID'] = df_raw['Order_No']
    if 'BatchNumber' not in df_raw.columns and 'Batch_No' in df_raw.columns:
        df_raw['BatchNumber'] = df_raw['Batch_No']

    order_df = df_raw[df_raw['OrderID'].astype(str).str.contains('2325066')].copy()
    order_df['BatchNumber_num'] = pd.to_numeric(order_df['BatchNumber'], errors='coerce')
    order_df = order_df.sort_values(by='BatchNumber_num').reset_index(drop=True)

    print("=" * 95)
    print(f"  ALL DATABASE ROWS FOR ORDER 2325066 (Total Rows: {len(order_df)})")
    print("=" * 95)

    cols_to_print = ['OrderID', 'BatchNumber_num', 'MNY', 'CompoundName']
    sample_cols = [c for c in ['MNY_SampleID', 'LabSampleID', 'PalletID', 'SampleID'] if c in order_df.columns]
    cols_to_print.extend(sample_cols)

    # Process features if available
    proc_cols = [c for c in ['Stage2_DryMixing_power_Mean', 'Stage6_BottomMixing_Torque_Mean', 'Stage6_BottomMixing_temp_Mean'] if c in order_df.columns]
    cols_to_print.extend(proc_cols)

    print(order_df[cols_to_print].to_string())

    print("\n--- MNY SUMMARY BY VALUE ---")
    for mny_val, group in order_df.groupby('MNY'):
        print(f"MNY = {mny_val:.2f} -> Batches: {group['BatchNumber_num'].tolist()}")

if __name__ == '__main__':
    inspect_order_2325066()
