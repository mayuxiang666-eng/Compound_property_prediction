import os
import sys
import pandas as pd

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
if 'OrderID' not in df_raw.columns and 'Order_No' in df_raw.columns:
    df_raw['OrderID'] = df_raw['Order_No']
if 'BatchNumber' not in df_raw.columns and 'Batch_No' in df_raw.columns:
    df_raw['BatchNumber'] = df_raw['Batch_No']

all_order_rows = df_raw[df_raw['OrderID'].astype(str).str.contains('2325066')].copy()
all_order_rows['BatchNumber_num'] = pd.to_numeric(all_order_rows['BatchNumber'], errors='coerce')
all_order_rows = all_order_rows.sort_values(by='BatchNumber_num').reset_index(drop=True)

print("Total raw CSV rows for Order 2325066:", len(all_order_rows))
print("Available BatchNumbers:", all_order_rows['BatchNumber_num'].tolist())
print("MNY null status:", all_order_rows['Mooney_Viscosity'].isna().sum() if 'Mooney_Viscosity' in all_order_rows.columns else all_order_rows['MNY'].isna().sum())
