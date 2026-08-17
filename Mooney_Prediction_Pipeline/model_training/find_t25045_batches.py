import os
import sys
import numpy as np
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

df = pd.read_csv(data_path, low_memory=False)

if 'MNY' not in df.columns and 'Mooney_Viscosity' in df.columns:
    df['MNY'] = df['Mooney_Viscosity']
if 'CompoundName' not in df.columns and 'Compound' in df.columns:
    df['CompoundName'] = df['Compound']
if 'OrderID' not in df.columns and 'Order_No' in df.columns:
    df['OrderID'] = df['Order_No']
if 'BatchNumber' not in df.columns and 'Batch_No' in df.columns:
    df['BatchNumber'] = df['Batch_No']

print("Searching for T25045 in dataset...")
t_mask = df['CompoundName'].astype(str).str.contains('T25045')
sub_t = df[t_mask].copy()
sub_t['MNY_num'] = pd.to_numeric(sub_t['MNY'], errors='coerce')

mny_73 = sub_t[np.isclose(sub_t['MNY_num'], 73.04, atol=0.5)]
print("--- MNY ~ 73.04 ---")
print(mny_73[['OrderID', 'BatchNumber', 'MNY', 'CompoundName']])

mny_62 = sub_t[np.isclose(sub_t['MNY_num'], 62.95, atol=0.5)]
print("\n--- MNY ~ 62.95 ---")
print(mny_62[['OrderID', 'BatchNumber', 'MNY', 'CompoundName']])
