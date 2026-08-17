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
if 'OrderID' not in df.columns and 'Order_No' in df.columns:
    df['OrderID'] = df['Order_No']
if 'BatchNumber' not in df.columns and 'Batch_No' in df.columns:
    df['BatchNumber'] = df['Batch_No']
if 'MNY' not in df.columns and 'Mooney_Viscosity' in df.columns:
    df['MNY'] = df['Mooney_Viscosity']
if 'CompoundName' not in df.columns and 'Compound' in df.columns:
    df['CompoundName'] = df['Compound']

sub_t = df[df['CompoundName'].astype(str).str.contains('T25045')].copy()
sub_t['MNY_num'] = pd.to_numeric(sub_t['MNY'], errors='coerce')
print("Highest MNY in T25045:")
print(sub_t.sort_values(by='MNY_num', ascending=False)[['OrderID', 'BatchNumber', 'MNY', 'CompoundName']].head(10))
