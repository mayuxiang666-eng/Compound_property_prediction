# ============================================================================
# Audit Real Production Timestamps in Dataset
# ============================================================================
import os
import sys
import pandas as pd

pipeline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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

date_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
print("Date columns found in dataset:", date_cols)

for c in ['OrderStartTime', 'test_result_start_time', 'production_time', 'datetime']:
    if c in df.columns:
        parsed = pd.to_datetime(df[c], errors='coerce').dropna()
        if len(parsed) > 0:
            print(f"Column '{c}': Min Date = {parsed.min()}, Max Date = {parsed.max()}, Total Count = {len(parsed)}")
