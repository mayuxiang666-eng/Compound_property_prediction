import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.append(r"d:\AntigravityIDE\Project")
import db_client
import compound_tracer
import pandas as pd
import numpy as np
import pickle

csv_path = r"C:\Users\uih83490\.gemini\antigravity-ide\brain\0f968684-3a6b-429f-82f1-62de38081140\scratch\mb_mooney_raw_dataset.csv"
df = pd.read_csv(csv_path)

mb_orders = df['MB_OrderID'].astype(str).unique().tolist()
in_orders = ",".join([f"'{o}'" for o in mb_orders])
sql = f"SELECT OrderID, BatchNumber, Curve1, Curve2, Curve3, Curve6 FROM BatchCurve WHERE OrderID IN ({in_orders})"
df_curves = db_client.query_df(sql, database="HESFESFEPLANT")

curve_dict = {}
for _, row in df_curves.iterrows():
    key = (int(row['OrderID']), int(row['BatchNumber']))
    c1 = compound_tracer.decode_curve_hex(row.get('Curve1'))
    c2 = compound_tracer.decode_curve_hex(row.get('Curve2'))
    c3 = compound_tracer.decode_curve_hex(row.get('Curve3'))
    c6 = compound_tracer.decode_curve_hex(row.get('Curve6'))
    if len(c1) == len(c2) == len(c3) == len(c6) and len(c1) > 0:
        curve_dict[key] = {
            'temp': np.array(c1, dtype=np.float32),
            'power': np.array(c2, dtype=np.float32),
            'ram': np.array(c3, dtype=np.float32),
            'speed': np.array(c6, dtype=np.float32)
        }

matched_records = []
for idx, r in df.iterrows():
    key = (int(r['MB_OrderID']), int(r['MB_BatchNumber']))
    if key in curve_dict:
        c = curve_dict[key]
        T_i = len(c['temp'])
        # Check for NaN or Inf
        has_nan = any(np.isnan(c[k]).any() or np.isinf(c[k]).any() for k in ['temp', 'power', 'ram', 'speed'])
        if has_nan:
            continue
        matched_records.append({
            'MB_OrderID': int(r['MB_OrderID']),
            'MB_BatchNumber': int(r['MB_BatchNumber']),
            'FM_OrderID': int(r['FM_OrderID']),
            'FM_PalletID': int(r['FM_PalletID']),
            'AgingHours': float(r['AgingHours']),
            'FM_Mooney': float(r['FM_Mooney']),
            'temp': c['temp'],
            'power': c['power'],
            'ram': c['ram'],
            'speed': c['speed'],
            'T_i': T_i
        })

print(f"Matched valid records without NaN/Inf: {len(matched_records)} out of {len(df)}")
lengths = [m['T_i'] for m in matched_records]
print(f"Curve length range: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

out_path = r"C:\Users\uih83490\.gemini\antigravity-ide\brain\0f968684-3a6b-429f-82f1-62de38081140\scratch\variable_length_dataset.pkl"
with open(out_path, "wb") as f:
    pickle.dump(matched_records, f)
print(f"Saved to {out_path}")
