import sys
sys.stdout.reconfigure(encoding='utf-8')
import pyodbc, json
import pandas as pd
import numpy as np

with open('Mooney_Prediction_Pipeline/data_processing/credentials.json') as f:
    creds = json.load(f)

# Target compounds (Top 5 Carbon Black + Top 5 Silica + A00205)
target_compounds = [
    # Carbon Black Top 5 (+ A00205)
    'M1-B15563R1-- 01 006',
    'M1-B00460B--- 04 010',
    'M1-B00163---- 08 010',
    'M1-R00218W2-- 10 011',
    'M1-B00458---- 12 012',
    'M1-A00205---- 09 004',
    # Silica Top 5
    'M1-T15760---- 06 002',
    'M1-T20153W--- 01 001',
    'M1-T09170---- 23 006',
    'M1-T33025W3-- 03 005',
    'M1-T17771---- 05 008'
]

# Check SFEPLANT orders for each of these compounds in recent weeks (Sep 15 - Oct 19)
c_sfe = creds['HF_MMS']
conn_sfe = pyodbc.connect(
    f"DRIVER={c_sfe['driver']};SERVER={c_sfe['server']};DATABASE=SFEPLANT;"
    f"UID={c_sfe['username']};PWD={c_sfe['password']};Encrypt=yes;TrustServerCertificate=yes;",
    timeout=15
)

placeholders = ','.join([f"'{c}'" for c in target_compounds])
sql_orders = f"""
SELECT CompoundName, Equipment, OrderID, OrderStartTime, NoBatchesSet, BatchWeight
FROM SFEPLANT.dbo.Orders WITH (NOLOCK)
WHERE CompoundName IN ({placeholders})
  AND OrderStartTime >= '2026-09-18'
  AND Equipment NOT LIKE 'SCW%'
  AND OrderID NOT LIKE 'H%'
  AND NoBatchesSet >= 20
ORDER BY CompoundName, OrderStartTime DESC
"""
df_orders = pd.read_sql(sql_orders, conn_sfe)
print("=== Recent Orders in SFEPLANT for Target Compounds ===")
print(df_orders.groupby(['CompoundName', 'Equipment']).agg(
    orders=('OrderID', 'count'),
    latest_order=('OrderID', 'first'),
    latest_time=('OrderStartTime', 'first'),
    avg_weight=('BatchWeight', 'mean')
).reset_index().to_string(index=False))

# For each target compound, pick up to 5 recent full orders (approx 200-400 batches per compound) to get full step-level telemetry from SFEPLANT.dbo.BatchData
selected_orders = df_orders.groupby('CompoundName').head(5)
order_list = selected_orders['OrderID'].unique().tolist()
print(f"\nFetching step-level telemetry for {len(order_list)} representative orders across {len(target_compounds)} compounds...")

order_Map = selected_orders.set_index('OrderID')[['CompoundName', 'Equipment', 'BatchWeight']].to_dict('index')

all_step_rows = []
cur = conn_sfe.cursor()
for oid in order_list:
    cur.execute(f"""
        SELECT OrderID, BatchNumber, EquipmentID, GroupName, StepNo, VariablePath, Value, Target
        FROM SFEPLANT.dbo.BatchData WITH (NOLOCK)
        WHERE OrderID = '{oid}'
          AND GroupName IN ('AVR_MST', 'AVR_MSB', 'AVR_RHT', 'AVR_RHB')
          AND VariablePath IN (
              'Rotor-Speed-1/min-Rpm',
              'SCP-1-Step-Time-rel-s',
              'SCP-2-Energy-rel-kWh',
              'SCP-3-Temperature-C-F',
              'SCP-5-Torque-Nm-ft/lbs',
              'Motor-Current-A',
              'MAX-Batch-Energy-kWh',
              'MAX-Batch-Temperature-C-F',
              'Mixing-Time-actual-s'
          )
    """)
    rows = cur.fetchall()
    cname = order_Map[oid]['CompoundName']
    w_nom = order_Map[oid]['BatchWeight']
    for r in rows:
        try:
            val = float(r[6]) if r[6] is not None else np.nan
        except:
            val = np.nan
        try:
            tgt = float(r[7]) if r[7] is not None else np.nan
        except:
            tgt = np.nan
        all_step_rows.append({
            'CompoundName': cname,
            'OrderID': r[0],
            'BatchNumber': int(r[1]),
            'EquipmentID': r[2],
            'GroupName': r[3],
            'StepNo': int(r[4]),
            'VariablePath': r[5],
            'Value': val,
            'Target': tgt,
            'NominalWeight_kg': float(w_nom) if w_nom else 270.0
        })

cur.close()
conn_sfe.close()

df_steps_raw = pd.DataFrame(all_step_rows)
print(f"Fetched {len(df_steps_raw)} step variable rows! Saving to scratch/top_compounds_steps_raw.csv...")
df_steps_raw.to_csv('scratch/top_compounds_steps_raw.csv', index=False)
print("Done!")
