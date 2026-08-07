"""
V4.2 Module A0: BatchData Pivot Builder
Pivots EAV long table SFEPlant.dbo.BatchData into step-level SP/AV wide DataFrame.
"""
import pandas as pd
import pyodbc
from typing import List, Optional

PIVOT_SQL = """
SELECT
    bd.OrderID,
    bd.BatchNumber,
    bd.EquipmentID,
    bd.GroupName,
    bd.StepNo,
    MAX(CASE WHEN bd.VariablePath='SCP-1-Step-Time-rel-s' THEN bd.Target END) AS time_SP,
    MAX(CASE WHEN bd.VariablePath='SCP-1-Step-Time-rel-s' THEN bd.Value  END) AS time_AV,
    MAX(CASE WHEN bd.VariablePath='SCP-3-Temperature-C-F' THEN bd.Target END) AS temp_SP,
    MAX(CASE WHEN bd.VariablePath='SCP-3-Temperature-C-F' THEN bd.Value  END) AS temp_AV,
    MAX(CASE WHEN bd.VariablePath='Rotor-Speed-1/min-Rpm' THEN bd.Target END) AS rotor_SP,
    MAX(CASE WHEN bd.VariablePath='Rotor-Speed-1/min-Rpm' THEN bd.Value  END) AS rotor_AV,
    MAX(CASE WHEN bd.VariablePath='RAM-Pressure-percent'  THEN bd.Target END) AS ram_SP,
    MAX(CASE WHEN bd.VariablePath='RAM-Pressure-percent'  THEN bd.Value  END) AS ram_AV,
    MAX(CASE WHEN bd.VariablePath='SCP-2-Energy-rel-kWh'  THEN bd.Value  END) AS energy_AV,
    MAX(bd.Timestamp) AS ts
FROM [SFEPlant].[dbo].[BatchData] bd
WHERE bd.GroupName IN ('AVR_MST','AVR_MSB')
  AND bd.VariablePath IN (
    'SCP-1-Step-Time-rel-s',
    'SCP-3-Temperature-C-F',
    'Rotor-Speed-1/min-Rpm',
    'RAM-Pressure-percent',
    'SCP-2-Energy-rel-kWh'
  )
{where_clause}
GROUP BY bd.OrderID, bd.BatchNumber, bd.EquipmentID, bd.GroupName, bd.StepNo
ORDER BY bd.OrderID, bd.BatchNumber, bd.GroupName, bd.StepNo;
"""

def pivot_batchdata_to_step_wide(
    conn: pyodbc.Connection,
    order_ids: Optional[List[str]] = None,
    batch_numbers: Optional[List[int]] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pivots SFEPlant.dbo.BatchData long table to step SP/AV wide table.
    """
    where_parts = []
    if order_ids:
        ids_str = ",".join(f"'{oid}'" for oid in order_ids)
        where_parts.append(f"AND bd.OrderID IN ({ids_str})")
    if batch_numbers:
        bn_str = ",".join(str(b) for b in batch_numbers)
        where_parts.append(f"AND bd.BatchNumber IN ({bn_str})")
    if date_start:
        where_parts.append(f"AND bd.Timestamp >= '{date_start}'")
    if date_end:
        where_parts.append(f"AND bd.Timestamp <= '{date_end}'")

    where_clause = " ".join(where_parts)
    sql = PIVOT_SQL.format(where_clause=where_clause)
    
    df = pd.read_sql(sql, conn)
    
    num_cols = ['time_SP', 'time_AV', 'temp_SP', 'temp_AV', 
                'rotor_SP', 'rotor_AV', 'ram_SP', 'ram_AV', 'energy_AV']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

if __name__ == '__main__':
    import os, sys
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
    conn = get_sfe_connection()
    df_sample = pivot_batchdata_to_step_wide(conn, order_ids=['7951751'], batch_numbers=[75])
    print(f"[OK] Task 2: Pivot Builder verified. Rows fetched: {len(df_sample)}")
    print(df_sample[['GroupName', 'StepNo', 'time_SP', 'time_AV', 'temp_SP', 'temp_AV']].to_string(index=False))
    conn.close()
