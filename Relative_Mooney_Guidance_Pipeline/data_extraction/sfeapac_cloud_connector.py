"""
Mustang Cloud SFEAPAC (HESFESFEPLANT) Read-Only Connector
Connects to AWS RDS read-only replica HESFESFEPLANT using credentials.json["SFEAPAC"].
Strictly enforces:
1. ApplicationIntent=ReadOnly
2. SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED
3. Zero access to local HF_MMS or local archive
4. Scoped queries (no full table scans)
"""
import os
import sys
import json
import pyodbc
import pandas as pd
from typing import List, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
CREDENTIALS_PATH = os.path.join(ROOT_DIR, 'Mooney_Prediction_Pipeline', 'data_processing', 'credentials.json')
ENDPOINT = "mustang-apac-rds-01-dbinstance-01.ch6kpnfcmztf.eu-central-1.rds.amazonaws.com"
DATABASE = "HESFESFEPLANT"

VARIABLES = [
    "SCP-1-Step-Time-rel-s",
    "SCP-3-Temperature-C-F",
    "Rotor-Speed-1/min-Rpm",
    "RAM-Pressure-percent",
    "SCP-2-Energy-rel-kWh"
]

def get_sfeapac_cloud_connection() -> pyodbc.Connection:
    """Returns a secure read-only pyodbc connection to cloud RDS HESFESFEPLANT."""
    if not os.path.exists(CREDENTIALS_PATH):
        raise FileNotFoundError(f"Credentials not found at {CREDENTIALS_PATH}")

    with open(CREDENTIALS_PATH, 'r', encoding='utf-8') as f:
        creds = json.load(f)["SFEAPAC"]

    driver = next((item for item in reversed(pyodbc.drivers()) if "SQL Server" in item), "ODBC Driver 17 for SQL Server")
    conn_str = (
        f"DRIVER={{{driver}}};SERVER=tcp:{ENDPOINT},{creds.get('port', 1433)};DATABASE={DATABASE};"
        f"UID={creds['user']};PWD={creds['password']};Encrypt=yes;TrustServerCertificate=yes;"
        "ApplicationIntent=ReadOnly;MultiSubnetFailover=yes;Connection Timeout=30;"
    )
    conn = pyodbc.connect(conn_str, autocommit=False)
    conn.timeout = 60
    conn.execute("SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED")
    return conn

def query_sfeapac_orders(
    compound_prefix: str = 'M1-T15760',
    limit: int = 100,
    date_start: str = '2024-01-01'
) -> pd.DataFrame:
    """Queries Orders scoped to a specific compound from cloud HESFESFEPLANT."""
    sql = f"""
    SELECT TOP ({limit})
        CAST(o.OrderID AS VARCHAR(64)) AS OrderID,
        o.CompoundDescription AS RecipeCode,
        o.EquipmentID,
        o.OrderStartTime,
        o.OrderEndTime,
        o.PlannedBatchQuantity AS BatchCount
    FROM HESFESFEPLANT.dbo.Orders o WITH (NOLOCK)
    WHERE o.CompoundDescription LIKE '{compound_prefix}%'
      AND o.OrderStartTime >= '{date_start}'
    ORDER BY o.OrderStartTime DESC;
    """
    with get_sfeapac_cloud_connection() as conn:
        df = pd.read_sql_query(sql, conn)
    return df

def query_sfeapac_step_data(
    order_ids: List[str]
) -> pd.DataFrame:
    """Queries step-level EAV BatchData pivoted into wide table from cloud HESFESFEPLANT."""
    if not order_ids:
        return pd.DataFrame()

    ids_str = ",".join(f"'{str(oid).strip()}'" for oid in order_ids)
    sql = f"""
    SELECT
        CAST(bd.OrderID AS VARCHAR(64)) AS OrderID,
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
    FROM HESFESFEPLANT.dbo.BatchData bd WITH (NOLOCK)
    WHERE bd.GroupName IN ('AVR_MST','AVR_MSB')
      AND bd.VariablePath IN (
        'SCP-1-Step-Time-rel-s',
        'SCP-3-Temperature-C-F',
        'Rotor-Speed-1/min-Rpm',
        'RAM-Pressure-percent',
        'SCP-2-Energy-rel-kWh'
      )
      AND CAST(bd.OrderID AS VARCHAR(64)) IN ({ids_str})
    GROUP BY bd.OrderID, bd.BatchNumber, bd.EquipmentID, bd.GroupName, bd.StepNo
    ORDER BY bd.OrderID, bd.BatchNumber, bd.GroupName, bd.StepNo;
    """
    with get_sfeapac_cloud_connection() as conn:
        df = pd.read_sql_query(sql, conn)
    
    numeric_cols = ['time_SP', 'time_AV', 'temp_SP', 'temp_AV', 'rotor_SP', 'rotor_AV', 'ram_SP', 'ram_AV', 'energy_AV']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

if __name__ == '__main__':
    print("[Testing] SFEAPAC Cloud Connection...")
    conn = get_sfeapac_cloud_connection()
    print("[PASS] SFEAPAC Cloud HESFESFEPLANT connected successfully!")
    conn.close()
