"""
Mooney Lab Database Connector
=============================
Database Server: 10.246.97.159\\AIDATASERVER
Database:        MooneyIntegratedData
Tables:          core.LabCurrent, core.LabGroup, core.LabelGroupMembership, core.Batch
Credentials:     Configured in db_config.json ['AIDATASERVER']
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.parse
import pandas as pd

CONFIG_FILE = Path(__file__).parent / "db_config.json"


def get_available_driver() -> str:
    """Detect available ODBC driver for SQL Server."""
    try:
        import pyodbc
        available = pyodbc.drivers()
        for drv in [
            "ODBC Driver 18 for SQL Server",
            "ODBC Driver 17 for SQL Server",
            "SQL Server Native Client 11.0",
            "SQL Server",
        ]:
            if drv in available:
                return drv
    except Exception:
        pass
    return "SQL Server"


def load_lab_config(env_key: str = "AIDATASERVER") -> Dict[str, Any]:
    """Load lab database configuration from db_config.json."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    if env_key not in config:
        raise KeyError(f"Key '{env_key}' not found in {CONFIG_FILE}. Available: {list(config.keys())}")

    return config[env_key]


def get_connection_string(
    env_key: str = "AIDATASERVER",
    database: Optional[str] = None,
    driver: Optional[str] = None,
) -> str:
    """Generate SQLAlchemy connection string."""
    cfg = load_lab_config(env_key)
    target_db = database or cfg.get("database", "MooneyIntegratedData")
    driver_name = driver or get_available_driver()

    odbc_str = (
        f"DRIVER={{{driver_name}}};"
        f"SERVER={cfg['host']};"
        f"DATABASE={target_db};"
        f"UID={cfg['user']};"
        f"PWD={cfg['password']};"
        f"TrustServerCertificate=yes;"
    )
    encoded = urllib.parse.quote_plus(odbc_str)
    return f"mssql+pyodbc:///?odbc_connect={encoded}"


def query_dataframe(
    query: str,
    env_key: str = "AIDATASERVER",
    database: Optional[str] = None,
    driver: Optional[str] = None,
) -> pd.DataFrame:
    """Execute SQL query and return DataFrame."""
    cfg = load_lab_config(env_key)
    target_db = database or cfg.get("database", "MooneyIntegratedData")
    driver_name = driver or get_available_driver()

    try:
        from sqlalchemy import create_engine
        conn_str = get_connection_string(env_key, database=target_db, driver=driver_name)
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
    except Exception:
        import pyodbc
        conn_str = (
            f"DRIVER={{{driver_name}}};"
            f"SERVER={cfg['host']};"
            f"DATABASE={target_db};"
            f"UID={cfg['user']};"
            f"PWD={cfg['password']};"
            f"TrustServerCertificate=yes;"
        )
        with pyodbc.connect(conn_str) as conn:
            return pd.read_sql(query, conn)


def get_mooney_by_order(order_id: str) -> pd.DataFrame:
    """
    Get all Mooney Viscosity test records for a specific OrderID.
    Returns: DataFrame with OrderID, PalletID, SampleID, EquipmentID, TestResult, TestStartTime
    """
    sql = f"""
    SELECT 
        OrderID,
        PalletID,
        SampleID,
        EquipmentID,
        CompoundName,
        TestResult AS Mooney_MU,
        TestTarget,
        TestStatus,
        TestStartTime,
        TestEndTime
    FROM core.LabCurrent
    WHERE OrderID = '{order_id}'
    ORDER BY TestStartTime ASC
    """
    return query_dataframe(sql)


def get_mooney_by_compound(compound_pattern: str, top_n: int = 50) -> pd.DataFrame:
    """Query recent Mooney test records for a specific rubber compound."""
    sql = f"""
    SELECT TOP {top_n}
        OrderID,
        PalletID,
        SampleID,
        EquipmentID,
        CompoundName,
        TestResult AS Mooney_MU,
        TestStartTime
    FROM core.LabCurrent
    WHERE CompoundName LIKE '%{compound_pattern}%'
    ORDER BY TestStartTime DESC
    """
    return query_dataframe(sql)


def test_connection():
    """Test connection to AIDATASERVER and print diagnostics."""
    print("=" * 65)
    print("Testing Connection to Mooney Lab Database (AIDATASERVER)...")
    print("=" * 65)
    cfg = load_lab_config("AIDATASERVER")
    print(f"Host:     {cfg['host']}")
    print(f"User:     {cfg['user']}")
    print(f"Database: {cfg['database']}")
    
    try:
        diag = query_dataframe("SELECT @@VERSION AS Ver, DB_NAME() AS CurrentDB, SYSTEM_USER AS CurrentUser")
        print(">> [OK] Connection Successful!")
        print(f"Current DB:     {diag['CurrentDB'].iloc[0]}")
        print(f"Server User:    {diag['CurrentUser'].iloc[0]}")
        print(f"Server Version: {diag['Ver'].iloc[0].splitlines()[0]}")
        
        print("\nVerifying Lab Mooney Data for Order 8067458:")
        df_8067458 = get_mooney_by_order("8067458")
        print(df_8067458[['OrderID', 'PalletID', 'EquipmentID', 'Mooney_MU', 'TestStartTime']].to_string(index=False))
        return True
    except Exception as e:
        print(">> [ERROR] Connection failed:", e)
        return False


if __name__ == "__main__":
    test_connection()
