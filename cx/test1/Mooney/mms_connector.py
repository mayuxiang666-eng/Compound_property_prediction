"""
MMS Tool Database Connector & Mathematical Correlation Analysis Helper
========================================================================
Tool Path: Q:\\Public\\PE\\MMSTool-V1.45\\MMSTool.exe
Config:    D:\\Antigravity\\Mooney\\db_config.json
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

CONFIG_FILE = Path(__file__).parent / "db_config.json"


def load_db_config(env_key: str = "SFEAPAC") -> Dict[str, Any]:
    """Load database credentials from db_config.json."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    if env_key not in config:
        raise KeyError(f"Key '{env_key}' not found in {CONFIG_FILE}. Available: {list(config.keys())}")
    
    return config[env_key]


def get_available_driver() -> str:
    """Auto-detect available ODBC driver installed on the system."""
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


def get_connection_string(
    env_key: str = "SFEAPAC",
    database: Optional[str] = None,
    driver: Optional[str] = None,
) -> str:
    """
    Generate SQLAlchemy / pyodbc compatible connection string.
    
    Format for pyodbc / sqlalchemy:
    mssql+pyodbc://<user>:<password>@<host>:<port>/<database>?driver=ODBC+Driver+17+for+SQL+Server
    """
    cfg = load_db_config(env_key)
    user = cfg["user"]
    password = cfg["password"]
    host = cfg["host"]
    port = cfg["port"]
    target_db = database or cfg.get("database", "HESFESFEPLANT")
    driver_name = driver or get_available_driver()
    
    return f"mssql+pyodbc://{user}:{password}@{host}:{port}/{target_db}?driver={driver_name.replace(' ', '+')}"


def query_dataframe(
    query: str,
    env_key: str = "SFEAPAC",
    database: Optional[str] = None,
    driver: Optional[str] = None,
) -> pd.DataFrame:
    """
    Execute SQL query and return results directly as a pandas DataFrame
    for mathematical correlation, regression, and data analysis.
    """
    cfg = load_db_config(env_key)
    target_db = database or cfg.get("database", "HESFESFEPLANT")
    driver_name = driver or get_available_driver()
    
    try:
        from sqlalchemy import create_engine
        conn_str = get_connection_string(env_key, database=target_db, driver=driver_name)
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            return df
    except Exception:
        # Fallback to pyodbc direct connection
        import pyodbc
        conn_str = (
            f"DRIVER={{{driver_name}}};"
            f"SERVER={cfg['host']},{cfg['port']};"
            f"DATABASE={target_db};"
            f"UID={cfg['user']};"
            f"PWD={cfg['password']}"
        )
        with pyodbc.connect(conn_str) as conn:
            df = pd.read_sql(query, conn)
            return df


if __name__ == "__main__":
    print(f"Loaded database configuration for SFEAPAC:")
    cfg = load_db_config("SFEAPAC")
    print(f"  Host:     {cfg['host']}:{cfg['port']}")
    print(f"  User:     {cfg['user']}")
    print(f"  Database: {cfg['database']}")
    print(f"  Driver:   {get_available_driver()}")
    
    print("\nAttempting to connect to MMS database...")
    try:
        df_diag = query_dataframe("SELECT @@VERSION AS Version, DB_NAME() AS CurrentDB, SYSTEM_USER AS CurrentUser")
        print(">> MMS Connection Succeeded!")
        print(f"Current Database: {df_diag['CurrentDB'].iloc[0]}")
        print(f"Current User:     {df_diag['CurrentUser'].iloc[0]}")
        print(f"Server Version:   {df_diag['Version'].iloc[0].splitlines()[0]}")

        print("\nQuerying recent mixing batch records from MMS (BatchHeader)...")
        sample_query = """
        SELECT TOP 5 
            OrderID, 
            BatchNumber, 
            EquipmentID, 
            BatchWeight, 
            BatchStartTime, 
            BatchCycleTimeSeconds, 
            MixingTimeSeconds
        FROM BatchHeader 
        ORDER BY BatchStartTime DESC
        """
        df_batches = query_dataframe(sample_query)
        print("Recent MMS Mixing Batches:")
        print(df_batches.to_string(index=False))
    except Exception as e:
        print(f">> Connection failed: {e}")


