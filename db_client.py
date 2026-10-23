"""
Database Client Module for SFEAPAC and other MSSQL RDS instances.
Supports pyodbc and SQLAlchemy for direct querying into pandas DataFrames.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.parse

import pandas as pd
import pyodbc
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Default config path
CONFIG_PATH = Path(__file__).resolve().parent / "db_config.json"


def load_config(profile: str = "SFEAPAC", config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load connection settings from the configuration file."""
    path = config_path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    if profile not in configs:
        raise KeyError(f"Profile '{profile}' not found in {path}. Available profiles: {list(configs.keys())}")

    return configs[profile]


def get_best_driver() -> str:
    """Find the best available ODBC driver on this system."""
    available_drivers = pyodbc.drivers()
    preferred_order = [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
        "SQL Server Native Client 11.0",
        "SQL Server",
    ]
    for driver in preferred_order:
        if driver in available_drivers:
            return driver
    # Fallback to the first driver found or SQL Server
    return available_drivers[0] if available_drivers else "SQL Server"


def get_connection_string(database: Optional[str] = None, profile: str = "SFEAPAC") -> str:
    """Construct an ODBC connection string for the specified profile and database."""
    cfg = load_config(profile)
    db = database or cfg.get("database", "master")
    driver = cfg.get("driver") or get_best_driver()

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={cfg['host']},{cfg.get('port', 1433)};"
        f"DATABASE={db};"
        f"UID={cfg['user']};"
        f"PWD={cfg['password']};"
    )

    # For ODBC Driver 18, encryption is enabled by default; trust certificate for RDS if needed
    if "ODBC Driver 18" in driver:
        conn_str += "TrustServerCertificate=yes;"

    return conn_str


def get_connection(database: Optional[str] = None, profile: str = "SFEAPAC", timeout: int = 15) -> pyodbc.Connection:
    """Return a raw pyodbc Connection."""
    conn_str = get_connection_string(database=database, profile=profile)
    return pyodbc.connect(conn_str, timeout=timeout)


def get_engine(database: Optional[str] = None, profile: str = "SFEAPAC") -> Engine:
    """Return a SQLAlchemy Engine (ideal for pandas)."""
    conn_str = get_connection_string(database=database, profile=profile)
    params = urllib.parse.quote_plus(conn_str)
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")


def query_df(sql: str, params: Optional[dict] = None, database: Optional[str] = None, profile: str = "SFEAPAC") -> pd.DataFrame:
    """
    Execute a SELECT SQL query and return the result as a pandas DataFrame.

    Example:
        df = query_df("SELECT TOP 10 * FROM sys.tables", database="master")
    """
    engine = get_engine(database=database, profile=profile)
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def list_databases(profile: str = "SFEAPAC") -> List[str]:
    """Retrieve all database names accessible on the server."""
    df = query_df("SELECT name FROM sys.databases ORDER BY name", database="master", profile=profile)
    return df["name"].tolist()


def list_tables(database: Optional[str] = None, profile: str = "SFEAPAC") -> pd.DataFrame:
    """List all user tables in the specified database."""
    sql = """
        SELECT 
            TABLE_SCHEMA AS [schema],
            TABLE_NAME AS [table_name],
            TABLE_TYPE AS [table_type]
        FROM INFORMATION_SCHEMA.TABLES
        ORDER BY TABLE_SCHEMA, TABLE_NAME
    """
    return query_df(sql, database=database, profile=profile)


def get_redshift_connection(profile: str = "RedshiftDatamart"):
    """Return an active connection to Amazon Redshift using redshift_connector."""
    import redshift_connector
    cfg = load_config(profile)
    return redshift_connector.connect(
        host=cfg["host"],
        database=cfg["database"],
        port=cfg.get("port", 5439),
        user=cfg["user"],
        password=cfg["password"],
    )


def query_redshift_df(sql: str, profile: str = "RedshiftDatamart") -> pd.DataFrame:
    """
    Execute a query on Amazon Redshift and return the result as a pandas DataFrame.
    """
    conn = get_redshift_connection(profile=profile)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        try:
            return cursor.fetch_dataframe()
        except Exception:
            cols = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


def list_redshift_schemas(profile: str = "RedshiftDatamart") -> List[str]:
    """Retrieve all distinct user schemas in the Redshift database."""
    sql = "SELECT DISTINCT schemaname FROM pg_tables WHERE schemaname NOT IN ('pg_catalog', 'pg_internal') ORDER BY schemaname"
    df = query_redshift_df(sql, profile=profile)
    return df["schemaname"].tolist() if not df.empty else []


def list_redshift_tables(schema: str = "he_datamarts", profile: str = "RedshiftDatamart") -> List[str]:
    """List tables in the specified Redshift schema."""
    sql = f"SELECT tablename FROM pg_tables WHERE schemaname = '{schema}' ORDER BY tablename"
    df = query_redshift_df(sql, profile=profile)
    return df["tablename"].tolist() if not df.empty else []


if __name__ == "__main__":
    import sys
    # Ensure stdout handles UTF-8 if supported
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== Testing Database Connection ===")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()[0]
        print("[OK] Successfully connected to remote SQL Server (SFEAPAC)!")
        print(f"Server Version:\n  {version.splitlines()[0]}")
        conn.close()

        print("\n=== Fetching Database List ===")
        dbs = list_databases()
        print(f"Found {len(dbs)} databases:")
        for idx, name in enumerate(dbs, start=1):
            print(f"  {idx:2d}. {name}")

    except Exception as exc:
        print(f"[ERROR] SFEAPAC Connection failed: {exc}")

    print("\n=== Testing Amazon Redshift Connection ===")
    try:
        rs_df = query_redshift_df("SELECT current_user, current_database(), version()")
        print("[OK] Successfully connected to Amazon Redshift (RedshiftDatamart)!")
        print(f"  User: {rs_df.iloc[0, 0]}")
        print(f"  DB:   {rs_df.iloc[0, 1]}")
        print(f"  Ver:  {rs_df.iloc[0, 2].splitlines()[0]}")

        schemas = list_redshift_schemas()
        print(f"\nFound {len(schemas)} user schemas in Redshift.")
        he_schemas = [s for s in schemas if "he" in s.lower()]
        print(f"Hefei related schemas: {he_schemas}")

        if "he_datamarts" in schemas:
            tables = list_redshift_tables("he_datamarts")
            print(f"he_datamarts contains {len(tables)} tables. Sample:")
            for t in tables[:10]:
                print(f"  - {t}")
    except Exception as exc:
        print(f"[ERROR] Redshift Connection failed: {exc}")


