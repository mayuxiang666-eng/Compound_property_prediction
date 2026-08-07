"""
V4.2 Step-Phase Architecture: SQL Connection Helper
"""
import os
import sys
import json
import pyodbc

def get_sfe_connection(server_key: str = 'HF_MMS') -> pyodbc.Connection:
    """Returns a pyodbc connection to SFEPlant using credentials.json."""
    cred_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../../Mooney_Prediction_Pipeline/data_processing/credentials.json'
    ))
    if not os.path.exists(cred_path):
        raise FileNotFoundError(f"Credentials file not found at: {cred_path}")
        
    with open(cred_path, 'r', encoding='utf-8') as f:
        creds = json.load(f)
    c = creds[server_key]
    conn_str = (
        f"DRIVER={c['driver']};"
        f"SERVER={c['server']};"
        f"DATABASE={c.get('database', 'SFEPlant')};"
        f"UID={c['username']};"
        f"PWD={c['password']};"
        f"Encrypt=yes;TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)

if __name__ == '__main__':
    conn = get_sfe_connection()
    print("[OK] Task 1: SQL Connection Helper verified successfully.")
    conn.close()
