from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import pyodbc


def connection_string() -> str:
    value = os.environ.get("MOONEY_SQL_CONNECTION_STRING", "").strip()
    if not value:
        raise RuntimeError("MOONEY_SQL_CONNECTION_STRING is not configured")
    return value


@contextmanager
def connect() -> Iterator[pyodbc.Connection]:
    conn = pyodbc.connect(connection_string(), timeout=15, autocommit=False)
    try:
        yield conn
    finally:
        conn.close()


def fetch_active_model() -> dict:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT TOP (1) ModelID, ModelCode, ModelVersion, PackagePath,
                   PackageHash, FeatureSchemaHash, ReleaseType
            FROM dbo.MooneyModel
            WHERE IsActive = 1
            ORDER BY ModelID DESC
            """
        ).fetchone()
        if row is None:
            raise RuntimeError("No active Mooney model is registered")
        columns = [d[0] for d in conn.cursor().description]
        return dict(zip(columns, row))

