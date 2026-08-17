"""Apply the Phase-group migration using an existing Windows stored credential.

No username/password is written to source, stdout, logs, or audit artifacts.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
from ctypes import wintypes
from pathlib import Path

import pyodbc


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "sql" / "mooney_v37_phase_group_migration.sql"
AUDIT = ROOT / "reports" / "sql_phase_group_migration_audit.json"
SERVER = r"10.246.97.159\AIDATASERVER"
DATABASE = "MooneyPredictionV37Shadow"
TARGET_PREFIX = r"Microsoft:SSMS:19:10.246.97.159\AIDATASERVER:sa:"


class CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def _enumerate_target() -> str:
    count = wintypes.DWORD()
    pointer = ctypes.POINTER(ctypes.POINTER(CREDENTIALW))()
    advapi = ctypes.WinDLL("Advapi32.dll")
    advapi.CredEnumerateW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    advapi.CredEnumerateW.restype = wintypes.BOOL
    advapi.CredFree.argtypes = [ctypes.c_void_p]
    if not advapi.CredEnumerateW(None, 0, ctypes.byref(count), ctypes.byref(pointer)):
        raise ctypes.WinError()
    try:
        matches = [pointer[index].contents.TargetName for index in range(count.value)]
        matches = [value for value in matches if value.startswith(TARGET_PREFIX)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one stored SQL credential target; found {len(matches)}")
        return matches[0]
    finally:
        advapi.CredFree(pointer)


def _read_stored_credential(target: str) -> tuple[str, str]:
    credential = ctypes.POINTER(CREDENTIALW)()
    advapi = ctypes.WinDLL("Advapi32.dll")
    advapi.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    advapi.CredReadW.restype = wintypes.BOOL
    advapi.CredFree.argtypes = [ctypes.c_void_p]
    if not advapi.CredReadW(target, 1, 0, ctypes.byref(credential)):
        raise ctypes.WinError()
    try:
        item = credential.contents
        blob = ctypes.string_at(item.CredentialBlob, item.CredentialBlobSize)
        try:
            password = blob.decode("utf-16-le").rstrip("\x00")
        except UnicodeDecodeError:
            password = blob.decode("utf-8").rstrip("\x00")
        return str(item.UserName), password
    finally:
        advapi.CredFree(credential)


def connect(database: str = DATABASE) -> pyodbc.Connection:
    target = _enumerate_target()
    username, password = _read_stored_credential(target)
    connection_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={database};"
        f"UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=yes;"
    )
    return pyodbc.connect(connection_string, timeout=30, autocommit=False)


def execute_migration() -> None:
    script = MIGRATION.read_text(encoding="utf-8")
    batches = [batch.strip() for batch in re.split(r"^\s*GO\s*$", script, flags=re.MULTILINE | re.IGNORECASE) if batch.strip()]
    with connect("master") as connection:
        cursor = connection.cursor()
        try:
            for batch in batches:
                cursor.execute(batch)
                while cursor.nextset():
                    pass
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def inspect() -> dict[str, object]:
    expected = [
        "MooneyPhaseBatchResult", "MooneyGroupResult",
        "MooneyFeatureQualityAudit", "MooneyFeatureParityAudit",
    ]
    with connect(DATABASE) as connection:
        rows = connection.execute(
            "SELECT name FROM sys.tables WHERE name IN (?,?,?,?) ORDER BY name", *expected,
        ).fetchall()
        tables = [str(row[0]) for row in rows]
        constraints = [
            str(row[0]) for row in connection.execute(
                """
                SELECT name FROM sys.key_constraints
                 WHERE parent_object_id IN
                       (OBJECT_ID('dbo.MooneyPhaseBatchResult'), OBJECT_ID('dbo.MooneyGroupResult'))
                 ORDER BY name
                """
            ).fetchall()
        ]
    return {
        "server": SERVER,
        "database": DATABASE,
        "expected_tables": expected,
        "existing_tables": tables,
        "all_tables_present": set(expected) == set(tables),
        "key_constraints": constraints,
        "credential_source": "WINDOWS_CREDENTIAL_MANAGER",
        "credential_written_to_output": False,
        "automatic_control": "DISABLED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    before = inspect()
    if args.apply:
        execute_migration()
    after = inspect()
    audit = {"apply_requested": bool(args.apply), "before": before, "after": after}
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
