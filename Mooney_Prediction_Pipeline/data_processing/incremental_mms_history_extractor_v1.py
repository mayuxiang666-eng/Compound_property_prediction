"""Guarded, chunked, read-only MMS historical extractor.

Safety contract:
- exact OrderID batches only; no unbounded table scans;
- default maximum is 10 orders per run;
- active and archive rows are cached as raw Parquet with source database;
- local resource probes run before and after every table query;
- remote SQL Server memory DMV probing is optional; when disabled, response
  time is used as a conservative operational slowdown guard;
- credentials are read but never written to logs or manifests.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyodbc

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - Windows fallback is tested in situ.
    psutil = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CREDENTIALS = ROOT / "Mooney_Prediction_Pipeline" / "data_processing" / "credentials.json"
DEFAULT_QUEUE = ROOT / "reports" / "mms_incremental_history" / "m1_order_queue_v1.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "mms_incremental_history" / "parquet"
DEFAULT_STATE = ROOT / "reports" / "mms_incremental_history" / "extractor_state_v1.json"
DEFAULT_AUDIT = ROOT / "reports" / "mms_incremental_history" / "extractor_audit_v1.jsonl"
DEFAULT_LOCK = ROOT / "reports" / "mms_incremental_history" / "extractor_v1.lock"

DATABASES = ("SFEPLANT", "SFEPLANT_ARCHIVE")
TABLES = ("Orders", "BatchHeader", "BatchCurve", "BatchData", "BatchMaterials", "OrderMaterials")
BATCHDATA_VARIABLES = (
    "SCP-1-Step-Time-rel-s",
    "SCP-3-Temperature-C-F",
    "Rotor-Speed-1/min-Rpm",
    "RAM-Pressure-percent",
    "SCP-2-Energy-rel-kWh",
)
BATCHDATA_BATCH_CHUNK_SIZE = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_credentials(path: Path, section: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if section not in payload:
        raise RuntimeError(
            f"BLOCKED_MISSING_CREDENTIAL_SECTION:{section}; "
            "no database connection was attempted"
        )
    value = payload[section]
    if not isinstance(value, dict):
        raise RuntimeError(f"BLOCKED_INVALID_CREDENTIAL_SECTION:{section}")
    # ``sever`` is retained as a compatibility alias for the existing local
    # MMS credentials file.  It is a read-only connection and the value is
    # never emitted to audit output.
    host = value.get("server") or value.get("host") or value.get("sever")
    username = value.get("username") or value.get("user")
    password = value.get("password")
    if not host or not username or not password:
        raise RuntimeError(f"BLOCKED_INCOMPLETE_CREDENTIAL_SECTION:{section}")
    return value


def connect(credentials: dict[str, object], database: str) -> pyodbc.Connection:
    driver = str(credentials.get("driver") or "ODBC Driver 17 for SQL Server").strip()
    # Accept both JSON forms: ``ODBC Driver ...`` and ``{ODBC Driver ...}``.
    # The connection string adds the required braces exactly once.
    if driver.startswith("{") and driver.endswith("}"):
        driver = driver[1:-1].strip()
    server = str(credentials.get("server") or credentials.get("host") or credentials.get("sever"))
    port = credentials.get("port")
    if port and "," not in server and "\\" not in server:
        server = f"{server},{port}"
    username = credentials.get("username") or credentials.get("user")
    connection_string = (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
        f"UID={username};PWD={credentials['password']};"
        "Encrypt=yes;TrustServerCertificate=yes;"
    )
    return pyodbc.connect(connection_string, timeout=30, readonly=True)


def _windows_memory_snapshot() -> tuple[float, float]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong),
            ("avail_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("avail_virtual", ctypes.c_ulonglong),
            ("avail_extended", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("LOCAL_MEMORY_MONITOR_UNAVAILABLE")
    total = status.total_phys / 1024**2
    available = status.avail_phys / 1024**2
    return total, available


def _windows_process_rss_mb() -> float:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("page_fault_count", ctypes.c_ulong),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    get_info = psapi.GetProcessMemoryInfo
    get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]
    get_info.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    if not get_info(handle, ctypes.byref(counters), counters.cb):
        raise RuntimeError("LOCAL_PROCESS_MEMORY_MONITOR_UNAVAILABLE")
    return counters.working_set_size / 1024**2


def local_resource_snapshot(process: object | None) -> dict[str, object]:
    if psutil is not None:
        memory = process.memory_info()
        rss_mb = memory.rss / 1024**2
        cpu_percent = float(process.cpu_percent(interval=0.2))
        available_mb = psutil.virtual_memory().available / 1024**2
    else:
        _, available_mb = _windows_memory_snapshot()
        rss_mb = _windows_process_rss_mb()
        cpu_percent = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rss_mb": round(rss_mb, 2),
        "cpu_percent": cpu_percent,
        "system_available_memory_mb": round(available_mb, 2),
    }


def server_resource_snapshot(connection: pyodbc.Connection) -> dict[str, object]:
    # Fail closed if server memory cannot be monitored.  This is deliberately
    # a DMV-only probe and does not inspect rows from production tables.
    sql = """
    SELECT total_physical_memory_kb, available_physical_memory_kb,
           system_memory_state_desc
      FROM sys.dm_os_sys_memory;
    """
    try:
        frame = pd.read_sql_query(sql, connection)
    except Exception as exc:
        message = str(exc)
        if "permission was denied" in message.lower() or "view server" in message.lower():
            raise RuntimeError("BLOCKED_SERVER_MEMORY_MONITOR_PERMISSION") from exc
        raise
    if frame.empty:
        raise RuntimeError("SERVER_RESOURCE_MONITOR_EMPTY")
    row = frame.iloc[0]
    return {
        "server_total_memory_mb": round(float(row["total_physical_memory_kb"]) / 1024, 2),
        "server_available_memory_mb": round(float(row["available_physical_memory_kb"]) / 1024, 2),
        "server_memory_state": str(row["system_memory_state_desc"]),
    }


def check_limits(local: dict[str, object], server: dict[str, object] | None, args: argparse.Namespace) -> None:
    if float(local["rss_mb"]) > args.max_local_rss_mb:
        raise RuntimeError("STOP_LOCAL_RSS_LIMIT")
    if server is not None and "server_available_memory_mb" in server and float(server["server_available_memory_mb"]) < args.min_server_available_memory_mb:
        raise RuntimeError("STOP_SERVER_MEMORY_LIMIT")


def query_table(
    connection: pyodbc.Connection,
    database: str,
    table: str,
    order_ids: list[str],
    max_rows: int,
    timeout_seconds: float,
) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in order_ids)
    sql = (
        f"SELECT TOP ({int(max_rows) + 1}) N'{database}' AS SourceDatabase, t.* "
        f"FROM [{database}].[dbo].[{table}] AS t WITH (NOLOCK) "
        f"WHERE t.OrderID IN ({placeholders})"
    )
    # pyodbc versions differ: some expose cursor.timeout and some do not.
    # The connection-level timeout is supported by the installed driver and
    # is set before every query by the caller.
    cursor = connection.cursor()
    cursor.execute(sql, order_ids)
    columns = [str(column[0]) for column in cursor.description]
    rows = cursor.fetchmany(int(max_rows) + 1)
    frame = pd.DataFrame.from_records(rows, columns=columns)
    cursor.close()
    if len(frame) > max_rows:
        raise RuntimeError(f"STOP_TABLE_ROW_LIMIT:{database}.{table}:{len(frame)}>{max_rows}")
    return frame


def query_batchdata_compact(
    connection: pyodbc.Connection,
    database: str,
    order_ids: list[str],
    max_rows: int,
    timeout_seconds: float,
    batch_numbers: list[object] | None = None,
) -> pd.DataFrame:
    """Read only the Step→Phase-relevant BatchData EAV columns.

    BatchData is a wide/high-volume EAV table.  The model path only needs the
    five process variables below, so selecting ``t.*`` is intentionally
    avoided.  This remains a raw row-level extract (not an imputed or
    aggregated feature table) and preserves Target/Value/Timestamp semantics.
    """
    variable_placeholders = ",".join("?" for _ in BATCHDATA_VARIABLES)
    batches = list(batch_numbers or [])
    batch_groups = [batches[i:i + BATCHDATA_BATCH_CHUNK_SIZE] for i in range(0, len(batches), BATCHDATA_BATCH_CHUNK_SIZE)] or [None]
    frames: list[pd.DataFrame] = []
    total_rows = 0
    for batch_group in batch_groups:
        order_placeholders = ",".join("?" for _ in order_ids)
        batch_clause = ""
        batch_params: list[object] = []
        if batch_group is not None:
            batch_placeholders = ",".join("?" for _ in batch_group)
            batch_clause = f"AND t.BatchNumber IN ({batch_placeholders}) "
            batch_params = list(batch_group)
        sql = (
            f"SELECT TOP ({int(max_rows - total_rows) + 1}) N'{database}' AS SourceDatabase, "
            "t.OrderID, t.BatchNumber, t.EquipmentID, t.GroupName, t.StepNo, "
            "t.VariablePath, t.Target, t.Value, t.Timestamp "
            f"FROM [{database}].[dbo].[BatchData] AS t WITH (NOLOCK) "
            f"WHERE t.OrderID IN ({order_placeholders}) {batch_clause}"
            "AND t.GroupName IN ('AVR_MST','AVR_MSB') "
            f"AND t.VariablePath IN ({variable_placeholders})"
        )
        cursor = connection.cursor()
        cursor.execute(sql, [*order_ids, *batch_params, *BATCHDATA_VARIABLES])
        columns = [str(column[0]) for column in cursor.description]
        rows = cursor.fetchmany(int(max_rows - total_rows) + 1)
        cursor.close()
        frame = pd.DataFrame.from_records(rows, columns=columns)
        frames.append(frame)
        total_rows += len(frame)
        if total_rows > max_rows:
            raise RuntimeError(f"STOP_TABLE_ROW_LIMIT:{database}.BatchData:{total_rows}>{max_rows}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def append_audit(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"completed_order_ids": [], "completed_chunks": []}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", type=Path, default=Path(os.environ.get("MOONEY_MMS_CREDENTIALS_PATH", DEFAULT_CREDENTIALS)))
    parser.add_argument("--credential-section", default="HF_MMS")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--max-rows-per-table", type=int, default=50000)
    parser.add_argument("--max-local-rss-mb", type=float, default=2048.0)
    parser.add_argument("--min-server-available-memory-mb", type=float, default=4096.0)
    parser.add_argument(
        "--skip-server-memory-monitor",
        action="store_true",
        help="Disable remote SQL Server DMV memory probes; response-time guard remains active.",
    )
    parser.add_argument(
        "--max-query-seconds",
        type=float,
        default=60.0,
        help="Stop the current chunk when any exact-order table query exceeds this elapsed time.",
    )
    parser.add_argument("--max-chunks", type=int, default=1)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.chunk_size > 10:
        raise RuntimeError("BLOCKED_CHUNK_SIZE_MUST_BE_BETWEEN_1_AND_10")
    if args.max_chunks < 1:
        raise RuntimeError("BLOCKED_MAX_CHUNKS_MUST_BE_POSITIVE")

    args.output.mkdir(parents=True, exist_ok=True)
    queue = pd.read_csv(args.queue, low_memory=False)
    queue["OrderID"] = queue["OrderID"].astype(str).str.strip()
    queue = queue.drop_duplicates("OrderID")
    queue_sort = ["QueueRank"] if "QueueRank" in queue.columns else ["OrderID"]
    queue = queue.sort_values(queue_sort, kind="stable")
    state = load_state(args.state)
    completed = set(map(str, state.get("completed_order_ids", [])))
    pending = [order_id for order_id in queue["OrderID"].tolist() if order_id not in completed]
    selected = pending[: args.chunk_size * args.max_chunks]

    run_payload: dict[str, object] = {
        "contract": "MMS_INCREMENTAL_READONLY_PARQUET_V1",
        "credential_section": args.credential_section,
        "credentials_written_to_output": False,
        "source_databases": list(DATABASES),
        "selected_order_ids": selected,
        "chunk_size": args.chunk_size,
        "max_chunks": args.max_chunks,
        "max_rows_per_table": args.max_rows_per_table,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Credential validation occurs before opening any ODBC connection.  A
    # blocked attempt is still written to the audit trail without secrets.
    try:
        credentials = read_credentials(args.credentials, args.credential_section)
    except Exception as exc:
        run_payload.update({"status": "BLOCKED_BEFORE_CONNECTION", "reason": str(exc)})
        append_audit(args.audit, run_payload)
        print(json.dumps(run_payload, ensure_ascii=False, indent=2))
        return
    process = psutil.Process() if psutil is not None else None
    if not selected:
        run_payload["status"] = "NO_PENDING_ORDERS"
        append_audit(args.audit, run_payload)
        print(json.dumps(run_payload, ensure_ascii=False, indent=2))
        return

    lock_fd = None
    try:
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(str(args.lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
    except FileExistsError:
        run_payload.update({"status": "BLOCKED_EXTRACTOR_ALREADY_RUNNING", "reason": "LOCAL_SINGLETON_LOCK"})
        append_audit(args.audit, run_payload)
        print(json.dumps(run_payload, ensure_ascii=False, indent=2))
        return

    chunk_id = len(state.get("completed_chunks", [])) + 1
    chunk_dir = args.output / f"chunk_{chunk_id:06d}"
    attempt = 1
    if chunk_dir.exists() and not (chunk_dir / "chunk_manifest.json").exists():
        attempt_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        chunk_dir = args.output / f"chunk_{chunk_id:06d}_retry_{attempt_stamp}"
        attempt = 2
    chunk_dir.mkdir(parents=True, exist_ok=True)
    resources: list[dict[str, object]] = []
    table_records: dict[str, int] = {}
    table_frames: dict[str, pd.DataFrame] = {}
    started = time.perf_counter()
    connection = None
    try:
        connection = connect(credentials, "SFEPLANT")
        # Apply a real statement timeout so a slow table cannot run forever
        # while the caller is using response time as the operational guard.
        connection.timeout = max(1, int(args.max_query_seconds))
        for database in DATABASES:
            for table in TABLES:
                local_before = local_resource_snapshot(process)
                server_before = (
                    {"server_memory_monitor_status": "DISABLED_BY_USER"}
                    if args.skip_server_memory_monitor
                    else server_resource_snapshot(connection)
                )
                check_limits(local_before, server_before, args)
                query_started = time.perf_counter()
                if table == "BatchData":
                    batch_numbers = None
                    header_frame = table_frames.get("BatchHeader")
                    if header_frame is not None and "BatchNumber" in header_frame.columns:
                        batch_numbers = (
                            header_frame["BatchNumber"]
                            .dropna()
                            .drop_duplicates()
                            .tolist()
                        )
                    frame = query_batchdata_compact(
                        connection,
                        database,
                        selected,
                        args.max_rows_per_table,
                        args.max_query_seconds,
                        batch_numbers=batch_numbers,
                    )
                else:
                    frame = query_table(
                        connection,
                        database,
                        table,
                        selected,
                        args.max_rows_per_table,
                        args.max_query_seconds,
                    )
                elapsed = round(time.perf_counter() - query_started, 3)
                local_after = local_resource_snapshot(process)
                server_after = (
                    {"server_memory_monitor_status": "DISABLED_BY_USER"}
                    if args.skip_server_memory_monitor
                    else server_resource_snapshot(connection)
                )
                check_limits(local_after, server_after, args)
                slow_response = elapsed > args.max_query_seconds
                resources.append({
                    "database": database,
                    "table": table,
                    "elapsed_seconds": elapsed,
                    "response_time_status": "SLOW_STOP" if slow_response else "OK",
                    "before": {**local_before, **server_before},
                    "after": {**local_after, **server_after},
                    "rows": int(len(frame)),
                    "extraction_mode": "STEP_PHASE_COMPACT" if table == "BatchData" else "RAW_TABLE_COLUMNS",
                })
                if slow_response:
                    raise RuntimeError(
                        f"STOP_SLOW_QUERY_RESPONSE:{database}.{table}:{elapsed}s>{args.max_query_seconds}s"
                    )
                if frame.empty:
                    continue
                table_frames[table] = frame
                path = chunk_dir / f"{database.lower()}__{table.lower()}.parquet"
                frame.to_parquet(path, index=False)
                table_records[f"{database}.{table}"] = int(len(frame))
                if float(local_after["rss_mb"]) > args.max_local_rss_mb * 0.85:
                    raise RuntimeError("STOP_LOCAL_RSS_HEADROOM")
        manifest = {
            "contract": "MMS_INCREMENTAL_READONLY_PARQUET_V1",
            "chunk_id": chunk_id,
            "attempt": attempt,
            "order_ids": selected,
            "table_records": table_records,
            "resources": resources,
            "files": [
                {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                for path in sorted(chunk_dir.glob("*.parquet"))
            ],
            "credentials_written_to_output": False,
            "batchdata_extract_contract": "STEP_PHASE_RELEVANT_EAV_COLUMNS_ONLY",
            "batchdata_variables": list(BATCHDATA_VARIABLES),
            "batchdata_batch_chunk_size": BATCHDATA_BATCH_CHUNK_SIZE,
            "remote_server_memory_monitoring": not args.skip_server_memory_monitor,
            "response_time_guard_seconds": args.max_query_seconds,
            "automatic_control": False,
            "started_utc": run_payload["started_utc"],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        (chunk_dir / "chunk_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        completed.update(selected)
        state["completed_order_ids"] = sorted(completed)
        state.setdefault("completed_chunks", []).append({"chunk_id": chunk_id, "order_count": len(selected), "path": str(chunk_dir)})
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        run_payload.update({"status": "COMPLETED", "chunk_id": chunk_id, "table_records": table_records, "elapsed_seconds": round(time.perf_counter() - started, 3)})
    except Exception as exc:
        run_payload.update({"status": "STOPPED", "reason": str(exc), "elapsed_seconds": round(time.perf_counter() - started, 3), "resources": resources})
        append_audit(args.audit, run_payload)
        raise
    finally:
        if connection is not None:
            connection.close()
        if lock_fd is not None:
            os.close(lock_fd)
            try:
                args.lock.unlink()
            except FileNotFoundError:
                pass
    append_audit(args.audit, run_payload)
    print(json.dumps(run_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
