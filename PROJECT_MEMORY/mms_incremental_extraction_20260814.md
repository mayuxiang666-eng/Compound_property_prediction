# 2026-08-14 本地 MMS 分批抽取记忆

## User-approved change

User允许在资源受控的前提下读取本地 MMS/Archive 历史，但禁止全量搜索；每次只能增加少量历史订单，监控服务器内存，历史数据落地为 Parquet，后续优先读取本地快照。

## Implementation

Added:

- `Mooney_Prediction_Pipeline/data_processing/prepare_m1_mms_history_queue_v1.py`
- `Mooney_Prediction_Pipeline/data_processing/incremental_mms_history_extractor_v1.py`

The extractor is read-only, exact-OrderID based, max 10 orders per chunk, one chunk per default run, row-limited, resource-gated, resumable, and SHA256-audited. It uses SQL Server memory DMV probes and Windows-native local memory monitoring when psutil is unavailable.

## First run

- Prepared 50 M1 order IDs from the existing snapshot.
- Selected 10 orders for the first chunk.
- Stopped before ODBC connection because the current credentials JSON has no `HF_MMS` section.
- No local MMS SQL was executed and no Parquet data was written.
- Audit: `reports/mms_incremental_history/extractor_audit_v1.jsonl`.

## Required before next run

Restore the correct local MMS `HF_MMS` credential section in the protected JSON. Do not substitute RDS/Mustang credentials. After that, run one chunk only, inspect resources and Parquet manifest, then decide whether to add the next chunk.

## 2026-08-14 connection diagnostic

- The JSON is now syntactically valid and contains `HF_MMS`; the extractor accepts the existing legacy `sever` key as a compatibility alias without writing credentials to output.
- The first 10-order run stopped during the read-only ODBC connection handshake. No business table query ran and no Parquet chunk was committed.
- The configured host resolves on the internal network, but TCP 1433 and SQL Browser UDP 1434 did not respond. The audit records `STOPPED` with the ODBC instance-location error.
- Next required input is a reachable SQL Server named-instance port (or an approved DSN/endpoint from the working database client). Do not retry repeated chunks until the endpoint is confirmed.

## 2026-08-14 controlled extraction findings

- User explicitly approved disabling the remote SQL Server memory DMV probe. The extractor now records local process/system memory and uses per-table response time as an operational guard.
- The first 10-order attempt showed `SFEPLANT.BatchData` at 84.49 seconds and stopped. A one-order attempt reached `SFEPLANT_ARCHIVE.BatchData` and was cancelled by the 60-second ODBC query timeout. No complete chunk manifest was produced.
- The extractor now has a local singleton lock so duplicate launcher processes cannot query concurrently.
- `BatchData` extraction was narrowed to Step→Phase-relevant EAV columns and variables (`AVR_MST`/`AVR_MSB`, step time, temperature, rotor, ram, energy) instead of `SELECT t.*`. This is recorded as `STEP_PHASE_RELEVANT_EAV_COLUMNS_ONLY` in the manifest.
- Existing project pivot code supports the same scoped variable set; no unknown database function or stored procedure was created or called.
