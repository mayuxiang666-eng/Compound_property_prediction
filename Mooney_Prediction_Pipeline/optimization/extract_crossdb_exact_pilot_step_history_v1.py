"""Read-only cross-database Step extraction for the two exact pilot recipes.

Contract highlights
-------------------
* Scope comes only from the no-future-fill Phase history.
* SFEPlant and SFEPlant_ARCHIVE are queried independently and retained in the
  raw cache with an explicit SourceDatabase column.
* Rows are normalised back to the original Phase-history OrderID before
  de-duplication.  Active SFEPlant wins an exact duplicate key.
* A conflict audit is emitted before de-duplication.
* No database write, recipe write, PLC action, or recommendation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.data_processing.canonical_material_contract import (
    deduplicate_cross_database_rows,
)
from Mooney_Prediction_Pipeline.optimization.extract_active_exact_pilot_step_history import (
    NUMERIC_STEP_COLUMNS,
    PILOTS,
    _discharge_audit,
    _step_readiness,
)
from Mooney_Prediction_Pipeline.optimization.recipe_scp_logic import (
    contracts_to_frame,
    parse_recipe_scp_logic,
)
from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import (
    build_settable_flags,
)
from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports" / "corrected_phase_history" / "phase_curve_timesafe_v1.csv"
OUTPUT = ROOT / "reports" / "corrected_phase_history" / "crossdb_exact_pilot_step_history_v1"
DATABASES = ("SFEPLANT", "SFEPLANT_ARCHIVE")
STEP_KEY = ["OrderID", "BatchNumber", "EquipmentID", "GroupName", "StepNo"]
COMPARE_COLUMNS = [*NUMERIC_STEP_COLUMNS, "ts"]


def _normalise_order_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def _normalise_scope(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["OrderID"] = _normalise_order_id(out["OrderID"])
    out["BatchNumber"] = pd.to_numeric(out["BatchNumber"], errors="raise").astype(int)
    out["OrderStartTime"] = pd.to_datetime(out["OrderStartTime"], errors="coerce")
    return out


def _scope_from_phase(source: Path) -> pd.DataFrame:
    frame = _normalise_scope(pd.read_csv(source, low_memory=False))
    recipe_ids = {item["recipe_id"] for item in PILOTS.values()}
    scope = frame.loc[frame["CompoundDescription"].astype(str).str.strip().isin(recipe_ids)].copy()
    if scope.empty:
        raise RuntimeError("No exact pilot RecipeID rows found in the no-leak Phase source")
    if scope.duplicated(["OrderID", "BatchNumber"]).any():
        raise ValueError("No-leak Phase source is not unique by OrderID+BatchNumber")
    lookup = {item["recipe_id"]: item["material_system"] for item in PILOTS.values()}
    scope["material_system"] = scope["CompoundDescription"].map(lookup)
    return scope


def build_order_candidates(scope: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    orders = scope[["OrderID", "OrderStartTime"]].drop_duplicates("OrderID")
    for row in orders.itertuples(index=False):
        original = str(row.OrderID)
        candidates = [original]
        if pd.notna(row.OrderStartTime) and "~" not in original:
            candidates.append(f"{int(row.OrderStartTime.year)}~{original}")
        for rank, candidate in enumerate(dict.fromkeys(candidates)):
            rows.append({
                "OriginalOrderID": original,
                "CandidateOrderID": candidate,
                "CandidateRank": rank,
            })
    return pd.DataFrame(rows)


def _query_orders(connection, database: str, candidates: list[str]) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in candidates)
    query = f"""
        SELECT CAST(OrderID AS VARCHAR(64)) AS ResolvedOrderID
        FROM {database}.dbo.Orders WITH (NOLOCK)
        WHERE OrderID IN ({placeholders})
    """
    return pd.read_sql_query(query, connection, params=candidates)


def resolve_orders(
    connection,
    candidates: pd.DataFrame,
    cache_dir: Path,
    *,
    chunk_size: int,
    refresh: bool,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidate_ids = sorted(candidates["CandidateOrderID"].unique())
    frames: list[pd.DataFrame] = []
    for database in DATABASES:
        for start in range(0, len(candidate_ids), chunk_size):
            chunk = candidate_ids[start:start + chunk_size]
            path = cache_dir / f"{database.lower()}_orders_{start // chunk_size:04d}.csv"
            if path.exists() and not refresh:
                found = pd.read_csv(path, dtype=str)
            else:
                found = _query_orders(connection, database, chunk)
                found["SourceDatabase"] = database
                found.to_csv(path, index=False, encoding="utf-8-sig")
            if not found.empty:
                found["ResolvedOrderID"] = _normalise_order_id(found["ResolvedOrderID"])
                found["SourceDatabase"] = database
                frames.append(found)
            print(json.dumps({
                "event": "order_resolution_chunk", "database": database,
                "chunk": start // chunk_size, "candidate_count": len(chunk),
                "matched": int(len(found)), "cache": str(path),
            }), flush=True)
    if not frames:
        return pd.DataFrame(columns=[
            "OriginalOrderID", "CandidateOrderID", "CandidateRank",
            "ResolvedOrderID", "SourceDatabase",
        ])
    found = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["SourceDatabase", "ResolvedOrderID"]
    )
    resolved = candidates.merge(
        found, left_on="CandidateOrderID", right_on="ResolvedOrderID",
        how="inner", validate="many_to_many",
    )
    return resolved.sort_values(
        ["OriginalOrderID", "SourceDatabase", "CandidateRank"], kind="stable"
    ).drop_duplicates(["OriginalOrderID", "SourceDatabase"], keep="first")


def _query_step_wide(connection, database: str, order_ids: list[str]) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in order_ids)
    query = f"""
        SELECT CAST(bd.OrderID AS VARCHAR(64)) AS ResolvedOrderID,
               bd.BatchNumber, bd.EquipmentID, bd.GroupName, bd.StepNo,
               MAX(CASE WHEN bd.VariablePath='SCP-1-Step-Time-rel-s' THEN bd.Target END) AS time_SP,
               MAX(CASE WHEN bd.VariablePath='SCP-1-Step-Time-rel-s' THEN bd.Value END) AS time_AV,
               MAX(CASE WHEN bd.VariablePath='SCP-3-Temperature-C-F' THEN bd.Target END) AS temp_SP,
               MAX(CASE WHEN bd.VariablePath='SCP-3-Temperature-C-F' THEN bd.Value END) AS temp_AV,
               MAX(CASE WHEN bd.VariablePath='Rotor-Speed-1/min-Rpm' THEN bd.Target END) AS rotor_SP,
               MAX(CASE WHEN bd.VariablePath='Rotor-Speed-1/min-Rpm' THEN bd.Value END) AS rotor_AV,
               MAX(CASE WHEN bd.VariablePath='RAM-Pressure-percent' THEN bd.Target END) AS ram_SP,
               MAX(CASE WHEN bd.VariablePath='RAM-Pressure-percent' THEN bd.Value END) AS ram_AV,
               MAX(CASE WHEN bd.VariablePath='SCP-2-Energy-rel-kWh' THEN bd.Value END) AS energy_AV,
               MAX(bd.Timestamp) AS ts
        FROM {database}.dbo.BatchData bd WITH (NOLOCK)
        WHERE bd.OrderID IN ({placeholders})
          AND bd.GroupName IN ('AVR_MST','AVR_MSB')
          AND bd.VariablePath IN (
              'SCP-1-Step-Time-rel-s','SCP-3-Temperature-C-F',
              'Rotor-Speed-1/min-Rpm','RAM-Pressure-percent','SCP-2-Energy-rel-kWh'
          )
        GROUP BY bd.OrderID, bd.BatchNumber, bd.EquipmentID, bd.GroupName, bd.StepNo
    """
    return pd.read_sql_query(query, connection, params=order_ids)


def fetch_steps(
    connection,
    resolved: pd.DataFrame,
    scope: pd.DataFrame,
    cache_dir: Path,
    *,
    chunk_size: int,
    refresh: bool,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    exact_pairs = scope[["OrderID", "BatchNumber"]].drop_duplicates()
    frames: list[pd.DataFrame] = []
    for database in DATABASES:
        mapping = resolved.loc[resolved["SourceDatabase"].eq(database), [
            "OriginalOrderID", "ResolvedOrderID",
        ]].drop_duplicates()
        order_ids = sorted(mapping["ResolvedOrderID"].unique())
        reverse = mapping.rename(columns={"OriginalOrderID": "OrderID"})
        for start in range(0, len(order_ids), chunk_size):
            chunk = order_ids[start:start + chunk_size]
            path = cache_dir / f"{database.lower()}_steps_{start // chunk_size:04d}.csv"
            if path.exists() and not refresh:
                fetched = pd.read_csv(path, low_memory=False)
            else:
                fetched = _query_step_wide(connection, database, chunk)
                fetched["SourceDatabase"] = database
                fetched.to_csv(path, index=False, encoding="utf-8-sig")
            if fetched.empty:
                print(json.dumps({
                    "event": "step_chunk", "database": database,
                    "chunk": start // chunk_size, "orders": len(chunk), "rows": 0,
                }), flush=True)
                continue
            fetched["ResolvedOrderID"] = _normalise_order_id(fetched["ResolvedOrderID"])
            fetched["BatchNumber"] = pd.to_numeric(fetched["BatchNumber"], errors="raise").astype(int)
            fetched["SourceDatabase"] = database
            fetched = fetched.merge(reverse, on="ResolvedOrderID", how="inner", validate="many_to_one")
            fetched = fetched.merge(exact_pairs, on=["OrderID", "BatchNumber"], how="inner", validate="many_to_one")
            frames.append(fetched)
            print(json.dumps({
                "event": "step_chunk", "database": database,
                "chunk": start // chunk_size, "orders": len(chunk),
                "rows": int(len(fetched)), "cache": str(path),
            }), flush=True)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_STEP_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def build_duplicate_conflict_audit(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, frame in raw.groupby(STEP_KEY, dropna=False, sort=False):
        sources = sorted(frame["SourceDatabase"].astype(str).unique())
        if len(sources) < 2:
            continue
        record = dict(zip(STEP_KEY, key))
        record["SourceDatabases"] = "|".join(sources)
        record["SourceRowCount"] = int(len(frame))
        conflicts: list[str] = []
        for column in COMPARE_COLUMNS:
            values = frame[column].dropna()
            if column == "ts":
                distinct = values.astype(str).nunique()
            else:
                numeric = pd.to_numeric(values, errors="coerce").dropna()
                distinct = 0 if numeric.empty else int(not np.isclose(numeric.max(), numeric.min(), rtol=0, atol=1e-6)) + 1
            record[f"{column}_conflict"] = bool(distinct > 1)
            if distinct > 1:
                conflicts.append(column)
        record["ConflictColumns"] = "|".join(conflicts)
        record["HasValueConflict"] = bool(conflicts)
        rows.append(record)
    return pd.DataFrame(rows)


def deduplicate_steps(raw: pd.DataFrame) -> pd.DataFrame:
    return deduplicate_cross_database_rows(raw, STEP_KEY, source_column="SourceDatabase")


def map_steps(step: pd.DataFrame, scope: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "OrderID", "BatchNumber", "CompoundName", "CompoundDescription", "MixerLine",
        "OrderStartTime", "PalletID", "MNY", "MNY target", "Target_Temperature",
        "material_system",
    ]
    metadata_columns = [column for column in metadata_columns if column in scope]
    metadata = scope[metadata_columns].drop_duplicates(["OrderID", "BatchNumber"])
    work = build_settable_flags(step).merge(
        metadata, on=["OrderID", "BatchNumber"], how="left", validate="many_to_one"
    )
    mapped: list[pd.DataFrame] = []
    for _, batch in work.groupby(["OrderID", "BatchNumber"], sort=False):
        target = pd.to_numeric(batch["Target_Temperature"], errors="coerce").dropna()
        valid_target = not target.empty and float(target.iloc[0]) > 0
        if valid_target:
            item = build_step_phase_mapping(batch, target_temp=float(target.iloc[0]))
            item["process_target_temperature_status"] = "AVAILABLE"
        else:
            item = batch.copy()
            item["mapped_phase"] = "BLOCKED_MISSING_PROCESS_TARGET_TEMPERATURE"
            item["mapping_confidence"] = 0.0
            item["mapping_source"] = "missing_process_target_temperature"
            item["auto_control_allowed"] = False
            item["process_target_temperature_status"] = "BLOCKED_MISSING_PROCESS_TARGET_TEMPERATURE"
        mapped.append(item)
    out = pd.concat(mapped, ignore_index=True)
    out = out.sort_values(["OrderID", "BatchNumber", "GroupName", "StepNo"], kind="stable")
    group_keys = [out["OrderID"], out["BatchNumber"], out["GroupName"]]
    out["dur_SP"] = out["time_SP"] - out["time_SP"].groupby(group_keys).shift(1, fill_value=0.0)
    out["dur_AV"] = out["time_AV"] - out["time_AV"].groupby(group_keys).shift(1, fill_value=0.0)
    out.loc[out["dur_SP"] < 0, "dur_SP"] = np.nan
    out.loc[out["dur_AV"] < 0, "dur_AV"] = np.nan
    out["duration_track_err"] = out["dur_AV"] - out["dur_SP"]
    return out


def _batch_template(batch: pd.DataFrame) -> pd.Series:
    rows: list[str] = []
    route: list[str] = []
    for row in batch.sort_values(["GroupName", "StepNo"], kind="stable").itertuples():
        controls = "".join([
            "D" if bool(row.is_time_settable) else "-",
            "T" if bool(row.is_temp_settable) else "-",
            "R" if bool(row.is_rotor_settable) else "-",
            "P" if bool(row.is_ram_settable) else "-",
        ])
        rows.append(f"{row.GroupName}:S{int(row.StepNo)}:{row.mapped_phase}:{controls}")
        route.append(f"{row.GroupName}:{row.EquipmentID}")
    signature = "|".join(rows)
    route_signature = "|".join(dict.fromkeys(route))
    material = str(batch["material_system"].iloc[0])
    recipe = str(batch["CompoundDescription"].iloc[0])
    mixer = str(batch["MixerLine"].iloc[0])
    digest = hashlib.sha256(
        f"{recipe}|{material}|{mixer}|{route_signature}|{signature}".encode("utf-8")
    ).hexdigest()[:12].upper()
    return pd.Series({
        "TemplateID": digest, "TemplateSignature": signature,
        "RouteSignature": route_signature,
    })


def build_templates(mapped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "CompoundDescription", "CompoundName", "material_system", "MixerLine",
        "OrderID", "BatchNumber",
    ]
    batches = mapped.groupby(keys, sort=False).apply(_batch_template, include_groups=False).reset_index()
    summary = batches.groupby([
        "CompoundDescription", "CompoundName", "material_system", "MixerLine",
        "RouteSignature", "TemplateID", "TemplateSignature",
    ], as_index=False).size().rename(columns={"size": "BatchCount"})
    totals = summary.groupby(["CompoundDescription", "MixerLine"])["BatchCount"].transform("sum")
    summary["TemplateShareWithinRecipeRoute"] = summary["BatchCount"] / totals
    summary["IsDominantTemplate"] = summary["BatchCount"].eq(
        summary.groupby(["CompoundDescription", "MixerLine"])["BatchCount"].transform("max")
    )
    return batches, summary.sort_values(
        ["CompoundDescription", "MixerLine", "BatchCount"], ascending=[True, True, False]
    )


def query_recipe_parameters(connection, recipe_ids: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    placeholders = ",".join("?" for _ in recipe_ids)
    for database in DATABASES:
        query = f"""
            SELECT RecipeID, StepNumber, ParameterID, ParameterName, ParameterValue, UOM
            FROM {database}.dbo.RecipeCBS3Parameters WITH (NOLOCK)
            WHERE RecipeID IN ({placeholders}) AND ParameterID LIKE '%SCP%'
        """
        frame = pd.read_sql_query(query, connection, params=recipe_ids)
        frame["SourceDatabase"] = database
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parameter_conflicts(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["RecipeID", "StepNumber", "ParameterID"]
    rows: list[dict[str, object]] = []
    for key, frame in raw.groupby(keys, dropna=False, sort=False):
        if frame["SourceDatabase"].nunique() < 2:
            continue
        values = frame["ParameterValue"].astype(str).str.strip().unique().tolist()
        rows.append({
            **dict(zip(keys, key)),
            "SourceDatabases": "|".join(sorted(frame["SourceDatabase"].unique())),
            "ParameterValues": "|".join(values),
            "HasValueConflict": len(values) > 1,
        })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scope = _scope_from_phase(args.source)
    candidates = build_order_candidates(scope)
    connection = get_sfe_connection()
    try:
        resolved = resolve_orders(
            connection, candidates, args.output_dir / "cache" / "order_resolution",
            chunk_size=args.resolve_chunk_size, refresh=args.refresh,
        )
        raw = fetch_steps(
            connection, resolved, scope, args.output_dir / "cache" / "steps",
            chunk_size=args.step_chunk_size, refresh=args.refresh,
        )
        parameters_raw = query_recipe_parameters(
            connection, sorted(item["recipe_id"] for item in PILOTS.values())
        )
    finally:
        connection.close()
    if raw.empty:
        raise RuntimeError("No Step rows returned from either database")

    conflicts = build_duplicate_conflict_audit(raw)
    step = deduplicate_steps(raw)
    mapped = map_steps(step, scope)
    batch_templates, template_summary = build_templates(mapped)
    mapped = mapped.merge(
        batch_templates[["OrderID", "BatchNumber", "TemplateID"]],
        on=["OrderID", "BatchNumber"], how="left", validate="many_to_one",
    )
    readiness = _step_readiness(mapped)
    readiness = readiness.merge(
        mapped[["CompoundDescription", "GroupName", "StepNo", "mapped_phase", "TemplateID"]]
        .drop_duplicates(),
        on=["CompoundDescription", "GroupName", "StepNo", "mapped_phase"],
        how="left", validate="many_to_many",
    )
    discharge, discharge_summary = _discharge_audit(mapped)

    parameter_conflict = parameter_conflicts(parameters_raw)
    parameter_keys = ["RecipeID", "StepNumber", "ParameterID"]
    parameters = deduplicate_cross_database_rows(
        parameters_raw, parameter_keys, source_column="SourceDatabase"
    )
    contracts = contracts_to_frame(parse_recipe_scp_logic(parameters))

    outputs = {
        "scope": scope,
        "order_candidates": candidates,
        "order_resolution": resolved,
        "raw_crossdb_steps": raw,
        "crossdb_duplicate_conflicts": conflicts,
        "deduplicated_steps": step,
        "mapped_step_vectors": mapped,
        "batch_template_assignments": batch_templates,
        "template_summary": template_summary,
        "template_step_readiness": readiness,
        "discharge_tracking": discharge,
        "recipe_scp_parameters_raw": parameters_raw,
        "recipe_scp_parameter_conflicts": parameter_conflict,
        "recipe_scp_parameters_effective": parameters,
        "recipe_scp_logic_effective": contracts,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    expected = scope.groupby("CompoundDescription").size()
    mapped_counts = mapped.groupby("CompoundDescription")[["OrderID", "BatchNumber"]].apply(
        lambda value: value.drop_duplicates().shape[0]
    )
    source_rows = raw.groupby("SourceDatabase").size().to_dict()
    source_batches = raw.groupby("SourceDatabase")[["OrderID", "BatchNumber"]].apply(
        lambda value: value.drop_duplicates().shape[0]
    ).to_dict()
    summary = {
        "contract": "CROSSDB-EXACT-PILOT-STEP-HISTORY-V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_source": str(args.source),
        "phase_source_contract": "NO_CROSS_BATCH_OR_FUTURE_CURVE_FILL",
        "source_databases": list(DATABASES),
        "archive_access": "READ_ONLY",
        "source_priority": "SFEPLANT_OVER_SFEPLANT_ARCHIVE_ON_EXACT_STEP_KEY",
        "dedupe_key": STEP_KEY,
        "scope_batches": {str(key): int(value) for key, value in expected.items()},
        "resolved_orders_by_source": {
            str(key): int(value) for key, value in resolved.groupby("SourceDatabase")["OriginalOrderID"].nunique().to_dict().items()
        },
        "raw_step_rows_by_source": {str(key): int(value) for key, value in source_rows.items()},
        "raw_step_batches_by_source": {str(key): int(value) for key, value in source_batches.items()},
        "duplicate_step_keys": int(len(conflicts)),
        "duplicate_step_keys_with_value_conflicts": int(conflicts.get("HasValueConflict", pd.Series(dtype=bool)).sum()),
        "deduplicated_step_rows": int(len(step)),
        "mapped_batches": {str(key): int(value) for key, value in mapped_counts.items()},
        "missing_process_target_temperature_batches": int(
            mapped.loc[mapped["process_target_temperature_status"].ne("AVAILABLE"), ["OrderID", "BatchNumber"]]
            .drop_duplicates().shape[0]
        ),
        "template_count": int(len(template_summary)),
        "scp_raw_rows": int(len(parameters_raw)),
        "scp_effective_rows": int(len(parameters)),
        "scp_crossdb_conflicts": int(parameter_conflict.get("HasValueConflict", pd.Series(dtype=bool)).sum()),
        "discharge_tracking": discharge_summary,
        "credentials_logged": False,
        "database_writes": False,
        "automatic_control": "DISABLED",
        "recommendation_status": "NOT_GENERATED_INPUT_AUDIT_ONLY",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--resolve-chunk-size", type=int, default=200)
    parser.add_argument("--step-chunk-size", type=int, default=30)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
