"""Extract and audit Active Step history for the two exact pilot recipes.

Read-only source scope: SFEPlant.  No Archive query, recipe write, PLC write,
prediction-table write, or automatic-control action is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from V4_2_Step_Phase_Architecture.data_extraction.pivot_batchdata import pivot_batchdata_to_step_wide
from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags
from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping
from Mooney_Prediction_Pipeline.optimization.recipe_scp_logic import contracts_to_frame, parse_recipe_scp_logic


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history"
SOURCE = BASE / "phase_corrected_v371_active.csv"
OUTPUT = BASE / "active_exact_pilot_step_history_v1"

PILOTS = {
    "b00458_12004": {
        "compound": "M1-B00458---- 12 004",
        "recipe_id": "9922A00458BT_PV77_004",
        "material_system": "CarbonBlack",
    },
    "t09170ws_02004": {
        "compound": "M1-T09170WS-- 02 004",
        "recipe_id": "9922A09170AX_PV50_004",
        "material_system": "Silica",
    },
}

NUMERIC_STEP_COLUMNS = (
    "time_SP", "time_AV", "temp_SP", "temp_AV", "rotor_SP", "rotor_AV",
    "ram_SP", "ram_AV", "energy_AV",
)


def _normalise_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["OrderID"] = out["OrderID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out["BatchNumber"] = pd.to_numeric(out["BatchNumber"], errors="raise").astype(int)
    return out


def _fetch_chunks(scope: pd.DataFrame, cache_dir: Path, *, chunk_size: int, refresh: bool) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    order_ids = sorted(scope["OrderID"].unique())
    exact_pairs = scope[["OrderID", "BatchNumber"]].drop_duplicates()
    frames: list[pd.DataFrame] = []
    connection = None
    try:
        for start in range(0, len(order_ids), chunk_size):
            chunk = order_ids[start:start + chunk_size]
            path = cache_dir / f"step_chunk_{start // chunk_size:04d}.csv"
            if path.exists() and not refresh:
                fetched = pd.read_csv(path, low_memory=False)
            else:
                if connection is None:
                    connection = get_sfe_connection()
                fetched = pivot_batchdata_to_step_wide(connection, order_ids=chunk)
                fetched = _normalise_ids(fetched)
                fetched = fetched.merge(exact_pairs, on=["OrderID", "BatchNumber"], how="inner", validate="many_to_one")
                fetched.to_csv(path, index=False, encoding="utf-8-sig")
            frames.append(_normalise_ids(fetched))
            print(json.dumps({
                "event": "active_step_chunk_ready", "chunk": start // chunk_size,
                "orders": len(chunk), "rows": len(fetched), "cache": str(path),
            }), flush=True)
    finally:
        if connection is not None:
            connection.close()
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates(
        ["OrderID", "BatchNumber", "EquipmentID", "GroupName", "StepNo"], keep="last"
    )


def _query_recipe_parameters(recipe_ids: list[str]) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in recipe_ids)
    query = f"""
        SELECT RecipeID, StepNumber, ParameterID, ParameterName, ParameterValue, UOM
        FROM SFEPlant.dbo.RecipeCBS3Parameters WITH (NOLOCK)
        WHERE RecipeID IN ({placeholders})
          AND ParameterID LIKE '%SCP%'
        ORDER BY RecipeID, StepNumber, ParameterID
    """
    connection = get_sfe_connection()
    try:
        return pd.read_sql_query(query, connection, params=recipe_ids)
    finally:
        connection.close()


def _map_batches(step: pd.DataFrame, scope: pd.DataFrame) -> pd.DataFrame:
    metadata = scope[[
        "OrderID", "BatchNumber", "CompoundName", "CompoundDescription", "MixerLine",
        "OrderStartTime", "Target_Temperature", "material_system",
    ]].drop_duplicates(["OrderID", "BatchNumber"])
    work = build_settable_flags(step).merge(
        metadata, on=["OrderID", "BatchNumber"], how="left", validate="many_to_one"
    )
    mapped: list[pd.DataFrame] = []
    for _, batch in work.groupby(["OrderID", "BatchNumber"], sort=False):
        target = pd.to_numeric(batch["Target_Temperature"], errors="coerce").dropna()
        target_temperature = float(target.iloc[0]) if not target.empty else 165.0
        mapped.append(build_step_phase_mapping(batch, target_temp=target_temperature))
    out = pd.concat(mapped, ignore_index=True) if mapped else pd.DataFrame()
    out = out.sort_values(["OrderID", "BatchNumber", "GroupName", "StepNo"], kind="stable")
    for column in NUMERIC_STEP_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    group_keys = [out["OrderID"], out["BatchNumber"], out["GroupName"]]
    out["dur_SP"] = out["time_SP"] - out["time_SP"].groupby(group_keys).shift(1, fill_value=0.0)
    out["dur_AV"] = out["time_AV"] - out["time_AV"].groupby(group_keys).shift(1, fill_value=0.0)
    out.loc[out["dur_SP"] < 0, "dur_SP"] = np.nan
    out.loc[out["dur_AV"] < 0, "dur_AV"] = np.nan
    out["duration_track_err"] = out["dur_AV"] - out["dur_SP"]
    return out


def _template_signature(batch: pd.DataFrame) -> str:
    rows = []
    for row in batch.sort_values(["GroupName", "StepNo"], kind="stable").itertuples():
        controls = "".join([
            "D" if bool(row.is_time_settable) else "-",
            "T" if bool(row.is_temp_settable) else "-",
            "R" if bool(row.is_rotor_settable) else "-",
            "P" if bool(row.is_ram_settable) else "-",
        ])
        rows.append(f"{row.GroupName}:S{int(row.StepNo)}:{row.mapped_phase}:{controls}")
    return "|".join(rows)


def _template_audit(mapped: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    batches = mapped.groupby(
        ["CompoundDescription", "CompoundName", "OrderID", "BatchNumber"], sort=False
    ).apply(_template_signature, include_groups=False).rename("TemplateSignature").reset_index()
    counts = batches.groupby(
        ["CompoundDescription", "CompoundName", "TemplateSignature"], as_index=False
    ).size().rename(columns={"size": "BatchCount"})
    totals = counts.groupby("CompoundDescription")["BatchCount"].transform("sum")
    counts["TemplateShare"] = counts["BatchCount"] / totals
    counts["IsDominantTemplate"] = counts["BatchCount"].eq(
        counts.groupby("CompoundDescription")["BatchCount"].transform("max")
    )
    summary = {}
    for recipe, frame in counts.groupby("CompoundDescription"):
        summary[str(recipe)] = {
            "batch_count": int(frame["BatchCount"].sum()),
            "template_count": int(len(frame)),
            "dominant_template_share": float(frame["TemplateShare"].max()),
            "template_consistency_pass_90pct": bool(frame["TemplateShare"].max() >= 0.90),
        }
    return counts.sort_values(["CompoundDescription", "BatchCount"], ascending=[True, False]), summary


def _distribution(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"n": 0, "p05": None, "p25": None, "median": None, "p75": None, "p95": None}
    p05, p25, median, p75, p95 = np.percentile(numeric, [5, 25, 50, 75, 95])
    return {"n": int(len(numeric)), "p05": float(p05), "p25": float(p25), "median": float(median), "p75": float(p75), "p95": float(p95)}


def _step_readiness(mapped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in mapped.groupby(
        ["CompoundDescription", "CompoundName", "GroupName", "StepNo", "mapped_phase"], dropna=False
    ):
        recipe, compound, group, step_no, phase = keys
        for parameter, sp, av, settable, error in (
            ("duration", "dur_SP", "dur_AV", "is_time_settable", "duration_track_err"),
            ("temperature", "temp_SP", "temp_AV", "is_temp_settable", "temp_track_err"),
            ("rotor", "rotor_SP", "rotor_AV", "is_rotor_settable", "rotor_track_err"),
            ("ram", "ram_SP", "ram_AV", "is_ram_settable", "ram_track_err"),
        ):
            eligible = frame.loc[frame[settable].fillna(False).astype(bool)]
            errors = pd.to_numeric(eligible[error], errors="coerce").abs().dropna()
            bounds = _distribution(eligible[sp])
            rows.append({
                "RecipeID": recipe, "CompoundName": compound, "GroupName": group,
                "StepNo": int(step_no), "mapped_phase": phase, "parameter": parameter,
                "observed_batches": int(frame[["OrderID", "BatchNumber"]].drop_duplicates().shape[0]),
                "settable_batches": int(eligible[["OrderID", "BatchNumber"]].drop_duplicates().shape[0]),
                "target_level_count": int(pd.to_numeric(eligible[sp], errors="coerce").dropna().nunique()),
                "target_p05": bounds["p05"], "target_p25": bounds["p25"],
                "target_median": bounds["median"], "target_p75": bounds["p75"], "target_p95": bounds["p95"],
                "tracking_n": int(len(errors)),
                "tracking_mae": float(errors.mean()) if len(errors) else None,
                "tracking_p95": float(np.percentile(errors, 95)) if len(errors) else None,
                "tracking_excursion_gt5_rate": float((errors > 5.0).mean()) if len(errors) else None,
            })
    return pd.DataFrame(rows)


def _discharge_audit(mapped: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    for (recipe, compound, order_id, batch_number), batch in mapped.groupby(
        ["CompoundDescription", "CompoundName", "OrderID", "BatchNumber"], sort=False
    ):
        msb = batch.loc[batch["GroupName"].eq("AVR_MSB")].sort_values("StepNo", kind="stable")
        controls = msb.loc[pd.to_numeric(msb["temp_SP"], errors="coerce").fillna(0).gt(0)]
        if msb.empty or controls.empty:
            continue
        control, final = controls.iloc[-1], msb.iloc[-1]
        actual = pd.to_numeric(pd.Series([final["temp_AV"]]), errors="coerce").iloc[0]
        target = float(control["temp_SP"])
        rows.append({
            "RecipeID": recipe, "CompoundName": compound, "OrderID": order_id,
            "BatchNumber": int(batch_number), "control_step_no": int(control["StepNo"]),
            "final_step_no": int(final["StepNo"]), "discharge_temp_target": target,
            "discharge_temp_actual": float(actual) if pd.notna(actual) else None,
            "tracking_error": float(actual - target) if pd.notna(actual) else None,
            "trailing_no_target_step": bool(int(final["StepNo"]) > int(control["StepNo"])),
        })
    result = pd.DataFrame(rows)
    summary = {}
    for recipe, frame in result.groupby("RecipeID"):
        errors = pd.to_numeric(frame["tracking_error"], errors="coerce").abs().dropna()
        mae = float(errors.mean()) if len(errors) else None
        p95 = float(np.percentile(errors, 95)) if len(errors) else None
        excursion = float((errors > 5.0).mean()) if len(errors) else None
        summary[str(recipe)] = {
            "batches": int(len(frame)), "mae": mae, "p95_abs_error": p95,
            "excursion_gt5_rate": excursion,
            "passed": bool(len(errors) and mae <= 2.0 and p95 <= 5.0 and excursion <= 0.05),
            "trailing_no_target_batch_rate": float(frame["trailing_no_target_step"].mean()) if len(frame) else None,
        }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    source = _normalise_ids(pd.read_csv(args.source, low_memory=False))
    exact_compounds = {item["compound"] for item in PILOTS.values()}
    scope = source.loc[source["CompoundName"].astype(str).isin(exact_compounds)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = _fetch_chunks(scope, args.output_dir / "cache", chunk_size=args.chunk_size, refresh=args.refresh)
    if step.empty:
        raise RuntimeError("No Active BatchData Step rows were returned for the exact pilot scope")
    mapped = _map_batches(step, scope)
    templates, template_summary = _template_audit(mapped)
    readiness = _step_readiness(mapped)
    discharge, discharge_summary = _discharge_audit(mapped)

    recipe_ids = sorted(item["recipe_id"] for item in PILOTS.values())
    parameters = _query_recipe_parameters(recipe_ids)
    contracts = parse_recipe_scp_logic(parameters)
    contract_frame = contracts_to_frame(contracts)

    mapped.to_csv(args.output_dir / "active_exact_pilot_step_vectors.csv", index=False, encoding="utf-8-sig")
    templates.to_csv(args.output_dir / "active_exact_pilot_template_signatures.csv", index=False, encoding="utf-8-sig")
    readiness.to_csv(args.output_dir / "active_exact_pilot_step_readiness.csv", index=False, encoding="utf-8-sig")
    discharge.to_csv(args.output_dir / "active_exact_pilot_discharge_tracking.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(args.output_dir / "active_exact_pilot_recipe_scp_parameters.csv", index=False, encoding="utf-8-sig")
    contract_frame.to_csv(args.output_dir / "active_exact_pilot_scp_logic.csv", index=False, encoding="utf-8-sig")

    expected = scope.groupby("CompoundDescription").size().to_dict()
    actual = mapped.groupby("CompoundDescription")[["OrderID", "BatchNumber"]].apply(
        lambda value: value.drop_duplicates().shape[0]
    ).to_dict()
    summary = {
        "contract": "ACTIVE-EXACT-PILOT-STEP-HISTORY-V1",
        "source_database": "SFEPlant",
        "archive_queried": False,
        "expected_phase_batches": {str(key): int(value) for key, value in expected.items()},
        "mapped_step_rows": int(len(mapped)),
        "mapped_batches": {str(key): int(value) for key, value in actual.items()},
        "template_audit": template_summary,
        "discharge_tracking": discharge_summary,
        "scp_parameter_rows": int(len(parameters)),
        "scp_contract_rows": int(len(contract_frame)),
        "unknown_scp_contracts": int((contract_frame.get("scp_logic_status", pd.Series(dtype=str)) != "KNOWN").sum()),
        "credentials_logged": False,
        "database_writes": False,
        "automatic_control": "DISABLED",
        "recommendation_status": "NOT_GENERATED_INPUT_AUDIT_ONLY",
    }
    (args.output_dir / "active_exact_pilot_step_history_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
