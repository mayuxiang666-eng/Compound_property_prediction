"""Build and audit the real Step -> Phase -> V3.7 path for strict labels.

This is an audit only.  It never writes a recipe target and treats any
missing V3.7 feature as a blocked path, even though the legacy builder has a
zero-fill compatibility behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection
from V4_2_Step_Phase_Architecture.data_extraction.settable_flag_builder import build_settable_flags
from V4_2_Step_Phase_Architecture.mapping.step_phase_mapping import build_step_phase_mapping
from V4_2_Step_Phase_Architecture.feature_engineering.aggregate_sp_to_phase import aggregate_steps_to_phase_features
from Mooney_Prediction_Pipeline.optimization.v37_feature_row_builder import build_v37_baseline_feature_row
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor


ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "Mooney_Prediction_Pipeline" / "models" / "hefei_group_training_dataset_clean.csv"
RAW_BATCH = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow" / "t09170ws_02004_raw_material_lot_by_batch_v3.csv"
OUT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"
PACKAGE = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v37_production_model_package"


def _pair_clause(pairs: list[tuple[str, int]]) -> str:
    terms = []
    for order_id, batch_number in pairs:
        if not order_id.isdigit():
            raise ValueError(f"Unexpected non-numeric OrderID: {order_id!r}")
        terms.append(f"(bd.OrderID = {int(order_id)} AND bd.BatchNumber = {int(batch_number)})")
    return " OR ".join(terms) or "1 = 0"


def _query_step_wide(connection, pairs: list[tuple[str, int]], *, archive: bool) -> pd.DataFrame:
    table = "SFEPLANT_ARCHIVE.dbo.BatchData" if archive else "SFEPlant.dbo.BatchData"
    clause = _pair_clause(pairs)
    query = f"""
        SELECT bd.OrderID, bd.BatchNumber, bd.EquipmentID, bd.GroupName, bd.StepNo,
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
        FROM {table} bd WITH (NOLOCK)
        WHERE bd.GroupName IN ('AVR_MST','AVR_MSB')
          AND bd.VariablePath IN ('SCP-1-Step-Time-rel-s','SCP-3-Temperature-C-F',
                                 'Rotor-Speed-1/min-Rpm','RAM-Pressure-percent',
                                 'SCP-2-Energy-rel-kWh')
          AND ({clause})
        GROUP BY bd.OrderID, bd.BatchNumber, bd.EquipmentID, bd.GroupName, bd.StepNo
        ORDER BY bd.OrderID, bd.BatchNumber, bd.GroupName, bd.StepNo
    """
    return pd.read_sql_query(query, connection)


def main() -> None:
    history = pd.read_csv(HISTORY, low_memory=False)
    history = history[history["CompoundName"].eq("M1-T09170WS-- 02 004")].copy()
    history["OrderID"] = history["OrderID"].astype(str).str.strip()
    history["PalletID"] = history["PalletID"].astype(str).str.strip()
    raw = pd.read_csv(RAW_BATCH, low_memory=False)
    raw["OrderID"] = raw["OrderID"].astype(str).str.strip()
    raw["PalletID"] = raw["PalletID"].astype(str).str.strip()
    raw["BatchNumber"] = pd.to_numeric(raw["BatchNumber"], errors="coerce")
    strict = history[history["N_batches"].eq(1)].merge(
        raw[["OrderID", "PalletID", "BatchNumber", "batch_lot_status"]],
        on=["OrderID", "PalletID"], how="inner",
    )
    strict = strict[strict["batch_lot_status"].eq("PASS")].drop_duplicates(["OrderID", "BatchNumber"])
    pairs = [(str(row.OrderID), int(row.BatchNumber)) for row in strict.itertuples()]

    connection = get_sfe_connection()
    try:
        online = _query_step_wide(connection, pairs, archive=False)
        archive = _query_step_wide(connection, pairs, archive=True)
    finally:
        connection.close()
    step = pd.concat([online, archive], ignore_index=True).drop_duplicates(
        subset=["OrderID", "BatchNumber", "EquipmentID", "GroupName", "StepNo"], keep="last"
    )
    numeric = ["time_SP", "time_AV", "temp_SP", "temp_AV", "rotor_SP", "rotor_AV", "ram_SP", "ram_AV", "energy_AV"]
    for column in numeric:
        step[column] = pd.to_numeric(step[column], errors="coerce")
    flagged = build_settable_flags(step)
    mapped = build_step_phase_mapping(flagged, target_temp=165.0)
    phase = aggregate_steps_to_phase_features(mapped)

    OUT.mkdir(parents=True, exist_ok=True)
    mapped_path = OUT / "t09170ws_02004_strict_step_phase_mapping.csv"
    phase_path = OUT / "t09170ws_02004_strict_phase_features.csv"
    mapped.to_csv(mapped_path, index=False, encoding="utf-8-sig")
    phase.to_csv(phase_path, index=False, encoding="utf-8-sig")

    predictor = V37RawPredictor(PACKAGE).load()
    v37_rows = []
    for order_id, batch_number in pairs:
        record = {"OrderID": order_id, "BatchNumber": batch_number}
        try:
            built = build_v37_baseline_feature_row(order_id, batch_number, predictor)
            record.update({
                "missing_feature_count": built.missing_feature_count,
                "missing_feature_names": "|".join(built.missing_feature_names),
                "v37_feature_count": len(predictor.feature_columns),
                "model_artifact_loaded": True,
                "model_path_status": "MODEL_PATH_OK" if built.missing_feature_count == 0 else "BLOCKED_MISSING_RAW_FEATURES",
            })
            if built.missing_feature_count == 0:
                record["baseline_prediction"] = float(predictor.predict_raw(pd.DataFrame([built.feature_row])).final_prediction[0])
        except Exception as exc:
            record.update({
                "missing_feature_count": None,
                "missing_feature_names": "",
                "v37_feature_count": len(predictor.feature_columns),
                "model_artifact_loaded": True,
                "model_path_status": "BLOCKED_V37_BASELINE_BUILD_ERROR",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            })
        v37_rows.append(record)
    v37 = pd.DataFrame(v37_rows)
    v37_path = OUT / "t09170ws_02004_strict_v37_input_path_audit.csv"
    v37.to_csv(v37_path, index=False, encoding="utf-8-sig")

    summary = {
        "candidate_strict_raw_lot_batches": len(pairs),
        "step_rows_online": len(online),
        "step_rows_archive": len(archive),
        "step_rows_combined": len(step),
        "mapped_batch_count": int(mapped[["OrderID", "BatchNumber"]].drop_duplicates().shape[0]) if not mapped.empty else 0,
        "phase_batch_count": int(phase[["OrderID", "BatchNumber"]].drop_duplicates().shape[0]) if not phase.empty else 0,
        "mapping_phase_count": int(mapped["mapped_phase"].nunique()) if not mapped.empty else 0,
        "settable_step_count": int((mapped[["is_time_settable", "is_temp_settable", "is_rotor_settable"]].any(axis=1)).sum()) if not mapped.empty else 0,
        "v37_model_feature_count": len(predictor.feature_columns),
        "v37_model_path_ok_batches": int(v37["model_path_status"].eq("MODEL_PATH_OK").sum()),
        "v37_blocked_missing_raw_features": int(v37["model_path_status"].eq("BLOCKED_MISSING_RAW_FEATURES").sum()),
        "v37_blocked_build_errors": int(v37["model_path_status"].eq("BLOCKED_V37_BASELINE_BUILD_ERROR").sum()),
        "status": "MODEL_PATH_OK" if not v37.empty and v37["model_path_status"].eq("MODEL_PATH_OK").all() else "BLOCKED_V37_INPUT_PATH",
        "automatic_control": "DISABLED",
        "note": "The legacy baseline builder may zero-fill missing artifact columns; this audit treats those missing columns as a hard block per P0 data-quality policy.",
    }
    summary_path = OUT / "t09170ws_02004_strict_step_phase_v37_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"mapping={mapped_path}")
    print(f"phase={phase_path}")
    print(f"v37={v37_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
