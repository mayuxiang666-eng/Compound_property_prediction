"""Create auditable label/spec/SCP-LOG artifacts for the selected silica pilot.

All database operations are SELECT-only. Credentials are read only at runtime and
are never printed or written to output artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import psycopg2 as psy

from Mooney_Prediction_Pipeline.optimization.recipe_scp_logic import contracts_to_frame, parse_recipe_scp_logic
from Mooney_Prediction_Pipeline.optimization.silica_label_alignment import (
    audit_pallet_label_alignment,
    resolve_mooney_spec_contract,
    write_alignment_audit,
)
from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection


COMPOUND = "M1-T09170WS-- 02 004"
RECIPE_ID = "9922A09170AX_PV50_004"
ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = ROOT / "Mooney_Prediction_Pipeline" / "models" / "hefei_group_training_dataset_clean.csv"
OUTPUT_DIRECTORY = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"


def fetch_lab_rows() -> pd.DataFrame:
    credential_path = ROOT / "Mooney_Prediction_Pipeline" / "data_processing" / "credentials.json"
    credentials = json.loads(credential_path.read_text(encoding="utf-8"))["mustangmaster"]
    connection = psy.connect(
        host=credentials["host"],
        port=credentials["port"],
        database=credentials["database"],
        user=credentials["user"],
        password=credentials["password"],
        sslmode="require",
    )
    try:
        query = """
            SELECT
                test_result, test_target, tolerance_lower, tolerance_upper,
                corp_tolerance_lower, corp_tolerance_upper,
                test_result_start_time, retest_number, test_status,
                test_status_name, compound_name, master_recipe, datamart_pk,
                last_modified_timestamp
            FROM he_datamarts.compound_excellence_datamart
            WHERE test_variable = %s
              AND master_recipe = %s
              AND test_status <> %s
              AND test_status_name <> %s
        """
        return pd.read_sql_query(query, connection, params=["MS1+3", RECIPE_ID, "Failed", "D"])
    finally:
        connection.close()


def fetch_recipe_scp_parameters() -> pd.DataFrame:
    connection = get_sfe_connection()
    try:
        query = """
            SELECT RecipeID, StepNumber, ParameterID, ParameterName, ParameterValue, UOM
            FROM dbo.RecipeCBS3Parameters
            WHERE RecipeID = ?
              AND ParameterID LIKE ?
            ORDER BY StepNumber, ParameterID
        """
        return pd.read_sql_query(query, connection, params=[RECIPE_ID, "%SCP%"])
    finally:
        connection.close()


def main() -> None:
    history = pd.read_csv(HISTORY_PATH, low_memory=False)
    alignment = audit_pallet_label_alignment(history, compound_name=COMPOUND)
    lab_rows = fetch_lab_rows()
    spec = resolve_mooney_spec_contract(lab_rows, compound_name=COMPOUND)
    recipe_parameters = fetch_recipe_scp_parameters()
    contracts = parse_recipe_scp_logic(recipe_parameters, recipe_id=RECIPE_ID)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    alignment_csv, summary_json = write_alignment_audit(
        alignment,
        spec,
        output_directory=OUTPUT_DIRECTORY,
        stem="t09170ws_02004",
    )
    contracts_to_frame(contracts).to_csv(
        OUTPUT_DIRECTORY / "t09170ws_02004_scp_logic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lab_rows.to_csv(
        OUTPUT_DIRECTORY / "t09170ws_02004_lab_spec_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"alignment_csv={alignment_csv}")
    print(f"summary_json={summary_json}")
    print(f"records={len(alignment)} one_batch={int(alignment['eligible_for_batch_step_replay'].sum())}")
    print(f"spec_status={spec.status} target={spec.target} lower={spec.lower} upper={spec.upper}")
    print(f"scp_contracts={len(contracts)}")


if __name__ == "__main__":
    main()
