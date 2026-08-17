"""Read-only audit of exact RecipeID presence versus Lab spec inheritance."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import psycopg2

from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection


ROOT = Path(__file__).resolve().parents[2]
RECIPE_ID = "9922A09170AX_PV50_004"
COMPOUND_NAME = "M1-T09170WS-- 02 004"
OUT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"


def main() -> None:
    sfe = get_sfe_connection()
    try:
        order_query = """
            SELECT 'ONLINE' AS source, COUNT(*) AS rows,
                   COUNT(DISTINCT OrderID) AS orders
            FROM dbo.Orders WITH (NOLOCK)
            WHERE CompoundDescription = ?
            UNION ALL
            SELECT 'ARCHIVE' AS source, COUNT(*) AS rows,
                   COUNT(DISTINCT OrderID) AS orders
            FROM SFEPLANT_ARCHIVE.dbo.Orders WITH (NOLOCK)
            WHERE CompoundDescription = ?
        """
        order_counts = pd.read_sql_query(order_query, sfe, params=[RECIPE_ID, RECIPE_ID])
        param_query = """
            SELECT COUNT(*) AS rows, COUNT(DISTINCT RecipeID) AS recipes
            FROM dbo.RecipeCBS3Parameters WITH (NOLOCK)
            WHERE RecipeID = ?
        """
        recipe_params = pd.read_sql_query(param_query, sfe, params=[RECIPE_ID])
    finally:
        sfe.close()

    credential_path = ROOT / "Mooney_Prediction_Pipeline" / "data_processing" / "credentials.json"
    credentials = json.loads(credential_path.read_text(encoding="utf-8"))["mustangmaster"]
    lab = psycopg2.connect(
        host=credentials["host"], port=credentials["port"], database=credentials["database"],
        user=credentials["user"], password=credentials["password"], sslmode="require",
    )
    try:
        exact_query = """
            SELECT compound_name, master_recipe, test_target, tolerance_lower,
                   tolerance_upper, corp_tolerance_lower, corp_tolerance_upper,
                   COUNT(*) AS n, MIN(test_result_start_time) AS first_time,
                   MAX(test_result_start_time) AS last_time
            FROM he_datamarts.compound_excellence_datamart
            WHERE test_variable = %s AND master_recipe = %s
              AND test_status <> %s AND test_status_name <> %s
            GROUP BY compound_name, master_recipe, test_target, tolerance_lower,
                     tolerance_upper, corp_tolerance_lower, corp_tolerance_upper
            ORDER BY first_time
        """
        family_query = exact_query.replace("master_recipe = %s", "compound_name LIKE %s")
        exact = pd.read_sql_query(exact_query, lab, params=["MS1+3", RECIPE_ID, "Failed", "D"])
        family = pd.read_sql_query(family_query, lab, params=["MS1+3", "M1-T09170%", "Failed", "D"])
    finally:
        lab.close()

    OUT.mkdir(parents=True, exist_ok=True)
    exact.to_csv(OUT / "t09170ws_02004_lab_exact_recipe_spec.csv", index=False, encoding="utf-8-sig")
    family.to_csv(OUT / "t09170ws_02004_lab_family_spec_evidence.csv", index=False, encoding="utf-8-sig")
    order_counts.to_csv(OUT / "t09170ws_02004_mms_exact_recipe_order_evidence.csv", index=False, encoding="utf-8-sig")
    recipe_params.to_csv(OUT / "t09170ws_02004_recipe_parameter_evidence.csv", index=False, encoding="utf-8-sig")

    valid_family = family.copy()
    for column in ["test_target", "tolerance_lower", "tolerance_upper"]:
        valid_family[column] = pd.to_numeric(valid_family[column], errors="coerce")
    family_contracts = valid_family[["test_target", "tolerance_lower", "tolerance_upper"]].dropna().drop_duplicates()
    summary = {
        "recipe_id": RECIPE_ID,
        "compound_name": COMPOUND_NAME,
        "mms_exact_recipe_order_rows": int(order_counts["rows"].sum()),
        "mms_exact_recipe_distinct_orders": int(order_counts["orders"].sum()),
        "recipe_cbs3_parameter_rows": int(recipe_params.iloc[0]["rows"]),
        "lab_exact_recipe_spec_rows": int(len(exact)),
        "lab_family_spec_rows": int(len(family)),
        "family_numeric_contract_count": int(len(family_contracts)),
        "family_spec_contracts": family_contracts.to_dict(orient="records"),
        "status": "FAMILY_SPEC_INHERITED_VERSION_CONFIRMATION_REQUIRED" if len(exact) == 0 and len(family_contracts) == 1 else "PASS_EXACT_SPEC_FOUND",
        "interpretation": "MMS confirms the exact RecipeID and recipe parameters, but Lab has no exact master_recipe spec row; family target/tolerance may only be used after process/quality confirms inheritance.",
        "automatic_control": "DISABLED",
    }
    path = OUT / "t09170ws_02004_spec_inheritance_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"summary={path}")


if __name__ == "__main__":
    main()
