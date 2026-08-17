"""Read-only probe for family/recipe-level Mooney specification ownership."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import psycopg2 as psy


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
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
                compound_name, master_recipe,
                test_target, tolerance_lower, tolerance_upper,
                corp_tolerance_lower, corp_tolerance_upper,
                COUNT(*) AS n,
                MIN(test_result_start_time) AS first_time,
                MAX(test_result_start_time) AS last_time
            FROM he_datamarts.compound_excellence_datamart
            WHERE test_variable = %s
              AND compound_name LIKE %s
              AND test_status <> %s
              AND test_status_name <> %s
            GROUP BY
                compound_name, master_recipe,
                test_target, tolerance_lower, tolerance_upper,
                corp_tolerance_lower, corp_tolerance_upper
            ORDER BY compound_name, master_recipe, first_time
        """
        rows = pd.read_sql_query(query, connection, params=["MS1+3", "M1-T09170%", "Failed", "D"])
    finally:
        connection.close()
    output = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow" / "t09170ws_02004_spec_probe.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"rows={len(rows)}")
    print(rows.to_string(index=False))
    print(f"output={output}")


if __name__ == "__main__":
    main()
