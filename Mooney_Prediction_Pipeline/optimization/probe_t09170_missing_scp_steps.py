"""Read-only probe for current recipe parameters on SCP-uncovered T09170 Steps."""

from pathlib import Path

import pandas as pd

from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "corrected_phase_history" / "active_exact_pilot_step_history_v1" / "t09170_missing_scp_step_all_parameters.csv"


def main() -> None:
    query = """
        SELECT RecipeID, StepNumber, ParameterID, ParameterName, ParameterValue, UOM
        FROM SFEPlant.dbo.RecipeCBS3Parameters WITH (NOLOCK)
        WHERE RecipeID = ? AND StepNumber IN (2, 7, 8, 9, 10)
        ORDER BY StepNumber, ParameterID
    """
    connection = get_sfe_connection()
    try:
        frame = pd.read_sql_query(query, connection, params=["9922A09170AX_PV50_004"])
    finally:
        connection.close()
    frame.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"rows={len(frame)}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
