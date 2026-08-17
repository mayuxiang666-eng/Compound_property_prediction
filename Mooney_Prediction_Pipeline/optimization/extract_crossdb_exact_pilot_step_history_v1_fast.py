"""Index-seeking adapter for exact cross-database pilot Step extraction."""

from __future__ import annotations

import pandas as pd

from Mooney_Prediction_Pipeline.optimization import (
    extract_crossdb_exact_pilot_step_history_v1_exact as exact,
)


def _query_exact_pairs_fast(
    connection, database: str, pairs: list[tuple[str, int]],
) -> pd.DataFrame:
    values_clause = ",".join("(CAST(? AS VARCHAR(20)),CAST(? AS INT))" for _ in pairs)
    params: list[object] = []
    for order_id, batch_number in pairs:
        params.extend([order_id, int(batch_number)])
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
        FROM (VALUES {values_clause}) wanted(OrderID, BatchNumber)
        INNER LOOP JOIN {database}.dbo.BatchData bd WITH (NOLOCK, FORCESEEK)
          ON bd.OrderID = wanted.OrderID AND bd.BatchNumber = wanted.BatchNumber
        WHERE bd.GroupName IN ('AVR_MST','AVR_MSB')
          AND bd.VariablePath IN (
              'SCP-1-Step-Time-rel-s','SCP-3-Temperature-C-F',
              'Rotor-Speed-1/min-Rpm','RAM-Pressure-percent','SCP-2-Energy-rel-kWh'
          )
        GROUP BY bd.OrderID, bd.BatchNumber, bd.EquipmentID, bd.GroupName, bd.StepNo
        OPTION (FORCE ORDER)
    """
    return pd.read_sql_query(query, connection, params=params)


def main() -> None:
    exact._query_exact_pairs = _query_exact_pairs_fast
    exact.main()


if __name__ == "__main__":
    main()
