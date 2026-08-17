"""Read-only latest Active-order probe for the two exact pilot compounds."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "corrected_phase_history" / "active_exact_pilot_step_history_v1"
COMPOUNDS = ["M1-B00458---- 12 004", "M1-T09170WS-- 02 004"]


def main() -> None:
    query = """
        SELECT TOP (500)
            o.OrderID, o.CompoundDescription AS RecipeID, o.CompoundName,
            o.Equipment AS MixerLine, o.OrderStartTime,
            COUNT(bh.BatchNumber) AS BatchCount,
            MIN(bh.BatchNumber) AS FirstBatchNumber,
            MAX(bh.BatchNumber) AS LastBatchNumber
        FROM SFEPlant.dbo.Orders o WITH (NOLOCK)
        LEFT JOIN SFEPlant.dbo.BatchHeader bh WITH (NOLOCK) ON bh.OrderID = o.OrderID
        WHERE o.CompoundName IN (?, ?)
        GROUP BY o.OrderID, o.CompoundDescription, o.CompoundName, o.Equipment, o.OrderStartTime
        ORDER BY o.OrderStartTime DESC
    """
    connection = get_sfe_connection()
    try:
        frame = pd.read_sql_query(query, connection, params=COMPOUNDS)
    finally:
        connection.close()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "exact_pilot_latest_active_orders.csv", index=False, encoding="utf-8-sig")
    frame["OrderStartTime"] = pd.to_datetime(frame["OrderStartTime"], errors="coerce")
    summary = {"contract": "EXACT-PILOT-LATEST-ACTIVE-ORDER-PROBE-V1", "compounds": {}}
    for compound in COMPOUNDS:
        selected = frame.loc[frame["CompoundName"].astype(str).eq(compound)]
        summary["compounds"][compound] = {
            "orders_returned": int(len(selected)),
            "latest_order_time": selected["OrderStartTime"].max().isoformat() if not selected.empty else None,
            "latest_recipe_ids": selected.sort_values("OrderStartTime", ascending=False)["RecipeID"].astype(str).head(10).unique().tolist(),
            "latest_orders": selected.sort_values("OrderStartTime", ascending=False).head(10)[
                ["OrderID", "RecipeID", "OrderStartTime", "BatchCount"]
            ].astype({"OrderID": str}).to_dict("records"),
        }
    summary.update({"database_writes": False, "automatic_control": "DISABLED"})
    (OUTPUT / "exact_pilot_latest_active_orders_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
