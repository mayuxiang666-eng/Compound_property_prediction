"""Read-only screen for currently active B00458/T09170 family recipe candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from V4_2_Step_Phase_Architecture.data_extraction.sql_connection import get_sfe_connection


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "corrected_phase_history"
ELIGIBILITY = REPORT / "v371_active_candidates_v1" / "operational_core_no_rubber_qm" / "group_eligibility_and_split.csv"
OUTPUT = REPORT / "active_family_recipe_screen_v1"


def main() -> None:
    query = """
        SELECT o.OrderID, o.CompoundDescription AS RecipeID, o.CompoundName,
               o.Equipment AS MixerLine, o.OrderStartTime,
               COUNT(bh.BatchNumber) AS BatchCount
        FROM SFEPlant.dbo.Orders o WITH (NOLOCK)
        LEFT JOIN SFEPlant.dbo.BatchHeader bh WITH (NOLOCK) ON bh.OrderID = o.OrderID
        WHERE (o.CompoundName LIKE '%B00458%' OR o.CompoundName LIKE '%T09170%')
          AND o.OrderID NOT LIKE 'H%'
        GROUP BY o.OrderID, o.CompoundDescription, o.CompoundName, o.Equipment, o.OrderStartTime
    """
    connection = get_sfe_connection()
    try:
        orders = pd.read_sql_query(query, connection)
    finally:
        connection.close()
    orders["OrderID"] = orders["OrderID"].astype(str).str.strip()
    orders["OrderStartTime"] = pd.to_datetime(orders["OrderStartTime"], errors="coerce")
    recipe = orders.groupby(["RecipeID", "CompoundName", "MixerLine"], as_index=False).agg(
        ActiveOrders=("OrderID", "nunique"), ActiveBatches=("BatchCount", "sum"),
        FirstActiveOrder=("OrderStartTime", "min"), LatestActiveOrder=("OrderStartTime", "max"),
    )

    groups = pd.read_csv(ELIGIBILITY, low_memory=False)
    complete = groups.loc[groups["GroupComplete"] & ~groups["MNYConflict"]].copy()
    # CompoundDescription is not present in the eligibility table; the corrected source
    # provides the exact RecipeID for the same batch/group identities.
    source = pd.read_csv(
        REPORT / "phase_corrected_v371_active.csv",
        usecols=["OrderID", "BatchNumber", "CompoundDescription"], low_memory=False,
    )
    for frame in (complete, source):
        frame["OrderID"] = frame["OrderID"].astype(str).str.replace(r"\.0$", "", regex=True)
    source["BatchNumber"] = pd.to_numeric(source["BatchNumber"], errors="raise").astype(int)
    group_recipe = complete[["LabelGroupID", "OrderID"]].merge(
        source[["OrderID", "CompoundDescription"]].drop_duplicates(),
        on="OrderID", how="left", validate="many_to_one",
    ).drop_duplicates(["LabelGroupID", "CompoundDescription"])
    support = group_recipe.groupby("CompoundDescription", as_index=False).agg(
        CompleteLabelGroups=("LabelGroupID", "nunique")
    ).rename(columns={"CompoundDescription": "RecipeID"})
    result = recipe.merge(support, on="RecipeID", how="left")
    result["CompleteLabelGroups"] = result["CompleteLabelGroups"].fillna(0).astype(int)
    result["RecentAfter2026"] = result["LatestActiveOrder"].ge(pd.Timestamp("2026-01-01"))
    result["BaselineMin150"] = result["CompleteLabelGroups"].ge(150)
    result["NextAuditPriority"] = result["RecentAfter2026"] & result["ActiveBatches"].ge(150)
    result = result.sort_values(["LatestActiveOrder", "ActiveBatches"], ascending=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    orders.to_csv(OUTPUT / "family_active_orders.csv", index=False, encoding="utf-8-sig")
    result.to_csv(OUTPUT / "family_recipe_screen.csv", index=False, encoding="utf-8-sig")
    summary = {
        "contract": "ACTIVE-FAMILY-RECIPE-SCREEN-V1",
        "recipe_count": int(len(result)),
        "recent_recipe_count": int(result["RecentAfter2026"].sum()),
        "recent_with_150_batches": int(result["NextAuditPriority"].sum()),
        "recent_with_150_complete_label_groups": int((result["RecentAfter2026"] & result["BaselineMin150"]).sum()),
        "priority_candidates": result.loc[result["NextAuditPriority"], [
            "RecipeID", "CompoundName", "MixerLine", "ActiveOrders", "ActiveBatches",
            "CompleteLabelGroups", "LatestActiveOrder", "BaselineMin150",
        ]].to_dict("records"),
        "database_writes": False, "automatic_control": "DISABLED",
    }
    (OUTPUT / "family_recipe_screen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
