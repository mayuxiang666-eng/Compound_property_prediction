"""Active-SFEPLANT-only corrected Phase builder for the first V3.7.1 trial."""

from __future__ import annotations

from Mooney_Prediction_Pipeline.data_processing import build_corrected_phase_history_v2 as v2
from Mooney_Prediction_Pipeline.data_processing import build_corrected_phase_history_v6 as v6  # noqa: F401


def _active_only(select_sql: str) -> str:
    """Retain V2's two parameter lists while making the second branch empty."""
    return select_sql


v2.MATERIAL_SQL = _active_only("""
SELECT N'SFEPLANT' SourceDatabase, CAST(o.OrderID AS VARCHAR(64)) OrderID,
       o.CompoundDescription RecipeID, om.MaterialCode, om.MaterialName,
       TRY_CAST(om.BatchWeight AS FLOAT) MaterialWeight
  FROM SFEPLANT.dbo.Orders o WITH (NOLOCK)
  JOIN SFEPLANT.dbo.OrderMaterials om WITH (NOLOCK) ON om.OrderID=o.OrderID
 WHERE o.OrderID IN ({placeholders})
UNION ALL
SELECT N'SFEPLANT', CAST(o.OrderID AS VARCHAR(64)), o.CompoundDescription,
       om.MaterialCode, om.MaterialName, TRY_CAST(om.BatchWeight AS FLOAT)
  FROM SFEPLANT.dbo.Orders o WITH (NOLOCK)
  JOIN SFEPLANT.dbo.OrderMaterials om WITH (NOLOCK) ON om.OrderID=o.OrderID
 WHERE 1=0 AND o.OrderID IN ({placeholders})
""")

v2.HEADER_SQL = _active_only("""
SELECT N'SFEPLANT' SourceDatabase, CAST(OrderID AS VARCHAR(64)) OrderID,
       BatchNumber, CAST(PalletID AS VARCHAR(128)) PalletID,
       TRY_CAST(BatchWeight AS FLOAT) BatchWeight, EquipmentID
  FROM SFEPLANT.dbo.BatchHeader WITH (NOLOCK) WHERE OrderID IN ({placeholders})
UNION ALL
SELECT N'SFEPLANT', CAST(OrderID AS VARCHAR(64)), BatchNumber,
       CAST(PalletID AS VARCHAR(128)), TRY_CAST(BatchWeight AS FLOAT), EquipmentID
  FROM SFEPLANT.dbo.BatchHeader WITH (NOLOCK) WHERE 1=0 AND OrderID IN ({placeholders})
""")

v2.CHARGE_SQL = _active_only("""
SELECT N'SFEPLANT' SourceDatabase, CAST(OrderID AS VARCHAR(64)) OrderID,
       BatchNumber, MaterialCode, MaterialName, CAST(ChargeID AS VARCHAR(256)) ChargeID,
       TRY_CAST(ActualWeight AS FLOAT) ActualWeight
  FROM SFEPLANT.dbo.BatchMaterials WITH (NOLOCK)
 WHERE OrderID IN ({placeholders}) AND ChargeID IS NOT NULL
UNION ALL
SELECT N'SFEPLANT', CAST(OrderID AS VARCHAR(64)), BatchNumber,
       MaterialCode, MaterialName, CAST(ChargeID AS VARCHAR(256)), TRY_CAST(ActualWeight AS FLOAT)
  FROM SFEPLANT.dbo.BatchMaterials WITH (NOLOCK)
 WHERE 1=0 AND OrderID IN ({placeholders}) AND ChargeID IS NOT NULL
""")

v2.RECIPE_SQL = _active_only("""
SELECT N'SFEPLANT' SourceDatabase, RecipeID, StepNumber, MaterialCode, MaterialName,
       TRY_CAST(Pphr AS FLOAT) Pphr
  FROM SFEPLANT.dbo.RecipeMaterials WITH (NOLOCK) WHERE RecipeID IN ({placeholders})
UNION ALL
SELECT N'SFEPLANT', RecipeID, StepNumber, MaterialCode, MaterialName, TRY_CAST(Pphr AS FLOAT)
  FROM SFEPLANT.dbo.RecipeMaterials WITH (NOLOCK) WHERE 1=0 AND RecipeID IN ({placeholders})
""")


if __name__ == "__main__":
    v2.main()
