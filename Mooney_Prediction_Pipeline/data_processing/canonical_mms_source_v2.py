"""V2 facade that includes corrected, deduplicated recipe silica PHR."""

from __future__ import annotations

import pandas as pd

from .canonical_material_contract import aggregate_batch_qm, corrected_silica_phr
from .canonical_mms_source import (
    query_batch_headers,
    query_batch_material_charges,
    query_order_materials,
    query_recipe_materials,
    query_supplier_qm,
)
from .canonical_material_contract import aggregate_order_material_percentages


def build_corrected_static_features(order_id: str) -> pd.DataFrame:
    materials = query_order_materials(order_id)
    headers = query_batch_headers(order_id)
    if materials.empty or headers.empty:
        raise ValueError(f"No material/header rows for OrderID={order_id}")
    base = aggregate_order_material_percentages(materials, headers)
    charges = query_batch_material_charges(order_id)
    if charges.empty:
        qm = pd.DataFrame(columns=["OrderID", "BatchNumber"])
    else:
        qm = aggregate_batch_qm(charges, query_supplier_qm(charges))
    result = base.merge(qm, on=["OrderID", "BatchNumber"], how="left", validate="one_to_one")

    recipe_ids = materials["RecipeID"].dropna().astype(str).drop_duplicates().tolist()
    recipes = query_recipe_materials(recipe_ids)
    phr = corrected_silica_phr(recipes)
    recipe_map = materials[["OrderID", "RecipeID"]].drop_duplicates("OrderID")
    result = result.merge(recipe_map, on="OrderID", how="left", validate="many_to_one")
    result = result.merge(phr, on="RecipeID", how="left", validate="many_to_one")
    result["MaterialContractVersion"] = "MATERIAL-TAXONOMY-HF-V2"
    return result.drop(columns=["RecipeID"], errors="ignore")
