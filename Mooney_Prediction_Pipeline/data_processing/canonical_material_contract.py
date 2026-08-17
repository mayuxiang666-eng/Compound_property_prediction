"""Versioned material/QM taxonomy for corrected Phase features.

V3.7 legacy data treated only ``CS100`` as silica and every other ``CS`` code
as oil.  Hefei recipe evidence shows silica grades such as ``CS1056``; the
canonical contract therefore uses CS10/CS12 for silica and CS4 for process oil.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


CONTRACT_VERSION = "MATERIAL-TAXONOMY-HF-V2"


def material_family(code: object) -> str:
    value = str(code or "").strip().upper()
    if (value.startswith("CE") and not value.startswith("CE19")):
        return "solid_elastomer"
    if value.startswith("CN"):
        return "natural_rubber"
    if value.startswith(("CS10", "CS12")):
        return "silica"
    if value.startswith("CS4"):
        return "oil"
    if value.startswith("CA551"):
        return "silane"
    if value.startswith("CC"):
        return "carbon_black"
    return "other"


def deduplicate_cross_database_rows(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    source_column: str = "SourceDatabase",
) -> pd.DataFrame:
    """Prefer active SFEPLANT while retaining archive-only historical rows."""
    work = frame.copy()
    if source_column not in work:
        return work.drop_duplicates(keys, keep="first")
    work["_source_priority"] = np.where(
        work[source_column].astype(str).str.upper().eq("SFEPLANT"), 0, 1,
    )
    work = work.sort_values([*keys, "_source_priority"], kind="stable")
    return work.drop_duplicates(keys, keep="first").drop(columns=["_source_priority"])


def aggregate_order_material_percentages(
    order_materials: pd.DataFrame,
    batch_headers: pd.DataFrame,
) -> pd.DataFrame:
    required = {"OrderID", "MaterialCode", "MaterialWeight"}
    if not required.issubset(order_materials.columns):
        raise ValueError(f"Order material columns missing: {sorted(required - set(order_materials.columns))}")
    headers_required = {"OrderID", "BatchNumber", "BatchWeight"}
    if not headers_required.issubset(batch_headers.columns):
        raise ValueError(f"Batch header columns missing: {sorted(headers_required - set(batch_headers.columns))}")
    materials = deduplicate_cross_database_rows(
        order_materials,
        [column for column in ("OrderID", "MaterialCode", "MaterialName", "MaterialWeight") if column in order_materials],
    )
    materials["family"] = materials["MaterialCode"].map(material_family)
    materials["MaterialWeight"] = pd.to_numeric(materials["MaterialWeight"], errors="coerce")
    totals = materials.loc[materials["family"] != "other"].pivot_table(
        index="OrderID", columns="family", values="MaterialWeight", aggfunc="sum",
    ).reset_index()
    result = batch_headers.merge(totals, on="OrderID", how="left", validate="many_to_one")
    result["BatchWeight"] = pd.to_numeric(result["BatchWeight"], errors="coerce")
    output_names = {
        "solid_elastomer": "weight_pct_solid_elastomer",
        "natural_rubber": "weight_pct_natural_rubber",
        "silica": "weight_pct_silica",
        "oil": "weight_pct_oil",
        "silane": "weight_pct_silian",
        "carbon_black": "weight_pct_carbon_black",
    }
    for family, output in output_names.items():
        numerator = pd.to_numeric(result.get(family, 0.0), errors="coerce").fillna(0.0)
        result[output] = numerator / result["BatchWeight"] * 100.0
    return result[["OrderID", "BatchNumber", *output_names.values()]]


def corrected_silica_phr(recipe_materials: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ("RecipeID", "StepNumber", "MaterialCode", "MaterialName", "Pphr") if column in recipe_materials]
    work = deduplicate_cross_database_rows(recipe_materials, keys)
    work["Pphr"] = pd.to_numeric(work["Pphr"], errors="coerce")
    work = work.loc[work["MaterialCode"].map(material_family).eq("silica")]
    return work.groupby("RecipeID", as_index=False)["Pphr"].sum().rename(columns={"Pphr": "silica_phr"})


def qm_parameter_category(material: object, parameter: object) -> str | None:
    code = str(material or "").strip().upper()
    text = re.sub(r"\s+", " ", str(parameter or "").strip().lower())
    family = material_family(code)
    if family == "silica" and "loss on heating" in text:
        return "supplier_silica_moisture_avg"
    if family == "silica" and "nitrogen surface area" in text:
        return "supplier_silica_surface_area_avg"
    if family == "carbon_black" and "dbp" in text:
        return "supplier_carbon_black_structure_avg"
    if family == "carbon_black" and ("stsa" in text or "iodine" in text):
        return "supplier_carbon_black_surface_area_avg"
    if family == "carbon_black" and "loss on heating" in text:
        return "supplier_carbon_black_moisture_avg"
    if family in {"solid_elastomer", "natural_rubber"} and "uml 1+4" in text:
        return "supplier_rubber_viscosity_avg"
    return None


def aggregate_batch_qm(charges: pd.DataFrame, qm: pd.DataFrame) -> pd.DataFrame:
    charge = deduplicate_cross_database_rows(
        charges,
        [column for column in ("OrderID", "BatchNumber", "MaterialCode", "ChargeID") if column in charges],
    )
    charge["ChargeID_clean"] = charge["ChargeID"].astype(str).str.strip().str.upper()
    quality = qm.copy()
    quality["ChargeID_clean"] = quality["batch"].astype(str).str.strip().str.upper()
    quality["category"] = [
        qm_parameter_category(material, parameter)
        for material, parameter in zip(quality["material"], quality["parameter"])
    ]
    quality["mean_value"] = pd.to_numeric(quality["mean_value"], errors="coerce")
    quality = quality.dropna(subset=["category", "mean_value"])
    per_charge = quality.groupby(["ChargeID_clean", "category"], as_index=False)["mean_value"].mean()
    linked = charge.merge(per_charge, on="ChargeID_clean", how="inner", validate="many_to_many")
    result = linked.pivot_table(
        index=["OrderID", "BatchNumber"], columns="category", values="mean_value", aggfunc="mean",
    ).reset_index()
    expected = [
        "supplier_rubber_viscosity_avg", "supplier_silica_moisture_avg",
        "supplier_silica_surface_area_avg", "supplier_carbon_black_structure_avg",
        "supplier_carbon_black_surface_area_avg", "supplier_carbon_black_moisture_avg",
    ]
    for column in expected:
        if column not in result:
            result[column] = np.nan
    return result[["OrderID", "BatchNumber", *expected]]
