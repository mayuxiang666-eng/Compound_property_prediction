import pandas as pd


STAGE1_RECIPE_COLUMNS = (
    "Top_Fill_Factor",
    "Bot_Fill_Factor",
    "Target_Temperature",
    "weight_pct_solid_elastomer",
    "weight_pct_natural_rubber",
    "weight_pct_silica",
    "weight_pct_oil",
    "weight_pct_silian",
    "weight_pct_carbon_black",
    "silica_phr",
    "is_oil_loading_present",
    "ratio_nr_rubber",
    "ratio_filler_polymer",
    "ratio_oil_polymer",
    "ratio_oil_filler",
    "supplier_rubber_viscosity_avg",
    "supplier_silica_moisture_avg",
    "supplier_silica_surface_area_avg",
    "supplier_carbon_black_structure_avg",
    "supplier_carbon_black_surface_area_avg",
    "supplier_carbon_black_moisture_avg",
)


def extract_stage1_recipe_features(df: pd.DataFrame) -> list[str]:
    """Return production-known recipe and COA features available in ``df``.

    The allowlist prevents targets, identifiers, current-batch process signals,
    and time-derived fields from entering the static baseline.
    """
    return [column for column in STAGE1_RECIPE_COLUMNS if column in df.columns]
