"""Recompute all features that depend on corrected recipe percentages."""

from __future__ import annotations

import numpy as np
import pandas as pd


STAGES = (
    "Stage1_Loading", "Stage2_DryMixing", "Stage3_OilLoading",
    "Stage4_WetMixing", "Stage5_PID", "Stage6_BottomMixing",
)


def recompute_material_dependent_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    numeric = {}
    for column in (
        "weight_pct_solid_elastomer", "weight_pct_natural_rubber",
        "weight_pct_silica", "weight_pct_oil", "weight_pct_carbon_black",
    ):
        numeric[column] = pd.to_numeric(work.get(column, 0.0), errors="coerce").fillna(0.0)
    rubber = numeric["weight_pct_solid_elastomer"] + numeric["weight_pct_natural_rubber"]
    filler = numeric["weight_pct_silica"] + numeric["weight_pct_carbon_black"]
    solids = rubber + filler
    work["ratio_nr_rubber"] = np.divide(
        numeric["weight_pct_natural_rubber"], rubber,
        out=np.zeros(len(work), dtype=float), where=rubber.to_numpy() > 0,
    )
    work["ratio_filler_polymer"] = np.divide(
        filler, rubber, out=np.zeros(len(work), dtype=float), where=rubber.to_numpy() > 0,
    )
    work["ratio_oil_polymer"] = np.divide(
        numeric["weight_pct_oil"], rubber,
        out=np.zeros(len(work), dtype=float), where=rubber.to_numpy() > 0,
    )
    work["ratio_oil_filler"] = np.divide(
        numeric["weight_pct_oil"], filler,
        out=np.zeros(len(work), dtype=float), where=filler.to_numpy() > 0,
    )
    for stage in STAGES:
        integral = pd.to_numeric(work.get(f"{stage}_power_Integral", 0.0), errors="coerce").fillna(0.0)
        work[f"{stage}_Specific_Energy"] = np.divide(
            integral, solids, out=np.zeros(len(work), dtype=float), where=solids.to_numpy() > 0,
        )
    return work
