"""Corrected, online-safe Phase feature contract."""

from __future__ import annotations

import pandas as pd

from .cb_dispersion_feature_builder import build_cb_dispersion_features
from .clustering import cluster_silica_carbon_black
from .material_dependent_features import recompute_material_dependent_features
from .silica_pid_feature_builder import build_silica_pid_features


CONTRACT_VERSION = "PHASE-CORRECTED-V2"


def apply_corrected_phase_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute dependent fields and remove the synthetic bottom-mixer ram.

    Curve7 belongs to the top mixer.  Bottom mixing has no physical ram signal;
    historical non-zero Stage6 ram values were created by cross-batch time-index
    imputation and are not valid online process measurements.
    """
    work = recompute_material_dependent_features(frame)
    work["Stage6_BottomMixing_WayofRam_Mean"] = 0.0
    work["Stage6_BottomMixing_WayofRam_Std"] = 0.0
    work["Stage6RamApplicability"] = "NOT_APPLICABLE_BOTTOM_MIXER"
    for derived in (build_silica_pid_features(work), build_cb_dispersion_features(work)):
        for column in derived.columns:
            work[column] = derived[column]
    work = cluster_silica_carbon_black(work)
    work["PhaseContractVersion"] = CONTRACT_VERSION
    return work
