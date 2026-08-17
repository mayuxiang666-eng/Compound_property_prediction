"""Hard gate against label aliases entering model feature contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd


FORBIDDEN_EXACT = {
    "mny", "actual_mny", "lab_mny", "result_mny", "mooney", "actual_mooney",
    "lab_time", "levelprediction", "finalprediction", "ooflevelprediction", "residualtarget",
}


def find_target_leakage(frame: pd.DataFrame, features: list[str], target: str = "MNY") -> list[str]:
    leaks: set[str] = set()
    target_values = pd.to_numeric(frame[target], errors="coerce") if target in frame else None
    for feature in features:
        normalized = feature.lower().replace("_", "").replace(" ", "")
        exact = feature.lower().strip()
        if exact in FORBIDDEN_EXACT or normalized in FORBIDDEN_EXACT:
            leaks.add(feature)
            continue
        if any(token in normalized for token in ("actualmny", "labmny", "resultmny", "mooneyresult")):
            leaks.add(feature)
            continue
        if target_values is not None and feature in frame:
            values = pd.to_numeric(frame[feature], errors="coerce")
            valid = values.notna() & target_values.notna()
            if int(valid.sum()) >= 20 and np.allclose(values.loc[valid], target_values.loc[valid], equal_nan=True):
                leaks.add(feature)
    return sorted(leaks)


def assert_no_target_leakage(frame: pd.DataFrame, features: list[str], target: str = "MNY") -> None:
    leaks = find_target_leakage(frame, features, target)
    if leaks:
        raise ValueError(f"TARGET_LEAKAGE_BLOCKED: {leaks}")


__all__ = ["assert_no_target_leakage", "find_target_leakage"]
