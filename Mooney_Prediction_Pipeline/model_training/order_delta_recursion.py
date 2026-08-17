"""Type-stable recursive fusion of level and within-order delta predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def recursive_order_prediction(
    frame: pd.DataFrame,
    level_prediction: np.ndarray,
    pair_prediction: pd.DataFrame,
    beta: float,
) -> np.ndarray:
    """Fuse absolute level and delta path estimates inside each OrderID.

    beta=1 keeps the level estimate; beta=0 follows the delta path after the
    first group in each order.  LabelGroupID is normalized to string on both
    sides so integer/string CSV inference cannot silently disable recursion.
    """

    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    ids = frame["LabelGroupID"].astype(str)
    result = pd.Series(np.asarray(level_prediction, dtype=float), index=ids, dtype=float)
    pairs = pair_prediction.copy()
    pairs["to_group"] = pairs["to_group"].astype(str)
    pair_map = pd.to_numeric(pairs.set_index("to_group")["DeltaPrediction"], errors="coerce").to_dict()
    ordered = frame.assign(_group_id=ids).sort_values(
        ["OrderID", "production_start", "PalletID", "LabelGroupID"], kind="stable"
    )
    applied = 0
    for _, order in ordered.groupby("OrderID", sort=False):
        previous_id = None
        for current_id in order["_group_id"]:
            if previous_id is not None and current_id in pair_map and pd.notna(pair_map[current_id]):
                path_prediction = float(result.loc[previous_id]) + float(pair_map[current_id])
                result.loc[current_id] = float(beta) * float(result.loc[current_id]) + (1.0 - float(beta)) * path_prediction
                applied += 1
            previous_id = current_id
    output = ids.map(result).to_numpy(dtype=float)
    if applied == 0 and len(pair_prediction) > 0 and float(beta) < 1.0:
        raise RuntimeError("No delta predictions were applied; check LabelGroupID contract")
    return output


__all__ = ["recursive_order_prediction"]
