"""Label-group-aware weighting for Mooney model training and evaluation."""

import numpy as np
import pandas as pd

from .label_group_handler import add_label_group_information


def compute_effective_sample_weights(
    df: pd.DataFrame,
    compound_col: str = "CompoundName",
    enable_compound_balance: bool = True,
    enable_risk_weight: bool = False,
    max_compound_weight: float = 3.0,
    max_final_weight: float = 5.0,
) -> pd.DataFrame:
    """Add distinct label, loss, and metric weights without mixing their roles.

    ``_w_label_raw`` and ``_w_metric`` preserve one unit of support per lab
    label group. ``_w_loss`` additionally balances compounds and is normalized
    for stable estimator loss scaling.
    """
    if compound_col not in df.columns:
        raise ValueError(f"Missing required compound column: {compound_col}")
    if max_compound_weight < 1.0 or max_final_weight <= 0:
        raise ValueError("Weight caps must be positive, with max_compound_weight >= 1.")

    work = df.copy()
    required_label_columns = {"_label_group_id", "_label_group_size"}
    if not required_label_columns.issubset(work.columns):
        work = add_label_group_information(work)

    if work["_label_group_id"].isna().any() or work["_label_group_size"].isna().any():
        raise ValueError("Label group information contains missing values.")

    label_size = pd.to_numeric(work["_label_group_size"], errors="coerce")
    if label_size.isna().any() or (label_size <= 0).any():
        raise ValueError("_label_group_size must contain positive numeric values.")

    w_label_raw = 1.0 / label_size
    w_metric = w_label_raw.copy()

    if enable_compound_balance:
        n_eff_compound = work.groupby(compound_col)["_label_group_id"].transform("nunique").clip(lower=1)
        w_compound_balance = 1.0 / np.sqrt(n_eff_compound.astype(float))
        median_weight = float(np.nanmedian(w_compound_balance))
        if not np.isfinite(median_weight) or median_weight <= 0:
            raise ValueError("Unable to derive finite compound-balance weights.")
        w_compound_balance = (w_compound_balance / median_weight).clip(
            lower=1.0 / max_compound_weight,
            upper=max_compound_weight,
        )
    else:
        w_compound_balance = pd.Series(1.0, index=work.index)

    if enable_risk_weight:
        if "distance_to_limit" not in work.columns:
            raise ValueError("distance_to_limit is required when risk weighting is enabled.")
        distance = pd.to_numeric(work["distance_to_limit"], errors="coerce")
        w_risk = pd.Series(np.where(distance.abs() <= 2.0, 1.5, 1.0), index=work.index)
    else:
        w_risk = pd.Series(1.0, index=work.index)

    w_model_raw = (w_label_raw * w_compound_balance * w_risk).clip(
        lower=0.05,
        upper=max_final_weight,
    )
    weight_sum = float(w_model_raw.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Invalid model sample weights.")

    work["_w_label_raw"] = w_label_raw
    work["_w_compound_balance"] = w_compound_balance
    work["_w_risk"] = w_risk
    work["_w_model_raw"] = w_model_raw
    work["_w_loss"] = w_model_raw * len(work) / weight_sum
    work["_w_metric"] = w_metric
    work["_sample_weight"] = w_label_raw
    return work
