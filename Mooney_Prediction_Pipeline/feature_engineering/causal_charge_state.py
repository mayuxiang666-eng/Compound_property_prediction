"""Causal material-Charge state features using only already available Lab results."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def material_family(code: object) -> str:
    text = str(code).strip().upper()
    if text.startswith("CE"):
        return "ELASTOMER"
    if text.startswith(("CS10", "CS12")):
        return "SILICA"
    if text.startswith("CS4"):
        return "PROCESS_OIL"
    if text.startswith("CR"):
        return "RESIN"
    return "OTHER"


def build_causal_charge_features(
    groups: pd.DataFrame,
    charges: pd.DataFrame,
    *,
    shrinkage: float = 10.0,
) -> pd.DataFrame:
    """Create deployment-time Charge priors without current/future Lab leakage."""

    required_groups = {"LabelGroupID", "production_start", "lab_time", "MNY"}
    required_charges = {"LabelGroupID", "MaterialFamily", "ChargeID", "ActualWeight"}
    if missing := required_groups.difference(groups.columns):
        raise ValueError(f"Missing group columns: {sorted(missing)}")
    if missing := required_charges.difference(charges.columns):
        raise ValueError(f"Missing charge columns: {sorted(missing)}")
    group = groups[list(required_groups)].copy()
    group["production_start"] = pd.to_datetime(group["production_start"], errors="coerce")
    group["lab_time"] = pd.to_datetime(group["lab_time"], errors="coerce")
    group["MNY"] = pd.to_numeric(group["MNY"], errors="coerce")
    charge = charges.copy()
    charge["ActualWeight"] = pd.to_numeric(charge["ActualWeight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    charge["ChargeID"] = charge["ChargeID"].astype(str).str.strip()
    charge = charge.loc[charge["ChargeID"].ne("") & charge["ChargeID"].ne("nan")]
    by_group = {key: part for key, part in charge.groupby("LabelGroupID", sort=False)}

    events = group.dropna(subset=["lab_time", "MNY"]).sort_values("lab_time", kind="stable").to_dict("records")
    queries = group.sort_values(["production_start", "LabelGroupID"], kind="stable")
    charge_sum, charge_count = defaultdict(float), defaultdict(float)
    global_sum = 0.0
    global_count = 0.0
    event_index = 0
    rows = []
    families = sorted(set(charge["MaterialFamily"].dropna().astype(str)))
    for query in queries.itertuples(index=False):
        query_time = query.production_start
        while event_index < len(events) and pd.notna(query_time) and events[event_index]["lab_time"] < query_time:
            event = events[event_index]
            event_id = event["LabelGroupID"]
            value = float(event["MNY"])
            if event_id != query.LabelGroupID:
                global_sum += value
                global_count += 1.0
                for item in by_group.get(event_id, pd.DataFrame()).itertuples(index=False):
                    key = (str(item.MaterialFamily), str(item.ChargeID))
                    charge_sum[key] += value
                    charge_count[key] += 1.0
            event_index += 1
        global_mean = global_sum / global_count if global_count else np.nan
        result: dict[str, object] = {
            "LabelGroupID": query.LabelGroupID,
            "charge_state__available_lab_count": global_count,
        }
        current = by_group.get(query.LabelGroupID, pd.DataFrame())
        for family in families:
            family_rows = current.loc[current["MaterialFamily"].astype(str).eq(family)] if len(current) else current
            if not len(family_rows):
                result[f"charge_state__{family}__known_weight_ratio"] = 0.0
                result[f"charge_state__{family}__prior_count"] = 0.0
                result[f"charge_state__{family}__prior_delta"] = 0.0
                result[f"charge_state__{family}__unique_charge_count"] = 0.0
                continue
            weight = family_rows["ActualWeight"].to_numpy(dtype=float)
            if float(weight.sum()) <= 0:
                weight = np.ones(len(family_rows), dtype=float)
            weight = weight / weight.sum()
            known_weight, weighted_count, weighted_delta = 0.0, 0.0, 0.0
            for item, item_weight in zip(family_rows.itertuples(index=False), weight):
                key = (family, str(item.ChargeID))
                count = charge_count[key]
                if count > 0 and pd.notna(global_mean):
                    raw_mean = charge_sum[key] / count
                    shrunk = (count * raw_mean + shrinkage * global_mean) / (count + shrinkage)
                    known_weight += item_weight
                    weighted_count += item_weight * count
                    weighted_delta += item_weight * (shrunk - global_mean)
            result[f"charge_state__{family}__known_weight_ratio"] = known_weight
            result[f"charge_state__{family}__prior_count"] = weighted_count
            result[f"charge_state__{family}__prior_delta"] = weighted_delta
            result[f"charge_state__{family}__unique_charge_count"] = float(family_rows["ChargeID"].nunique())
        rows.append(result)
    output = pd.DataFrame(rows)
    return groups[["LabelGroupID"]].merge(output, on="LabelGroupID", how="left", validate="one_to_one")


__all__ = ["build_causal_charge_features", "material_family"]
