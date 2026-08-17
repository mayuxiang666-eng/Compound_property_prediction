"""Gate-0 natural-action support audit for M1 Step controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = (
    "compound",
    "recipe_id",
    "process_version",
    "mixer_line",
    "phase_route",
    "structural_template_key",
    "label_group_id",
    "order_id",
    "event_time",
)


@dataclass(frozen=True)
class ActionSupportThresholds:
    minimum_independent_transitions: int = 30
    minimum_levels: int = 3
    minimum_groups_per_major_level: int = 8
    minimum_effective_sample: int = 25
    maximum_transition_gap_days: float = 30.0


def _first_consistent(series: pd.Series):
    values = series.dropna().astype(str).unique()
    return values[0] if len(values) == 1 else pd.NA


def _rounded_level(kind: str, value: float) -> float:
    quantum = {"duration": 1.0, "rotor": 0.5, "temperature": 0.5}.get(kind, 1.0)
    return float(np.round(float(value) / quantum) * quantum)


def build_group_action_frame(step_long: pd.DataFrame) -> pd.DataFrame:
    required = {*IDENTITY_COLUMNS, "group_name", "step_no", "kind", "target", "actual", "phase"}
    missing = sorted(required.difference(step_long.columns))
    if missing:
        raise ValueError(f"Gate-0 Step input missing columns: {missing}")
    data = step_long.copy()
    data = data[data["kind"].isin(["duration", "rotor", "temperature"])].copy()
    data["target"] = pd.to_numeric(data["target"], errors="coerce")
    data["actual"] = pd.to_numeric(data["actual"], errors="coerce")
    data["step_no"] = pd.to_numeric(data["step_no"], errors="coerce")
    data["event_time"] = pd.to_datetime(data["event_time"], errors="coerce", utc=True)
    data = data[data["target"].notna() & data["event_time"].notna()].copy()
    data["lever"] = (
        data["group_name"].astype(str)
        + "|S"
        + data["step_no"].astype("Int64").astype(str)
        + "|"
        + data["kind"].astype(str)
    )
    keys = [*IDENTITY_COLUMNS, "group_name", "step_no", "kind", "phase", "lever"]
    aggregations = {"target": "median", "actual": "median"}
    for optional in ("material_stratum", "environment_stratum", "process_version"):
        if optional in data.columns and optional not in keys:
            aggregations[optional] = _first_consistent
    grouped = data.groupby(keys, as_index=False, dropna=False).agg(aggregations)
    grouped["target_level"] = [
        _rounded_level(kind, value) for kind, value in zip(grouped["kind"], grouped["target"])
    ]
    grouped["action_allowed"] = (
        grouped["step_no"].gt(2)
        & grouped["target"].gt(0)
        & ~grouped["kind"].eq("ram")
    )
    return grouped


def _transitions(part: pd.DataFrame, max_gap_days: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordering = part.sort_values(["event_time", "order_id", "label_group_id"], kind="stable")
    # Prefer transitions inside an order; additionally include one boundary
    # transition between adjacent orders in the same structural template.
    candidates: list[tuple[pd.Series, pd.Series, str]] = []
    for _, order in ordering.groupby("order_id", sort=False):
        records = [row for _, row in order.drop_duplicates("label_group_id").iterrows()]
        candidates.extend((a, b, "SAME_ORDER") for a, b in zip(records, records[1:]))
    order_last = ordering.drop_duplicates("label_group_id").groupby("order_id", sort=False).tail(1)
    order_first = ordering.drop_duplicates("label_group_id").groupby("order_id", sort=False).head(1)
    ordered_orders = (
        ordering.groupby("order_id", as_index=False)["event_time"].min().sort_values("event_time")["order_id"].tolist()
    )
    for prior_order, next_order in zip(ordered_orders, ordered_orders[1:]):
        a = order_last[order_last["order_id"].eq(prior_order)].iloc[-1]
        b = order_first[order_first["order_id"].eq(next_order)].iloc[0]
        candidates.append((a, b, "ADJACENT_ORDER_WINDOW"))
    seen: set[tuple[str, str]] = set()
    for previous, current, transition_type in candidates:
        pair = (str(previous["label_group_id"]), str(current["label_group_id"]))
        if pair in seen:
            continue
        seen.add(pair)
        gap = (current["event_time"] - previous["event_time"]).total_seconds() / 86400.0
        if gap <= 0 or gap > max_gap_days:
            continue
        delta = float(current["target_level"] - previous["target_level"])
        if abs(delta) < 1e-12:
            continue
        rows.append({
            "from_group": pair[0],
            "to_group": pair[1],
            "transition_type": transition_type,
            "gap_days": gap,
            "target_delta": delta,
            "actual_delta": (
                float(current["actual"] - previous["actual"])
                if pd.notna(current["actual"]) and pd.notna(previous["actual"])
                else np.nan
            ),
            "from_level": float(previous["target_level"]),
            "to_level": float(current["target_level"]),
        })
    return pd.DataFrame(rows)


def audit_action_support(
    group_actions: pd.DataFrame,
    thresholds: ActionSupportThresholds = ActionSupportThresholds(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    transitions: list[pd.DataFrame] = []
    keys = ["compound", "recipe_id", "process_version", "mixer_line", "structural_template_key", "lever"]
    for key, part in group_actions.groupby(keys, dropna=False, sort=False):
        identity = dict(zip(keys, key))
        kind = str(part["kind"].iloc[0])
        action_allowed = bool(part["action_allowed"].all())
        level_counts = part.groupby("target_level")["label_group_id"].nunique().sort_values(ascending=False)
        major = level_counts[level_counts >= thresholds.minimum_groups_per_major_level]
        trans = _transitions(part, thresholds.maximum_transition_gap_days)
        if not trans.empty:
            for column, value in identity.items():
                trans[column] = value
            transitions.append(trans)
        independent = int(trans["to_group"].nunique()) if not trans.empty else 0
        effective = float(independent)
        represented_months = part.assign(month=part["event_time"].dt.strftime("%Y-%m")).groupby("target_level")["month"].nunique()
        major_month_support = bool(len(major) and represented_months.reindex(major.index).fillna(0).ge(2).all())
        bidirectional = bool(not trans.empty and trans["target_delta"].gt(0).any() and trans["target_delta"].lt(0).any())
        actual_response_fraction = float(trans["actual_delta"].notna().mean()) if not trans.empty else 0.0
        if not action_allowed:
            status = "LOCKED_EXPERT_RULE"
        elif part["label_group_id"].nunique() < thresholds.minimum_groups_per_major_level:
            status = "INSUFFICIENT_LAB_SUPPORT"
        elif len(level_counts) <= 1:
            status = "FIXED_IN_HISTORY"
        elif len(major) < thresholds.minimum_levels:
            status = "ASSOCIATION_ONLY"
        elif not major_month_support:
            status = "CONFOUNDED_ACTION_SUPPORT"
        elif independent < thresholds.minimum_independent_transitions or effective < thresholds.minimum_effective_sample:
            status = "ASSOCIATION_ONLY"
        else:
            status = "CAUSAL_ACTION_SUPPORTED"
        rows.append({
            **identity,
            "kind": kind,
            "phase": _first_consistent(part["phase"]),
            "action_allowed": action_allowed,
            "lab_groups": int(part["label_group_id"].nunique()),
            "orders": int(part["order_id"].nunique()),
            "control_levels": int(len(level_counts)),
            "major_control_levels": int(len(major)),
            "minimum_groups_in_major_level": int(major.min()) if len(major) else 0,
            "independent_transitions": independent,
            "effective_sample": effective,
            "bidirectional_history": bidirectional,
            "actual_response_observed_fraction": actual_response_fraction,
            "major_levels_span_multiple_months": major_month_support,
            "causal_support_status": status,
            "thresholds_json": str(asdict(thresholds)),
        })
    transition_frame = pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame()
    return pd.DataFrame(rows), transition_frame
