"""Leakage-safe static adjustment and historical low-Mooney envelope."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


@dataclass(frozen=True)
class EnvelopeResult:
    adjusted_groups: pd.DataFrame
    template_audit: pd.DataFrame
    envelope_members: pd.DataFrame


def _pipeline(categorical: list[str], numeric: list[str]) -> Pipeline:
    transform = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical),
        ("num", Pipeline([( "impute", SimpleImputer(strategy="median"))]), numeric),
    ])
    model = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.05, max_iter=250,
        max_leaf_nodes=15, l2_regularization=10.0, random_state=17,
    )
    return Pipeline([("transform", transform), ("model", model)])


def crossfit_static_adjustment(
    groups: pd.DataFrame,
    *,
    categorical_features: list[str],
    numeric_features: list[str],
) -> pd.DataFrame:
    required = {"LabelGroupID", "OrderID", "MNY", "split", *categorical_features, *numeric_features}
    missing = sorted(required.difference(groups.columns))
    if missing:
        raise ValueError(f"Static adjustment input missing columns: {missing}")
    frame = groups.copy()
    if frame["LabelGroupID"].astype(str).duplicated().any():
        raise ValueError("DUPLICATE_LAB_GROUP")
    frame["MNY"] = pd.to_numeric(frame["MNY"], errors="raise")
    train_mask = frame["split"].eq("train")
    train = frame.loc[train_mask]
    unique_orders = train["OrderID"].astype(str).nunique()
    folds = min(5, unique_orders)
    if folds < 2:
        raise ValueError("INSUFFICIENT_TRAIN_ORDERS_FOR_CROSSFIT")
    features = [*categorical_features, *numeric_features]
    prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    splitter = GroupKFold(n_splits=folds)
    for fit_idx, hold_idx in splitter.split(train, groups=train["OrderID"].astype(str)):
        fit_rows, hold_rows = train.iloc[fit_idx], train.iloc[hold_idx]
        model = _pipeline(categorical_features, numeric_features)
        model.fit(fit_rows[features], fit_rows["MNY"])
        prediction.loc[hold_rows.index] = model.predict(hold_rows[features])
    final_model = _pipeline(categorical_features, numeric_features)
    final_model.fit(train[features], train["MNY"])
    prediction.loc[~train_mask] = final_model.predict(frame.loc[~train_mask, features])
    if prediction.isna().any():
        raise ValueError("STATIC_CROSSFIT_PREDICTION_MISSING")
    frame["StaticExpectedMNY"] = prediction
    frame["AdjustedMNY"] = frame["MNY"] - frame["StaticExpectedMNY"]
    frame["StaticContractHash"] = hashlib.sha256("|".join(features).encode("utf-8")).hexdigest()
    return frame


def _bootstrap_interval(low: np.ndarray, rest: np.ndarray, seed: int = 17) -> tuple[float, float]:
    if len(low) < 3 or len(rest) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    effects = []
    for _ in range(500):
        a = rng.choice(low, len(low), replace=True)
        b = rng.choice(rest, len(rest), replace=True)
        effects.append(float(np.median(a) - np.median(b)))
    return float(np.quantile(effects, 0.05)), float(np.quantile(effects, 0.95))


def build_historical_envelopes(
    adjusted: pd.DataFrame,
    *,
    template_column: str = "TemplateKey",
    minimum_template_groups: int = 20,
) -> EnvelopeResult:
    required = {template_column, "MNY", "AdjustedMNY", "lab_time", "split", "LabelGroupID"}
    missing = sorted(required.difference(adjusted.columns))
    if missing:
        raise ValueError(f"Envelope input missing columns: {missing}")
    train = adjusted[adjusted["split"].eq("train")].copy()
    train["lab_time"] = pd.to_datetime(train["lab_time"], errors="coerce", utc=True)
    audit_rows: list[dict[str, object]] = []
    member_rows: list[pd.DataFrame] = []
    for template, part in train.groupby(template_column, dropna=False, sort=False):
        part = part.sort_values("lab_time", kind="stable").copy()
        n = len(part)
        if n < minimum_template_groups:
            audit_rows.append({template_column: template, "groups": n, "envelope_status": "INSUFFICIENT_TEMPLATE_HISTORY"})
            continue
        raw_cut = float(part["MNY"].quantile(0.20))
        adjusted_cut = float(part["AdjustedMNY"].quantile(0.20))
        part["raw_low"] = part["MNY"].le(raw_cut)
        part["adjusted_low"] = part["AdjustedMNY"].le(adjusted_cut)
        part["envelope_member"] = part["raw_low"] & part["adjusted_low"]
        low, rest = part[part["raw_low"]], part[~part["raw_low"]]
        raw_effect = float(low["MNY"].median() - rest["MNY"].median()) if len(rest) else np.nan
        adjusted_effect = float(low["AdjustedMNY"].median() - rest["AdjustedMNY"].median()) if len(rest) else np.nan
        attenuation = 1.0 - abs(adjusted_effect) / abs(raw_effect) if raw_effect and np.isfinite(raw_effect) else np.nan
        # Baseline expected MNY is a leakage-safe joint material/environment score.
        try:
            part["material_environment_stratum"] = pd.qcut(
                part["StaticExpectedMNY"].rank(method="first"), 3, labels=False, duplicates="drop"
            )
        except ValueError:
            part["material_environment_stratum"] = 0
        directions = []
        for _, stratum in part.groupby("material_environment_stratum", dropna=False):
            a, b = stratum[stratum["raw_low"]]["AdjustedMNY"], stratum[~stratum["raw_low"]]["AdjustedMNY"]
            if len(a) >= 2 and len(b) >= 2:
                directions.append(float(a.median() - b.median()) < 0)
        stratum_consistency = float(np.mean(directions)) if directions else 0.0
        part["rolling_window"] = np.asarray(
            pd.qcut(part.reset_index().index, 3, labels=False, duplicates="drop")
        )
        window_directions = []
        for _, window in part.groupby("rolling_window"):
            a, b = window[window["raw_low"]]["AdjustedMNY"], window[~window["raw_low"]]["AdjustedMNY"]
            if len(a) >= 2 and len(b) >= 2:
                window_directions.append(float(a.median() - b.median()) < 0)
        window_passes = int(sum(window_directions))
        ci_low, ci_high = _bootstrap_interval(low["AdjustedMNY"].to_numpy(), rest["AdjustedMNY"].to_numpy())
        low_with_stratum = part[part["raw_low"]]
        dominant_expected_bin = (
            float(low_with_stratum["material_environment_stratum"].value_counts(normalize=True).max())
            if len(low_with_stratum) else 1.0
        )
        confounded = bool(
            (np.isfinite(attenuation) and attenuation > 0.50)
            or dominant_expected_bin >= 0.80
            or (ci_high >= 0 and stratum_consistency < 0.70)
        )
        supported = bool(
            not confounded
            and part["envelope_member"].sum() >= 3
            and stratum_consistency >= 0.70
            and window_passes >= 2
        )
        status = "HISTORICAL_ENVELOPE_SUPPORTED" if supported else (
            "CONFOUNDED_BY_MATERIAL" if confounded else "HISTORICAL_ENVELOPE_SHADOW_ONLY"
        )
        audit_rows.append({
            template_column: template,
            "groups": n,
            "raw_low_groups": int(part["raw_low"].sum()),
            "envelope_groups": int(part["envelope_member"].sum()),
            "raw_p20": raw_cut,
            "adjusted_p20": adjusted_cut,
            "raw_low_effect_mu": raw_effect,
            "adjusted_low_effect_mu": adjusted_effect,
            "advantage_attenuation": attenuation,
            "stratum_direction_consistency": stratum_consistency,
            "rolling_windows_with_low_direction": window_passes,
            "adjusted_effect_bootstrap_p05": ci_low,
            "adjusted_effect_bootstrap_p95": ci_high,
            "dominant_material_environment_stratum_share": dominant_expected_bin,
            "confounding_status": "CONFOUNDED_BY_MATERIAL" if confounded else "NO_DOMINANT_MEASURED_CONFOUNDING",
            "envelope_status": status,
        })
        selected = part[part["envelope_member"]].copy()
        selected["envelope_status"] = status
        member_rows.append(selected)
    members = pd.concat(member_rows, ignore_index=True) if member_rows else pd.DataFrame()
    return EnvelopeResult(adjusted, pd.DataFrame(audit_rows), members)
