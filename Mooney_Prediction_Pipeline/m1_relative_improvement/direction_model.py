"""B2 pairwise direction and quantile magnitude heads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, recall_score


@dataclass
class DirectionMagnitudeResult:
    transitions: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    release_gate_passed: bool
    feature_columns: tuple[str, ...]


def build_pairwise_transitions(
    groups: pd.DataFrame,
    *,
    feature_columns: list[str],
    maximum_gap_days: float = 30.0,
) -> pd.DataFrame:
    identity = ["OrderID", "RecipeID", "ProcessVersion", "MixerLine", "TemplateKey"]
    required = {"LabelGroupID", "MNY", "lab_time", "split", *identity, *feature_columns}
    missing = sorted(required.difference(groups.columns))
    if missing:
        raise ValueError(f"B2 group frame missing columns: {missing}")
    rows: list[dict[str, object]] = []
    source = groups.copy()
    source["lab_time"] = pd.to_datetime(source["lab_time"], errors="raise", utc=True)
    for _, part in source.sort_values("lab_time").groupby(identity, dropna=False, sort=False):
        records = list(part.drop_duplicates("LabelGroupID").sort_values("lab_time").to_dict("records"))
        for previous, current in zip(records, records[1:]):
            gap = (current["lab_time"] - previous["lab_time"]).total_seconds() / 86400.0
            if gap <= 0 or gap > maximum_gap_days or current["split"] != previous["split"]:
                continue
            delta = float(current["MNY"] - previous["MNY"])
            row: dict[str, object] = {
                "from_group": previous["LabelGroupID"],
                "to_group": current["LabelGroupID"],
                "OrderID": current["OrderID"],
                "RecipeID": current["RecipeID"],
                "CompoundName": current.get("CompoundName", ""),
                "ProcessVersion": current["ProcessVersion"],
                "TemplateKey": current["TemplateKey"],
                "split": current["split"],
                "to_lab_time": current["lab_time"],
                "previous_mny": float(previous["MNY"]),
                "actual_mny": float(current["MNY"]),
                "actual_delta_mny": delta,
                "actual_reduction": -delta,
                "is_reduction": int(delta < 0),
            }
            for feature in feature_columns:
                a = pd.to_numeric(pd.Series([previous[feature]]), errors="coerce").iloc[0]
                b = pd.to_numeric(pd.Series([current[feature]]), errors="coerce").iloc[0]
                row[f"delta__{feature}"] = b - a if pd.notna(a) and pd.notna(b) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _classifier() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=160, learning_rate=0.025,
        num_leaves=7, min_child_samples=25, reg_alpha=1.0, reg_lambda=10.0,
        subsample=1.0, colsample_bytree=0.75, random_state=17,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )


def _quantile(alpha: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile", alpha=alpha, n_estimators=180, learning_rate=0.025,
        num_leaves=7, min_child_samples=25, reg_alpha=1.0, reg_lambda=10.0,
        subsample=1.0, colsample_bytree=0.75, random_state=17,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )


def _metric_row(part: pd.DataFrame, split: str, scope: str) -> dict[str, object]:
    significant = part["actual_delta_mny"].abs().ge(2.0)
    direction = np.sign(part.loc[significant, "predicted_delta_mny"]) == np.sign(part.loc[significant, "actual_delta_mny"])
    actual_reduction = part["actual_reduction"].to_numpy(float)
    predicted_reduction = part["reduction_p50"].to_numpy(float)
    actual_std = float(np.std(actual_reduction, ddof=0))
    predicted_std = float(np.std(predicted_reduction, ddof=0))
    actual_positive = part["is_reduction"].to_numpy(int)
    predicted_positive = part["reduction_probability"].ge(0.5).astype(int).to_numpy()
    return {
        "split": split,
        "scope": scope,
        "pairs": int(len(part)),
        "significant_pairs": int(significant.sum()),
        "direction_accuracy_abs_actual_ge_2": float(direction.mean()) if len(direction) else math.nan,
        "reduction_recall": float(recall_score(actual_positive, predicted_positive, zero_division=0)),
        "delta_mae": float(mean_absolute_error(part["actual_delta_mny"], part["predicted_delta_mny"])),
        "actual_reduction_std": actual_std,
        "predicted_reduction_std": predicted_std,
        "delta_std_ratio": predicted_std / actual_std if actual_std else math.nan,
        "p10_p90_coverage": float(
            ((part["actual_reduction"] >= part["reduction_p10"]) & (part["actual_reduction"] <= part["reduction_p90"])).mean()
        ),
    }


def fit_direction_magnitude(transitions: pd.DataFrame) -> DirectionMagnitudeResult:
    if transitions.empty:
        raise ValueError("NO_B2_TRANSITIONS")
    features = tuple(column for column in transitions.columns if column.startswith("delta__"))
    train = transitions[transitions["split"].eq("train")].copy()
    if len(train) < 50 or train["is_reduction"].nunique() < 2:
        raise ValueError("INSUFFICIENT_B2_TRAIN_TRANSITIONS")
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_train = imputer.fit_transform(train[list(features)])
    direction = _classifier().fit(x_train, train["is_reduction"])
    quantiles = {alpha: _quantile(alpha).fit(x_train, train["actual_reduction"]) for alpha in (0.1, 0.5, 0.9)}
    output: list[pd.DataFrame] = []
    for split in ("train", "val", "oot"):
        part = transitions[transitions["split"].eq(split)].copy()
        if part.empty:
            continue
        x = imputer.transform(part[list(features)])
        part["reduction_probability"] = direction.predict_proba(x)[:, 1]
        raw = {alpha: quantiles[alpha].predict(x) for alpha in quantiles}
        ordered = np.sort(np.column_stack([raw[0.1], raw[0.5], raw[0.9]]), axis=1)
        part["reduction_p10"], part["reduction_p50"], part["reduction_p90"] = ordered.T
        part["predicted_delta_mny"] = -part["reduction_p50"]
        part["predicted_mny"] = part["previous_mny"] + part["predicted_delta_mny"]
        output.append(part)
    predictions = pd.concat(output, ignore_index=True)
    metric_rows: list[dict[str, object]] = []
    for split in predictions["split"].unique():
        part = predictions[predictions["split"].eq(split)]
        metric_rows.append(_metric_row(part, split, "GLOBAL_M1"))
        for marker in ("T15760", "B00458"):
            pilot = part[part["CompoundName"].astype(str).str.contains(marker, regex=False)]
            if len(pilot):
                metric_rows.append(_metric_row(pilot, split, marker))
    metrics = pd.DataFrame(metric_rows)
    required_rows = metrics[(metrics["split"].isin(["val", "oot"])) & metrics["scope"].eq("GLOBAL_M1")]
    release = bool(
        len(required_rows) == 2
        and required_rows["direction_accuracy_abs_actual_ge_2"].ge(0.65).all()
        and required_rows["reduction_recall"].ge(0.70).all()
        and required_rows["delta_std_ratio"].ge(0.70).all()
        and required_rows["p10_p90_coverage"].ge(0.80).all()
    )
    return DirectionMagnitudeResult(transitions, predictions, metrics, release, features)
