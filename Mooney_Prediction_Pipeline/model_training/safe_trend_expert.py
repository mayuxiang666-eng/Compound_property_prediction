"""Prequential residual-trend expert using only already-returned Lab groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ANCHOR_SCOPE_ORDER = {
    "SAME_ORDER_RECIPE_VERSION_MIXER": 0,
    "EXACT_RECIPE_VERSION_MIXER": 1,
    "COMPOUND_MIXER_ROUTE": 2,
}


def _normalized_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def attach_safe_prior_lab_anchors(
    groups: pd.DataFrame,
    *,
    production_time_col: str,
    lab_time_col: str = "lab_time",
    split_col: str = "split",
) -> pd.DataFrame:
    """Attach the newest Lab known at production completion, never a future Lab.

    Validation and OOT are replayed prequentially: a prior validation/OOT label
    may anchor a later event only after its Lab timestamp has passed.
    """

    required = {
        "LabelGroupID", "RecipeID", "ProcessVersion", "MixerLine", "CompoundName",
        "material_system", "phase_route", "MNY", production_time_col, lab_time_col, split_col,
    }
    missing = sorted(required.difference(groups.columns))
    if missing:
        raise ValueError(f"Safe trend anchor input missing columns: {missing}")
    frame = groups.copy().reset_index(drop=True)
    if frame["LabelGroupID"].astype(str).duplicated().any():
        raise ValueError("Safe trend anchors require one row per Lab group")
    frame[production_time_col] = _normalized_time(frame[production_time_col])
    frame[lab_time_col] = _normalized_time(frame[lab_time_col])
    if frame[[production_time_col, lab_time_col]].isna().any().any():
        raise ValueError("Safe trend anchor timestamps contain invalid values")
    frame["_row_id"] = np.arange(len(frame), dtype=int)
    frame["anchor_row_id"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["anchor_scope"] = "NONE"

    same_order_cols = ["OrderID", "RecipeID", "ProcessVersion", "MixerLine"]
    exact_cols = ["RecipeID", "ProcessVersion", "MixerLine"]
    pooled_cols = ["CompoundName", "MixerLine", "material_system", "phase_route"]
    split_rank = {"train": 0, "val": 1, "oot": 2}

    for current_index, current in frame.sort_values(production_time_col, kind="stable").iterrows():
        current_split_rank = split_rank.get(str(current[split_col]), -1)
        prior = frame.loc[
            frame[production_time_col].lt(current[production_time_col])
            & frame[lab_time_col].le(current[production_time_col])
            & frame[split_col].map(split_rank).fillna(99).le(current_split_rank)
        ]
        if prior.empty:
            continue
        same_order = prior
        for column in same_order_cols:
            same_order = same_order.loc[
                same_order[column].astype(str).eq(str(current[column]))
            ]
        if not same_order.empty:
            candidates = same_order
            scope = "SAME_ORDER_RECIPE_VERSION_MIXER"
        else:
            exact = prior
            for column in exact_cols:
                exact = exact.loc[exact[column].astype(str).eq(str(current[column]))]
            if not exact.empty:
                candidates = exact
                scope = "EXACT_RECIPE_VERSION_MIXER"
            else:
                pooled = prior
                for column in pooled_cols:
                    pooled = pooled.loc[pooled[column].astype(str).eq(str(current[column]))]
                if pooled.empty:
                    continue
                candidates = pooled
                scope = "COMPOUND_MIXER_ROUTE"
        anchor_index = candidates[lab_time_col].idxmax()
        frame.at[current_index, "anchor_row_id"] = int(frame.at[anchor_index, "_row_id"])
        frame.at[current_index, "anchor_scope"] = scope

    anchor_source = frame.set_index("_row_id")
    anchored = frame["anchor_row_id"].notna()
    for column in frame.columns:
        if column.startswith("anchor__") or column in {"anchor_row_id", "anchor_scope"}:
            continue
        frame[f"anchor__{column}"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    for current_index in frame.index[anchored]:
        anchor_id = int(frame.at[current_index, "anchor_row_id"])
        source = anchor_source.loc[anchor_id]
        for column in groups.columns:
            frame.at[current_index, f"anchor__{column}"] = source[column]
    frame["anchor_lab_age_hours"] = (
        frame[production_time_col] - _normalized_time(frame[f"anchor__{lab_time_col}"])
    ).dt.total_seconds() / 3600.0
    frame["anchor_production_gap_hours"] = (
        frame[production_time_col] - _normalized_time(frame[f"anchor__{production_time_col}"])
    ).dt.total_seconds() / 3600.0
    frame["anchor_is_same_template"] = (
        frame.get("TemplateKey", pd.Series("", index=frame.index)).astype(str)
        == frame.get("anchor__TemplateKey", pd.Series("", index=frame.index)).astype(str)
    ).astype(float)
    frame["anchor_available"] = anchored.astype(int)
    invalid = frame.loc[anchored, f"anchor__{lab_time_col}"].pipe(_normalized_time).gt(
        frame.loc[anchored, production_time_col]
    )
    if invalid.any():
        raise AssertionError("Future Lab leakage detected in safe trend anchors")
    return frame.drop(columns=["_row_id"])


@dataclass
class SafeTrendResidualExpert:
    alpha: float = 10.0
    top_k_process_deltas: int = 30
    process_features: tuple[str, ...] = ()
    selected_process_features_: tuple[str, ...] = ()
    feature_columns_: tuple[str, ...] = ()
    pipeline_: Pipeline | None = None

    def _base_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        result["anchor_residual"] = (
            pd.to_numeric(frame["anchor__MNY"], errors="coerce")
            - pd.to_numeric(frame["anchor__RawPrediction"], errors="coerce")
        )
        result["raw_prediction_delta"] = (
            pd.to_numeric(frame["RawPrediction"], errors="coerce")
            - pd.to_numeric(frame["anchor__RawPrediction"], errors="coerce")
        )
        result["anchor_lab_age_hours"] = pd.to_numeric(frame["anchor_lab_age_hours"], errors="coerce")
        result["anchor_production_gap_hours"] = pd.to_numeric(frame["anchor_production_gap_hours"], errors="coerce")
        result["anchor_is_same_template"] = pd.to_numeric(frame["anchor_is_same_template"], errors="coerce")
        result["anchor_scope_same_order"] = frame["anchor_scope"].eq(
            "SAME_ORDER_RECIPE_VERSION_MIXER"
        ).astype(float)
        result["anchor_scope_exact"] = frame["anchor_scope"].isin([
            "SAME_ORDER_RECIPE_VERSION_MIXER", "EXACT_RECIPE_VERSION_MIXER"
        ]).astype(float)
        for feature in self.process_features:
            anchor_column = f"anchor__{feature}"
            if feature in frame and anchor_column in frame:
                result[f"delta__{feature}"] = (
                    pd.to_numeric(frame[feature], errors="coerce")
                    - pd.to_numeric(frame[anchor_column], errors="coerce")
                )
        return result

    def fit(self, frame: pd.DataFrame) -> "SafeTrendResidualExpert":
        train = frame.loc[frame["anchor_available"].eq(1)].copy()
        if len(train) < 50:
            raise ValueError("Safe trend expert requires at least 50 anchored training groups")
        features = self._base_features(train)
        target = (
            pd.to_numeric(train["MNY"], errors="coerce")
            - pd.to_numeric(train["RawPrediction"], errors="coerce")
        )
        delta_columns = [column for column in features if column.startswith("delta__")]
        scores = {}
        for column in delta_columns:
            valid = features[column].notna() & target.notna()
            corr = features.loc[valid, column].corr(target.loc[valid], method="spearman") if valid.sum() >= 10 else np.nan
            scores[column] = abs(float(corr)) if pd.notna(corr) else 0.0
        selected_delta = sorted(delta_columns, key=lambda column: (-scores[column], column))[
            : self.top_k_process_deltas
        ]
        base = [column for column in features if not column.startswith("delta__")]
        self.selected_process_features_ = tuple(column.removeprefix("delta__") for column in selected_delta)
        self.feature_columns_ = tuple(base + selected_delta)
        self.pipeline_ = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        self.pipeline_.fit(features.loc[:, self.feature_columns_], target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("Safe trend expert is not fitted")
        raw = pd.to_numeric(frame["RawPrediction"], errors="coerce").to_numpy(dtype=float)
        output = raw.copy()
        available = frame["anchor_available"].eq(1).to_numpy()
        if available.any():
            features = self._base_features(frame.loc[available])
            output[available] = raw[available] + self.pipeline_.predict(
                features.loc[:, self.feature_columns_]
            )
        return output


__all__ = ["attach_safe_prior_lab_anchors", "SafeTrendResidualExpert"]
