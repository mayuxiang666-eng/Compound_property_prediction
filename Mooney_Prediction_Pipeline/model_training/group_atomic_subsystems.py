"""Group-atomic temporal OOF variants of V3.7 material subexperts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from .cb_subsystem import CarbonBlackSubsystemPredictor
from .silica_subsystem import SilicaSubsystemPredictor
from .temporal_validation import expanding_window_folds


def expanding_group_row_folds(
    group_ids: pd.Series,
    n_splits: int = 5,
    min_train_fraction: float = 0.20,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, int]]]:
    """Create expanding folds that never split a Lab group across train/valid."""

    ids = pd.Series(group_ids).astype(str).reset_index(drop=True)
    group_order = list(dict.fromkeys(ids.tolist()))
    group_folds = expanding_window_folds(
        len(group_order), n_splits=n_splits, min_train_fraction=min_train_fraction
    )
    row_folds = []
    audit = []
    values = ids.to_numpy()
    for fold_no, (train_groups, valid_groups) in enumerate(group_folds, start=1):
        train_names = {group_order[index] for index in train_groups}
        valid_names = {group_order[index] for index in valid_groups}
        if train_names.intersection(valid_names):
            raise RuntimeError("Lab group leakage inside temporal subsystem fold")
        train_rows = np.flatnonzero(np.isin(values, list(train_names)))
        valid_rows = np.flatnonzero(np.isin(values, list(valid_names)))
        row_folds.append((train_rows, valid_rows))
        audit.append(
            {
                "fold": fold_no,
                "train_groups": int(len(train_names)),
                "valid_groups": int(len(valid_names)),
                "train_rows": int(len(train_rows)),
                "valid_rows": int(len(valid_rows)),
            }
        )
    return row_folds, audit


class _GroupAtomicMixin:
    objective: str
    min_train_fraction: float

    def _new_expert(self, n_estimators: int, reg_lambda: float) -> LGBMRegressor:
        return LGBMRegressor(
            objective=self.objective,
            n_estimators=n_estimators,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            reg_lambda=reg_lambda,
            random_state=42,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
        )


class GroupAtomicSilicaSubsystem(_GroupAtomicMixin, SilicaSubsystemPredictor):
    """Silica 4-subexpert blender with Lab-group-atomic temporal OOF."""

    def __init__(
        self,
        n_splits: int = 5,
        shrinkage_alpha: float = 1.0,
        objective: str = "regression",
        min_train_fraction: float = 0.20,
    ):
        super().__init__(n_splits, shrinkage_alpha, True)
        self.objective = objective
        self.min_train_fraction = float(min_train_fraction)
        self.fold_audit_: list[dict[str, int]] = []

    def fit(self, X_delta, y_residual, sample_weights=None, df_meta=None):
        if df_meta is None or "_label_group_id" not in df_meta.columns:
            raise ValueError("Silica group-atomic subsystem requires _label_group_id")
        n_samples = len(X_delta)
        weights = np.ones(n_samples) if sample_weights is None else np.asarray(sample_weights, dtype=float)
        target = np.asarray(y_residual, dtype=float)
        self._select_sub_features(X_delta, df_meta)
        self.feature_names_pid_ = sorted(set(self.feature_names_pid_))
        self.feature_names_wet_ = sorted(set(self.feature_names_wet_))
        self.feature_names_bottom_ = sorted(set(self.feature_names_bottom_))
        self.feature_names_material_ = sorted(set(self.feature_names_material_))
        folds, self.fold_audit_ = expanding_group_row_folds(
            df_meta["_label_group_id"], self.n_splits, self.min_train_fraction
        )
        feature_sets = (
            self.feature_names_pid_, self.feature_names_wet_,
            self.feature_names_bottom_, self.feature_names_material_,
        )
        oof = np.full((n_samples, 4), np.nan, dtype=float)
        for train_rows, valid_rows in folds:
            for column, features in enumerate(feature_sets):
                expert = self._new_expert(100, 3.0).fit(
                    X_delta.iloc[train_rows][features],
                    target[train_rows],
                    sample_weight=weights[train_rows],
                )
                oof[valid_rows, column] = expert.predict(X_delta.iloc[valid_rows][features])
        valid = np.isfinite(oof).all(axis=1)
        self.combiner_ = Ridge(
            alpha=self.shrinkage_alpha, fit_intercept=False, positive=True
        ).fit(oof[valid], target[valid], sample_weight=weights[valid])
        self.pid_expert_ = self._new_expert(120, 3.0).fit(
            X_delta[self.feature_names_pid_], target, sample_weight=weights
        )
        self.wet_expert_ = self._new_expert(120, 3.0).fit(
            X_delta[self.feature_names_wet_], target, sample_weight=weights
        )
        self.bottom_expert_ = self._new_expert(120, 3.0).fit(
            X_delta[self.feature_names_bottom_], target, sample_weight=weights
        )
        self.material_expert_ = self._new_expert(120, 3.0).fit(
            X_delta[self.feature_names_material_], target, sample_weight=weights
        )
        return self


class GroupAtomicCarbonBlackSubsystem(_GroupAtomicMixin, CarbonBlackSubsystemPredictor):
    """Carbon Black 3-subexpert blender with Lab-group-atomic temporal OOF."""

    def __init__(
        self,
        n_splits: int = 5,
        shrinkage_alpha: float = 1.0,
        objective: str = "regression",
        min_train_fraction: float = 0.20,
    ):
        super().__init__(n_splits, shrinkage_alpha)
        self.objective = objective
        self.min_train_fraction = float(min_train_fraction)
        self.fold_audit_: list[dict[str, int]] = []

    def fit(self, X_delta, y_residual, sample_weights=None, df_meta=None):
        if df_meta is None or "_label_group_id" not in df_meta.columns:
            raise ValueError("Carbon Black group-atomic subsystem requires _label_group_id")
        n_samples = len(X_delta)
        weights = np.ones(n_samples) if sample_weights is None else np.asarray(sample_weights, dtype=float)
        target = np.asarray(y_residual, dtype=float)
        self._select_sub_features(X_delta)
        self.feature_names_prep_ = sorted(set(self.feature_names_prep_))
        self.feature_names_bottom_ = sorted(set(self.feature_names_bottom_))
        self.feature_names_material_ = sorted(set(self.feature_names_material_))
        folds, self.fold_audit_ = expanding_group_row_folds(
            df_meta["_label_group_id"], self.n_splits, self.min_train_fraction
        )
        feature_sets = (
            self.feature_names_prep_, self.feature_names_bottom_, self.feature_names_material_
        )
        oof = np.full((n_samples, 3), np.nan, dtype=float)
        for train_rows, valid_rows in folds:
            for column, features in enumerate(feature_sets):
                expert = self._new_expert(100, 2.0).fit(
                    X_delta.iloc[train_rows][features],
                    target[train_rows],
                    sample_weight=weights[train_rows],
                )
                oof[valid_rows, column] = expert.predict(X_delta.iloc[valid_rows][features])
        valid = np.isfinite(oof).all(axis=1)
        self.combiner_ = Ridge(
            alpha=self.shrinkage_alpha, fit_intercept=False, positive=True
        ).fit(oof[valid], target[valid], sample_weight=weights[valid])
        self.prep_expert_ = self._new_expert(120, 2.0).fit(
            X_delta[self.feature_names_prep_], target, sample_weight=weights
        )
        self.bottom_expert_ = self._new_expert(120, 2.0).fit(
            X_delta[self.feature_names_bottom_], target, sample_weight=weights
        )
        self.material_expert_ = self._new_expert(120, 2.0).fit(
            X_delta[self.feature_names_material_], target, sample_weight=weights
        )
        return self


__all__ = [
    "expanding_group_row_folds",
    "GroupAtomicSilicaSubsystem",
    "GroupAtomicCarbonBlackSubsystem",
]
