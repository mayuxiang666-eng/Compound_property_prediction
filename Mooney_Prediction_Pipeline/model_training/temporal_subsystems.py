"""Chronological variants of the V3.7 Silica and Carbon Black subexperts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from .cb_subsystem import CarbonBlackSubsystemPredictor
from .silica_subsystem import SilicaSubsystemPredictor
from .temporal_validation import expanding_window_folds, fold_audit


class _TemporalExpertMixin:
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
            verbose=-1,
        )

    def _folds(self, n_samples: int):
        folds = expanding_window_folds(
            n_samples,
            n_splits=self.n_splits,
            min_train_fraction=self.min_train_fraction,
        )
        self.fold_audit_ = fold_audit(folds)
        return folds


class TemporalSilicaSubsystemPredictor(_TemporalExpertMixin, SilicaSubsystemPredictor):
    """Silica four-expert subsystem with strictly chronological OOF blending."""

    def __init__(
        self,
        n_splits: int = 5,
        shrinkage_alpha: float = 1.0,
        positive_constraint: bool = True,
        objective: str = "regression_l1",
        min_train_fraction: float = 0.20,
    ):
        super().__init__(n_splits, shrinkage_alpha, positive_constraint)
        self.objective = objective
        self.min_train_fraction = float(min_train_fraction)
        self.fold_audit_: list[dict[str, int]] = []
        self.oof_valid_rows_ = 0

    def fit(
        self,
        X_delta: pd.DataFrame,
        y_residual: np.ndarray,
        sample_weights: np.ndarray | None = None,
        df_meta: pd.DataFrame | None = None,
    ):
        n_samples = len(X_delta)
        if sample_weights is None:
            sample_weights = np.ones(n_samples)
        sample_weights = np.asarray(sample_weights, dtype=float)
        y_residual = np.asarray(y_residual, dtype=float)
        self._select_sub_features(X_delta, df_meta)
        self.feature_names_pid_ = list(dict.fromkeys(self.feature_names_pid_))
        self.feature_names_wet_ = list(dict.fromkeys(self.feature_names_wet_))
        self.feature_names_bottom_ = list(dict.fromkeys(self.feature_names_bottom_))
        self.feature_names_material_ = list(dict.fromkeys(self.feature_names_material_))

        oof = np.full((n_samples, 4), np.nan, dtype=float)
        for train_idx, valid_idx in self._folds(n_samples):
            train = X_delta.iloc[train_idx]
            valid = X_delta.iloc[valid_idx]
            targets = y_residual[train_idx]
            weights = sample_weights[train_idx]
            feature_sets = (
                self.feature_names_pid_, self.feature_names_wet_,
                self.feature_names_bottom_, self.feature_names_material_,
            )
            for column, features in enumerate(feature_sets):
                model = self._new_expert(100, 3.0)
                model.fit(train[features], targets, sample_weight=weights)
                oof[valid_idx, column] = model.predict(valid[features])

        valid_oof = np.isfinite(oof).all(axis=1)
        if int(valid_oof.sum()) < 2:
            raise ValueError("Silica temporal subsystem has insufficient OOF rows")
        self.oof_valid_rows_ = int(valid_oof.sum())
        self.combiner_ = Ridge(
            alpha=self.shrinkage_alpha,
            fit_intercept=False,
            positive=self.positive_constraint,
        ).fit(oof[valid_oof], y_residual[valid_oof], sample_weight=sample_weights[valid_oof])

        self.pid_expert_ = self._new_expert(120, 3.0).fit(
            X_delta[self.feature_names_pid_], y_residual, sample_weight=sample_weights
        )
        self.wet_expert_ = self._new_expert(120, 3.0).fit(
            X_delta[self.feature_names_wet_], y_residual, sample_weight=sample_weights
        )
        self.bottom_expert_ = self._new_expert(120, 3.0).fit(
            X_delta[self.feature_names_bottom_], y_residual, sample_weight=sample_weights
        )
        self.material_expert_ = self._new_expert(120, 3.0).fit(
            X_delta[self.feature_names_material_], y_residual, sample_weight=sample_weights
        )
        return self


class TemporalCarbonBlackSubsystemPredictor(_TemporalExpertMixin, CarbonBlackSubsystemPredictor):
    """Carbon Black three-expert subsystem with chronological OOF blending."""

    def __init__(
        self,
        n_splits: int = 5,
        shrinkage_alpha: float = 1.0,
        objective: str = "regression_l1",
        min_train_fraction: float = 0.20,
    ):
        super().__init__(n_splits, shrinkage_alpha)
        self.objective = objective
        self.min_train_fraction = float(min_train_fraction)
        self.fold_audit_: list[dict[str, int]] = []
        self.oof_valid_rows_ = 0

    def fit(
        self,
        X_delta: pd.DataFrame,
        y_residual: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ):
        n_samples = len(X_delta)
        if sample_weights is None:
            sample_weights = np.ones(n_samples)
        sample_weights = np.asarray(sample_weights, dtype=float)
        y_residual = np.asarray(y_residual, dtype=float)
        self._select_sub_features(X_delta)
        self.feature_names_prep_ = list(dict.fromkeys(self.feature_names_prep_))
        self.feature_names_bottom_ = list(dict.fromkeys(self.feature_names_bottom_))
        self.feature_names_material_ = list(dict.fromkeys(self.feature_names_material_))

        oof = np.full((n_samples, 3), np.nan, dtype=float)
        for train_idx, valid_idx in self._folds(n_samples):
            train = X_delta.iloc[train_idx]
            valid = X_delta.iloc[valid_idx]
            targets = y_residual[train_idx]
            weights = sample_weights[train_idx]
            for column, features in enumerate(
                (self.feature_names_prep_, self.feature_names_bottom_, self.feature_names_material_)
            ):
                model = self._new_expert(100, 2.0)
                model.fit(train[features], targets, sample_weight=weights)
                oof[valid_idx, column] = model.predict(valid[features])

        valid_oof = np.isfinite(oof).all(axis=1)
        if int(valid_oof.sum()) < 2:
            raise ValueError("Carbon Black temporal subsystem has insufficient OOF rows")
        self.oof_valid_rows_ = int(valid_oof.sum())
        self.combiner_ = Ridge(
            alpha=self.shrinkage_alpha, fit_intercept=False, positive=True
        ).fit(oof[valid_oof], y_residual[valid_oof], sample_weight=sample_weights[valid_oof])

        self.prep_expert_ = self._new_expert(120, 2.0).fit(
            X_delta[self.feature_names_prep_], y_residual, sample_weight=sample_weights
        )
        self.bottom_expert_ = self._new_expert(120, 2.0).fit(
            X_delta[self.feature_names_bottom_], y_residual, sample_weight=sample_weights
        )
        self.material_expert_ = self._new_expert(120, 2.0).fit(
            X_delta[self.feature_names_material_], y_residual, sample_weight=sample_weights
        )
        return self


__all__ = ["TemporalSilicaSubsystemPredictor", "TemporalCarbonBlackSubsystemPredictor"]
