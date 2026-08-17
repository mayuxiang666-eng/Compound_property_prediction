"""Batch-prediction V3.7.1 model with Lab-group-atomic temporal validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor, Ridge

from .bias_shrinkage import RegularizedBiasEstimator
from .group_atomic_subsystems import (
    GroupAtomicCarbonBlackSubsystem,
    GroupAtomicSilicaSubsystem,
    expanding_group_row_folds,
)
from .hybrid_unified_model import HybridUnifiedMooneyModel


class V371GroupAtomicBatchModel(HybridUnifiedMooneyModel):
    """Predict batches, but split/weight/evaluate by independent Lab group."""

    contract_version = "V3.7.1-R1B-GROUP-ATOMIC-BATCH-V1"

    def __init__(
        self,
        *args,
        crossfit_splits: int = 5,
        time_col: str = "lab_time",
        objective: str = "regression",
        min_train_fraction: float = 0.20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.crossfit_splits = int(crossfit_splits)
        self.time_col = time_col
        self.objective = objective
        self.min_train_fraction = float(min_train_fraction)
        self.crossfit_audit_: dict[str, object] = {}

    def _new_stage1(self) -> LGBMRegressor:
        params = self.stage1_params.copy()
        params["objective"] = self.objective
        params.setdefault("deterministic", True)
        params.setdefault("force_col_wise", True)
        return LGBMRegressor(**params)

    def _prepare_batch_frame(self, source: pd.DataFrame, target_col: str) -> pd.DataFrame:
        required = {"LabelGroupID", self.time_col, target_col}
        missing = sorted(required.difference(source.columns))
        if missing:
            raise ValueError(f"Missing batch model columns: {missing}")
        frame = source.copy()
        frame["LabelGroupID"] = frame["LabelGroupID"].astype(str)
        frame[self.time_col] = pd.to_datetime(frame[self.time_col], errors="coerce", utc=True)
        frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
        if frame[[self.time_col, target_col]].isna().any().any():
            raise ValueError("Batch model target/time contains missing values")
        target_conflicts = frame.groupby("LabelGroupID")[target_col].agg(
            lambda values: float(values.max() - values.min())
        )
        if (target_conflicts.abs() > 1e-9).any():
            raise ValueError("CONFLICTING_LAB_LABEL_WITHIN_GROUP")
        time_conflicts = frame.groupby("LabelGroupID")[self.time_col].nunique()
        if (time_conflicts > 1).any():
            raise ValueError("CONFLICTING_LAB_TIME_WITHIN_GROUP")

        group_order = frame.groupby("LabelGroupID", as_index=False).agg(
            GroupTime=(self.time_col, "first"),
            OrderID=("OrderID", "first") if "OrderID" in frame.columns else (target_col, "size"),
        ).sort_values(["GroupTime", "OrderID", "LabelGroupID"], kind="stable")
        rank = pd.Series(np.arange(len(group_order)), index=group_order["LabelGroupID"])
        frame["_group_time_rank"] = frame["LabelGroupID"].map(rank)
        sort_columns = ["_group_time_rank"]
        sort_columns.extend(column for column in ("OrderID", "BatchNumber") if column in frame.columns)
        frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        frame["_label_group_id"] = frame["LabelGroupID"]
        frame["_label_group_size"] = frame.groupby("LabelGroupID")["LabelGroupID"].transform("size")
        weight = 1.0 / frame["_label_group_size"].astype(float)
        frame["_w_label_raw"] = weight
        frame["_w_loss"] = weight
        frame["_w_metric"] = weight
        frame["_sample_weight"] = weight
        sums = frame.groupby("LabelGroupID")["_w_loss"].sum().to_numpy(dtype=float)
        if not np.allclose(sums, 1.0):
            raise RuntimeError("Each Lab group must have one effective training vote")
        return frame

    def _crossfit_stage1_bias(self, frame, y, compound_col, cluster_col):
        folds, fold_details = expanding_group_row_folds(
            frame["LabelGroupID"], self.crossfit_splits, self.min_train_fraction
        )
        oof_stage1 = np.full(len(frame), np.nan, dtype=float)
        oof_bias = np.full(len(frame), np.nan, dtype=float)
        for train_rows, valid_rows in folds:
            model = self._new_stage1().fit(
                frame.iloc[train_rows][self.feature_names_s1_],
                y[train_rows],
                sample_weight=frame.iloc[train_rows]["_w_loss"],
            )
            oof_stage1[valid_rows] = model.predict(
                frame.iloc[valid_rows][self.feature_names_s1_]
            )
            prior_rows = train_rows[np.isfinite(oof_stage1[train_rows])]
            if len(prior_rows):
                bias = RegularizedBiasEstimator(
                    shrinkage_k=self.shrinkage_k,
                    compound_col=compound_col,
                    cluster_col=cluster_col,
                ).fit(y[prior_rows], oof_stage1[prior_rows], frame.iloc[prior_rows])
                oof_bias[valid_rows] = bias.predict_bias(frame.iloc[valid_rows])
            else:
                oof_bias[valid_rows] = 0.0
        eligible = np.isfinite(oof_stage1) & np.isfinite(oof_bias)
        return oof_stage1, oof_bias, eligible, folds, fold_details

    def _fit_weighted_process_nominals(self, frame: pd.DataFrame) -> None:
        work = frame.copy()
        work["_phase_route"] = self._phase_route(work)
        work["_material_system"] = work["material_system"].fillna("GLOBAL").astype(str)
        keys = ["_phase_route", "_material_system"]
        values = work[self.feature_names_s2_].apply(pd.to_numeric, errors="coerce")
        weights = work["_w_loss"].astype(float)
        numerator = pd.concat([work[keys], values.mul(weights, axis=0)], axis=1).groupby(keys).sum()
        denominator = pd.concat(
            [work[keys], values.notna().astype(float).mul(weights, axis=0)], axis=1
        ).groupby(keys).sum()
        self.process_nominals_ = numerator.divide(denominator.replace(0.0, np.nan))
        self.process_nominals_.index = self.process_nominals_.index.set_names(
            ["phase_route", "material_system"]
        )
        global_numerator = values.mul(weights, axis=0).sum()
        global_denominator = values.notna().astype(float).mul(weights, axis=0).sum().replace(0.0, np.nan)
        self.global_process_nominal_ = global_numerator.divide(global_denominator)

    def fit(
        self,
        df_train,
        feature_cols_stage1,
        feature_cols_stage2,
        target_col="MNY",
        compound_col="CompoundName",
        cluster_col="material_system",
    ):
        self.feature_names_s1_ = list(feature_cols_stage1)
        self.feature_names_s2_ = list(feature_cols_stage2)
        frame = self._prepare_batch_frame(df_train, target_col)
        y = frame[target_col].to_numpy(dtype=float)
        oof_stage1, oof_bias, eligible, folds, fold_details = self._crossfit_stage1_bias(
            frame, y, compound_col, cluster_col
        )

        self.stage1_model_ = self._new_stage1().fit(
            frame[self.feature_names_s1_], y, sample_weight=frame["_w_loss"]
        )
        self.stage1_feature_importance_ = pd.DataFrame(
            {"feature": self.feature_names_s1_, "importance": self.stage1_model_.feature_importances_}
        ).sort_values("importance", ascending=False, ignore_index=True)
        self.stage1b_bias_ = RegularizedBiasEstimator(
            shrinkage_k=self.shrinkage_k,
            compound_col=compound_col,
            cluster_col=cluster_col,
        ).fit(y[eligible], oof_stage1[eligible], frame.loc[eligible])
        self._fit_weighted_process_nominals(frame)

        stage2_frame = frame.loc[eligible].copy().reset_index(drop=True)
        stage2_frame["_res_s1b"] = y[eligible] - oof_stage1[eligible] - oof_bias[eligible]
        process_delta, routes = self._transform_process_deltas(stage2_frame, cluster_col)
        self.stage2_experts_ = {}
        report = []
        for route_key in sorted(routes.unique(), key=str):
            mask = routes == route_key
            x_route = process_delta.loc[mask].reset_index(drop=True)
            route_frame = stage2_frame.loc[mask].reset_index(drop=True)
            y_route = route_frame["_res_s1b"].to_numpy(dtype=float)
            weights = route_frame["_w_loss"].to_numpy(dtype=float)
            route_groups = int(route_frame["LabelGroupID"].nunique())
            is_no_oil = "no_oil_dry" in route_key if isinstance(route_key, tuple) else route_key == "no_oil_dry"
            is_silica = "Silica" in route_key if isinstance(route_key, tuple) else route_frame[cluster_col].iloc[0] == "Silica"
            is_cb = "CarbonBlack" in route_key if isinstance(route_key, tuple) else route_frame[cluster_col].iloc[0] == "CarbonBlack"
            if route_groups < 30:
                expert = Ridge(alpha=10.0).fit(x_route, y_route, sample_weight=weights)
                model_type = "ridge_small_route"
            elif self.use_silica_subsystem and is_silica:
                expert = GroupAtomicSilicaSubsystem(
                    n_splits=self.crossfit_splits,
                    shrinkage_alpha=1.0,
                    objective=self.objective,
                    min_train_fraction=self.min_train_fraction,
                ).fit(x_route, y_route, weights, route_frame)
                model_type = "silica_4expert_group_atomic_batch"
            elif self.use_cb_subsystem and is_cb:
                expert = GroupAtomicCarbonBlackSubsystem(
                    n_splits=self.crossfit_splits,
                    shrinkage_alpha=1.0,
                    objective=self.objective,
                    min_train_fraction=self.min_train_fraction,
                ).fit(x_route, y_route, weights, route_frame)
                model_type = "cb_3expert_group_atomic_batch"
            elif is_no_oil and self.no_oil_expert_type == "ridge":
                expert = Ridge(alpha=5.0).fit(x_route, y_route, sample_weight=weights)
                model_type = "ridge_linear"
            elif is_no_oil and self.no_oil_expert_type == "huber":
                expert = HuberRegressor(alpha=1.0).fit(x_route, y_route, sample_weight=weights)
                model_type = "huber_linear"
            else:
                params = self.stage2_params.copy()
                params.update(
                    objective=self.objective,
                    n_estimators=180 if is_no_oil else 150,
                    learning_rate=0.03 if is_no_oil else 0.02,
                    max_depth=5 if is_no_oil else 4,
                    min_child_samples=15 if is_no_oil else 25,
                    reg_alpha=0.1 if is_no_oil else 0.5,
                    reg_lambda=2.0 if is_no_oil else 5.0,
                )
                params.setdefault("deterministic", True)
                params.setdefault("force_col_wise", True)
                expert = LGBMRegressor(**params).fit(x_route, y_route, sample_weight=weights)
                model_type = "lightgbm_group_atomic_batch"
            self.stage2_experts_[route_key] = expert
            report.append(
                {
                    "phase_route": str(route_key),
                    "n_rows": int(len(route_frame)),
                    "n_label_groups": route_groups,
                    "effective_weight": float(weights.sum()),
                    "model_type": model_type,
                }
            )
        self.stage2_expert_report_ = pd.DataFrame(report)
        for detail, (train_rows, valid_rows) in zip(fold_details, folds):
            detail["train_max_time"] = frame.iloc[train_rows][self.time_col].max().isoformat()
            detail["valid_min_time"] = frame.iloc[valid_rows][self.time_col].min().isoformat()
        self.crossfit_audit_ = {
            "contract_version": self.contract_version,
            "split_mode": "EXPANDING_WINDOW_BY_LAB_GROUP_TIME",
            "folds": fold_details,
            "batch_rows": int(len(frame)),
            "lab_groups": int(frame["LabelGroupID"].nunique()),
            "oof_eligible_rows": int(eligible.sum()),
            "oof_eligible_groups": int(frame.loc[eligible, "LabelGroupID"].nunique()),
            "each_group_effective_weight_one": True,
            "objective": self.objective,
            "stage2_residual_source": "GROUP_ATOMIC_TEMPORAL_OOF",
            "metric_contract": "BATCH_PREDICTION_MEAN_BY_LABEL_GROUP_THEN_COMPARE_ONCE",
            "unknown_route_policy": "BLOCKED_UNKNOWN_EXPERT_ROUTE",
            "stage3_calibration_enabled": False,
        }
        return self

    def predict(self, df_test, compound_col="CompoundName", cluster_col="material_system"):
        stage1 = self.stage1_model_.predict(df_test[self.feature_names_s1_])
        stage1b = self.stage1b_bias_.predict_bias(df_test)
        process_delta, routes = self._transform_process_deltas(df_test, cluster_col)
        unknown = [key for key in routes.unique() if key not in self.stage2_experts_]
        if unknown:
            raise ValueError("BLOCKED_UNKNOWN_EXPERT_ROUTE: " + ", ".join(sorted(map(str, unknown))))
        stage2 = np.zeros(len(df_test), dtype=float)
        for route_key in routes.unique():
            mask = routes == route_key
            stage2[mask] = self.stage2_experts_[route_key].predict(process_delta.loc[mask])
        return stage1 + stage1b + stage2, stage1, stage1b, stage2


__all__ = ["V371GroupAtomicBatchModel"]
