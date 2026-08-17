"""R1: equal-Lab-group, chronological, robust V3.7.1 challenger."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor, Ridge

from .bias_shrinkage import RegularizedBiasEstimator
from .group_metric_contract import assert_one_row_per_lab_group
from .hybrid_unified_model import HybridUnifiedMooneyModel
from .temporal_subsystems import (
    TemporalCarbonBlackSubsystemPredictor,
    TemporalSilicaSubsystemPredictor,
)
from .temporal_validation import expanding_window_folds, fold_audit


class V371CorrectedTemporalGroupModel(HybridUnifiedMooneyModel):
    """Preserve V3.7 stages while removing group-weight and time leakage defects."""

    contract_version = "V3.7.1-R1-EQUAL-GROUP-TEMPORAL-V2"

    def __init__(
        self,
        *args,
        crossfit_splits: int = 5,
        time_col: str = "lab_time",
        objective: str = "regression_l1",
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
        return LGBMRegressor(**params)

    def _transform_process_deltas(self, df: pd.DataFrame, material_col: str):
        """Build process deltas without converting missing measurements to zero.

        A missing process measurement is not the same thing as a measurement
        exactly equal to the route nominal. LightGBM can consume NaN and use
        the explicit ``__missing`` contract columns. This override is kept
        in V3.7.1 so the frozen V3.7 implementation remains unchanged.
        """
        route = self._get_route_keys(df, material_col)
        if material_col in df.columns:
            material = df[material_col].fillna("GLOBAL").astype(str)
        else:
            material = pd.Series("GLOBAL", index=df.index)
        phase_series = pd.Series(self._phase_route(df), index=df.index, name="phase_route")
        keys = pd.MultiIndex.from_arrays(
            [phase_series, material], names=["phase_route", "material_system"]
        )
        nominals = self.process_nominals_.reindex(keys)
        nominals.index = df.index
        nominals = nominals.reindex(columns=self.feature_names_s2_)
        nominals = nominals.fillna(self.global_process_nominal_)
        actual = df[self.feature_names_s2_].apply(pd.to_numeric, errors="coerce")
        return actual.subtract(nominals, axis="columns"), route

    def _prepare_group_frame(self, df_train: pd.DataFrame) -> pd.DataFrame:
        assert_one_row_per_lab_group(df_train)
        if self.time_col not in df_train.columns:
            raise ValueError(f"Missing chronological event column: {self.time_col}")
        frame = df_train.copy()
        frame[self.time_col] = pd.to_datetime(frame[self.time_col], errors="coerce", utc=True)
        if frame[self.time_col].isna().any():
            raise ValueError("Chronological event time contains missing/invalid values")
        stable_keys = [self.time_col]
        stable_keys.extend(column for column in ("OrderID", "LabelGroupID") if column in frame.columns)
        frame = frame.sort_values(stable_keys, kind="stable").reset_index(drop=True)
        frame["_label_group_id"] = frame["LabelGroupID"].astype(str)
        frame["_label_group_size"] = 1.0
        frame["_w_label_raw"] = 1.0
        frame["_w_compound_balance"] = 1.0
        frame["_w_risk"] = 1.0
        frame["_w_model_raw"] = 1.0
        frame["_w_loss"] = 1.0
        frame["_w_metric"] = 1.0
        frame["_sample_weight"] = 1.0
        return frame

    def _crossfit_stage1_bias(self, frame, y, compound_col, cluster_col):
        folds = expanding_window_folds(
            len(frame), self.crossfit_splits, self.min_train_fraction
        )
        oof_stage1 = np.full(len(frame), np.nan, dtype=float)
        oof_bias = np.full(len(frame), np.nan, dtype=float)
        for train_index, valid_index in folds:
            model = self._new_stage1()
            model.fit(
                frame.iloc[train_index][self.feature_names_s1_],
                y[train_index],
                sample_weight=np.ones(len(train_index)),
            )
            oof_stage1[valid_index] = model.predict(
                frame.iloc[valid_index][self.feature_names_s1_]
            )
            prior_oof = train_index[np.isfinite(oof_stage1[train_index])]
            if len(prior_oof):
                estimator = RegularizedBiasEstimator(
                    shrinkage_k=self.shrinkage_k,
                    compound_col=compound_col,
                    cluster_col=cluster_col,
                )
                estimator.fit(y[prior_oof], oof_stage1[prior_oof], frame.iloc[prior_oof])
                oof_bias[valid_index] = estimator.predict_bias(frame.iloc[valid_index])
            else:
                oof_bias[valid_index] = 0.0
        eligible = np.isfinite(oof_stage1) & np.isfinite(oof_bias)
        if int(eligible.sum()) < 2:
            raise RuntimeError("Temporal Stage1/Stage1b has insufficient OOF rows")
        return oof_stage1, oof_bias, eligible, folds

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
        frame = self._prepare_group_frame(df_train)
        y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("V3.7.1 R1 target contains non-finite values")

        oof_stage1, oof_bias, eligible, folds = self._crossfit_stage1_bias(
            frame, y, compound_col, cluster_col
        )
        self.stage1_model_ = self._new_stage1()
        self.stage1_model_.fit(
            frame[self.feature_names_s1_], y, sample_weight=np.ones(len(frame))
        )
        self.stage1_feature_importance_ = pd.DataFrame(
            {"feature": self.feature_names_s1_, "importance": self.stage1_model_.feature_importances_}
        ).sort_values("importance", ascending=False, ignore_index=True)
        self.stage1b_bias_ = RegularizedBiasEstimator(
            shrinkage_k=self.shrinkage_k,
            compound_col=compound_col,
            cluster_col=cluster_col,
        )
        self.stage1b_bias_.fit(y[eligible], oof_stage1[eligible], frame.loc[eligible])

        frame["_phase_route"] = self._phase_route(frame)
        frame["_material_system"] = (
            frame[cluster_col].fillna("GLOBAL").astype(str)
            if cluster_col in frame.columns else "GLOBAL"
        )
        self.process_nominals_ = frame.groupby(
            ["_phase_route", "_material_system"]
        )[self.feature_names_s2_].mean()
        self.process_nominals_.index = self.process_nominals_.index.set_names(
            ["phase_route", "material_system"]
        )
        self.global_process_nominal_ = frame[self.feature_names_s2_].mean()

        stage2_frame = frame.loc[eligible].copy().reset_index(drop=True)
        stage2_frame["_res_s1b"] = (
            y[eligible] - oof_stage1[eligible] - oof_bias[eligible]
        )
        process_delta, routes = self._transform_process_deltas(stage2_frame, cluster_col)
        self.stage2_experts_ = {}
        report: list[dict[str, object]] = []
        for route_key in sorted(routes.unique(), key=str):
            mask = routes == route_key
            x_route = process_delta.loc[mask].reset_index(drop=True)
            route_frame = stage2_frame.loc[mask].reset_index(drop=True)
            y_route = route_frame["_res_s1b"].to_numpy(dtype=float)
            route_weights = np.ones(len(route_frame), dtype=float)
            is_no_oil = "no_oil_dry" in route_key if isinstance(route_key, tuple) else route_key == "no_oil_dry"
            is_silica = "Silica" in route_key if isinstance(route_key, tuple) else route_frame["_material_system"].iloc[0] == "Silica"
            is_cb = "CarbonBlack" in route_key if isinstance(route_key, tuple) else route_frame["_material_system"].iloc[0] == "CarbonBlack"

            if len(route_frame) < 30:
                expert = Ridge(alpha=10.0).fit(x_route, y_route, sample_weight=route_weights)
                model_type = "ridge_small_route"
            elif self.use_silica_subsystem and is_silica:
                expert = TemporalSilicaSubsystemPredictor(
                    n_splits=self.crossfit_splits,
                    shrinkage_alpha=1.0,
                    objective=self.objective,
                    min_train_fraction=self.min_train_fraction,
                ).fit(x_route, y_route, route_weights, route_frame)
                model_type = "silica_4expert_temporal"
            elif self.use_cb_subsystem and is_cb:
                expert = TemporalCarbonBlackSubsystemPredictor(
                    n_splits=self.crossfit_splits,
                    shrinkage_alpha=1.0,
                    objective=self.objective,
                    min_train_fraction=self.min_train_fraction,
                ).fit(x_route, y_route, route_weights)
                model_type = "cb_3expert_temporal"
            elif is_no_oil and self.no_oil_expert_type == "ridge":
                expert = Ridge(alpha=5.0).fit(x_route, y_route, sample_weight=route_weights)
                model_type = "ridge_linear"
            elif is_no_oil and self.no_oil_expert_type == "huber":
                expert = HuberRegressor(alpha=1.0).fit(x_route, y_route, sample_weight=route_weights)
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
                expert = LGBMRegressor(**params).fit(
                    x_route, y_route, sample_weight=route_weights
                )
                model_type = "lightgbm_temporal_residual"
            self.stage2_experts_[route_key] = expert
            report.append(
                {
                    "phase_route": str(route_key),
                    "n_rows": int(len(route_frame)),
                    "n_label_groups": int(route_frame["_label_group_id"].nunique()),
                    "model_type": model_type,
                }
            )
        self.stage2_expert_report_ = pd.DataFrame(report)
        fold_details = fold_audit(folds)
        for item, (train_index, valid_index) in zip(fold_details, folds):
            item["train_max_time"] = frame.iloc[train_index][self.time_col].max().isoformat()
            item["valid_min_time"] = frame.iloc[valid_index][self.time_col].min().isoformat()
        self.crossfit_audit_ = {
            "contract_version": self.contract_version,
            "split_mode": "EXPANDING_WINDOW_BY_LAB_TIME",
            "folds": fold_details,
            "rows": int(len(frame)),
            "oof_eligible_rows": int(eligible.sum()),
            "warmup_rows": int((~eligible).sum()),
            "all_group_weights_equal_one": bool(np.all(frame["_w_loss"].to_numpy() == 1.0)),
            "objective": self.objective,
            "stage2_residual_source": "TEMPORAL_OOF_STAGE1_PLUS_PRIOR_ONLY_STAGE1B",
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


__all__ = ["V371CorrectedTemporalGroupModel"]
