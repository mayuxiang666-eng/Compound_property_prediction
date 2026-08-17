"""V3.7.1 cross-fitted and fail-closed hybrid model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import KFold

from .bias_shrinkage import RegularizedBiasEstimator
from .cb_subsystem import CarbonBlackSubsystemPredictor
from .effective_weighting import compute_effective_sample_weights
from .hybrid_unified_model import HybridUnifiedMooneyModel
from .silica_subsystem import SilicaSubsystemPredictor


class V371CorrectedHybridModel(HybridUnifiedMooneyModel):
    """Keep V3.7 experts, but cross-fit residuals and block unknown routes."""

    contract_version = "V3.7.1-CORRECTED-HYBRID-V1"

    def __init__(self, *args, crossfit_splits: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.crossfit_splits = int(crossfit_splits)
        self.crossfit_audit_: dict[str, object] = {}

    def _new_stage1(self) -> LGBMRegressor:
        return LGBMRegressor(**self.stage1_params)

    def _crossfit_stage1_bias(self, frame, y, weights, compound_col, cluster_col):
        n_splits = min(self.crossfit_splits, len(frame))
        if n_splits < 2:
            raise ValueError("V3.7.1 cross-fitting requires at least two rows")
        folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(frame))
        oof_stage1 = np.full(len(frame), np.nan, dtype=float)
        for train_index, valid_index in folds:
            model = self._new_stage1()
            model.fit(
                frame.iloc[train_index][self.feature_names_s1_],
                y[train_index],
                sample_weight=weights[train_index],
            )
            oof_stage1[valid_index] = model.predict(
                frame.iloc[valid_index][self.feature_names_s1_]
            )
        if not np.isfinite(oof_stage1).all():
            raise RuntimeError("Stage1 cross-fit is incomplete")

        oof_bias = np.full(len(frame), np.nan, dtype=float)
        for train_index, valid_index in folds:
            estimator = RegularizedBiasEstimator(
                shrinkage_k=self.shrinkage_k,
                compound_col=compound_col,
                cluster_col=cluster_col,
            )
            estimator.fit(y[train_index], oof_stage1[train_index], frame.iloc[train_index])
            oof_bias[valid_index] = estimator.predict_bias(frame.iloc[valid_index])
        if not np.isfinite(oof_bias).all():
            raise RuntimeError("Stage1b cross-fit is incomplete")
        return oof_stage1, oof_bias

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
        frame = compute_effective_sample_weights(df_train.copy(), compound_col=compound_col)
        weights = frame["_w_loss"].to_numpy(dtype=float)
        y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("V3.7.1 target contains non-finite values")

        oof_stage1, oof_bias = self._crossfit_stage1_bias(
            frame, y, weights, compound_col, cluster_col
        )
        self.stage1_model_ = self._new_stage1()
        self.stage1_model_.fit(frame[self.feature_names_s1_], y, sample_weight=weights)
        self.stage1_feature_importance_ = pd.DataFrame(
            {"feature": self.feature_names_s1_, "importance": self.stage1_model_.feature_importances_}
        ).sort_values("importance", ascending=False, ignore_index=True)
        self.stage1b_bias_ = RegularizedBiasEstimator(
            shrinkage_k=self.shrinkage_k,
            compound_col=compound_col,
            cluster_col=cluster_col,
        )
        self.stage1b_bias_.fit(y, oof_stage1, frame)

        frame["_res_s1b"] = y - oof_stage1 - oof_bias
        frame["_phase_route"] = self._phase_route(frame)
        frame["_material_system"] = (
            frame[cluster_col].fillna("GLOBAL").astype(str)
            if cluster_col in frame.columns
            else "GLOBAL"
        )
        self.process_nominals_ = frame.groupby(
            ["_phase_route", "_material_system"]
        )[self.feature_names_s2_].mean()
        self.process_nominals_.index = self.process_nominals_.index.set_names(
            ["phase_route", "material_system"]
        )
        self.global_process_nominal_ = frame[self.feature_names_s2_].mean()
        process_delta, routes = self._transform_process_deltas(frame, cluster_col)

        self.stage2_experts_ = {}
        report = []
        for route_key in sorted(routes.unique(), key=str):
            mask = routes == route_key
            x_route = process_delta.loc[mask]
            y_route = frame.loc[mask, "_res_s1b"]
            route_weights = frame.loc[mask, "_w_loss"]
            is_no_oil = "no_oil_dry" in route_key if isinstance(route_key, tuple) else route_key == "no_oil_dry"
            is_silica = "Silica" in route_key if isinstance(route_key, tuple) else frame.loc[mask, "_material_system"].iloc[0] == "Silica"
            is_cb = "CarbonBlack" in route_key if isinstance(route_key, tuple) else frame.loc[mask, "_material_system"].iloc[0] == "CarbonBlack"

            if self.use_silica_subsystem and is_silica:
                expert = SilicaSubsystemPredictor(n_splits=5, shrinkage_alpha=1.0)
                expert.fit(x_route, y_route.to_numpy(), sample_weights=route_weights.to_numpy(), df_meta=frame.loc[mask])
                model_type = "silica_4expert_subsystem"
            elif self.use_cb_subsystem and is_cb:
                expert = CarbonBlackSubsystemPredictor(n_splits=5, shrinkage_alpha=1.0)
                expert.fit(x_route, y_route.to_numpy(), sample_weights=route_weights.to_numpy())
                model_type = "cb_3expert_subsystem"
            elif int(mask.sum()) < 30:
                expert = Ridge(alpha=10.0).fit(x_route, y_route, sample_weight=route_weights)
                model_type = "ridge_fallback"
            elif is_no_oil and self.no_oil_expert_type == "ridge":
                expert = Ridge(alpha=5.0).fit(x_route, y_route, sample_weight=route_weights)
                model_type = "ridge_linear"
            elif is_no_oil and self.no_oil_expert_type == "huber":
                expert = HuberRegressor(alpha=1.0).fit(x_route, y_route, sample_weight=route_weights)
                model_type = "huber_linear"
            else:
                params = self.stage2_params.copy()
                params.update(
                    n_estimators=180 if is_no_oil else 150,
                    learning_rate=0.03 if is_no_oil else 0.02,
                    max_depth=5 if is_no_oil else 4,
                    min_child_samples=15 if is_no_oil else 25,
                    reg_alpha=0.1 if is_no_oil else 0.5,
                    reg_lambda=2.0 if is_no_oil else 5.0,
                )
                expert = LGBMRegressor(**params).fit(x_route, y_route, sample_weight=route_weights)
                model_type = "lightgbm"
            self.stage2_experts_[route_key] = expert
            report.append(
                {
                    "phase_route": str(route_key),
                    "n_rows": int(mask.sum()),
                    "n_label_groups": int(frame.loc[mask, "_label_group_id"].nunique()),
                    "model_type": model_type,
                }
            )
        self.stage2_expert_report_ = pd.DataFrame(report)
        self.crossfit_audit_ = {
            "contract_version": self.contract_version,
            "crossfit_splits": min(self.crossfit_splits, len(frame)),
            "rows": len(frame),
            "oof_stage1_complete": bool(np.isfinite(oof_stage1).all()),
            "oof_stage1b_complete": bool(np.isfinite(oof_bias).all()),
            "stage2_residual_source": "OOF_STAGE1_PLUS_OOF_STAGE1B",
            "unknown_route_policy": "BLOCKED_UNKNOWN_EXPERT_ROUTE",
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


__all__ = ["V371CorrectedHybridModel"]
