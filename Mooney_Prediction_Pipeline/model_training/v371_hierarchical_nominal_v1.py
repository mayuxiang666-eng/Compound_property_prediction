"""R2 hierarchical process-nominal challenger for V3.7.1."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor, Ridge

from .temporal_subsystems import (
    TemporalCarbonBlackSubsystemPredictor,
    TemporalSilicaSubsystemPredictor,
)
from .v371_corrected_hybrid_v2 import V371CorrectedTemporalGroupModel


class V371HierarchicalNominalModel(V371CorrectedTemporalGroupModel):
    """Use exact template/version nominals before route-level fallback."""

    contract_version = "V3.7.1-R2-HIERARCHICAL-NOMINAL-V1"

    def __init__(
        self,
        *args,
        template_min_groups: int = 5,
        recipe_version_min_groups: int = 5,
        recipe_min_groups: int = 10,
        compound_min_groups: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.template_min_groups = int(template_min_groups)
        self.recipe_version_min_groups = int(recipe_version_min_groups)
        self.recipe_min_groups = int(recipe_min_groups)
        self.compound_min_groups = int(compound_min_groups)
        self.nominal_levels_: list[dict[str, object]] = []
        self._hierarchical_nominals_ready = False
        self.nominal_resolution_audit_: dict[str, int] = {}

    @staticmethod
    def _normalized_keys(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        normalized = pd.DataFrame(index=frame.index)
        for column in columns:
            if column not in frame.columns:
                normalized[column] = "__MISSING__"
            else:
                normalized[column] = frame[column].where(frame[column].notna(), "__MISSING__").astype(str)
        return normalized

    def _fit_hierarchical_nominals(self, frame: pd.DataFrame) -> None:
        work = frame.copy()
        work["_phase_route"] = self._phase_route(work)
        work["_material_system"] = work["material_system"].fillna("GLOBAL").astype(str)
        definitions = [
            (
                "TEMPLATE",
                ["TemplateKey", "_phase_route", "_material_system"],
                self.template_min_groups,
            ),
            (
                "RECIPE_VERSION_MIXER",
                ["RecipeID", "ProcessVersion", "MixerLine", "_phase_route", "_material_system"],
                self.recipe_version_min_groups,
            ),
            (
                "RECIPE_MIXER",
                ["RecipeID", "MixerLine", "_phase_route", "_material_system"],
                self.recipe_min_groups,
            ),
            (
                "COMPOUND_MIXER",
                ["CompoundName", "MixerLine", "_phase_route", "_material_system"],
                self.compound_min_groups,
            ),
            (
                "MATERIAL_ROUTE",
                ["_phase_route", "_material_system"],
                1,
            ),
        ]
        self.nominal_levels_ = []
        numeric = work[self.feature_names_s2_].apply(pd.to_numeric, errors="coerce")
        for name, keys, minimum in definitions:
            normalized = self._normalized_keys(work, keys)
            joined = pd.concat([normalized, numeric], axis=1)
            means = joined.groupby(keys, dropna=False)[self.feature_names_s2_].mean()
            counts = joined.groupby(keys, dropna=False).size()
            self.nominal_levels_.append(
                {"name": name, "keys": keys, "min_groups": int(minimum), "means": means, "counts": counts}
            )
        self.global_process_nominal_ = numeric.mean()
        material_route = self.nominal_levels_[-1]["means"]
        self.process_nominals_ = material_route
        self.process_nominals_.index = self.process_nominals_.index.set_names(
            ["phase_route", "material_system"]
        )
        self._hierarchical_nominals_ready = True

    def _resolve_nominals(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        work = frame.copy()
        work["_phase_route"] = self._phase_route(work)
        work["_material_system"] = work["material_system"].fillna("GLOBAL").astype(str)
        resolved = pd.DataFrame(np.nan, index=work.index, columns=self.feature_names_s2_)
        source = pd.Series("UNRESOLVED", index=work.index, dtype=object)
        for level in self.nominal_levels_:
            keys = level["keys"]
            normalized = self._normalized_keys(work, keys)
            lookup = pd.MultiIndex.from_frame(normalized[keys])
            means = level["means"].reindex(lookup)
            means.index = work.index
            counts = level["counts"].reindex(lookup).fillna(0).to_numpy(dtype=float)
            eligible = source.eq("UNRESOLVED") & (counts >= int(level["min_groups"]))
            if eligible.any():
                resolved.loc[eligible, self.feature_names_s2_] = means.loc[
                    eligible, self.feature_names_s2_
                ].to_numpy()
                source.loc[eligible] = str(level["name"])
        unresolved = source.eq("UNRESOLVED")
        if unresolved.any():
            resolved.loc[unresolved] = self.global_process_nominal_.reindex(
                self.feature_names_s2_
            ).to_numpy()
            source.loc[unresolved] = "GLOBAL"
        resolved = resolved.fillna(self.global_process_nominal_)
        return resolved, source

    def nominal_resolution(self, frame: pd.DataFrame) -> pd.Series:
        if not self._hierarchical_nominals_ready:
            raise RuntimeError("Hierarchical nominals are not fitted")
        return self._resolve_nominals(frame)[1]

    def _transform_process_deltas(self, df, material_col):
        if not self._hierarchical_nominals_ready:
            return super()._transform_process_deltas(df, material_col)
        nominals, source = self._resolve_nominals(df)
        self.nominal_resolution_audit_ = {
            str(key): int(value) for key, value in source.value_counts().items()
        }
        actual = df[self.feature_names_s2_].apply(pd.to_numeric, errors="coerce")
        return actual.subtract(nominals, axis="columns").fillna(0.0), self._get_route_keys(df, material_col)

    def _refit_stage2(self, frame, y, oof_stage1, oof_bias, eligible, cluster_col):
        stage2_frame = frame.loc[eligible].copy().reset_index(drop=True)
        stage2_frame["_res_s1b"] = y[eligible] - oof_stage1[eligible] - oof_bias[eligible]
        process_delta, routes = self._transform_process_deltas(stage2_frame, cluster_col)
        experts = {}
        report = []
        for route_key in sorted(routes.unique(), key=str):
            mask = routes == route_key
            x_route = process_delta.loc[mask].reset_index(drop=True)
            route_frame = stage2_frame.loc[mask].reset_index(drop=True)
            y_route = route_frame["_res_s1b"].to_numpy(dtype=float)
            weights = np.ones(len(route_frame), dtype=float)
            is_no_oil = "no_oil_dry" in route_key if isinstance(route_key, tuple) else route_key == "no_oil_dry"
            is_silica = "Silica" in route_key if isinstance(route_key, tuple) else route_frame["material_system"].iloc[0] == "Silica"
            is_cb = "CarbonBlack" in route_key if isinstance(route_key, tuple) else route_frame["material_system"].iloc[0] == "CarbonBlack"
            if len(route_frame) < 30:
                expert = Ridge(alpha=10.0).fit(x_route, y_route, sample_weight=weights)
                model_type = "ridge_small_route"
            elif self.use_silica_subsystem and is_silica:
                expert = TemporalSilicaSubsystemPredictor(
                    n_splits=self.crossfit_splits,
                    shrinkage_alpha=1.0,
                    objective=self.objective,
                    min_train_fraction=self.min_train_fraction,
                ).fit(x_route, y_route, weights, route_frame)
                model_type = "silica_4expert_temporal_hierarchical_nominal"
            elif self.use_cb_subsystem and is_cb:
                expert = TemporalCarbonBlackSubsystemPredictor(
                    n_splits=self.crossfit_splits,
                    shrinkage_alpha=1.0,
                    objective=self.objective,
                    min_train_fraction=self.min_train_fraction,
                ).fit(x_route, y_route, weights)
                model_type = "cb_3expert_temporal_hierarchical_nominal"
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
                expert = LGBMRegressor(**params).fit(x_route, y_route, sample_weight=weights)
                model_type = "lightgbm_temporal_hierarchical_nominal"
            experts[route_key] = expert
            report.append(
                {
                    "phase_route": str(route_key),
                    "n_rows": int(len(route_frame)),
                    "n_label_groups": int(route_frame["_label_group_id"].nunique()),
                    "model_type": model_type,
                }
            )
        self.stage2_experts_ = experts
        self.stage2_expert_report_ = pd.DataFrame(report)

    def fit(
        self,
        df_train,
        feature_cols_stage1,
        feature_cols_stage2,
        target_col="MNY",
        compound_col="CompoundName",
        cluster_col="material_system",
    ):
        super().fit(
            df_train,
            feature_cols_stage1,
            feature_cols_stage2,
            target_col=target_col,
            compound_col=compound_col,
            cluster_col=cluster_col,
        )
        frame = self._prepare_group_frame(df_train)
        y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
        oof_stage1, oof_bias, eligible, _ = self._crossfit_stage1_bias(
            frame, y, compound_col, cluster_col
        )
        self._fit_hierarchical_nominals(frame)
        self._refit_stage2(frame, y, oof_stage1, oof_bias, eligible, cluster_col)
        resolution = self.nominal_resolution(frame)
        self.crossfit_audit_.update(
            {
                "contract_version": self.contract_version,
                "process_nominal_contract": (
                    "TEMPLATE>RECIPE_VERSION_MIXER>RECIPE_MIXER>COMPOUND_MIXER>"
                    "MATERIAL_ROUTE>GLOBAL"
                ),
                "process_nominal_min_groups": {
                    "template": self.template_min_groups,
                    "recipe_version_mixer": self.recipe_version_min_groups,
                    "recipe_mixer": self.recipe_min_groups,
                    "compound_mixer": self.compound_min_groups,
                },
                "training_nominal_resolution": {
                    str(key): int(value) for key, value in resolution.value_counts().items()
                },
            }
        )
        return self


__all__ = ["V371HierarchicalNominalModel"]
