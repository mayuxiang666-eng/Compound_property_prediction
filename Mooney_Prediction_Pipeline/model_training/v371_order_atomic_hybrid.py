"""Order-atomic V3.7.1 raw expert used as the R5 base model."""

from __future__ import annotations

import numpy as np

from .bias_shrinkage import RegularizedBiasEstimator
from .order_atomic_validation import expanding_window_order_folds
from .v371_corrected_hybrid_v2 import V371CorrectedTemporalGroupModel


class V371OrderAtomicGroupModel(V371CorrectedTemporalGroupModel):
    """Keep V3.7 stages while making Stage1/Stage1b OOF order-atomic."""

    contract_version = "V3.7.1-R5-ORDER-ATOMIC-RAW-V1"

    def _crossfit_stage1_bias(self, frame, y, compound_col, cluster_col):
        folds = expanding_window_order_folds(
            frame,
            time_col=self.time_col,
            order_col="OrderID",
            n_splits=self.crossfit_splits,
            min_train_fraction=self.min_train_fraction,
        )
        oof_stage1 = np.full(len(frame), np.nan, dtype=float)
        oof_bias = np.full(len(frame), np.nan, dtype=float)
        audit = []
        for fold_no, (train_index, valid_index) in enumerate(folds, start=1):
            train_orders = set(frame.iloc[train_index]["OrderID"].astype(str))
            valid_orders = set(frame.iloc[valid_index]["OrderID"].astype(str))
            overlap = train_orders & valid_orders
            if overlap:
                raise AssertionError("Order leakage inside R5 crossfit")
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
                estimator.fit(
                    y[prior_oof], oof_stage1[prior_oof], frame.iloc[prior_oof]
                )
                oof_bias[valid_index] = estimator.predict_bias(
                    frame.iloc[valid_index]
                )
            else:
                oof_bias[valid_index] = 0.0
            audit.append(
                {
                    "fold": fold_no,
                    "train_rows": int(len(train_index)),
                    "valid_rows": int(len(valid_index)),
                    "train_orders": int(len(train_orders)),
                    "valid_orders": int(len(valid_orders)),
                    "order_overlap": 0,
                }
            )
        eligible = np.isfinite(oof_stage1) & np.isfinite(oof_bias)
        if int(eligible.sum()) < 2:
            raise RuntimeError("Order-atomic Stage1/Stage1b has insufficient OOF rows")
        self.order_atomic_fold_audit_ = audit
        return oof_stage1, oof_bias, eligible, folds

    def fit(self, *args, **kwargs):
        result = super().fit(*args, **kwargs)
        self.crossfit_audit_["split_mode"] = (
            "ORDER_ATOMIC_EXPANDING_WINDOW_BY_LATEST_LAB_TIME"
        )
        self.crossfit_audit_["order_atomic_folds"] = self.order_atomic_fold_audit_
        self.crossfit_audit_["order_overlap_total"] = 0
        return result


__all__ = ["V371OrderAtomicGroupModel"]
