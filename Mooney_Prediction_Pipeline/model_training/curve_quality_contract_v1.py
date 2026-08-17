"""Leakage-safe curve quality contract for V3.7.1 retraining.

This module deliberately does not read Lab/MNY values.  It separates hard
physical invalidity from statistical process unusualness:

* physical invalidity: excluded from the modelling population;
* statistical unusualness: retained, but exposed as an auditable risk flag.

Isolation Forest is fit on the chronological training portion only and is
stratified by material system and process route.  RobustScaler is used only
inside this detector because tree isolation is affected by feature scale.
The Mooney model itself keeps physical units for LightGBM features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


CONTRACT_VERSION = "CURVE-QUALITY-V1-PHYSICAL-FAIL-CLOSED-STATISTICAL-RISK"
STATUS_VALID = "CURVE_VALID"
STATUS_PHYSICAL_INVALID = "CURVE_PHYSICAL_INVALID"
STATUS_STATISTICAL_OUTLIER = "CURVE_STATISTICAL_OUTLIER"

IDENTITY_COLUMNS = {
    "OrderID", "BatchNumber", "PalletID", "LabelGroupID", "MNY", "lab_time",
    "split", "TargetVectorRaw", "TargetVectorHash", "SCP_LOG_contract_raw",
    "SCP_LOG_contract_hash", "TemplateKey",
}

PROCESS_PREFIXES = (
    "Stage1_", "Stage2_", "Stage3_", "Stage4_", "Stage5_", "Stage6_",
    "phys_", "time_", "setting_", "ratio_",
)

REQUIRED_CORE = (
    "idx_total_length", "time_total_mixing", "Stage1_Loading_Duration",
    "Stage2_DryMixing_Duration", "Stage5_PID_Duration", "phys_init_temp",
    "phys_max_temp", "phys_t_max_temp",
)

DURATION_COLUMNS = (
    "Stage1_Loading_Duration", "Stage2_DryMixing_Duration",
    "Stage3_OilLoading_Duration", "Stage4_WetMixing_Duration",
    "Stage5_PID_Duration", "Stage6_BottomMixing_Duration",
)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _route_key(frame: pd.DataFrame) -> pd.Series:
    material = frame.get("material_system", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN").astype(str)
    route = frame.get("phase_route", pd.Series("UNKNOWN", index=frame.index)).fillna("UNKNOWN").astype(str)
    return material + "|" + route


def _physical_reasons(frame: pd.DataFrame) -> pd.Series:
    reasons = pd.Series("", index=frame.index, dtype=object)

    def add(mask: pd.Series, code: str) -> None:
        mask = mask.fillna(False)
        reasons.loc[mask] = reasons.loc[mask].where(
            reasons.loc[mask].eq(""), reasons.loc[mask] + ";"
        ) + code

    if "BatchCompleteMsgReceived" in frame:
        add(_num(frame, "BatchCompleteMsgReceived").ne(1), "BATCH_NOT_COMPLETE")
    if "pid_data_valid_flag" in frame:
        add(_num(frame, "pid_data_valid_flag").ne(1), "PID_DATA_INVALID")

    for column in REQUIRED_CORE:
        add(_num(frame, column).isna(), f"MISSING_{column.upper()}")

    for column in DURATION_COLUMNS:
        values = _num(frame, column)
        add(values.lt(0), f"NEGATIVE_{column.upper()}")

    add(_num(frame, "idx_total_length").le(0), "NONPOSITIVE_CURVE_LENGTH")
    add(_num(frame, "time_total_mixing").le(0), "NONPOSITIVE_TOTAL_MIXING_TIME")

    for column in ("phys_init_temp", "phys_max_temp", "phys_discharge_temp"):
        values = _num(frame, column)
        add(values.lt(0) | values.gt(250), f"IMPOSSIBLE_{column.upper()}")

    for column in ("Stage1_Loading_power_Integral", "Stage2_DryMixing_power_Integral",
                   "Stage3_OilLoading_power_Integral", "Stage4_WetMixing_power_Integral",
                   "Stage5_PID_power_Integral", "Stage6_BottomMixing_power_Integral",
                   "phys_power_integral"):
        values = _num(frame, column)
        add(values.lt(0), f"NEGATIVE_{column.upper()}")

    return reasons


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in IDENTITY_COLUMNS or column in {"BatchNumber", "ProcessVersion"}:
            continue
        if not any(column.startswith(prefix) for prefix in PROCESS_PREFIXES):
            continue
        if column.endswith("__missing") or column.endswith("__group_std"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def build_quality_audit(
    batch_frame: pd.DataFrame,
    group_frame: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42,
    min_route_rows: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return batch audit, one-row-per-Lab-group flags, and manifest.

    `group_frame` supplies the chronological split only.  No target column is
    used by the detector or by the physical rules.
    """
    required_ids = ["OrderID", "BatchNumber", "LabelGroupID"]
    missing = sorted(set(required_ids).difference(batch_frame.columns))
    if missing:
        raise ValueError(f"Missing batch identity columns: {missing}")
    if "LabelGroupID" not in group_frame or "split" not in group_frame:
        raise ValueError("Group frame must contain LabelGroupID and split")

    work = batch_frame.copy()
    work = work.drop_duplicates(["OrderID", "BatchNumber"], keep="first").reset_index(drop=True)
    split_map = group_frame[["LabelGroupID", "split"]].drop_duplicates("LabelGroupID")
    if "split" in work.columns:
        work = work.drop(columns=["split"])
    work = work.merge(split_map, on="LabelGroupID", how="left", validate="many_to_one")
    if work["split"].isna().any():
        raise ValueError("Some physical batches have no group-atomic split")

    work["curve_quality_physical_reasons"] = _physical_reasons(work)
    work["curve_physical_valid"] = work["curve_quality_physical_reasons"].eq("")
    work["curve_route_key"] = _route_key(work)
    feature_columns = _feature_columns(work)
    if len(feature_columns) < 5:
        raise ValueError("Too few process features for statistical curve audit")

    work["curve_if_score"] = np.nan
    work["curve_if_outlier"] = False
    model_rows: list[dict[str, Any]] = []

    for route_key, route in work.groupby("curve_route_key", sort=True):
        train = route.loc[route["split"].eq("train") & route["curve_physical_valid"]]
        usable = train[feature_columns].replace([np.inf, -np.inf], np.nan)
        usable = usable.dropna(axis=1, how="all")
        cols = [c for c in usable.columns if train[c].notna().mean() >= 0.95]
        if len(train) < min_route_rows or len(cols) < 5:
            model_rows.append({"route_key": route_key, "status": "BLOCKED_SMALL_ROUTE_OR_FEATURE_COVERAGE", "train_rows": int(len(train)), "feature_count": int(len(cols))})
            continue

        # Training-fold medians only; never fit preprocessing on validation/OOT.
        medians = train[cols].apply(pd.to_numeric, errors="coerce").median()
        x_train = train[cols].apply(pd.to_numeric, errors="coerce").fillna(medians).replace([np.inf, -np.inf], np.nan).fillna(medians)
        scaler = RobustScaler().fit(x_train)
        detector = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
        detector.fit(scaler.transform(x_train))

        route_matrix = route[cols].apply(pd.to_numeric, errors="coerce").fillna(medians).replace([np.inf, -np.inf], np.nan).fillna(medians)
        scores = detector.decision_function(scaler.transform(route_matrix))
        predictions = detector.predict(scaler.transform(route_matrix))
        work.loc[route.index, "curve_if_score"] = scores
        work.loc[route.index, "curve_if_outlier"] = predictions == -1
        model_rows.append({"route_key": route_key, "status": "FITTED", "train_rows": int(len(train)), "feature_count": int(len(cols)), "feature_columns": cols})

    work["curve_statistical_outlier"] = work["curve_if_outlier"] & work["curve_physical_valid"]
    work["curve_quality_status"] = STATUS_VALID
    work.loc[~work["curve_physical_valid"], "curve_quality_status"] = STATUS_PHYSICAL_INVALID
    work.loc[work["curve_statistical_outlier"], "curve_quality_status"] = STATUS_STATISTICAL_OUTLIER
    work["curve_model_include"] = work["curve_physical_valid"]
    work["curve_risk_flag"] = (~work["curve_physical_valid"]) | work["curve_statistical_outlier"]

    group = work.groupby("LabelGroupID", as_index=False).agg(
        curve_batch_count=("BatchNumber", "size"),
        curve_physical_invalid_batch_count=("curve_physical_valid", lambda x: int((~x).sum())),
        curve_statistical_outlier_batch_count=("curve_statistical_outlier", "sum"),
        curve_model_include_batch_count=("curve_model_include", "sum"),
        curve_if_score_mean=("curve_if_score", "mean"),
        curve_if_score_min=("curve_if_score", "min"),
    )
    group["curve_physical_invalid_rate"] = group["curve_physical_invalid_batch_count"] / group["curve_batch_count"]
    group["curve_statistical_outlier_rate"] = group["curve_statistical_outlier_batch_count"] / group["curve_batch_count"]
    group["curve_valid_batch_rate"] = group["curve_model_include_batch_count"] / group["curve_batch_count"]
    group["curve_quality_risk_flag"] = (group["curve_physical_invalid_batch_count"] > 0) | (group["curve_statistical_outlier_batch_count"] > 0)

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "detector": "IsolationForest",
        "detector_fit_policy": "TRAIN_ONLY_BY_MATERIAL_SYSTEM_AND_PHASE_ROUTE",
        "contamination": contamination,
        "physical_invalid_excluded_from_model": True,
        "statistical_outlier_retained_with_risk_flag": True,
        "mny_used_by_detector": False,
        "lab_used_by_detector": False,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "batch_rows": int(len(work)),
        "group_rows": int(len(group)),
        "physical_invalid_batches": int((~work["curve_physical_valid"]).sum()),
        "statistical_outlier_batches": int(work["curve_statistical_outlier"].sum()),
        "model_included_batches": int(work["curve_model_include"].sum()),
        "route_models": model_rows,
    }
    return work, group, manifest


__all__ = ["CONTRACT_VERSION", "build_quality_audit"]
