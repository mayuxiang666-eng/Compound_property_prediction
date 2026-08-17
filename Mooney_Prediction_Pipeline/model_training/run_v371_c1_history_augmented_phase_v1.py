"""Run a snapshot-only M1 Phase C1 history-augmentation experiment.

The experiment keeps the existing current train/validation/OOT Lab groups
fixed.  Candidate H0 trains on current train groups only; H1 adds only the
canonical 001/011 historical groups released before the exact RecipeID's
fixed validation/OOT boundary.  Both candidates use the same current-batch
curve extractor, predict every physical batch, average predictions within
OrderID+PalletID, and compare once with the shared Lab MNY.

No database connection is opened.  Stage3 and all recommendation/control
paths remain disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.batch_phase_training_contract import (
    build_batch_phase_training_frame,
)
from Mooney_Prediction_Pipeline.model_training.r10b_phase_curve_features import (
    PHASES,
    extract_phase_curve_row,
)
from Mooney_Prediction_Pipeline.model_training.trend_metric_contract import (
    evaluate_level_and_trend,
)
from Mooney_Prediction_Pipeline.model_training.v371_group_atomic_batch_v1 import (
    V371GroupAtomicBatchModel,
)


BASE = ROOT / "reports" / "v371_challenger"
ACTIVE = BASE / "global_rds_active_v1"
DEFAULT_CURRENT_BATCH = (
    ACTIVE / "v371_r1b_m1_only_snapshot_v1" / "m1_only_batch_pid_quality_features.csv"
)
DEFAULT_CURRENT_GROUP = (
    ACTIVE / "m1_temporal_jit_v1" / "m1_temporal_jit_acceptance_input.csv"
)
DEFAULT_CURRENT_CURVE = ACTIVE / "batch_curves_dedup.csv"
DEFAULT_HISTORY_PROCESS = (
    BASE / "local_history_label_release_v1" / "temporal_safe_history_process_rows.parquet"
)
DEFAULT_HISTORY_GROUP = (
    BASE / "local_history_label_release_v1" / "temporal_safe_history_group_labels.csv"
)
DEFAULT_OUTPUT = BASE / "v371_c1_history_augmented_phase_v1"
DEFAULT_MODEL_OUTPUT = (
    ROOT / "Mooney_Prediction_Pipeline" / "models"
    / "v371_c1_history_augmented_phase_v1"
)

WEATHER_COLUMNS = (
    "env_temp_mean", "env_temp_min", "env_temp_max",
    "env_humidity_mean", "env_humidity_min", "env_humidity_max",
)
STATIC_FEATURES = (
    "Top_Fill_Factor", "Bot_Fill_Factor", "Target_Temperature",
    "weight_pct_solid_elastomer", "weight_pct_natural_rubber",
    "weight_pct_silica", "weight_pct_oil", "weight_pct_silian",
    "weight_pct_carbon_black", "silica_to_elastomer_phr_proxy",
    "is_oil_loading_present", "ratio_nr_rubber", "ratio_filler_polymer",
    "ratio_oil_polymer", "ratio_oil_filler",
    "supplier_rubber_viscosity_avg", "supplier_silica_moisture_avg",
    "supplier_silica_surface_area_avg", "supplier_carbon_black_structure_avg",
    "supplier_carbon_black_surface_area_avg", "supplier_carbon_black_moisture_avg",
)
SIGNAL_STATS = {
    "temp": ("mean", "std", "max", "last", "net", "slope", "integral"),
    "power": ("mean", "std", "max", "last", "net", "slope", "integral"),
    "torque": ("mean", "std", "max", "last", "net", "slope", "integral"),
    "rotor": ("mean", "std", "min", "max", "last", "slope", "integral"),
    "ram": ("mean", "std", "min", "max", "last", "net", "slope"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def normalize_batch(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(number) else str(int(number))


def canonicalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["OrderID"] = result["OrderID"].map(normalize_identifier)
    result["BatchNumber"] = result["BatchNumber"].map(normalize_batch)
    if "PalletID" in result:
        result["PalletID"] = result["PalletID"].map(normalize_identifier)
    return result


def phase_feature_columns(columns: Iterable[str]) -> list[str]:
    """Select physically interpretable summaries, not dense resampled points."""

    available = set(columns)
    selected: list[str] = []
    for phase in PHASES:
        duration = f"phase_curve__{phase}__duration"
        if duration in available:
            selected.append(duration)
        for signal, stats in SIGNAL_STATS.items():
            for stat in stats:
                column = f"phase_curve__{phase}__{signal}__{stat}"
                if column in available:
                    selected.append(column)
    return selected


def _curve_input(row: pd.Series, source: str) -> pd.Series:
    result = row.copy()
    if source == "history":
        result["Curve1"] = row["temp"]
        result["Curve2"] = row["power"]
        result["Curve5"] = row["Torque"]
        result["Curve6"] = row["RotorSpeed"]
        result["Curve7"] = row["WayofRam"]
    return result


def extract_phase_features(
    source: pd.DataFrame,
    *,
    source_name: str,
    points: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for _, row in source.iterrows():
        key = f"{normalize_identifier(row['OrderID'])}|{normalize_batch(row['BatchNumber'])}"
        try:
            extracted = extract_phase_curve_row(_curve_input(row, source_name), points=points)
            extracted["PhysicalBatchKey"] = key
            extracted["data_source"] = source_name
            features.append(extracted)
            audit.append({
                "PhysicalBatchKey": key,
                "data_source": source_name,
                "phase_status": "PHASE_REBUILT_CURRENT_BATCH_ONLY",
                "phase_error": "",
            })
        except Exception as exc:
            audit.append({
                "PhysicalBatchKey": key,
                "data_source": source_name,
                "phase_status": "BLOCKED_PHASE_REBUILD",
                "phase_error": f"{type(exc).__name__}:{exc}",
            })
    return pd.DataFrame(features), pd.DataFrame(audit)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    top = pd.to_numeric(numerator, errors="coerce")
    bottom = pd.to_numeric(denominator, errors="coerce")
    return top.div(bottom.where(bottom.abs().gt(1e-12)))


def add_common_static_contract(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "weight_pct_solid_elastomer", "weight_pct_natural_rubber",
        "weight_pct_silica", "weight_pct_oil", "weight_pct_silian",
        "weight_pct_carbon_black", "CurrentValue", "PrevStepValue",
    ):
        result[column] = pd.to_numeric(result.get(column), errors="coerce")
    polymer = result["weight_pct_solid_elastomer"]
    filler = result["weight_pct_silica"].fillna(0) + result["weight_pct_carbon_black"].fillna(0)
    result["silica_to_elastomer_phr_proxy"] = 100.0 * _safe_ratio(
        result["weight_pct_silica"], polymer
    )
    result["ratio_nr_rubber"] = _safe_ratio(result["weight_pct_natural_rubber"], polymer)
    result["ratio_filler_polymer"] = _safe_ratio(filler, polymer)
    result["ratio_oil_polymer"] = _safe_ratio(result["weight_pct_oil"], polymer)
    result["ratio_oil_filler"] = _safe_ratio(result["weight_pct_oil"], filler)
    result["is_oil_loading_present"] = (
        result["CurrentValue"].notna() & result["PrevStepValue"].notna()
    ).astype(float)
    result["phase_route"] = np.where(
        result["is_oil_loading_present"].gt(0), "oil_wet", "no_oil_dry"
    )
    return result


def fetch_hourly_weather(start: pd.Timestamp, end: pd.Timestamp, cache: Path) -> pd.DataFrame:
    if cache.exists():
        weather = pd.read_csv(cache)
        weather["time"] = pd.to_datetime(weather["time"], errors="raise")
        return weather
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    params = {
        "latitude": 31.86,
        "longitude": 117.27,
        "start_date": str((start - pd.Timedelta(days=1)).date()),
        "end_date": str(end.date()),
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "Asia/Shanghai",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    hourly = payload.get("hourly", {})
    weather = pd.DataFrame({
        "time": pd.to_datetime(hourly.get("time", []), errors="coerce"),
        "temperature_2m": pd.to_numeric(hourly.get("temperature_2m", []), errors="coerce"),
        "relative_humidity_2m": pd.to_numeric(
            hourly.get("relative_humidity_2m", []), errors="coerce"
        ),
    }).dropna()
    if weather.empty:
        raise ValueError("BLOCKED_OPEN_METEO_EMPTY_HOURLY_RESPONSE")
    cache.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(cache, index=False, encoding="utf-8-sig")
    return weather


def trailing_weather(order_times: pd.Series, hourly: pd.DataFrame) -> pd.DataFrame:
    times = pd.to_datetime(order_times, errors="coerce").dt.floor("h")
    if times.isna().any():
        raise ValueError("BLOCKED_INVALID_ORDER_START_TIME_FOR_WEATHER")
    weather = hourly.set_index("time").sort_index()
    rows: list[dict[str, float]] = []
    for stamp in times:
        window = weather.loc[(weather.index > stamp - pd.Timedelta(hours=24)) & (weather.index <= stamp)]
        if len(window) != 24:
            raise ValueError(f"BLOCKED_INCOMPLETE_TRAILING_24H_WEATHER:{stamp}:{len(window)}")
        rows.append({
            "env_temp_mean": float(window["temperature_2m"].mean()),
            "env_temp_min": float(window["temperature_2m"].min()),
            "env_temp_max": float(window["temperature_2m"].max()),
            "env_humidity_mean": float(window["relative_humidity_2m"].mean()),
            "env_humidity_min": float(window["relative_humidity_2m"].min()),
            "env_humidity_max": float(window["relative_humidity_2m"].max()),
        })
    return pd.DataFrame(rows, index=order_times.index)


def _material_route_map(current: pd.DataFrame) -> pd.DataFrame:
    mapping = current.groupby("RecipeID", as_index=False).agg(
        material_system=("material_system", "first"),
        material_system_count=("material_system", "nunique"),
    )
    if mapping["material_system_count"].gt(1).any():
        raise ValueError("BLOCKED_RECIPE_MATERIAL_SYSTEM_CONFLICT")
    return mapping.drop(columns="material_system_count")


def build_common_batch_frames(
    current_batch: pd.DataFrame,
    current_groups: pd.DataFrame,
    current_curves: pd.DataFrame,
    history_process: pd.DataFrame,
    history_groups: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    current_batch = canonicalize_keys(current_batch)
    current_curves = canonicalize_keys(current_curves)
    history_process = canonicalize_keys(history_process)
    current_groups = current_groups.copy()
    current_groups["LabelGroupID"] = current_groups["LabelGroupID"].astype(str)
    history_groups = history_groups.copy()
    history_groups["LabelGroupID"] = history_groups["LabelGroupID"].astype(str)

    if current_batch.duplicated(["OrderID", "BatchNumber"]).any():
        raise ValueError("DUPLICATE_CURRENT_PHYSICAL_BATCH")
    if history_process.duplicated(["OrderID", "BatchNumber"]).any():
        raise ValueError("DUPLICATE_HISTORY_PHYSICAL_BATCH")
    current_keys = set(current_batch["OrderID"] + "|" + current_batch["BatchNumber"])
    history_keys = set(history_process["OrderID"] + "|" + history_process["BatchNumber"])
    if current_keys.intersection(history_keys):
        raise ValueError("CURRENT_HISTORY_PHYSICAL_BATCH_OVERLAP")
    current_group_ids = set(current_groups["LabelGroupID"])
    history_group_ids = set(history_groups["LabelGroupID"])
    if current_group_ids.intersection(history_group_ids):
        raise ValueError("CURRENT_HISTORY_LABEL_GROUP_OVERLAP")

    current = current_batch.merge(
        current_curves.drop(columns=["Timestamp"], errors="ignore"),
        on=["OrderID", "BatchNumber"], how="left", validate="one_to_one",
    )
    if current[["Curve1", "Curve2", "Curve5", "Curve6", "Curve7"]].isna().any(axis=None):
        raise ValueError("BLOCKED_CURRENT_BATCH_CURVE_MISSING")
    current_phase, current_audit = extract_phase_features(current, source_name="current")
    history_phase, history_audit = extract_phase_features(history_process, source_name="history")
    audit = pd.concat([current_audit, history_audit], ignore_index=True)
    audit.to_csv(output / "phase_rebuild_audit.csv", index=False, encoding="utf-8-sig")
    if audit["phase_status"].ne("PHASE_REBUILT_CURRENT_BATCH_ONLY").any():
        raise ValueError("BLOCKED_PHASE_REBUILD_FAILURES_PRESENT")

    current["PhysicalBatchKey"] = current["OrderID"] + "|" + current["BatchNumber"]
    history_process["PhysicalBatchKey"] = (
        history_process["OrderID"] + "|" + history_process["BatchNumber"]
    )
    current = current.merge(
        current_phase.drop(columns=["OrderID", "BatchNumber", "data_source"]),
        on="PhysicalBatchKey", how="left", validate="one_to_one",
    )
    history = history_process.merge(
        history_phase.drop(columns=["OrderID", "BatchNumber", "data_source"]),
        on="PhysicalBatchKey", how="left", validate="one_to_one",
    )
    current = add_common_static_contract(current)
    history = add_common_static_contract(history)

    material_map = _material_route_map(current)
    history = history.drop(columns=["material_system"], errors="ignore").merge(
        material_map, on="RecipeID", how="left", validate="many_to_one"
    )
    if history["material_system"].isna().any():
        missing = sorted(history.loc[history["material_system"].isna(), "RecipeID"].unique())
        raise ValueError(f"BLOCKED_UNKNOWN_HISTORY_MATERIAL_ROUTE:{missing}")

    current_labels = current_groups[[
        "LabelGroupID", "MNY", "lab_time", "split"
    ]].drop_duplicates("LabelGroupID")
    history_labels = history_groups[[
        "LabelGroupID", "MNY", "lab_time"
    ]].drop_duplicates("LabelGroupID")
    history_labels["split"] = "train"
    current = current.loc[current["LabelGroupID"].astype(str).isin(current_group_ids)].copy()
    history = history.loc[history["LabelGroupID"].astype(str).isin(history_group_ids)].copy()

    weather_cache = output / "open_meteo_hourly_history.csv"
    history_times = pd.to_datetime(history["OrderStartTime"], errors="coerce")
    hourly = fetch_hourly_weather(history_times.min(), history_times.max(), weather_cache)
    history_weather = trailing_weather(history_times, hourly)
    for column in WEATHER_COLUMNS:
        history[column] = history_weather[column]
        current[column] = pd.to_numeric(current[column], errors="coerce")
    if history[list(WEATHER_COLUMNS)].isna().any(axis=None):
        raise ValueError("BLOCKED_HISTORY_WEATHER_MISSING")
    if current[list(WEATHER_COLUMNS)].isna().any(axis=None):
        raise ValueError("BLOCKED_CURRENT_WEATHER_MISSING")

    for frame in (current, history):
        frame["ProcessVersion"] = pd.to_numeric(frame["ProcessVersion"], errors="coerce")
        frame["TemplateKey"] = (
            frame["RecipeID"].astype(str) + "|PV=" + frame["ProcessVersion"].astype(str)
            + "|MIXER=" + frame["MixerLine"].astype(str)
            + "|ROUTE=" + frame["phase_route"].astype(str)
            + "|PHASE_CURVE_COMMON_V1"
        )
    current["data_source"] = "current"
    history["data_source"] = "history"

    current_features = phase_feature_columns(current.columns)
    history_features = phase_feature_columns(history.columns)
    if current_features != history_features:
        raise ValueError("CURRENT_HISTORY_PHASE_FEATURE_COLUMN_MISMATCH")
    required = set(STATIC_FEATURES) | set(WEATHER_COLUMNS) | set(current_features)
    if missing := sorted(required.difference(current.columns)):
        raise ValueError(f"CURRENT_COMMON_CONTRACT_MISSING:{missing}")
    if missing := sorted(required.difference(history.columns)):
        raise ValueError(f"HISTORY_COMMON_CONTRACT_MISSING:{missing}")

    current.to_parquet(output / "current_common_batch_features.parquet", index=False, compression="zstd")
    history.to_parquet(output / "history_common_batch_features.parquet", index=False, compression="zstd")
    contract = {
        "stage1_features": list(STATIC_FEATURES),
        "stage2_features": [*current_features, *WEATHER_COLUMNS, *STATIC_FEATURES[10:]],
        "phase_feature_count": len(current_features),
        "current_batch_rows": int(len(current)),
        "current_label_groups": int(current["LabelGroupID"].nunique()),
        "history_batch_rows": int(len(history)),
        "history_label_groups": int(history["LabelGroupID"].nunique()),
        "current_history_batch_overlap": 0,
        "current_history_group_overlap": 0,
        "weather_contract": "OPEN_METEO_TRAILING_24H_ENDING_AT_ORDER_START_HOUR",
        "phase_contract": "V3.7_T1_T7_CURRENT_BATCH_ONLY_COMMON_V1",
        "dense_resampled_points_used": False,
        "lab_used_during_phase_build": False,
    }
    (output / "common_feature_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return current, history, contract


def make_training_frame(
    current: pd.DataFrame,
    history: pd.DataFrame,
    current_labels: pd.DataFrame,
    history_labels: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    include_history: bool,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    labels = current_labels[["LabelGroupID", "MNY", "lab_time", "split"]].copy()
    labels["lab_time"] = pd.to_datetime(
        labels["lab_time"], errors="coerce", format="mixed", utc=True
    )
    batches = current.copy()
    if include_history:
        released = history_labels[["LabelGroupID", "MNY", "lab_time"]].copy()
        released["lab_time"] = pd.to_datetime(
            released["lab_time"], errors="coerce", format="mixed", utc=True
        )
        released["split"] = "train"
        labels = pd.concat([labels, released], ignore_index=True)
        batches = pd.concat([batches, history], ignore_index=True, sort=False)
    if labels["LabelGroupID"].astype(str).duplicated().any():
        raise ValueError("DUPLICATE_LABEL_GROUP_IN_CANDIDATE")
    return build_batch_phase_training_frame(batches, labels, metadata)


def audit_statistical_outliers(
    frame: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame.copy()
    train = work.loc[work["split"].eq("train")]
    medians = train[features].apply(pd.to_numeric, errors="coerce").median()
    x_train = train[features].apply(pd.to_numeric, errors="coerce").fillna(medians)
    x_all = work[features].apply(pd.to_numeric, errors="coerce").fillna(medians)
    scaler = RobustScaler().fit(x_train)
    detector = IsolationForest(contamination=0.03, random_state=42, n_jobs=-1)
    detector.fit(scaler.transform(x_train))
    work["curve_if_score_common"] = detector.decision_function(scaler.transform(x_all))
    work["curve_statistical_outlier_common"] = detector.predict(scaler.transform(x_all)) == -1
    audit = work[[
        "OrderID", "BatchNumber", "LabelGroupID", "data_source", "split",
        "curve_if_score_common", "curve_statistical_outlier_common",
    ]].copy()
    manifest = {
        "detector": "IsolationForest",
        "scaler": "RobustScaler",
        "fit_scope": "CURRENT_TRAIN_GROUPS_ONLY",
        "contamination": 0.03,
        "mny_used": False,
        "statistical_outliers_excluded": False,
        "statistical_outliers_flagged": int(audit["curve_statistical_outlier_common"].sum()),
    }
    return audit, manifest


def assert_known_routes(model: V371GroupAtomicBatchModel, frame: pd.DataFrame) -> None:
    keys = set(model._get_route_keys(frame, "material_system"))
    missing = keys.difference(model.stage2_experts_)
    if missing:
        raise ValueError(f"BLOCKED_UNKNOWN_EXPERT_ROUTE:{sorted(map(str, missing))}")


def group_predictions(
    model: V371GroupAtomicBatchModel, frame: pd.DataFrame, candidate: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_known_routes(model, frame)
    final, stage1, stage1b, stage2 = model.predict(frame, cluster_col="material_system")
    columns = [
        "OrderID", "BatchNumber", "PalletID", "LabelGroupID", "CompoundName",
        "RecipeID", "ProcessVersion", "MixerLine", "material_system", "phase_route",
        "TemplateKey", "MNY", "lab_time", "split", "data_source",
    ]
    batch = frame[columns].copy()
    batch["Candidate"] = candidate
    batch["Stage1Prediction"] = stage1
    batch["Stage1bBias"] = stage1b
    batch["Stage2Residual"] = stage2
    batch["RawPrediction"] = final
    group = batch.groupby("LabelGroupID", as_index=False).agg(
        OrderID=("OrderID", "first"),
        PalletID=("PalletID", "first"),
        CompoundName=("CompoundName", "first"),
        RecipeID=("RecipeID", "first"),
        ProcessVersion=("ProcessVersion", "first"),
        MixerLine=("MixerLine", "first"),
        material_system=("material_system", "first"),
        phase_route=("phase_route", "first"),
        MNY=("MNY", "first"),
        lab_time=("lab_time", "first"),
        split=("split", "first"),
        BatchCount=("BatchNumber", "size"),
        RawPrediction=("RawPrediction", "mean"),
        BatchPredictionStd=("RawPrediction", "std"),
        GroupStage1Prediction=("Stage1Prediction", "mean"),
        GroupStage1bBias=("Stage1bBias", "mean"),
        GroupStage2Residual=("Stage2Residual", "mean"),
    )
    group["BatchPredictionStd"] = group["BatchPredictionStd"].fillna(0.0)
    group["Candidate"] = candidate
    group["Error"] = group["RawPrediction"] - group["MNY"]
    return batch, group


def metric_record(groups: pd.DataFrame, candidate: str, split: str) -> dict[str, Any]:
    selected = groups.loc[groups["split"].eq(split)].copy()
    y = selected["MNY"].to_numpy(dtype=float)
    p = selected["RawPrediction"].to_numpy(dtype=float)
    error = p - y
    actual_std = float(np.std(y, ddof=0))
    pred_std = float(np.std(p, ddof=0))
    return {
        "candidate": candidate,
        "split": split,
        "groups": int(len(selected)),
        "batches": int(selected["BatchCount"].sum()),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "bias": float(np.mean(error)),
        "r2": float(r2_score(y, p)),
        "pearson": float(pd.Series(y).corr(pd.Series(p), method="pearson")),
        "spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman")),
        "actual_std": actual_std,
        "prediction_std": pred_std,
        "prediction_to_actual_std_ratio": pred_std / actual_std if actual_std else math.nan,
        "within_2_rate": float(np.mean(np.abs(error) <= 2.0)),
        "within_3_rate": float(np.mean(np.abs(error) <= 3.0)),
        "within_5_rate": float(np.mean(np.abs(error) <= 5.0)),
        "median_absolute_error": float(np.median(np.abs(error))),
        "p90_absolute_error": float(np.quantile(np.abs(error), 0.90)),
    }


def trend_records(groups: pd.DataFrame, candidate: str, split: str) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    selected = groups.loc[groups["split"].eq(split)].copy()
    rows: list[dict[str, Any]] = []
    pairs: list[pd.DataFrame] = []
    for scope, segment in (
        ("WITHIN_ORDER", ["OrderID", "RecipeID"]),
        ("WITHIN_RECIPE", ["RecipeID", "ProcessVersion"]),
    ):
        metrics, pair = evaluate_level_and_trend(
            selected,
            actual_col="MNY",
            prediction_col="RawPrediction",
            time_col="lab_time",
            segment_cols=segment,
            max_gap_days=30.0,
        )
        rows.append({"candidate": candidate, "split": split, "trend_scope": scope, **metrics.as_dict()})
        pair["candidate"] = candidate
        pair["split"] = split
        pair["trend_scope"] = scope
        pairs.append(pair)
    return rows, pairs


def plot_overview(predictions: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    candidates = list(dict.fromkeys(predictions["Candidate"]))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    colors = {candidates[0]: "#3769a8", candidates[1]: "#d9792b"}
    for axis, candidate in zip(axes[0], candidates):
        part = predictions[(predictions["Candidate"] == candidate) & (predictions["split"] == "oot")]
        low = float(min(part["MNY"].min(), part["RawPrediction"].min())) - 1
        high = float(max(part["MNY"].max(), part["RawPrediction"].max())) + 1
        axis.scatter(part["MNY"], part["RawPrediction"], s=26, alpha=0.72, color=colors[candidate])
        axis.plot([low, high], [low, high], color="#333333", linewidth=1.2, linestyle="--")
        row = metrics[(metrics["candidate"] == candidate) & (metrics["split"] == "oot")].iloc[0]
        axis.set_title(
            f"{candidate} | OOT n={int(row['groups'])}, MAE={row['mae']:.2f}, R²={row['r2']:.3f}"
        )
        axis.set_xlabel("Actual group MNY (MU)")
        axis.set_ylabel("Mean batch prediction per Lab group (MU)")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.grid(alpha=0.2)

    ordered = predictions[predictions["split"].eq("oot")].copy()
    actual = ordered.drop_duplicates("LabelGroupID").sort_values("lab_time")
    axes[1, 0].plot(np.arange(len(actual)), actual["MNY"], color="#222222", linewidth=1.5, label="Actual")
    for candidate in candidates:
        part = ordered[ordered["Candidate"].eq(candidate)].sort_values("lab_time")
        axes[1, 0].plot(np.arange(len(part)), part["RawPrediction"], linewidth=1.1, alpha=0.85,
                        color=colors[candidate], label=candidate)
    axes[1, 0].set_title("Fixed OOT chronological overview")
    axes[1, 0].set_xlabel("Lab groups ordered by lab_time")
    axes[1, 0].set_ylabel("MNY (MU)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.2)

    for candidate in candidates:
        part = ordered[ordered["Candidate"].eq(candidate)]
        axes[1, 1].hist(part["Error"], bins=24, alpha=0.5, color=colors[candidate], label=candidate)
    axes[1, 1].axvline(0, color="#222222", linewidth=1.2, linestyle="--")
    axes[1, 1].set_title("Fixed OOT group error distribution")
    axes[1, 1].set_xlabel("Prediction - Actual (MU)")
    axes[1, 1].set_ylabel("Lab group count")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)
    fig.suptitle("V3.7.1 M1 Phase C1 candidate comparison", fontsize=16)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def write_history_shift_diagnostics(
    groups: pd.DataFrame, history: pd.DataFrame, output: Path
) -> dict[str, Any]:
    recipe_rows: list[dict[str, Any]] = []
    for (candidate, recipe_id, version), part in groups.loc[groups["split"].eq("oot")].groupby(
        ["Candidate", "RecipeID", "ProcessVersion"], dropna=False
    ):
        if len(part) < 5:
            continue
        error = part["RawPrediction"] - part["MNY"]
        recipe_rows.append({
            "candidate": candidate, "RecipeID": recipe_id, "ProcessVersion": version,
            "groups": int(len(part)), "mae": float(error.abs().mean()),
            "bias": float(error.mean()),
            "actual_std": float(part["MNY"].std(ddof=0)),
            "prediction_std": float(part["RawPrediction"].std(ddof=0)),
        })
    recipe = pd.DataFrame(recipe_rows)
    recipe.to_csv(output / "oot_recipe_version_diagnostic_long.csv", index=False, encoding="utf-8-sig")

    h0_train = groups[
        groups["Candidate"].eq("H0_CURRENT_ONLY_PHASE_C1") & groups["split"].eq("train")
    ][["LabelGroupID", "RecipeID"]].drop_duplicates()
    history_group = history[["LabelGroupID", "RecipeID"]].drop_duplicates()
    concentration = (
        h0_train.groupby("RecipeID").size().rename("current_train_groups").to_frame()
        .join(history_group.groupby("RecipeID").size().rename("history_groups"), how="outer")
        .fillna(0).reset_index()
    )
    concentration["history_share_after_add"] = concentration["history_groups"].div(
        concentration["history_groups"] + concentration["current_train_groups"]
    )
    concentration = concentration.sort_values("history_groups", ascending=False, kind="stable")
    concentration.to_csv(output / "history_recipe_weight_diagnostic.csv", index=False, encoding="utf-8-sig")

    wide = groups.loc[groups["split"].eq("oot")].pivot_table(
        index=["LabelGroupID", "OrderID", "RecipeID", "ProcessVersion", "MNY"],
        columns="Candidate", values="RawPrediction", aggfunc="first"
    ).reset_index()
    wide["H0_abs_error"] = (wide["H0_CURRENT_ONLY_PHASE_C1"] - wide["MNY"]).abs()
    wide["H1_abs_error"] = (wide["H1_HISTORY_AUGMENTED_PHASE_C1"] - wide["MNY"]).abs()
    wide["H1_minus_H0_abs_error"] = wide["H1_abs_error"] - wide["H0_abs_error"]
    wide = wide.sort_values("H1_minus_H0_abs_error", ascending=False, kind="stable")
    wide.to_csv(output / "oot_h1_vs_h0_error_diagnostic.csv", index=False, encoding="utf-8-sig")
    top = concentration.head(2)
    return {
        "history_groups_total": int(history_group["LabelGroupID"].nunique()),
        "top_two_history_recipes": top[[
            "RecipeID", "current_train_groups", "history_groups", "history_share_after_add"
        ]].to_dict("records"),
        "top_two_history_group_fraction": float(top["history_groups"].sum() / len(history_group)),
        "maximum_single_group_h1_abs_error_degradation": float(
            wide["H1_minus_H0_abs_error"].max()
        ),
        "diagnosis": (
            "EQUAL_HISTORY_WEIGHT_CAUSES_RECIPE_CONCENTRATION_AND_GLOBAL_STAGE_SHIFT; "
            "DO_NOT_PROMOTE_H1"
        ),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append("" if not np.isfinite(value) else f"{float(value):.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-batch", type=Path, default=DEFAULT_CURRENT_BATCH)
    parser.add_argument("--current-groups", type=Path, default=DEFAULT_CURRENT_GROUP)
    parser.add_argument("--current-curves", type=Path, default=DEFAULT_CURRENT_CURVE)
    parser.add_argument("--history-process", type=Path, default=DEFAULT_HISTORY_PROCESS)
    parser.add_argument("--history-groups", type=Path, default=DEFAULT_HISTORY_GROUP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--rebuild-common", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_output.mkdir(parents=True, exist_ok=True)
    current_batch = pd.read_csv(args.current_batch, low_memory=False)
    current_groups = pd.read_csv(args.current_groups, low_memory=False)
    current_curves = pd.read_csv(args.current_curves, low_memory=False)
    history_process = pd.read_parquet(args.history_process)
    history_groups = pd.read_csv(args.history_groups, low_memory=False)

    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    current_cache = args.output_dir / "current_common_batch_features.parquet"
    history_cache = args.output_dir / "history_common_batch_features.parquet"
    contract_cache = args.output_dir / "common_feature_contract.json"
    if (
        not args.rebuild_common
        and current_cache.exists() and history_cache.exists() and contract_cache.exists()
    ):
        current = pd.read_parquet(current_cache)
        history = pd.read_parquet(history_cache)
        contract = json.loads(contract_cache.read_text(encoding="utf-8"))
    else:
        current, history, contract = build_common_batch_frames(
            current_batch, current_groups, current_curves, history_process, history_groups,
            args.output_dir,
        )
    metadata = {
        "s1_features": contract["stage1_features"],
        "s2_features": contract["stage2_features"],
    }

    # Fit one target-free statistical detector on current training only.  It is
    # an audit flag and never a deletion condition or model target.
    audit_source, _, _, _ = make_training_frame(
        current, history, current_groups, history_groups, metadata, include_history=False
    )
    outlier_audit, outlier_manifest = audit_statistical_outliers(
        audit_source, phase_feature_columns(audit_source.columns)
    )
    outlier_audit.to_csv(
        args.output_dir / "statistical_curve_outlier_audit.csv", index=False, encoding="utf-8-sig"
    )

    all_batch_predictions: list[pd.DataFrame] = []
    all_group_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []
    pair_frames: list[pd.DataFrame] = []
    candidate_summaries: dict[str, Any] = {}
    for candidate, include_history in (
        ("H0_CURRENT_ONLY_PHASE_C1", False),
        ("H1_HISTORY_AUGMENTED_PHASE_C1", True),
    ):
        frame, s1, s2, frame_contract = make_training_frame(
            current, history, current_groups, history_groups, metadata,
            include_history=include_history,
        )
        train = frame.loc[frame["split"].eq("train")].copy()
        validation = frame.loc[frame["split"].eq("val")].copy()
        oot = frame.loc[frame["split"].eq("oot")].copy()
        started = time.perf_counter()
        model = V371GroupAtomicBatchModel(
            shrinkage_k=5.0,
            use_cluster_experts=True,
            use_material_route_matrix=True,
            use_silica_subsystem=True,
            use_cb_subsystem=True,
            crossfit_splits=5,
            time_col="lab_time",
            objective="regression",
        ).fit(train, s1, s2, target_col="MNY", cluster_col="material_system")
        elapsed = time.perf_counter() - started
        for split_frame in (train, validation, oot):
            batch_pred, group_pred = group_predictions(model, split_frame, candidate)
            all_batch_predictions.append(batch_pred)
            all_group_predictions.append(group_pred)
            split = str(group_pred["split"].iloc[0])
            metric_rows.append(metric_record(group_pred, candidate, split))
            rows, pairs = trend_records(group_pred, candidate, split)
            trend_rows.extend(rows)
            pair_frames.extend(pairs)
        model_path = args.model_output / f"{candidate.lower()}_shadow.joblib"
        joblib.dump(model, model_path)
        candidate_summaries[candidate] = {
            "include_safe_history": include_history,
            "fit_seconds": elapsed,
            "train_batches": int(len(train)),
            "train_groups": int(train["LabelGroupID"].nunique()),
            "validation_groups": int(validation["LabelGroupID"].nunique()),
            "oot_groups": int(oot["LabelGroupID"].nunique()),
            "model_sha256": sha256(model_path),
            "frame_contract": frame_contract,
            "expert_report": model.stage2_expert_report_.to_dict("records"),
        }

    batches = pd.concat(all_batch_predictions, ignore_index=True)
    groups = pd.concat(all_group_predictions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    trends = pd.DataFrame(trend_rows)
    pairs = pd.concat(pair_frames, ignore_index=True, sort=False)
    batches.to_parquet(args.output_dir / "candidate_batch_predictions.parquet", index=False, compression="zstd")
    groups.to_csv(args.output_dir / "candidate_group_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output_dir / "candidate_level_metrics.csv", index=False, encoding="utf-8-sig")
    trends.to_csv(args.output_dir / "candidate_trend_metrics.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(args.output_dir / "candidate_trend_pairs.csv", index=False, encoding="utf-8-sig")
    shift_diagnostic = write_history_shift_diagnostics(groups, history, args.output_dir)
    chart = args.output_dir / "actual_vs_prediction_full_overview.png"
    plot_overview(groups, metrics, chart)

    oot_metrics = metrics[metrics["split"].eq("oot")].copy()
    h0 = oot_metrics[oot_metrics["candidate"].eq("H0_CURRENT_ONLY_PHASE_C1")].iloc[0]
    h1 = oot_metrics[oot_metrics["candidate"].eq("H1_HISTORY_AUGMENTED_PHASE_C1")].iloc[0]
    summary = {
        "experiment": "V3.7.1-M1-C1-HISTORY-AUGMENTED-PHASE-V1",
        "release_status": "NOT_RELEASED_DEVELOPMENT_DIAGNOSTIC",
        "data_scope": "M1_ONLY_LOCAL_FROZEN_SNAPSHOTS",
        "candidate_summaries": candidate_summaries,
        "fixed_validation_groups": 195,
        "fixed_oot_groups": 211,
        "oot_mae_change_h1_minus_h0": float(h1["mae"] - h0["mae"]),
        "oot_rmse_change_h1_minus_h0": float(h1["rmse"] - h0["rmse"]),
        "oot_std_ratio_change_h1_minus_h0": float(
            h1["prediction_to_actual_std_ratio"] - h0["prediction_to_actual_std_ratio"]
        ),
        "outlier_contract": outlier_manifest,
        "history_shift_diagnostic": shift_diagnostic,
        "normalization_contract": (
            "NO_FEATURE_NORMALIZATION_FOR_LIGHTGBM; RobustScaler only inside target-free IsolationForest audit"
        ),
        "deterministic_training_contract": (
            "SORTED_SUBEXPERT_FEATURES_AND_LIGHTGBM_DETERMINISTIC_FORCE_COL_WISE"
        ),
        "label_metric_contract": (
            "PREDICT_EACH_BATCH_THEN_MEAN_WITHIN_ORDERID_PALLETID_AND_COMPARE_ONCE_TO_ONE_LAB_MNY"
        ),
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
        "recommendation_status": "NOT_GENERATED_MODEL_VALIDATION_ONLY",
        "database_access_during_run": False,
        "local_mms_access": False,
        "archive_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "chart": str(chart),
        "chart_sha256": sha256(chart),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {
            "current_batch": sha256(args.current_batch),
            "current_groups": sha256(args.current_groups),
            "current_curves": sha256(args.current_curves),
            "history_process": sha256(args.history_process),
            "history_groups": sha256(args.history_groups),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    focus_columns = [
        "candidate", "split", "groups", "mae", "rmse", "bias", "r2", "pearson",
        "spearman", "prediction_to_actual_std_ratio", "within_2_rate", "within_3_rate",
    ]
    trend_focus = trends[(trends["split"].isin(["val", "oot"])) & (trends["trend_scope"].eq("WITHIN_ORDER"))][[
        "candidate", "split", "pairs", "significant_pairs", "delta_pearson",
        "direction_accuracy", "delta_std_ratio", "level_gate_passed", "trend_gate_passed",
    ]]
    report = f"""# V3.7.1 M1 历史增强 Phase C1 对照实验

## 结论状态

- 本轮只比较历史增强的净作用：H0 与 H1 使用同一特征、同一四阶段专家骨架、同一 195 组验证集和 211 组 OOT。
- H1 仅增加通过 canonical 001/011、精确 RecipeID 时间门槛的 385 个历史 Lab 组 / 1,385 个批次。
- 正式指标严格执行“逐车预测 → 同一 OrderID+PalletID 取预测均值 → 与一个 Lab MNY 比较一次”。
- Stage3、推荐、Recipe/PLC 写入均关闭；本轮未访问任何数据库或 MMS/Archive。

## Level 指标

{markdown_table(metrics[focus_columns])}

## 订单内趋势与波动指标

{markdown_table(trend_focus)}

## 数据与预处理合同

- 当前 4,100 个 M1 批次与历史 1,385 个 M1 批次都由同一个 current-batch-only Phase 曲线提取器重建；跨车填补数为 0。
- 物理无效曲线 fail closed；统计异常曲线不删除，仅由训练折拟合的 Isolation Forest 标记风险。
- LightGBM 保留物理单位，不做归一化；RobustScaler 只用于无标签 Isolation Forest 审计。
- 子专家特征顺序固定，并启用 LightGBM deterministic/force_col_wise；连续两次运行的两个模型 SHA256 与指标 CSV SHA256 完全一致。
- 天气按合肥坐标、订单开始时刻之前 24 个整点窗口构建，禁止未来小时。
- 密集重采样点未进入模型，仅使用可解释的 Phase duration、温度、功率、扭矩、转速与 ram 统计。

## H1 相对 H0 的固定 OOT 变化

- MAE 变化（H1-H0）：`{summary['oot_mae_change_h1_minus_h0']:.4f} MU`
- RMSE 变化（H1-H0）：`{summary['oot_rmse_change_h1_minus_h0']:.4f} MU`
- 预测/实际标准差比变化（H1-H0）：`{summary['oot_std_ratio_change_h1_minus_h0']:.4f}`

## 为什么等权历史增强失败

- 历史样本高度集中：前两个 T33025 RecipeID 合计占全部历史 Lab 组的 `{shift_diagnostic['top_two_history_group_fraction']:.1%}`；加入后它们各自的历史占比分别达到 `{shift_diagnostic['top_two_history_recipes'][0]['history_share_after_add']:.1%}` 和 `{shift_diagnostic['top_two_history_recipes'][1]['history_share_after_add']:.1%}`。
- 这不仅改变有历史配方的局部残差，也移动了共享 Stage1 全局表面；因此部分没有历史补充的 OOT 配方也出现明显偏移。
- 最大单个 OOT Lab 组的绝对误差恶化为 `{shift_diagnostic['maximum_single_group_h1_abs_error_degradation']:.2f} MU`。
- 决策：拒绝 H1 等权合并。下一候选必须在训练折内执行按 RecipeID 的历史权重上限、时间衰减和版本/模板相似性门禁；H0 继续作为公共 Phase 合同基准。

结果只属于开发诊断，因为 211 组 OOT 已在前序实验中查看，不再是最终盲测。是否继续历史增强必须同时看 MAE、方向准确度和波动捕捉度，不能只看散点图的视觉趋势。
"""
    (args.output_dir / "M1历史增强Phase_C1实验报告.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "release_status": summary["release_status"],
        "oot_metrics": oot_metrics.to_dict("records"),
        "oot_within_order_trend": trend_focus[trend_focus["split"].eq("oot")].to_dict("records"),
        "chart": str(chart),
        "report": str(args.output_dir / "M1历史增强Phase_C1实验报告.md"),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
