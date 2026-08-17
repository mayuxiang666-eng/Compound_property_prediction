"""B3 chronological Target-vector to actual-process response validation."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error


OUTPUT_THRESHOLDS = {
    "dry_end_temp_actual": 2.0,
    "post_oil_temp_actual": 2.0,
    "wet_end_temp_actual": 2.0,
    "wet_duration_actual": 5.0,
    "pid_duration_actual": 5.0,
    "discharge_temp_actual": 2.0,
}


@dataclass
class ProcessGateResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    control_transfer_status: str
    model_path_status: str


def chronological_order_split(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="raise", utc=True)
    orders = frame.groupby("order_id", as_index=False).agg(
        event_time=("event_time", "min"), rows=("order_id", "size")
    ).sort_values(["event_time", "order_id"], kind="stable")
    cumulative = orders["rows"].cumsum() / orders["rows"].sum()
    orders["split"] = np.select([cumulative <= 0.70, cumulative <= 0.85], ["train", "val"], default="oot")
    return frame.merge(orders[["order_id", "split"]], on="order_id", validate="many_to_one")


def _model(alpha: float | None) -> lgb.LGBMRegressor:
    objective = "regression_l1" if alpha is None else "quantile"
    kwargs = {"alpha": alpha} if alpha is not None else {}
    return lgb.LGBMRegressor(
        objective=objective, **kwargs, n_estimators=220, learning_rate=0.025,
        num_leaves=7, min_child_samples=20, reg_alpha=1.0, reg_lambda=10.0,
        colsample_bytree=0.8, random_state=17, deterministic=True,
        force_col_wise=True, verbosity=-1,
    )


def evaluate_process_response(
    history: pd.DataFrame,
    *,
    pilot: str,
    minimum_samples: int = 150,
) -> ProcessGateResult:
    required = {"order_id", "event_time", *OUTPUT_THRESHOLDS}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"B3 history missing columns: {missing}")
    target_features = sorted(column for column in history.columns if column.startswith("target__"))
    if not target_features:
        raise ValueError("B3 history has no target-vector features")
    frame = chronological_order_split(history)
    train = frame[frame["split"].eq("train")]
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_train_all = imputer.fit_transform(train[target_features])
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    all_pass = True
    for output, threshold in OUTPUT_THRESHOLDS.items():
        available = frame[output].notna()
        train_available = train[output].notna()
        if int(available.sum()) < minimum_samples or int(train_available.sum()) < 50:
            rows.append({
                "pilot": pilot, "output": output, "available_samples": int(available.sum()),
                "oot_samples": 0, "oot_mae": np.nan, "p10_p90_coverage": np.nan,
                "mae_threshold": threshold, "passed": False,
                "reason": "INSUFFICIENT_PROCESS_RESPONSE_HISTORY",
            })
            all_pass = False
            continue
        train_indices = np.flatnonzero(train_available.to_numpy())
        x_train = x_train_all[train_indices]
        y_train = train.loc[train_available, output].to_numpy(float)
        models = {
            "p10": _model(0.1).fit(x_train, y_train),
            "p50": _model(None).fit(x_train, y_train),
            "p90": _model(0.9).fit(x_train, y_train),
        }
        valid = frame[frame["split"].eq("val") & frame[output].notna()].copy()
        oot = frame[frame["split"].eq("oot") & frame[output].notna()].copy()
        if valid.empty or oot.empty:
            rows.append({
                "pilot": pilot, "output": output, "available_samples": int(available.sum()),
                "oot_samples": int(len(oot)), "oot_mae": np.nan, "p10_p90_coverage": np.nan,
                "mae_threshold": threshold, "passed": False, "reason": "MISSING_VAL_OR_OOT",
            })
            all_pass = False
            continue
        x_valid = imputer.transform(valid[target_features])
        valid_mid = models["p50"].predict(x_valid)
        conformal = float(np.quantile(np.abs(valid[output].to_numpy(float) - valid_mid), 0.90))
        x_oot = imputer.transform(oot[target_features])
        p10 = np.minimum(models["p10"].predict(x_oot), models["p90"].predict(x_oot)) - conformal
        p90 = np.maximum(models["p10"].predict(x_oot), models["p90"].predict(x_oot)) + conformal
        p50 = models["p50"].predict(x_oot)
        actual = oot[output].to_numpy(float)
        mae = float(mean_absolute_error(actual, p50))
        coverage = float(((actual >= p10) & (actual <= p90)).mean())
        passed = bool(len(oot) >= 20 and mae <= threshold and coverage >= 0.80)
        all_pass &= passed
        rows.append({
            "pilot": pilot, "output": output, "available_samples": int(available.sum()),
            "oot_samples": int(len(oot)), "oot_mae": mae, "p10_p90_coverage": coverage,
            "conformal_radius": conformal, "mae_threshold": threshold, "passed": passed,
            "reason": "PASSED" if passed else "PROCESS_RESPONSE_GATE_FAILED",
        })
        audit = oot[["order_id", "event_time", "split"]].copy()
        audit["pilot"] = pilot
        audit["output"] = output
        audit["actual"] = actual
        audit["p10"] = p10
        audit["p50"] = p50
        audit["p90"] = p90
        prediction_frames.append(audit)
    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    # Endpoint-only transfer is insufficient until every affected V3.7 Phase
    # feature (means/integrals) has an explicit response path.
    actual_phase_path_complete = bool(history.attrs.get("actual_phase_feature_path_complete", False))
    approved = all_pass and actual_phase_path_complete
    return ProcessGateResult(
        metrics=metrics,
        predictions=predictions,
        control_transfer_status="CONTROL_TRANSFER_AVAILABLE" if approved else "CONTROL_TRANSFER_UNAVAILABLE",
        model_path_status="MODEL_PATH_OK" if approved else "NO_MODEL_PATH",
    )
