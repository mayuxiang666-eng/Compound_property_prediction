"""Historical bidirectional screen for the selected silica pilot.

This module deliberately separates an observational surrogate screen from the
V3.7 control-transfer gate. It can show whether a minimum action is promising,
but it cannot promote a recommendation when the V3.7 feature path is missing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


@dataclass(frozen=True)
class ReplayGate:
    process_response_status: str
    thermal_budget_status: str
    temperature_cap_status: str
    model_path_status: str
    minimum_response_status: str
    specification_status: str
    final_status: str


def build_strata(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    frame["OrderStartTime"] = pd.to_datetime(frame["OrderStartTime"], errors="coerce")
    frame["time_stratum"] = frame["OrderStartTime"].dt.to_period("Q").astype(str)
    frame["equipment_stratum"] = frame["MixerLine"].astype(str)
    frame["route_stratum"] = frame["CompoundDescription"].astype(str) + "|" + frame["MixerLine"].astype(str)
    frame["raw_material_lot_stratum"] = "MISSING_RAW_MATERIAL_LOT"
    initial = pd.to_numeric(frame["phys_init_temp"], errors="coerce")
    frame["initial_temperature_stratum"] = (initial / 2.0).round() * 2.0
    frame["recipe_version_stratum"] = frame["CompoundDescription"].astype(str)
    frame["stratum_key"] = frame[
        [
            "time_stratum",
            "equipment_stratum",
            "route_stratum",
            "raw_material_lot_stratum",
            "initial_temperature_stratum",
            "recipe_version_stratum",
        ]
    ].astype(str).agg("|".join, axis=1)
    return frame


def _numeric_features(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "MNY",
        "OrderID",
        "PalletID",
        "CompoundName",
        "CompoundDescription",
        "MixerLine",
        "OrderStartTime",
        "test_result_start_time",
        "batch_information_fk_final",
        "N_batches",
    }
    columns = []
    for column in frame.columns:
        if column in excluded or column.endswith("_stratum") or column == "stratum_key":
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().mean() >= 0.95 and values.nunique(dropna=True) > 1:
            columns.append(column)
    return columns


def _fit_time_model(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[GradientBoostingRegressor, pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("OrderStartTime", kind="stable").dropna(subset=["MNY"]).copy()
    split = max(1, int(len(ordered) * 0.8))
    train, valid = ordered.iloc[:split].copy(), ordered.iloc[split:].copy()
    x_train = train[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(train[feature_columns].median())
    model = GradientBoostingRegressor(random_state=17, loss="squared_error")
    model.fit(x_train, train["MNY"].astype(float))
    return model, train, valid


def _mutated_prediction(
    model: GradientBoostingRegressor,
    row: pd.DataFrame,
    feature_columns: list[str],
    feature: str,
    delta: float,
) -> tuple[float, float]:
    base = row[feature_columns].apply(pd.to_numeric, errors="coerce")
    base = base.fillna(row[feature_columns].median(numeric_only=True))
    baseline = float(model.predict(base)[0])
    mutated = base.copy()
    mutated.loc[:, feature] = float(mutated.iloc[0][feature]) + delta
    candidate = float(model.predict(mutated)[0])
    return baseline, candidate


def run_bidirectional_replay(
    history: pd.DataFrame,
    *,
    spec_low: float,
    spec_high: float,
    process_response_status: str = "PASS_FROM_CALIBRATED_800_BATCH_VALIDATION",
    v37_model_path_status: str = "NO_MODEL_PATH",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "MNY",
        "OrderStartTime",
        "MixerLine",
        "CompoundDescription",
        "Stage6_BottomMixing_Duration",
        "phys_discharge_temp",
        "phys_init_temp",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Replay history missing columns: {sorted(missing)}")
    frame = build_strata(history)
    frame["MNY"] = pd.to_numeric(frame["MNY"], errors="coerce")
    frame = frame.dropna(subset=["MNY", "OrderStartTime"]).copy()
    feature_columns = _numeric_features(frame)
    model, train, valid = _fit_time_model(frame, feature_columns)
    validation_pred = model.predict(
        valid[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(train[feature_columns].median())
    )
    model_mae = float(mean_absolute_error(valid["MNY"], validation_pred))
    rows: list[dict[str, Any]] = []
    for parameter, feature, deltas in (
        ("Stage6_duration", "Stage6_BottomMixing_Duration", (-10.0, 10.0)),
        ("discharge_temperature", "phys_discharge_temp", (-1.0, 1.0)),
    ):
        for delta in deltas:
            responses = []
            baselines = []
            candidates = []
            for _, row in valid.iterrows():
                base, candidate = _mutated_prediction(model, row.to_frame().T, feature_columns, feature, delta)
                baselines.append(base)
                candidates.append(candidate)
                responses.append(candidate - base)
            response = np.asarray(responses, dtype=float)
            candidate_mny = np.asarray(candidates, dtype=float)
            abs_response = float(np.median(np.abs(response)))
            min_response = "PASS" if abs_response >= 0.25 else "REJECTED_MINIMUM_MOONEY_RESPONSE"
            spec_status = "PASS" if np.all((candidate_mny >= spec_low) & (candidate_mny <= spec_high)) else "REJECTED_MOONEY_OUT_OF_SPEC"
            cap_status = "NOT_EVALUATED_NO_PROCESS_RESPONSE_VECTOR"
            thermal_status = "NOT_EVALUATED_NO_PROCESS_RESPONSE_VECTOR"
            final_status = "REJECTED_MODEL_PATH"
            rows.append(
                {
                    "parameter": parameter,
                    "delta": delta,
                    "direction": "DOWN" if delta < 0 else "UP",
                    "validation_rows": len(valid),
                    "validation_model_mae": model_mae,
                    "baseline_mny_median": float(np.median(baselines)),
                    "candidate_mny_median": float(np.median(candidate_mny)),
                    "response_median": float(np.median(response)),
                    "response_abs_median": abs_response,
                    "response_p10": float(np.quantile(response, 0.10)),
                    "response_p90": float(np.quantile(response, 0.90)),
                    "process_response_status": process_response_status,
                    "thermal_budget_status": thermal_status,
                    "temperature_cap_status": cap_status,
                    "model_path_status": v37_model_path_status,
                    "minimum_response_status": min_response,
                    "specification_status": spec_status,
                    "status": final_status,
                    "replay_evidence": "OBSERVATIONAL_SURROGATE_ONLY",
                }
            )
    audit = pd.DataFrame(rows)
    summary = {
        "history_rows": int(len(frame)),
        "validation_rows": int(len(valid)),
        "time_split_model_mae": model_mae,
        "feature_columns": feature_columns,
        "strata": {
            "time": int(frame["time_stratum"].nunique()),
            "equipment": int(frame["equipment_stratum"].nunique()),
            "route": int(frame["route_stratum"].nunique()),
            "raw_material_lot": "MISSING",
            "initial_temperature": int(frame["initial_temperature_stratum"].nunique()),
            "recipe_version": int(frame["recipe_version_stratum"].nunique()),
        },
        "candidate_count": int(len(audit)),
        "accepted_count": int((audit["status"] == "MANUAL_REVIEW_ONLY").sum()),
        "replay_status": "DIAGNOSTIC_ONLY_MODEL_PATH_BLOCKED",
    }
    return audit, summary


def write_replay_artifacts(
    audit: pd.DataFrame,
    summary: dict[str, Any],
    *,
    output_directory: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"{stem}_bidirectional_replay.csv"
    json_path = destination / f"{stem}_bidirectional_replay_summary.json"
    audit.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path
