"""Run a temporal, pooled, regularized process residual expert for M1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from Mooney_Prediction_Pipeline.model_training.run_m1_order_delta_residual_expert_v1 import (
    PROCESS_FEATURES,
    _build_transitions,
    _group_frame,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    os.environ.get(
        "M1_REGULARIZED_SOURCE",
        str(
            ROOT
            / "reports"
            / "v371_challenger"
            / "global_rds_active_v1"
            / "v371_r1b_m1_only_snapshot_v1"
            / "m1_only_group_label_snapshot_with_pid_quality.csv"
        ),
    )
)
OUTPUT = Path(
    os.environ.get(
        "M1_REGULARIZED_OUTPUT",
        str(
            ROOT
            / "reports"
            / "v371_challenger"
            / "global_rds_active_v1"
            / "m1_regularized_transition_residual_v1"
        ),
    )
)

MIN_HISTORY = 20
RIDGE_ALPHA = 25.0


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_frame(transitions: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    process = [feature for feature in PROCESS_FEATURES if feature in transitions.columns]
    columns: dict[str, pd.Series] = {}
    for feature in process:
        columns[f"delta__{feature}"] = pd.to_numeric(
            transitions[f"delta__{feature}"], errors="coerce"
        )
    # Absolute state at the current completed batch helps distinguish the same
    # delta at different operating points.
    absolute = [
        "Stage2_DryMixing_Duration",
        "Stage2_DryMixing_temp_Mean",
        "Stage4_WetMixing_Duration",
        "Stage4_WetMixing_temp_Mean",
        "Stage4_WetMixing_RotorSpeed_Mean",
        "Stage5_PID_Duration",
        "Stage5_PID_temp_Mean",
        "Stage5_PID_Specific_Energy",
        "Stage6_BottomMixing_Duration",
        "Stage6_BottomMixing_temp_Mean",
        "Stage6_BottomMixing_Specific_Energy",
        "phys_discharge_temp",
        "pid_actual_total_duration_s",
        "pid_control_instability_proxy",
        "pid_high_temperature_risk_proxy",
        "curve_statistical_outlier",
    ]
    for feature in absolute:
        if feature in transitions.columns:
            columns[f"current__{feature}"] = pd.to_numeric(
                transitions[feature], errors="coerce"
            )
    columns["transition_gap_hours"] = (
        pd.to_datetime(transitions["lab_time"], utc=True)
        - pd.to_datetime(transitions["previous_lab_time"], utc=True)
    ).dt.total_seconds() / 3600.0
    result = pd.DataFrame(columns, index=transitions.index)
    return result, list(result.columns)


def _fit_predict(history: pd.DataFrame, query: pd.Series, feature_columns: list[str]) -> tuple[float, str]:
    if len(history) < MIN_HISTORY:
        return float("nan"), "BLOCKED_INSUFFICIENT_TRANSITION_HISTORY"
    valid = history.loc[history["delta_MNY"].notna()].copy()
    if len(valid) < MIN_HISTORY:
        return float("nan"), "BLOCKED_INSUFFICIENT_TRANSITION_HISTORY"
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RobustScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )
    model.fit(valid[feature_columns], valid["delta_MNY"])
    return float(model.predict(pd.DataFrame([query[feature_columns].to_dict()]))[0]), "MODEL_PATH_OK"


def _adapter(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[frame["RegularizedPrediction"].notna()].copy()
    output["Candidate"] = "M1_REGULARIZED_TRANSITION_RESIDUAL_V1"
    output["ActualMNY"] = output["MNY"]
    output["RawPrediction"] = output["RegularizedPrediction"]
    return output


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in frame.loc[frame["transition_available"]].groupby("split", sort=True):
        for candidate, column in [
            ("CARRY_FORWARD_PREVIOUS_MNY", "CarryForwardPrediction"),
            ("REGULARIZED_TRANSITION_RESIDUAL", "RegularizedPrediction"),
        ]:
            valid = group.loc[group[column].notna()]
            error = valid[column] - valid["MNY"]
            rows.append(
                {
                    "split": split,
                    "candidate": candidate,
                    "transition_groups": int(len(valid)),
                    "mae": float(error.abs().mean()) if len(valid) else np.nan,
                    "rmse": float(np.sqrt(np.mean(np.square(error)))) if len(valid) else np.nan,
                    "bias": float(error.mean()) if len(valid) else np.nan,
                    "prediction_to_actual_std_ratio": (
                        float(valid[column].std(ddof=1) / valid["MNY"].std(ddof=1))
                        if len(valid) > 1 and valid["MNY"].std(ddof=1) > 0 else np.nan
                    ),
                    "model_path_ok_rate": float((valid["regularized_status"] == "MODEL_PATH_OK").mean())
                    if candidate == "REGULARIZED_TRANSITION_RESIDUAL" and len(valid) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE, low_memory=False)
    groups = _group_frame(raw)
    transitions = _build_transitions(groups)
    feature_frame, feature_columns = _feature_frame(transitions)
    # _build_transitions already contains raw delta columns.  Replace only
    # the model-ready numeric view so sklearn receives unique column names.
    transitions = transitions.drop(columns=[column for column in feature_columns if column in transitions.columns])
    transitions = pd.concat([transitions, feature_frame], axis=1)
    transitions = transitions.sort_values(["lab_time", "LabelGroupID"], kind="stable").reset_index(drop=True)
    predictions: list[dict[str, object]] = []
    for index, query in transitions.iterrows():
        if not bool(query["transition_available"]):
            predictions.append({
                "RegularizedDeltaPrediction": np.nan,
                "regularized_status": "BLOCKED_NO_PRIOR_SAME_ORDER_LAB",
                "regularized_history_count": 0,
            })
            continue
        history = transitions.iloc[:index].loc[
            transitions.iloc[:index]["transition_available"]
            & transitions.iloc[:index]["lab_time"].lt(query["lab_time"])
        ].copy()
        delta, status = _fit_predict(history, query, feature_columns)
        predictions.append({
            "RegularizedDeltaPrediction": delta,
            "regularized_status": status,
            "regularized_history_count": int(len(history)),
        })
    result = pd.concat([transitions, pd.DataFrame(predictions)], axis=1)
    result["CarryForwardPrediction"] = result["previous_MNY"]
    result["RegularizedPrediction"] = result["previous_MNY"] + result["RegularizedDeltaPrediction"]
    result["RegularizedDeltaError"] = result["RegularizedDeltaPrediction"] - result["delta_MNY"]
    result.to_csv(OUTPUT / "m1_regularized_transition_residual_predictions.csv", index=False)
    _adapter(result).to_csv(OUTPUT / "m1_regularized_transition_residual_acceptance_input.csv", index=False)
    summary = _summary(result)
    summary.to_csv(OUTPUT / "m1_regularized_transition_residual_summary.csv", index=False)
    metadata = {
        "contract": "M1_REGULARIZED_TRANSITION_PROCESS_RESIDUAL_V1",
        "scope": "M1_ONLY",
        "source": str(SOURCE),
        "source_sha256": _hash_file(SOURCE),
        "feature_columns": feature_columns,
        "ridge_alpha": RIDGE_ALPHA,
        "min_history": MIN_HISTORY,
        "temporal_rule": "history transitions strictly earlier than current transition lab_time",
        "uses_current_lab_value": False,
        "uses_future_group": False,
        "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "stage3_calibration_enabled": False,
        "automatic_control": False,
        "all_groups": int(len(result)),
        "transition_available_groups": int(result["transition_available"].sum()),
        "model_path_ok_groups": int((result["regularized_status"] == "MODEL_PATH_OK").sum()),
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (OUTPUT / "m1_regularized_transition_residual_metadata_v1.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nStatus:")
    print(result["regularized_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
