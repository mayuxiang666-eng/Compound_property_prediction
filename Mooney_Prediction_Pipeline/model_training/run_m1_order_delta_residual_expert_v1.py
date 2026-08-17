"""Run a causal same-order Delta-MNY process residual expert.

This is a diagnostic challenger.  A transition is eligible only when the
immediately preceding Lab group belongs to the same OrderID and has an
earlier Lab timestamp.  Transition neighbors are selected from historical
transitions only; current and future Lab values are never used.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.model_training.run_m1_temporal_jit_expert_v1 import (
    _group_frame,
    _scope_candidates,
    _weighted_quantile,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    os.environ.get(
        "M1_DELTA_SOURCE",
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
        "M1_DELTA_OUTPUT",
        str(
            ROOT
            / "reports"
            / "v371_challenger"
            / "global_rds_active_v1"
            / "m1_order_delta_residual_v1"
        ),
    )
)

PROCESS_FEATURES = [
    "Stage2_DryMixing_Duration",
    "Stage2_DryMixing_temp_Mean",
    "Stage2_DryMixing_Torque_Integral",
    "Stage3_OilLoading_Duration",
    "Stage3_OilLoading_temp_Mean",
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
]
K_NEIGHBORS = 12
MIN_EXACT = 5
MIN_RECIPE = 5
MIN_ROUTE = 8


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_transitions(groups: pd.DataFrame) -> pd.DataFrame:
    ordered = groups.sort_values(["OrderID", "lab_time", "LabelGroupID"], kind="stable").copy()
    previous = ordered.groupby("OrderID", sort=False)
    ordered["previous_LabelGroupID"] = previous["LabelGroupID"].shift(1)
    ordered["previous_lab_time"] = previous["lab_time"].shift(1)
    ordered["previous_MNY"] = pd.to_numeric(previous["MNY"].shift(1), errors="coerce")
    for column in ["RecipeID", "TemplateKey", "ProcessVersion", "MixerLine", "phase_route"]:
        ordered[f"previous_{column}"] = previous[column].shift(1)
    for feature in PROCESS_FEATURES:
        current = pd.to_numeric(ordered[feature], errors="coerce")
        prior = pd.to_numeric(previous[feature].shift(1), errors="coerce")
        ordered[f"delta__{feature}"] = current - prior
    ordered["delta_MNY"] = ordered["MNY"] - ordered["previous_MNY"]
    ordered["prior_same_recipe"] = ordered["RecipeID"].eq(ordered["previous_RecipeID"])
    ordered["prior_same_template"] = ordered["TemplateKey"].eq(ordered["previous_TemplateKey"])
    ordered["transition_available"] = (
        ordered["previous_LabelGroupID"].notna()
        & ordered["previous_lab_time"].lt(ordered["lab_time"])
        & ordered["previous_MNY"].notna()
        & ordered["prior_same_recipe"]
        & ordered["prior_same_template"]
    )
    return ordered


def _distance(query: pd.Series, history: pd.DataFrame, features: list[str]) -> np.ndarray:
    matrix = history[features].apply(pd.to_numeric, errors="coerce")
    q = pd.to_numeric(query[features], errors="coerce")
    med = matrix.median(axis=0)
    matrix = matrix.fillna(med)
    q = q.fillna(med)
    scale = (matrix.quantile(0.75) - matrix.quantile(0.25)).replace(0, np.nan)
    scale = scale.fillna(matrix.std(ddof=1)).replace(0, 1.0).fillna(1.0)
    z = (matrix - med) / scale
    qz = (q - med) / scale
    return np.mean(np.abs(z.to_numpy(dtype=float) - qz.to_numpy(dtype=float)), axis=1)


def _transition_scope(query: pd.Series, history: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    # _scope_candidates operates on current-row metadata, which is exactly
    # how the transition table is represented: one row per current group.
    return _scope_candidates(query, history)


def _predict_delta(query: pd.Series, history: pd.DataFrame, features: list[str]) -> dict[str, object]:
    scopes = _transition_scope(query, history)
    if not scopes:
        return {
            "DeltaPrediction": np.nan,
            "DeltaP10": np.nan,
            "DeltaP90": np.nan,
            "delta_neighbor_count": 0,
            "delta_neighbor_scope": "NO_HISTORY_SCOPE",
            "delta_nearest_distance": np.nan,
            "delta_ood_status": "OOD_NO_HISTORY_SCOPE",
        }
    scope, pool = scopes[0]
    distance = _distance(query, pool, features)
    order = np.argsort(distance)[: min(K_NEIGHBORS, len(pool))]
    selected = pool.iloc[order]
    selected_distance = np.maximum(distance[order], 1e-6)
    weights = 1.0 / selected_distance
    values = pd.to_numeric(selected["delta_MNY"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return {
            "DeltaPrediction": np.nan,
            "DeltaP10": np.nan,
            "DeltaP90": np.nan,
            "delta_neighbor_count": 0,
            "delta_neighbor_scope": "NO_VALID_DELTA_HISTORY",
            "delta_nearest_distance": np.nan,
            "delta_ood_status": "OOD_NO_VALID_DELTA_HISTORY",
        }
    values = values[valid]
    weights = weights[valid]
    nearest = float(selected_distance[valid].min())
    return {
        "DeltaPrediction": float(np.average(values, weights=weights)),
        "DeltaP10": _weighted_quantile(values, weights, 0.10),
        "DeltaP90": _weighted_quantile(values, weights, 0.90),
        "delta_neighbor_count": int(len(values)),
        "delta_neighbor_scope": scope,
        "delta_nearest_distance": nearest,
        "delta_ood_status": (
            "IN_DOMAIN" if nearest <= 1.5 and len(values) >= K_NEIGHBORS
            else "OOD_DISTANCE_OR_SPARSE"
        ),
    }


def _adapter(frame: pd.DataFrame, prediction_column: str, candidate: str) -> pd.DataFrame:
    output = frame.loc[frame[prediction_column].notna()].copy()
    output["Candidate"] = candidate
    output["ActualMNY"] = output["MNY"]
    output["RawPrediction"] = output[prediction_column]
    return output


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = frame.loc[frame["transition_available"]].copy()
    for split, group in valid.groupby("split", sort=True):
        for candidate, column in [
            ("CARRY_FORWARD_PREVIOUS_MNY", "CarryForwardPrediction"),
            ("SAME_ORDER_DELTA_JIT", "DeltaJITPrediction"),
            ("LEVEL_DELTA_BLEND_50_50", "HybridPrediction"),
        ]:
            values = group.loc[group[column].notna()]
            error = values[column] - values["MNY"]
            rows.append(
                {
                    "split": split,
                    "candidate": candidate,
                    "transition_groups": int(len(values)),
                    "mae": float(error.abs().mean()) if len(values) else np.nan,
                    "rmse": float(np.sqrt(np.mean(np.square(error)))) if len(values) else np.nan,
                    "bias": float(error.mean()) if len(values) else np.nan,
                    "prediction_to_actual_std_ratio": (
                        float(values[column].std(ddof=1) / values["MNY"].std(ddof=1))
                        if len(values) > 1 and values["MNY"].std(ddof=1) > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE, low_memory=False)
    groups = _group_frame(raw)
    process = [feature for feature in PROCESS_FEATURES if feature in groups.columns]
    transitions = _build_transitions(groups)
    delta_features = [f"delta__{feature}" for feature in process]
    # Add a small absolute process context to distinguish identical changes at
    # different operating points without allowing labels into the features.
    absolute_features = [
        feature for feature in process
        if feature in {"Stage4_WetMixing_Duration", "Stage4_WetMixing_temp_Mean", "Stage5_PID_Duration", "Stage6_BottomMixing_Duration", "phys_discharge_temp"}
    ]
    distance_features = delta_features + absolute_features
    transitions = transitions.sort_values(["lab_time", "LabelGroupID"], kind="stable").reset_index(drop=True)
    predictions = []
    for index, query in transitions.iterrows():
        if not bool(query["transition_available"]):
            predictions.append({
                "DeltaPrediction": np.nan, "DeltaP10": np.nan, "DeltaP90": np.nan,
                "delta_neighbor_count": 0, "delta_neighbor_scope": "BLOCKED_NO_PRIOR_SAME_ORDER_LAB",
                "delta_nearest_distance": np.nan, "delta_ood_status": "BLOCKED_NO_PRIOR_SAME_ORDER_LAB",
            })
            continue
        history = transitions.iloc[:index].loc[
            transitions.iloc[:index]["transition_available"]
            & transitions.iloc[:index]["lab_time"].lt(query["lab_time"])
        ].copy()
        predictions.append(_predict_delta(query, history, distance_features))
    result = pd.concat([transitions, pd.DataFrame(predictions)], axis=1)
    result["CarryForwardPrediction"] = result["previous_MNY"]
    result["DeltaJITPrediction"] = result["previous_MNY"] + result["DeltaPrediction"]
    # A fixed, pre-declared blend is diagnostic only; it is not selected on OOT.
    result["HybridPrediction"] = 0.5 * result["DeltaJITPrediction"] + 0.5 * result["previous_MNY"]
    result["DeltaError"] = result["DeltaPrediction"] - result["delta_MNY"]
    result.to_csv(OUTPUT / "m1_order_delta_residual_predictions.csv", index=False)
    for candidate, column in [
        ("CARRY_FORWARD_PREVIOUS_MNY", "CarryForwardPrediction"),
        ("SAME_ORDER_DELTA_JIT", "DeltaJITPrediction"),
        ("LEVEL_DELTA_BLEND_50_50", "HybridPrediction"),
    ]:
        _adapter(result, column, candidate).to_csv(
            OUTPUT / f"{candidate.lower()}_acceptance_input.csv", index=False
        )
    summary = _summary(result)
    summary.to_csv(OUTPUT / "m1_order_delta_residual_summary.csv", index=False)
    metadata = {
        "contract": "M1_SAME_ORDER_DELTA_PROCESS_RESIDUAL_EXPERT_V1",
        "scope": "M1_ONLY",
        "source": str(SOURCE),
        "source_sha256": _hash_file(SOURCE),
        "process_features": process,
        "distance_features": distance_features,
        "k_neighbors": K_NEIGHBORS,
        "min_exact": MIN_EXACT,
        "min_recipe": MIN_RECIPE,
        "min_route": MIN_ROUTE,
        "transition_rule": "immediate prior Lab group in same OrderID, same RecipeID and TemplateKey, strictly earlier Lab time",
        "history_rule": "only earlier transition current lab_time",
        "uses_current_lab_value": False,
        "uses_future_group": False,
        "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "stage3_calibration_enabled": False,
        "automatic_control": False,
        "transition_available_groups": int(result["transition_available"].sum()),
        "all_groups": int(len(result)),
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (OUTPUT / "m1_order_delta_residual_metadata_v1.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nTransition coverage:")
    print(result.groupby("split")["transition_available"].agg(["count", "sum", "mean"]).to_string())


if __name__ == "__main__":
    main()
