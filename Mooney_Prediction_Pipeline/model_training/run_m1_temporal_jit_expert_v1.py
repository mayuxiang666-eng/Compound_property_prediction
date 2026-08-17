"""Run a strict M1 temporal JIT local expert diagnostic.

The expert is intentionally a transparent challenger, not a production model.
For every scored Lab group it uses only earlier Lab groups, with an explicit
scope hierarchy and robust distance.  It emits a point estimate, a neighbor
interval, and an OOD diagnostic.  No online calibration or control action is
performed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    os.environ.get(
        "M1_JIT_GROUP_SOURCE",
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
        "M1_JIT_OUTPUT",
        str(
            ROOT
            / "reports"
            / "v371_challenger"
            / "global_rds_active_v1"
            / "m1_temporal_jit_v1"
        ),
    )
)

K_NEIGHBORS = 15
MIN_EXACT = 8
MIN_RECIPE = 8
MIN_ROUTE = 12

# Kept deliberately small and interpretable.  All are available after the
# completed batch and before the delayed Lab result is consumed.
FEATURES = [
    "Target_Temperature",
    "silica_phr",
    "weight_pct_oil",
    "weight_pct_silica",
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
    "phys_temp_integral",
    "phys_shear_history_total",
    "pid_actual_total_duration_s",
]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_frame(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "LabelGroupID", "OrderID", "CompoundName", "RecipeID", "ProcessVersion",
        "MixerLine", "material_system", "phase_route", "TemplateKey", "MNY",
        "lab_time", "split",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"JIT source missing columns: {missing}")
    frame = raw.copy()
    frame = frame.loc[frame["CompoundName"].astype(str).str.startswith("M1-")].copy()
    frame["lab_time"] = pd.to_datetime(frame["lab_time"], errors="coerce", utc=True)
    frame["MNY"] = pd.to_numeric(frame["MNY"], errors="coerce")
    frame = frame.dropna(subset=["LabelGroupID", "lab_time", "MNY"])
    numeric = [feature for feature in FEATURES if feature in frame.columns]
    if len(numeric) < 12:
        raise ValueError(f"Too few JIT features present: {len(numeric)}")
    # The snapshot is already group-level.  Defensive deduplication ensures a
    # Lab label is never given extra weight if an upstream export is repeated.
    conflicts = frame.groupby("LabelGroupID")["MNY"].nunique(dropna=True)
    if (conflicts > 1).any():
        raise ValueError("Conflicting MNY values inside a LabelGroupID")
    return frame.drop_duplicates("LabelGroupID", keep="first").reset_index(drop=True)


def _scope_candidates(query: pd.Series, history: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    exact = history.loc[
        history["RecipeID"].eq(query["RecipeID"])
        & history["TemplateKey"].eq(query["TemplateKey"])
        & history["MixerLine"].eq(query["MixerLine"])
        & history["phase_route"].eq(query["phase_route"])
    ]
    if len(exact) >= MIN_EXACT:
        return [("RECIPE_TEMPLATE_MIXER_ROUTE", exact)]
    recipe = history.loc[history["RecipeID"].eq(query["RecipeID"])]
    if len(recipe) >= MIN_RECIPE:
        return [("RECIPE", recipe)]
    route = history.loc[
        history["material_system"].eq(query["material_system"])
        & history["phase_route"].eq(query["phase_route"])
        & history["MixerLine"].eq(query["MixerLine"])
    ]
    if len(route) >= MIN_ROUTE:
        return [("MATERIAL_ROUTE_MIXER", route)]
    material = history.loc[history["material_system"].eq(query["material_system"])]
    if len(material) >= MIN_ROUTE:
        return [("MATERIAL_SYSTEM", material)]
    return []


def _robust_distance(query: pd.Series, history: pd.DataFrame, features: list[str]) -> np.ndarray:
    matrix = history[features].apply(pd.to_numeric, errors="coerce")
    q = pd.to_numeric(query[features], errors="coerce")
    med = matrix.median(axis=0)
    q = q.fillna(med)
    matrix = matrix.fillna(med)
    scale = (matrix.quantile(0.75) - matrix.quantile(0.25)).replace(0, np.nan)
    scale = scale.fillna(matrix.std(ddof=1)).replace(0, 1.0).fillna(1.0)
    z = (matrix - med) / scale
    qz = (q - med) / scale
    # Mean absolute robust distance is easier to audit than a black-box kernel.
    return np.mean(np.abs(z.to_numpy(dtype=float) - qz.to_numpy(dtype=float)), axis=1)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(values[min(np.searchsorted(cumulative, quantile), len(values) - 1)])


def _predict_one(query: pd.Series, history: pd.DataFrame, features: list[str]) -> dict[str, object]:
    scopes = _scope_candidates(query, history)
    if not scopes:
        return {
            "JITPrediction": float("nan"),
            "JITP10": float("nan"),
            "JITP90": float("nan"),
            "neighbor_count": 0,
            "neighbor_scope": "NO_HISTORY_SCOPE",
            "nearest_distance": float("nan"),
            "ood_status": "OOD_NO_HISTORY_SCOPE",
        }
    scope, pool = scopes[0]
    distances = _robust_distance(query, pool, features)
    order = np.argsort(distances)[: min(K_NEIGHBORS, len(pool))]
    selected = pool.iloc[order]
    selected_distances = np.maximum(distances[order], 1e-6)
    weights = 1.0 / selected_distances
    values = pd.to_numeric(selected["MNY"], errors="coerce").to_numpy(dtype=float)
    prediction = float(np.average(values, weights=weights))
    p10 = _weighted_quantile(values, weights, 0.10)
    p90 = _weighted_quantile(values, weights, 0.90)
    nearest = float(selected_distances.min())
    # OOD is deliberately conservative: a distant nearest neighbor or a
    # fallback scope cannot produce a recommendation-grade result.
    ood = "IN_DOMAIN" if nearest <= 1.5 and len(selected) >= K_NEIGHBORS else "OOD_DISTANCE_OR_SPARSE"
    return {
        "JITPrediction": prediction,
        "JITP10": p10,
        "JITP90": p90,
        "neighbor_count": int(len(selected)),
        "neighbor_scope": scope,
        "nearest_distance": nearest,
        "ood_status": ood,
    }


def _metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, compound), group in frame.groupby(["split", "CompoundName"], sort=True):
        error = group["JITPrediction"] - group["MNY"]
        valid = error.notna()
        if not valid.any():
            continue
        actual = group.loc[valid, "MNY"]
        prediction = group.loc[valid, "JITPrediction"]
        actual_std = float(actual.std(ddof=1)) if len(actual) > 1 else float("nan")
        prediction_std = float(prediction.std(ddof=1)) if len(prediction) > 1 else float("nan")
        rows.append(
            {
                "split": split,
                "CompoundName": compound,
                "groups": int(valid.sum()),
                "mae": float(error.loc[valid].abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error.loc[valid])))),
                "bias": float(error.loc[valid].mean()),
                "prediction_to_actual_std_ratio": (
                    prediction_std / actual_std
                    if np.isfinite(actual_std) and actual_std > 0 else np.nan
                ),
                "coverage_rate": float(valid.mean()),
                "in_domain_rate": float((group.loc[valid, "ood_status"] == "IN_DOMAIN").mean()),
                "median_neighbor_count": float(group.loc[valid, "neighbor_count"].median()),
                "median_interval_width": float((group.loc[valid, "JITP90"] - group.loc[valid, "JITP10"]).median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE, low_memory=False)
    groups = _group_frame(raw)
    features = [feature for feature in FEATURES if feature in groups.columns]
    groups = groups.sort_values(["lab_time", "LabelGroupID"], kind="stable").reset_index(drop=True)
    predictions = []
    for index, query in groups.iterrows():
        # Strict temporal eligibility.  The current group and all future Lab
        # groups are excluded, even if their process features are available.
        history = groups.iloc[:index].loc[groups.iloc[:index]["lab_time"] < query["lab_time"]].copy()
        result = _predict_one(query, history, features)
        predictions.append(result)
    result = pd.concat([groups, pd.DataFrame(predictions)], axis=1)
    result["JITError"] = result["JITPrediction"] - result["MNY"]
    # Adapter columns keep the established acceptance contract reusable.  The
    # source remains one row per Lab group, so the audit cannot double-count a
    # shared Lab value.
    result["Candidate"] = "M1_TEMPORAL_JIT_LOCAL_EXPERT_V1"
    result["ActualMNY"] = result["MNY"]
    result["RawPrediction"] = result["JITPrediction"]
    result.to_csv(OUTPUT / "m1_temporal_jit_group_predictions.csv", index=False)
    result.loc[result["RawPrediction"].notna()].to_csv(
        OUTPUT / "m1_temporal_jit_acceptance_input.csv", index=False
    )
    metrics = _metrics(result)
    metrics.to_csv(OUTPUT / "m1_temporal_jit_metrics.csv", index=False)
    metadata = {
        "contract": "M1_TEMPORAL_JIT_LOCAL_EXPERT_V1",
        "scope": "M1_ONLY",
        "source": str(SOURCE),
        "source_sha256": _hash_file(SOURCE),
        "features": features,
        "k_neighbors": K_NEIGHBORS,
        "min_exact": MIN_EXACT,
        "min_recipe": MIN_RECIPE,
        "min_route": MIN_ROUTE,
        "temporal_rule": "only lab_time strictly earlier than current query",
        "uses_lab_value_for_current_query": False,
        "uses_future_group": False,
        "uses_local_mms": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "stage3_calibration_enabled": False,
        "automatic_control": False,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (OUTPUT / "m1_temporal_jit_metadata_v1.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
