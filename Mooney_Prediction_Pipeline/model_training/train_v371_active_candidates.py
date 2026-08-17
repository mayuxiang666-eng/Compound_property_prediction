"""Train leak-free V3.7.1 candidates on corrected Active-order Phase history.

The laboratory contract is one MNY label per OrderID+PalletID group.  Models
still predict individual batches, while training weights and evaluation give
each laboratory group one effective observation.  Orders are assigned
atomically to chronological train/validation/OOT partitions.

This experiment never writes production model packages or prediction tables.
Artifacts are shadow-only and automatic control remains disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.mooney_predict_service.phase_group_inference import prepare_phase_batches
from Mooney_Prediction_Pipeline.model_training.hybrid_unified_model import HybridUnifiedMooneyModel


PILOT_PATTERNS = {
    "overall": None,
    "b00458": "B00458",
    "t09170": "T09170",
}


def load_schema(path: Path, drop_features: tuple[str, ...] = ()) -> dict[str, Any]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    result = deepcopy(schema)
    dropped = set(drop_features)
    result["s1_features"] = [c for c in result["s1_features"] if c not in dropped]
    result["s2_features"] = [c for c in result["s2_features"] if c not in dropped]
    result["dropped_features"] = sorted(dropped)
    return result


def assign_order_time_splits(source: pd.DataFrame) -> pd.DataFrame:
    """Assign complete orders to 70/15/15 chronological partitions."""
    order_times = source.groupby("OrderID", as_index=False).agg(
        OrderTime=("OrderStartTime", "max"),
        GroupCount=("LabelGroupID", "nunique"),
    ).sort_values(["OrderTime", "OrderID"], kind="stable").reset_index(drop=True)
    if len(order_times) < 3:
        raise ValueError("At least three Active orders are required for chronological splitting")
    cumulative = order_times["GroupCount"].cumsum()
    total = float(order_times["GroupCount"].sum())
    order_times["Split"] = np.select(
        [cumulative <= 0.70 * total, cumulative <= 0.85 * total],
        ["train", "validation"],
        default="oot_test",
    )
    if set(order_times["Split"]) != {"train", "validation", "oot_test"}:
        raise ValueError("Chronological split did not produce all three partitions")
    return order_times


def build_candidate_frame(
    source: pd.DataFrame,
    schema: dict[str, Any],
    order_splits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepared = prepare_phase_batches(source, schema)
    quality = prepared.quality.copy()
    quality["BatchAccepted"] = quality["FeatureQualityStatus"].eq("MODEL_INPUT_OK")
    quality["GroupComplete"] = quality.groupby("LabelGroupID")["BatchAccepted"].transform("all")

    group_quality = quality.groupby("LabelGroupID", as_index=False).agg(
        OrderID=("OrderID", "first"),
        PalletID=("PalletID", "first"),
        BatchCount=("BatchNumber", "size"),
        AcceptedBatchCount=("BatchAccepted", "sum"),
        GroupComplete=("GroupComplete", "first"),
    )
    labels = source.groupby("LabelGroupID", as_index=False).agg(
        MNY=("MNY", "median"),
        MNYMin=("MNY", "min"),
        MNYMax=("MNY", "max"),
        CompoundName=("CompoundName", "first"),
        OrderStartTime=("OrderStartTime", "max"),
    )
    labels["MNYConflict"] = (labels["MNYMax"] - labels["MNYMin"]).abs().gt(1e-9)
    group_quality = group_quality.merge(labels, on="LabelGroupID", how="left", validate="one_to_one")
    group_quality = group_quality.merge(
        order_splits[["OrderID", "Split"]], on="OrderID", how="left", validate="many_to_one"
    )

    eligible_ids = set(group_quality.loc[
        group_quality["GroupComplete"] & ~group_quality["MNYConflict"] & group_quality["MNY"].notna(),
        "LabelGroupID",
    ])
    features = prepared.features.loc[prepared.features["LabelGroupID"].isin(eligible_ids)].copy()
    target = source[["OrderID", "BatchNumber", "MNY"]].copy()
    features = features.merge(target, on=["OrderID", "BatchNumber"], how="left", validate="one_to_one")
    features = features.merge(
        order_splits[["OrderID", "Split"]], on="OrderID", how="left", validate="many_to_one"
    )
    features["_label_group_id"] = features["LabelGroupID"]
    features["_label_group_size"] = features.groupby("_label_group_id")["_label_group_id"].transform("size")
    features["_sample_weight"] = 1.0 / features["_label_group_size"]
    return features, quality, group_quality


def prediction_rows(model: HybridUnifiedMooneyModel, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    final, stage1, stage1b, stage2 = model.predict(frame, cluster_col="material_system")
    result = frame[[
        "OrderID", "BatchNumber", "PalletID", "LabelGroupID", "CompoundName",
        "OrderStartTime", "Split", "MNY", "material_system",
    ]].copy()
    result["Stage1Pred"] = stage1
    result["Stage1bBias"] = stage1b
    result["Stage2Delta"] = stage2
    result["RawPrediction"] = final
    return result


def aggregate_groups(batch: pd.DataFrame) -> pd.DataFrame:
    if batch.empty:
        return pd.DataFrame()
    return batch.groupby("LabelGroupID", as_index=False).agg(
        OrderID=("OrderID", "first"),
        PalletID=("PalletID", "first"),
        CompoundName=("CompoundName", "first"),
        OrderStartTime=("OrderStartTime", "max"),
        Split=("Split", "first"),
        BatchCount=("BatchNumber", "size"),
        ActualMNY=("MNY", "median"),
        ActualMNYMin=("MNY", "min"),
        ActualMNYMax=("MNY", "max"),
        GroupRawPrediction=("RawPrediction", "mean"),
        BatchPredictionStd=("RawPrediction", "std"),
    ).assign(BatchPredictionStd=lambda x: x["BatchPredictionStd"].fillna(0.0))


def metric_block(groups: pd.DataFrame) -> dict[str, Any]:
    if groups.empty:
        return {"groups": 0, "mae": None, "rmse": None, "bias": None, "r2": None,
                "within_2_mny_rate": None, "within_3_mny_rate": None}
    y = groups["ActualMNY"].to_numpy(dtype=float)
    p = groups["GroupRawPrediction"].to_numpy(dtype=float)
    error = p - y
    return {
        "groups": int(len(groups)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "bias": float(np.mean(error)),
        "r2": float(r2_score(y, p)) if len(groups) > 1 else None,
        "within_2_mny_rate": float(np.mean(np.abs(error) <= 2.0)),
        "within_3_mny_rate": float(np.mean(np.abs(error) <= 3.0)),
    }


def scoped_metrics(groups: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = groups.loc[groups["Split"].eq(split)].copy()
    result: dict[str, Any] = {}
    for name, pattern in PILOT_PATTERNS.items():
        subset = selected if pattern is None else selected.loc[
            selected["CompoundName"].astype(str).str.upper().str.contains(pattern, regex=False)
        ]
        result[name] = metric_block(subset)
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def train_candidate(
    name: str,
    source: pd.DataFrame,
    schema: dict[str, Any],
    order_splits: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    candidate_dir = output_dir / name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    features, quality, groups = build_candidate_frame(source, schema, order_splits)
    train = features.loc[features["Split"].eq("train")].copy()
    validation = features.loc[features["Split"].eq("validation")].copy()
    oot = features.loc[features["Split"].eq("oot_test")].copy()
    if train.empty or validation.empty or oot.empty:
        raise ValueError(f"{name} has an empty chronological partition")

    s1 = list(schema["s1_features"])
    s2 = list(schema["s2_features"])
    model = HybridUnifiedMooneyModel(
        use_material_route_matrix=True,
        use_silica_subsystem=True,
        use_cb_subsystem=True,
    )
    model.fit(train, s1, s2, target_col="MNY", cluster_col="material_system")

    batches = pd.concat(
        [prediction_rows(model, frame) for frame in (train, validation, oot)],
        ignore_index=True,
    )
    group_predictions = aggregate_groups(batches)
    artifact = candidate_dir / "hybrid_model_shadow.joblib"
    joblib.dump(model, artifact)
    (candidate_dir / "feature_metadata_shadow.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    quality.to_csv(candidate_dir / "feature_quality_audit.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(candidate_dir / "group_eligibility_and_split.csv", index=False, encoding="utf-8-sig")
    batches.to_csv(candidate_dir / "batch_predictions.csv", index=False, encoding="utf-8-sig")
    group_predictions.to_csv(candidate_dir / "group_predictions.csv", index=False, encoding="utf-8-sig")

    eligible = groups.loc[groups["GroupComplete"] & ~groups["MNYConflict"] & groups["MNY"].notna()]
    summary = {
        "candidate": name,
        "data_contract": "CORRECTED_ACTIVE_PHASE_NO_CROSS_BATCH_CURVE_FILL",
        "label_contract": "BATCH_PREDICT_THEN_ORDERID_PALLETID_MEAN",
        "split_contract": "ORDER_ATOMIC_CHRONOLOGICAL_70_15_15_BY_GROUP_CUMULATIVE_COUNT",
        "stage3_calibration": "DISABLED_FOR_BASE_MODEL_COMPARISON",
        "dropped_features": schema.get("dropped_features", []),
        "feature_counts": {"stage1": len(s1), "stage2": len(s2)},
        "source_batches": int(len(source)),
        "accepted_batches": int(len(features)),
        "eligible_groups": int(len(eligible)),
        "partition_batches": {k: int((features["Split"] == k).sum()) for k in ("train", "validation", "oot_test")},
        "partition_groups": {k: int(features.loc[features["Split"].eq(k), "LabelGroupID"].nunique()) for k in ("train", "validation", "oot_test")},
        "metrics": {
            split: scoped_metrics(group_predictions, split)
            for split in ("train", "validation", "oot_test")
        },
        "lab_used_as_feature": False,
        "automatic_control": "DISABLED",
        "recommendation_status": "NOT_GENERATED_MODEL_VALIDATION_ONLY",
        "model_sha256": file_sha256(artifact),
    }
    (candidate_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    base = ROOT / "reports" / "corrected_phase_history"
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=base / "phase_corrected_v371_active.csv")
    parser.add_argument(
        "--schema", type=Path,
        default=ROOT / "Mooney_Prediction_Pipeline" / "models" / "v37_production_model_package" / "feature_metadata.json",
    )
    parser.add_argument("--output-dir", type=Path, default=base / "v371_active_candidates_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.source, low_memory=False)
    source["OrderID"] = source["OrderID"].astype(str).str.strip()
    source["PalletID"] = source["PalletID"].astype(str).str.strip()
    source["BatchNumber"] = pd.to_numeric(source["BatchNumber"], errors="raise").astype(int)
    source["MNY"] = pd.to_numeric(source["MNY"], errors="coerce")
    source["OrderStartTime"] = pd.to_datetime(source["OrderStartTime"], errors="raise")
    source["LabelGroupID"] = source["OrderID"] + "|" + source["PalletID"]
    source = source.dropna(subset=["MNY"]).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    order_splits = assign_order_time_splits(source)
    order_splits.to_csv(args.output_dir / "order_time_splits.csv", index=False, encoding="utf-8-sig")

    strict = train_candidate(
        "strict_full_qm", source, load_schema(args.schema), order_splits, args.output_dir
    )
    operational = train_candidate(
        "operational_core_no_rubber_qm",
        source,
        load_schema(args.schema, ("supplier_rubber_viscosity_avg",)),
        order_splits,
        args.output_dir,
    )
    combined = {
        "experiment": "V3.7.1-CORRECTED-ACTIVE-CANDIDATES-V1",
        "strict_full_qm": strict,
        "operational_core_no_rubber_qm": operational,
        "production_release": False,
        "database_writes": False,
        "automatic_control": "DISABLED",
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(combined, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
