"""V3.7 shadow runner using the original Phase feature contract.

This entry point intentionally never calls ``query_latest_order_rows`` or
``extract_features_for_row``.  It reads a versioned Phase feature store,
predicts completed batches, and then computes ``OrderID + PalletID`` means.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .model_registry import load_active_package
from .phase_feature_store import load_phase_feature_store
from .phase_group_contract import PhaseFeatureContractError
from .phase_group_inference import prepare_phase_batches, predict_batches_and_groups, quality_summary


LOG = logging.getLogger("mooney-v37-phase-group-shadow")


def _load_explicit_package(package_dir: Path) -> tuple[object, dict, dict]:
    model_path = package_dir / "hybrid_model.joblib"
    metadata_path = package_dir / "feature_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete V3.7 package: {package_dir}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest().upper()
    meta = {
        "ModelID": int(os.environ.get("MOONEY_MODEL_ID", "0")),
        "ModelVersion": str(schema.get("model_version", "V3.7 UNKNOWN")),
        "PackagePath": str(package_dir.resolve()),
        "PackageHash": digest,
        "ReleaseType": "SHADOW",
    }
    return joblib.load(model_path), schema, meta


def load_model(write: bool) -> tuple[object, dict, dict]:
    explicit = os.environ.get("MOONEY_MODEL_PACKAGE_DIR", "").strip()
    if explicit:
        model, schema, meta = _load_explicit_package(Path(explicit))
        if write and int(meta["ModelID"]) <= 0:
            raise RuntimeError("MOONEY_MODEL_ID is required for database writes with an explicit package")
        return model, schema, meta
    return load_active_package()


def _complete_group_outputs(
    quality: pd.DataFrame,
    group_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = quality.groupby("LabelGroupID", as_index=False).agg(
        ExpectedBatchCount=("BatchNumber", "size"),
        AcceptedBatchCount=("FeatureQualityStatus", lambda values: int((values == "MODEL_INPUT_OK").sum())),
    )
    group = group_predictions.merge(expected, on="LabelGroupID", how="right", validate="one_to_one")
    complete = group["ExpectedBatchCount"] == group["AcceptedBatchCount"]
    blocked = group.loc[~complete].copy()
    blocked["GroupPredictionStatus"] = "BLOCKED_INCOMPLETE_GROUP_FEATURES"
    valid = group.loc[complete].copy()
    if not valid.empty:
        valid["GroupPredictionStatus"] = "PREDICTED_COMPLETE_GROUP_MEAN"
    return valid, blocked


def _write_outputs(
    output_dir: Path,
    batch: pd.DataFrame,
    group: pd.DataFrame,
    blocked_group: pd.DataFrame,
    quality: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch.to_csv(output_dir / "phase_v37_batch_predictions.csv", index=False)
    group.to_csv(output_dir / "phase_v37_complete_group_predictions.csv", index=False)
    blocked_group.to_csv(output_dir / "phase_v37_blocked_groups.csv", index=False)
    quality.to_csv(output_dir / "phase_v37_feature_quality_audit.csv", index=False)
    with (output_dir / "phase_v37_run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)


def run(args: argparse.Namespace) -> dict[str, object]:
    source = args.phase_store or os.environ.get("MOONEY_PHASE_FEATURE_STORE", "").strip()
    if not source:
        raise RuntimeError("MOONEY_PHASE_FEATURE_STORE or --phase-store is required")
    frame, source_audit = load_phase_feature_store(
        source, order_id=args.order_id, compound_name=args.compound_name, since=args.since,
    )
    if frame.empty:
        raise PhaseFeatureContractError("No Phase rows remain after filters")
    model, schema, model_meta = load_model(write=args.write)
    prepared = prepare_phase_batches(frame, schema)
    predicted = predict_batches_and_groups(prepared, model, str(model_meta["ModelVersion"]))
    complete_groups, blocked_groups = _complete_group_outputs(prepared.quality, predicted.group_predictions)

    # Determinism is a runtime gate, not merely a unit test.
    repeated = predict_batches_and_groups(prepared, model, str(model_meta["ModelVersion"]))
    delta = 0.0
    if not predicted.batch_predictions.empty:
        delta = float(np.max(np.abs(
            predicted.batch_predictions["RawPrediction"].to_numpy()
            - repeated.batch_predictions["RawPrediction"].to_numpy()
        )))
    if delta > 1e-12:
        raise RuntimeError(f"V3.7 deterministic repeat gate failed: max delta={delta}")

    summary = {
        **source_audit,
        **quality_summary(prepared.quality, predicted.batch_predictions),
        "complete_group_count": int(len(complete_groups)),
        "blocked_incomplete_group_count": int(len(blocked_groups)),
        "deterministic_max_abs_prediction_delta": delta,
        "model_version": str(model_meta["ModelVersion"]),
        "model_package_hash": str(model_meta.get("PackageHash", "")),
        "model_path_status": "MODEL_PATH_OK" if len(complete_groups) else "BLOCKED_NO_COMPLETE_GROUP",
        "database_write": bool(args.write),
        "manual_review_only_issued": False,
    }

    if args.output_dir:
        _write_outputs(
            Path(args.output_dir), predicted.batch_predictions, complete_groups,
            blocked_groups, prepared.quality, summary,
        )
    if args.write:
        if os.environ.get("MOONEY_RELEASE_TYPE") != "SHADOW":
            raise RuntimeError("Database write requires MOONEY_RELEASE_TYPE=SHADOW")
        from .phase_group_writers import write_batch_rows, write_quality_rows, upsert_group_rows

        summary["quality_rows_written"] = write_quality_rows(prepared.quality)
        summary["batch_rows_written"] = write_batch_rows(
            predicted.batch_predictions, prepared.quality, int(model_meta["ModelID"]),
        )
        summary["group_rows_upserted"] = upsert_group_rows(complete_groups, int(model_meta["ModelID"]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-store")
    parser.add_argument("--order-id")
    parser.add_argument("--compound-name")
    parser.add_argument("--since")
    parser.add_argument("--output-dir")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    print(json.dumps(run(args), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
