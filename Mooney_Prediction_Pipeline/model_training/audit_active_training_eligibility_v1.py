"""Audit complete label groups and frozen-V3.7 metrics on corrected active data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _metrics(frame: pd.DataFrame) -> dict[str, object]:
    valid = frame.dropna(subset=["MNY", "GroupRawPrediction"])
    if valid.empty:
        return {"groups": 0, "mae": None, "rmse": None, "bias_pred_minus_actual": None}
    error = valid["GroupRawPrediction"] - valid["MNY"]
    return {
        "groups": int(len(valid)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias_pred_minus_actual": float(error.mean()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    source = pd.read_csv(args.source, low_memory=False)
    quality = pd.read_csv(args.quality, low_memory=False)
    predicted = pd.read_csv(args.group_predictions, low_memory=False)
    for frame in (source, quality):
        frame["OrderID"] = frame["OrderID"].astype(str).str.strip()
        frame["BatchNumber"] = pd.to_numeric(frame["BatchNumber"], errors="coerce").astype("Int64")
    joined = quality.merge(
        source[["OrderID", "BatchNumber", "CompoundName", "MNY", "OrderStartTime"]],
        on=["OrderID", "BatchNumber"], how="left", validate="one_to_one",
    )
    joined["BatchAccepted"] = joined["FeatureQualityStatus"].eq("MODEL_INPUT_OK")
    joined["GroupComplete"] = joined.groupby("LabelGroupID")["BatchAccepted"].transform("all")

    groups = joined.groupby("LabelGroupID", as_index=False).agg(
        OrderID=("OrderID", "first"), PalletID=("PalletID", "first"),
        CompoundName=("CompoundName", "first"), MNY=("MNY", "median"),
        MNYMin=("MNY", "min"), MNYMax=("MNY", "max"),
        BatchCount=("BatchNumber", "size"), AcceptedBatchCount=("BatchAccepted", "sum"),
        GroupComplete=("GroupComplete", "first"), OrderStartTime=("OrderStartTime", "first"),
    )
    groups["MNYConflict"] = (groups["MNYMax"] - groups["MNYMin"]).abs().gt(1e-9)
    groups = groups.merge(
        predicted[["LabelGroupID", "GroupRawPrediction"]],
        on="LabelGroupID", how="left", validate="one_to_one",
    )
    groups["FrozenV37Error"] = groups["GroupRawPrediction"] - groups["MNY"]

    missing_counts: dict[str, int] = {}
    for value in joined.loc[~joined["BatchAccepted"], "MissingFeatureCells"].dropna().astype(str):
        for column in value.split("|"):
            missing_counts[column] = missing_counts.get(column, 0) + 1
    missing_counts = dict(sorted(missing_counts.items(), key=lambda item: (-item[1], item[0])))

    def scope(pattern: str | None) -> dict[str, object]:
        mask = pd.Series(True, index=joined.index) if pattern is None else joined["CompoundName"].astype(str).str.upper().str.contains(pattern, regex=False)
        batch = joined.loc[mask]
        group_ids = set(batch["LabelGroupID"])
        group = groups.loc[groups["LabelGroupID"].isin(group_ids)]
        complete = group.loc[group["GroupComplete"] & ~group["MNYConflict"]]
        return {
            "batches": int(len(batch)), "accepted_batches": int(batch["BatchAccepted"].sum()),
            "label_groups": int(len(group)), "complete_label_groups": int(len(complete)),
            "complete_group_rate": float(len(complete) / max(len(group), 1)),
            "frozen_v37_metrics": _metrics(complete),
        }

    audit = {
        "contract": "CORRECTED-ACTIVE-TRAINING-ELIGIBILITY-V1",
        "overall": scope(None), "b00458": scope("B00458"), "t09170": scope("T09170"),
        "mny_conflict_groups": int(groups["MNYConflict"].sum()),
        "missing_feature_batch_counts": missing_counts,
        "pilot_minimum_complete_groups": 150,
        "b00458_threshold_met": bool(scope("B00458")["complete_label_groups"] >= 150),
        "t09170_threshold_met": bool(scope("T09170")["complete_label_groups"] >= 150),
        "lab_used_as_feature": False, "automatic_control": "DISABLED",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups.to_csv(args.output_dir / "corrected_active_group_eligibility.csv", index=False, encoding="utf-8-sig")
    joined.to_csv(args.output_dir / "corrected_active_batch_eligibility.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "corrected_active_eligibility_summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    base = ROOT / "reports" / "corrected_phase_history"
    parser.add_argument("--source", type=Path, default=base / "phase_corrected_v371_active.csv")
    parser.add_argument("--quality", type=Path, default=base / "frozen_v37_on_corrected_active" / "phase_v37_feature_quality_audit.csv")
    parser.add_argument("--group-predictions", type=Path, default=base / "frozen_v37_on_corrected_active" / "phase_v37_complete_group_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=base / "active_training_eligibility_v1")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
