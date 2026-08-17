"""C1 pooled four-route model selected by strict within-order validation score."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training import run_v371_r1_temporal_group as r1
from Mooney_Prediction_Pipeline.model_training.feature_leakage_guard import assert_no_target_leakage
from Mooney_Prediction_Pipeline.model_training.order_atomic_trend_metrics import metric_record
from Mooney_Prediction_Pipeline.model_training.run_v371_r3_expert_ablation import fit_candidate


OUTPUT = r1.DEFAULT_BASE / "v371_c1_strict_order_selected"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_c1_strict_order_selected"
CANDIDATE = "C1_SPECIALIZED_BOTH_K5_STRICT"


def add_production_start(frame: pd.DataFrame) -> pd.DataFrame:
    phase = pd.read_csv(r1.DEFAULT_BASE / "phase_batch_features_active_v1.csv", usecols=["LabelGroupID", "BatchStartTime"], low_memory=False)
    phase["BatchStartTime"] = pd.to_datetime(phase["BatchStartTime"], errors="coerce")
    start = phase.groupby("LabelGroupID", as_index=False)["BatchStartTime"].min().rename(columns={"BatchStartTime": "production_start"})
    result = frame.merge(start, on="LabelGroupID", how="left", validate="one_to_one")
    if result["production_start"].isna().any():
        raise ValueError("BLOCKED_MISSING_GROUP_PRODUCTION_START")
    return result


def scope_mask(frame: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "global":
        return pd.Series(True, index=frame.index)
    return frame["CompoundName"].astype(str).str.contains(scope, regex=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, metadata = r1.load_contract(r1.DEFAULT_FRAME, r1.DEFAULT_C3_METADATA)
    assert_no_target_leakage(frame, metadata["s1_features"] + metadata["s2_features"])
    train = frame.loc[frame["split"].eq("train")].copy()
    validation = frame.loc[frame["split"].eq("val")].copy()
    oot = frame.loc[frame["split"].eq("oot")].copy()
    model, val_prediction, _, elapsed = fit_candidate(
        CANDIDATE, train, validation, metadata,
        silica=True, carbon_black=True, shrinkage_k=5.0,
    )
    oot_prediction = r1.add_predictions(model, oot, CANDIDATE)
    combined = add_production_start(pd.concat([val_prediction, oot_prediction], ignore_index=True))
    combined["Prediction"] = pd.to_numeric(combined["RawPrediction"], errors="coerce")
    records = []
    for split_name in ("val", "oot"):
        for scope in ("global", "T15760", "B00458"):
            part = combined.loc[scope_mask(combined, scope)]
            if part.loc[part["split"].eq(split_name)].empty:
                continue
            records.append({"scope": scope, **metric_record(part, "Prediction", split_name, CANDIDATE)})
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "c1_strict_metrics.csv", index=False)
    combined.to_csv(OUTPUT / "c1_group_predictions.csv", index=False)
    model.stage2_expert_report_.to_csv(OUTPUT / "c1_expert_report.csv", index=False)
    model_path = MODEL_OUTPUT / "hybrid_model_shadow.joblib"
    joblib.dump(model, model_path)
    summary = {
        "model_name": "V3.7.1-Corrected-Challenger-C1-Strict-Order",
        "candidate": CANDIDATE,
        "selection_source": "minimum strict validation score among existing R3 candidates",
        "architecture": "four route pooled model plus Silica and CarbonBlack temporal subsystems",
        "process_version_contract": "not a numeric model feature; TemplateKey/ProcessVersion used for audit and routing only",
        "target_leakage_gate": "PASSED",
        "trend_contract": "within OrderID, adjacent LabelGroups ordered by minimum BatchStartTime",
        "metrics": metrics.to_dict("records"),
        "fit_seconds": elapsed,
        "release_status": "NOT_RELEASED_GATES_FAILED_AND_OOT_ALREADY_VIEWED",
        "stage3_calibration_enabled": False, "automatic_control": "DISABLED",
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (MODEL_OUTPUT / "feature_metadata.json").write_text(json.dumps({**summary, "s1_features": metadata["s1_features"], "s2_features": metadata["s2_features"]}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
