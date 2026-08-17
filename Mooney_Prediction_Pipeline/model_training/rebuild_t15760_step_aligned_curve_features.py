"""Rebuild all T15760 process features with BatchData Step boundaries."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.feature_engineering.batchdata_step_boundary_builder import TIME_VARIABLE
from Mooney_Prediction_Pipeline.feature_engineering.step_aligned_curve_features import extract_step_aligned_curve_features
from Mooney_Prediction_Pipeline.model_training.run_v371_r10_curve_state import load_t15760_curves, normalize_id
from Mooney_Prediction_Pipeline.model_training.run_v371_r8_batch_group_mean import BASE, PHASE_SOURCE, T15760_NAME


STEP_SOURCE = BASE / "t15760_within_order_contrast" / "step_execution" / "t15760_batch_step_execution_long.csv"
OUTPUT = BASE / "t15760_step_aligned_curve_v2"
MAPPING = {
    ("AVR_MST", 1): "Stage1_Loading",
    ("AVR_MST", 2): "Stage1_Loading",
    ("AVR_MST", 3): "Stage2_DryMixing",
    ("AVR_MST", 4): "Stage3_OilLoading",
    ("AVR_MST", 5): "Stage4_WetMixing",
    ("AVR_MST", 6): "Stage4_WetMixing",
    ("AVR_MST", 7): "Stage4_WetMixing",
    ("AVR_MST", 8): "Stage4_WetMixing",
    ("AVR_MST", 9): "Stage4_WetMixing",
    ("AVR_MST", 10): "Stage5_PID_Controlled",
    ("AVR_MST", 11): "PostPID_Tail",
    ("AVR_MSB", 1): "Stage6_MSB_PID_Step1",
    ("AVR_MSB", 2): "Stage6_MSB_PID_Step2",
    ("AVR_MSB", 3): "Stage6_MSB_PID_Step3",
}


def time_rows(part: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GroupName": part["GroupName"],
            "StepNo": part["StepNo"],
            "VariablePath": TIME_VARIABLE,
            "Target": part["time_target"],
            "Value": part["time"],
        }
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    phase = pd.read_csv(PHASE_SOURCE, low_memory=False)
    metadata = phase.loc[phase["CompoundName"].astype(str).eq(T15760_NAME), [
        "OrderID", "BatchNumber", "LabelGroupID", "StepPatternHash", "TargetVectorHash",
        "TemplateKey", "ProcessVersion", "BatchStartTime",
    ]].copy()
    metadata["OrderID"] = metadata["OrderID"].map(normalize_id)
    metadata["BatchNumber"] = metadata["BatchNumber"].map(normalize_id)
    keys = set(zip(metadata["OrderID"], metadata["BatchNumber"]))
    curves = load_t15760_curves(keys)
    source = metadata.merge(curves, on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one")

    steps = pd.read_csv(STEP_SOURCE, low_memory=False)
    steps["OrderID"] = steps["OrderID"].map(normalize_id)
    steps["BatchNumber"] = steps["BatchNumber"].map(normalize_id)
    grouped_steps = {(order, batch): part for (order, batch), part in steps.groupby(["OrderID", "BatchNumber"], sort=False)}
    features, audits, blocked = [], [], []
    for row in source.itertuples(index=False):
        key = (row.OrderID, row.BatchNumber)
        part = grouped_steps.get(key)
        if part is None:
            blocked.append({"OrderID": row.OrderID, "BatchNumber": row.BatchNumber, "reason": "BLOCKED_MISSING_BATCHDATA"})
            continue
        series = pd.Series(row._asdict())
        try:
            result, boundary = extract_step_aligned_curve_features(series, time_rows(part), MAPPING, points=8, long_tail_threshold_s=75.0)
        except ValueError as exc:
            blocked.append({"OrderID": row.OrderID, "BatchNumber": row.BatchNumber, "reason": str(exc)})
            continue
        result.update(
            {
                "LabelGroupID": row.LabelGroupID, "StepPatternHash": row.StepPatternHash,
                "TargetVectorHash": row.TargetVectorHash, "TemplateKey": row.TemplateKey,
                "ProcessVersion": row.ProcessVersion, "BatchStartTime": row.BatchStartTime,
            }
        )
        features.append(result)
        audits.append({"OrderID": row.OrderID, "BatchNumber": row.BatchNumber, "LabelGroupID": row.LabelGroupID, **boundary.audit})

    feature_frame = pd.DataFrame(features)
    audit_frame = pd.DataFrame(audits)
    blocked_frame = pd.DataFrame(blocked, columns=["OrderID", "BatchNumber", "reason"])
    feature_frame.to_csv(OUTPUT / "t15760_step_aligned_curve_features_v2.csv", index=False)
    audit_frame.to_csv(OUTPUT / "t15760_step_boundary_audit_v2.csv", index=False)
    blocked_frame.to_csv(OUTPUT / "t15760_step_boundary_blocked_v2.csv", index=False)
    status_counts = Counter(audit_frame["BoundaryStatus"]) if len(audit_frame) else Counter()
    summary = {
        "contract": "T15760-BATCHDATA-STEP-ALIGNED-CURVE-V2",
        "expected_batches": len(keys),
        "curve_rows": len(curves),
        "feature_rows": len(feature_frame),
        "blocked_rows": len(blocked_frame),
        "boundary_status_counts": dict(status_counts),
        "step_pattern_counts": metadata["StepPatternHash"].value_counts(dropna=False).to_dict(),
        "phase_mapping": {f"{group}|{step}": phase for (group, step), phase in MAPPING.items()},
        "cross_batch_fill": False,
        "database_access_during_run": False,
        "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
