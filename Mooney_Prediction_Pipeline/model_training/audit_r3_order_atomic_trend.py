"""Re-audit frozen R3 predictions with strict within-OrderID trend metrics."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.order_atomic_trend_metrics import metric_record
from Mooney_Prediction_Pipeline.model_training.run_v371_r1_temporal_group import T15760_NAME, B00458_NAME


BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
SOURCE = BASE / "v371_r3_expert_ablation"
OUTPUT = BASE / "v371_r3_order_atomic_trend_audit"
SELECTED = "R3_SILICA_CB_K20"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(SOURCE / "r3_validation_predictions.csv", low_memory=False)
    val = val.loc[val["Candidate"].eq(SELECTED)].copy()
    oot = pd.read_csv(SOURCE / "r3_selected_oot_predictions.csv", low_memory=False)
    frame = pd.concat([val, oot], ignore_index=True)
    frame["Prediction"] = pd.to_numeric(frame["RawPrediction"], errors="coerce")
    frame["lab_time"] = pd.to_datetime(frame["lab_time"], errors="coerce")
    frame["production_start"] = frame["lab_time"]
    records = []
    for split_name in ("val", "oot"):
        records.append({"scope": "global", **metric_record(frame, "Prediction", split_name, SELECTED)})
        for scope, compound in (("T15760", T15760_NAME), ("B00458", B00458_NAME)):
            part = frame.loc[frame["CompoundName"].astype(str).eq(compound)]
            if not part.loc[part["split"].eq(split_name)].empty:
                records.append({"scope": scope, **metric_record(part, "Prediction", split_name, SELECTED)})
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "r3_order_atomic_metrics.csv", index=False)
    summary = {
        "contract": "R3-STRICT-WITHIN-ORDERID-TREND-AUDIT-V1",
        "candidate": SELECTED,
        "metrics": metrics.to_dict("records"),
        "trend_pair_rule": "Consecutive LabelGroups within same OrderID only",
        "selection_status": "DIAGNOSTIC_ONLY_EXISTING_OOT_ALREADY_VIEWED",
        "database_access_during_run": False,
        "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
