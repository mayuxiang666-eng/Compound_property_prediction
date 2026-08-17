"""R3 trend audit ordered by true group production start from frozen Phase data."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.order_atomic_trend_metrics import metric_record, within_order_pairs


BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
SOURCE = BASE / "v371_r3_expert_ablation"
PHASE = BASE / "phase_batch_features_active_v1.csv"
OUTPUT = BASE / "v371_r3_order_atomic_trend_audit_v3"
SELECTED = "R3_SILICA_CB_K20"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(SOURCE / "r3_validation_predictions.csv", low_memory=False)
    val = val.loc[val["Candidate"].eq(SELECTED)].copy()
    oot = pd.read_csv(SOURCE / "r3_selected_oot_predictions.csv", low_memory=False)
    frame = pd.concat([val, oot], ignore_index=True)
    production = pd.read_csv(PHASE, usecols=["LabelGroupID", "BatchStartTime"], low_memory=False)
    production["BatchStartTime"] = pd.to_datetime(production["BatchStartTime"], errors="coerce")
    production = production.groupby("LabelGroupID", as_index=False)["BatchStartTime"].min().rename(columns={"BatchStartTime": "production_start"})
    frame = frame.merge(production, on="LabelGroupID", how="left", validate="one_to_one")
    if frame["production_start"].isna().any():
        raise ValueError("BLOCKED_MISSING_GROUP_PRODUCTION_START")
    frame["Prediction"] = pd.to_numeric(frame["RawPrediction"], errors="coerce")
    scopes = {
        "global": pd.Series(True, index=frame.index),
        "T15760": frame["CompoundName"].astype(str).str.contains("T15760", regex=False),
        "B00458": frame["CompoundName"].astype(str).str.contains("B00458", regex=False),
    }
    records, pairs = [], []
    for split_name in ("val", "oot"):
        for scope, mask in scopes.items():
            part = frame.loc[mask]
            if part.loc[part["split"].eq(split_name)].empty:
                continue
            records.append({"scope": scope, **metric_record(part, "Prediction", split_name, SELECTED)})
            pair = within_order_pairs(part.loc[part["split"].eq(split_name)], "Prediction")
            if len(pair):
                pair.insert(0, "scope", scope)
                pair.insert(1, "evaluation_split", split_name)
                pairs.append(pair)
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "r3_order_atomic_metrics.csv", index=False)
    if pairs:
        pd.concat(pairs, ignore_index=True).to_csv(OUTPUT / "r3_order_atomic_pairs.csv", index=False)
    summary = {
        "contract": "R3-STRICT-WITHIN-ORDERID-PRODUCTION-TIME-TREND-AUDIT-V3",
        "candidate": SELECTED, "production_start_coverage": float(frame["production_start"].notna().mean()),
        "metrics": metrics.to_dict("records"),
        "trend_pair_rule": "Consecutive LabelGroups within same OrderID ordered by minimum BatchStartTime",
        "selection_status": "DIAGNOSTIC_ONLY_EXISTING_OOT_ALREADY_VIEWED",
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
