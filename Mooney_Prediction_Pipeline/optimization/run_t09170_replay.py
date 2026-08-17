"""Run the current historical Stage6 bidirectional screen and issue a fail-closed
next-batch review artifact when the control-transfer gates are unavailable."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.silica_label_alignment import (
    alignment_summary,
    audit_pallet_label_alignment,
)
from Mooney_Prediction_Pipeline.optimization.silica_stage6_replay import (
    run_bidirectional_replay,
    write_replay_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
COMPOUND = "M1-T09170WS-- 02 004"
HISTORY_PATH = ROOT / "Mooney_Prediction_Pipeline" / "models" / "hefei_group_training_dataset_clean.csv"
OUTPUT_DIRECTORY = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"


def main() -> None:
    history = pd.read_csv(HISTORY_PATH, low_memory=False)
    candidate = history[history["CompoundName"].astype(str).eq(COMPOUND)].copy()
    alignment = audit_pallet_label_alignment(history, compound_name=COMPOUND)
    audit, summary = run_bidirectional_replay(candidate, spec_low=45.0, spec_high=77.0)
    summary["label_alignment"] = alignment_summary(alignment)
    summary["spec_contract"] = {
        "source": "M1-T09170 family Lab contract",
        "target": 61.0,
        "lower": 45.0,
        "upper": 77.0,
        "status": "FAMILY_SPEC_INHERITED_VERSION_CONFIRMATION_REQUIRED",
    }
    summary["next_batch_report"] = {
        "status": "MANUAL_REVIEW_ONLY_NOT_ISSUED",
        "reason": [
            "BLOCKED_INSUFFICIENT_ONE_BATCH_LABELS",
            "NO_MODEL_PATH",
            "NO_ACTIVE_NEXT_BATCH_CONTEXT",
        ],
        "automatic_control": "DISABLED",
    }
    replay_csv, replay_json = write_replay_artifacts(
        audit,
        summary,
        output_directory=OUTPUT_DIRECTORY,
        stem="t09170ws_02004",
    )
    next_batch = pd.DataFrame(
        [
            {
                "compound_name": COMPOUND,
                "report_status": "MANUAL_REVIEW_ONLY_NOT_ISSUED",
                "decision": "BLOCKED",
                "reason_codes": json.dumps(summary["next_batch_report"]["reason"]),
                "energy_status": "NOT_RANKED",
                "automatic_control": "DISABLED",
            }
        ]
    )
    next_batch_path = OUTPUT_DIRECTORY / "t09170ws_02004_next_batch_manual_review.csv"
    next_batch.to_csv(next_batch_path, index=False, encoding="utf-8-sig")
    print(f"replay_csv={replay_csv}")
    print(f"replay_json={replay_json}")
    print(f"next_batch_report={next_batch_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
