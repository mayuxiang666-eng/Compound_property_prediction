"""Run the silica replay with a fixed train-median imputation contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.optimization import silica_stage6_replay as replay
from Mooney_Prediction_Pipeline.optimization.silica_label_alignment import alignment_summary, audit_pallet_label_alignment


ROOT = Path(__file__).resolve().parents[2]
COMPOUND = "M1-T09170WS-- 02 004"
HISTORY_PATH = ROOT / "Mooney_Prediction_Pipeline" / "models" / "hefei_group_training_dataset_clean.csv"
OUTPUT_DIRECTORY = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"


def main() -> None:
    history = pd.read_csv(HISTORY_PATH, low_memory=False)
    candidate = history[history["CompoundName"].astype(str).eq(COMPOUND)].copy()
    alignment = audit_pallet_label_alignment(history, compound_name=COMPOUND)
    frame = replay.build_strata(candidate)
    frame["MNY"] = pd.to_numeric(frame["MNY"], errors="coerce")
    frame = frame.dropna(subset=["MNY", "OrderStartTime"]).copy()
    features = replay._numeric_features(frame)
    ordered = frame.sort_values("OrderStartTime", kind="stable")
    train = ordered.iloc[: max(1, int(len(ordered) * 0.8))]
    fill_values = train[features].apply(pd.to_numeric, errors="coerce").median()

    def fixed_mutated(model, row, feature_columns, feature, delta):
        base = row[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
        baseline = float(model.predict(base)[0])
        mutated = base.copy()
        mutated.loc[:, feature] = float(mutated.iloc[0][feature]) + delta
        return baseline, float(model.predict(mutated)[0])

    original = replay._mutated_prediction
    replay._mutated_prediction = fixed_mutated
    try:
        audit, summary = replay.run_bidirectional_replay(candidate, spec_low=45.0, spec_high=77.0)
    finally:
        replay._mutated_prediction = original

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
        "reason": ["BLOCKED_INSUFFICIENT_ONE_BATCH_LABELS", "NO_MODEL_PATH", "NO_ACTIVE_NEXT_BATCH_CONTEXT"],
        "automatic_control": "DISABLED",
    }
    replay_csv, replay_json = replay.write_replay_artifacts(
        audit, summary, output_directory=OUTPUT_DIRECTORY, stem="t09170ws_02004"
    )
    next_batch = pd.DataFrame([{
        "compound_name": COMPOUND,
        "report_status": "MANUAL_REVIEW_ONLY_NOT_ISSUED",
        "decision": "BLOCKED",
        "reason_codes": json.dumps(summary["next_batch_report"]["reason"]),
        "energy_status": "NOT_RANKED",
        "automatic_control": "DISABLED",
    }])
    next_batch_path = OUTPUT_DIRECTORY / "t09170ws_02004_next_batch_manual_review.csv"
    next_batch.to_csv(next_batch_path, index=False, encoding="utf-8-sig")
    print(f"replay_csv={replay_csv}")
    print(f"replay_json={replay_json}")
    print(f"next_batch_report={next_batch_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
