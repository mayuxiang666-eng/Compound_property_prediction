"""Rerun Stage6 duration and discharge-temperature bidirectional replay on true strict batches."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.optimization import silica_stage6_replay as replay


ROOT = Path(__file__).resolve().parents[2]
COMPOUND = "M1-T09170WS-- 02 004"
HISTORY = ROOT / "Mooney_Prediction_Pipeline" / "models" / "hefei_group_training_dataset_clean.csv"
PALLET_AUDIT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow" / "t09170ws_02004_raw_material_lot_by_pallet_v3.csv"
OUT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"


def main() -> None:
    history = pd.read_csv(HISTORY, low_memory=False)
    history = history[history["CompoundName"].astype(str).eq(COMPOUND)].copy()
    pallet = pd.read_csv(PALLET_AUDIT, low_memory=False)
    allowed = pallet.loc[
        pallet["mapped_batch_count"].eq(1) & pallet["raw_material_lot_status"].eq("PASS"),
        ["OrderID", "PalletID", "raw_material_lot_key"],
    ].drop_duplicates()
    history["OrderID"] = history["OrderID"].astype(str).str.strip()
    history["PalletID"] = history["PalletID"].astype(str).str.strip()
    strict = history[history["N_batches"].eq(1)].merge(allowed, on=["OrderID", "PalletID"], how="inner")
    strict = strict.drop_duplicates(["OrderID", "PalletID"])

    frame_for_medians = replay.build_strata(strict)
    frame_for_medians["MNY"] = pd.to_numeric(frame_for_medians["MNY"], errors="coerce")
    numeric = replay._numeric_features(frame_for_medians)
    ordered = frame_for_medians.sort_values("OrderStartTime", kind="stable")
    train = ordered.iloc[: max(1, int(len(ordered) * 0.8))]
    fill_values = train[numeric].apply(pd.to_numeric, errors="coerce").median()

    original_mutation = replay._mutated_prediction

    def fixed_mutated(model, row, feature_columns, feature, delta):
        base = row[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
        baseline = float(model.predict(base)[0])
        mutated = base.copy()
        mutated.loc[:, feature] = float(mutated.iloc[0][feature]) + delta
        return baseline, float(model.predict(mutated)[0])

    replay._mutated_prediction = fixed_mutated
    try:
        audit, summary = replay.run_bidirectional_replay(
            strict,
            spec_low=45.0,
            spec_high=77.0,
            process_response_status="PASS_FROM_CALIBRATED_800_BATCH_VALIDATION",
            v37_model_path_status="BLOCKED_V37_INPUT_PATH",
        )
    finally:
        replay._mutated_prediction = original_mutation

    summary.update({
        "strict_alignment_definition": "N_batches=1 AND BatchHeader mapped_batch_count=1 AND complete raw lot coverage",
        "strict_alignment_rows": int(len(strict)),
        "spec_contract_status": "FAMILY_SPEC_INHERITED_VERSION_CONFIRMATION_REQUIRED",
        "next_batch_report": {
            "status": "MANUAL_REVIEW_ONLY_NOT_ISSUED",
            "reason": ["BLOCKED_INSUFFICIENT_ONE_BATCH_LABELS", "BLOCKED_V37_INPUT_PATH", "FAMILY_SPEC_INHERITANCE_UNCONFIRMED"],
            "automatic_control": "DISABLED",
        },
    })
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "t09170ws_02004_strict33_bidirectional_replay.csv"
    json_path = OUT / "t09170ws_02004_strict33_bidirectional_replay_summary.json"
    audit.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(audit.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"replay={csv_path}")
    print(f"summary={json_path}")


if __name__ == "__main__":
    main()
