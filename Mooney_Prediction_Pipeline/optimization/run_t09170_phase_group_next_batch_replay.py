"""Replay one changed next batch inside each Pallet label group."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.run_t09170_phase_group_v37_reproduction import (
    COMPOUND,
    DATA,
    _prepare_original_phase_features,
)
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v37_production_model_package"
OUT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"


def main() -> None:
    source = pd.read_csv(DATA, low_memory=False)
    candidate = source[source["CompoundName"].astype(str).eq(COMPOUND)].copy()
    candidate["MNY"] = pd.to_numeric(candidate["MNY"], errors="coerce")
    candidate = candidate.dropna(subset=["MNY", "OrderID", "PalletID", "BatchNumber"]).copy()
    predictor = V37RawPredictor(PACKAGE).load()
    work, missing_columns, _ = _prepare_original_phase_features(candidate, predictor)
    if missing_columns:
        raise ValueError(f"Missing V3.7 columns: {missing_columns}")
    work["BatchNumber"] = pd.to_numeric(work["BatchNumber"], errors="coerce")
    baseline = predictor.predict_raw(work).final_prediction
    base = work[["_label_group_id", "BatchNumber"]].copy()
    base["baseline_batch_prediction"] = baseline
    baseline_group = base.groupby("_label_group_id")["baseline_batch_prediction"].mean()
    target_indices = work.groupby("_label_group_id")["BatchNumber"].idxmax()

    rows = []
    detail_frames = []
    for parameter, feature, deltas in (
        ("Stage6_duration", "Stage6_BottomMixing_Duration", (-10.0, 10.0)),
        ("discharge_temperature", "phys_discharge_temp", (-1.0, 1.0)),
    ):
        for delta in deltas:
            changed = work.copy()
            changed.loc[target_indices, feature] = pd.to_numeric(
                changed.loc[target_indices, feature], errors="coerce"
            ) + delta
            candidate_prediction = predictor.predict_raw(changed).final_prediction
            arm = changed[["_label_group_id", "OrderID", "PalletID", "BatchNumber"]].copy()
            arm["baseline_batch_prediction"] = baseline
            arm["candidate_batch_prediction"] = candidate_prediction
            arm["is_replayed_next_batch"] = arm.index.isin(target_indices)
            projected_group = arm.groupby("_label_group_id", as_index=False).agg(
                OrderID=("OrderID", "first"),
                PalletID=("PalletID", "first"),
                group_batch_count=("BatchNumber", "size"),
                replayed_batch_number=("BatchNumber", "max"),
                baseline_group_prediction=("baseline_batch_prediction", "mean"),
                projected_group_prediction=("candidate_batch_prediction", "mean"),
            )
            projected_group["group_response"] = (
                projected_group["projected_group_prediction"] - projected_group["baseline_group_prediction"]
            )
            projected_group["parameter"] = parameter
            projected_group["phase_feature"] = feature
            projected_group["delta"] = delta
            detail_frames.append(projected_group)
            response = projected_group["group_response"]
            abs_median = float(response.abs().median())
            path_ok = bool(response.abs().max() > 1e-12)
            rows.append({
                "parameter": parameter,
                "phase_feature": feature,
                "delta": delta,
                "direction": "DOWN" if delta < 0 else "UP",
                "label_group_count": int(len(projected_group)),
                "baseline_group_mny_median": float(baseline_group.median()),
                "projected_group_mny_median": float(projected_group["projected_group_prediction"].median()),
                "group_response_median": float(response.median()),
                "group_response_abs_median": abs_median,
                "group_response_p10": float(response.quantile(0.10)),
                "group_response_p90": float(response.quantile(0.90)),
                "nonzero_group_response_rate": float((response.abs() > 1e-12).mean()),
                "model_path_status": "MODEL_PATH_OK" if path_ok else "NO_MODEL_PATH",
                "minimum_response_status": "PASS" if abs_median >= 0.25 else "REJECTED_MINIMUM_MOONEY_RESPONSE",
                "action_scope": "ONLY_LAST_BATCH_IN_EACH_PALLET_GROUP",
                "process_transfer_status": "DIAGNOSTIC_DIRECT_PHASE_PERTURBATION",
                "thermal_budget_status": "NOT_EVALUATED_DIRECT_PHASE_REPLAY",
                "temperature_cap_status": "NOT_EVALUATED_DIRECT_PHASE_REPLAY",
                "specification_status": "FAMILY_SPEC_INHERITANCE_UNCONFIRMED",
                "final_status": "DIAGNOSTIC_ONLY",
            })

    summary_frame = pd.DataFrame(rows)
    detail = pd.concat(detail_frames, ignore_index=True)
    accepted = (
        summary_frame["model_path_status"].eq("MODEL_PATH_OK")
        & summary_frame["minimum_response_status"].eq("PASS")
    )
    summary = {
        "compound_name": COMPOUND,
        "batch_rows": int(len(work)),
        "label_groups": int(work["_label_group_id"].nunique()),
        "replay_contract": "One Lab MNY per Pallet group; perturb only the final/next batch and recompute projected group mean",
        "candidate_actions": int(len(summary_frame)),
        "model_path_ok_actions": int(summary_frame["model_path_status"].eq("MODEL_PATH_OK").sum()),
        "minimum_response_pass_actions": int(summary_frame["minimum_response_status"].eq("PASS").sum()),
        "accepted_actions": int(accepted.sum()),
        "manual_review_status": "MANUAL_REVIEW_ONLY_NOT_ISSUED" if not accepted.any() else "READY_FOR_REMAINING_PHYSICAL_GATES",
        "automatic_control": "DISABLED",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    summary_csv = OUT / "t09170ws_02004_phase_v37_next_batch_replay_summary.csv"
    detail_csv = OUT / "t09170ws_02004_phase_v37_next_batch_replay_detail.csv"
    json_path = OUT / "t09170ws_02004_phase_v37_next_batch_replay_summary.json"
    summary_frame.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_frame.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_csv={summary_csv}")
    print(f"detail_csv={detail_csv}")
    print(f"summary_json={json_path}")


if __name__ == "__main__":
    main()
