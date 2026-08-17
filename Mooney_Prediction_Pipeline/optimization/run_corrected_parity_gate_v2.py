"""Execute the formal 144-column corrected Phase/MMS parity gate."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.optimization.audit_corrected_operational_parity_v2 import PACKAGE
from Mooney_Prediction_Pipeline.optimization.corrected_parity_dataset import build_corrected_parity_frames
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor
from services.mooney_predict_service.feature_parity_gate_v2 import evaluate_feature_parity_v2


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "phase_parity_gate_v2"


def main() -> None:
    reference, candidate = build_corrected_parity_frames("2016836")
    predictor = V37RawPredictor(PACKAGE).load()
    feature_columns = predictor.feature_columns
    missing_columns_reference = sorted(set(feature_columns) - set(reference.columns))
    missing_columns_candidate = sorted(set(feature_columns) - set(candidate.columns))
    for frame in (reference, candidate):
        for column in feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    missing_cells_reference = int(reference[feature_columns].isna().sum().sum())
    missing_cells_candidate = int(candidate[feature_columns].isna().sum().sum())
    if missing_columns_reference or missing_columns_candidate or missing_cells_reference or missing_cells_candidate:
        raise ValueError({
            "missing_columns_reference": missing_columns_reference,
            "missing_columns_candidate": missing_columns_candidate,
            "missing_cells_reference": missing_cells_reference,
            "missing_cells_candidate": missing_cells_candidate,
        })
    reference["RawPrediction"] = predictor.predict_raw(reference).final_prediction
    candidate["RawPrediction"] = predictor.predict_raw(candidate).final_prediction
    summary_rows, detail = evaluate_feature_parity_v2(reference, candidate, feature_columns)
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows.to_csv(OUT / "order_2016836_parity_batch_summary.csv", index=False)
    detail.to_csv(OUT / "order_2016836_parity_feature_detail.csv", index=False)
    failed = detail.loc[detail["FeatureParityStatus"].eq("FAIL")]
    result = {
        "order_id": "2016836",
        "same_batch_count": int(len(summary_rows)),
        "model_feature_count": int(len(feature_columns)),
        "missing_model_columns_reference": missing_columns_reference,
        "missing_model_columns_candidate": missing_columns_candidate,
        "missing_feature_cells_reference": missing_cells_reference,
        "missing_feature_cells_candidate": missing_cells_candidate,
        "batch_status_counts": {
            str(key): int(value) for key, value in summary_rows["ParityStatus"].value_counts().items()
        },
        "failed_feature_cells": int(len(failed)),
        "failed_features": sorted(failed["Feature"].drop_duplicates().astype(str).tolist()),
        "max_prediction_abs_difference": float(summary_rows["PredictionAbsDifference"].max()),
        "prediction_tolerance_mny": 0.1,
        "tolerance_version": "PHASE-PARITY-TOLERANCE-V2",
        "parity_gate_status": "PARITY_PASS" if summary_rows["ParityStatus"].eq("PARITY_PASS").all() else "PARITY_BLOCKED",
        "automatic_control": "DISABLED",
    }
    with (OUT / "order_2016836_parity_gate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
