"""Column-by-column parity gate between canonical Phase and MMS candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


PREDICTION_TOLERANCE_MNY = 0.1


@dataclass(frozen=True)
class FeatureTolerance:
    absolute: float
    relative: float = 0.0


def tolerance_for_feature(name: str) -> FeatureTolerance:
    lower = name.lower()
    if "duration" in lower or lower.startswith("idx_"):
        return FeatureTolerance(1.0, 0.0)
    if "temp" in lower or "temperature" in lower:
        return FeatureTolerance(0.5, 0.001)
    if "rotorspeed" in lower or "rpm" in lower:
        return FeatureTolerance(0.5, 0.005)
    if "power" in lower or "torque" in lower or "energy" in lower:
        return FeatureTolerance(0.05, 0.02)
    if "humidity" in lower:
        return FeatureTolerance(0.1, 0.001)
    if lower.startswith("weight_pct_") or lower.endswith("_phr"):
        return FeatureTolerance(1e-6, 1e-6)
    return FeatureTolerance(1e-6, 1e-6)


def evaluate_feature_parity(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    reference_prediction: str = "RawPrediction",
    candidate_prediction: str = "RawPrediction",
    prediction_tolerance: float = PREDICTION_TOLERANCE_MNY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["OrderID", "BatchNumber"]
    features = list(dict.fromkeys(str(column) for column in feature_columns))
    missing_reference = sorted(set(keys + features + [reference_prediction]) - set(reference.columns))
    missing_candidate = sorted(set(keys + features + [candidate_prediction]) - set(candidate.columns))
    missing_columns = sorted(set(missing_reference + missing_candidate))
    if any(key in missing_columns for key in keys):
        raise ValueError(f"Parity key columns are missing: {missing_columns}")

    ref = reference.copy()
    cand = candidate.copy()
    ref["OrderID"] = ref["OrderID"].astype(str).str.strip()
    cand["OrderID"] = cand["OrderID"].astype(str).str.strip()
    ref["BatchNumber"] = pd.to_numeric(ref["BatchNumber"], errors="coerce")
    cand["BatchNumber"] = pd.to_numeric(cand["BatchNumber"], errors="coerce")
    if ref.duplicated(keys).any() or cand.duplicated(keys).any():
        raise ValueError("Parity inputs must be unique by OrderID+BatchNumber")

    usable = [column for column in features if column in ref.columns and column in cand.columns]
    ref_columns = keys + usable + ([reference_prediction] if reference_prediction in ref else [])
    cand_columns = keys + usable + ([candidate_prediction] if candidate_prediction in cand else [])
    merged = ref[ref_columns].merge(
        cand[cand_columns], on=keys, how="outer", suffixes=("__reference", "__candidate"), indicator=True,
        validate="one_to_one",
    )

    summaries: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        failed: list[str] = []
        max_normalized = 0.0
        row_missing = list(missing_columns)
        if row["_merge"] != "both":
            row_missing.append("REFERENCE_OR_CANDIDATE_ROW")
        if row["_merge"] == "both":
            for feature in usable:
                reference_value = pd.to_numeric(pd.Series([row[f"{feature}__reference"]]), errors="coerce").iloc[0]
                candidate_value = pd.to_numeric(pd.Series([row[f"{feature}__candidate"]]), errors="coerce").iloc[0]
                tolerance = tolerance_for_feature(feature)
                allowed = tolerance.absolute
                if pd.notna(reference_value):
                    allowed += tolerance.relative * abs(float(reference_value))
                if pd.isna(reference_value) or pd.isna(candidate_value):
                    passed, absolute_error, normalized = False, np.nan, np.inf
                else:
                    absolute_error = abs(float(candidate_value) - float(reference_value))
                    normalized = absolute_error / allowed if allowed > 0 else np.inf
                    passed = absolute_error <= allowed
                if not passed:
                    failed.append(feature)
                max_normalized = max(max_normalized, float(normalized))
                details.append({
                    "OrderID": row["OrderID"], "BatchNumber": row["BatchNumber"],
                    "Feature": feature, "ReferenceValue": reference_value,
                    "CandidateValue": candidate_value, "AbsoluteError": absolute_error,
                    "AllowedError": allowed, "NormalizedError": normalized,
                    "FeatureParityStatus": "PASS" if passed else "FAIL",
                })

        ref_pred = row.get(f"{reference_prediction}__reference", np.nan)
        cand_pred = row.get(f"{candidate_prediction}__candidate", np.nan)
        pred_diff = (
            abs(float(cand_pred) - float(ref_pred))
            if pd.notna(ref_pred) and pd.notna(cand_pred) else np.nan
        )
        if row_missing:
            status = "BLOCKED_MISSING_FEATURE"
        elif failed:
            status = "BLOCKED_FEATURE_DRIFT"
        elif pd.isna(pred_diff) or pred_diff > prediction_tolerance:
            status = "BLOCKED_PREDICTION_DRIFT"
        else:
            status = "PARITY_PASS"
        summaries.append({
            "OrderID": row["OrderID"], "BatchNumber": row["BatchNumber"],
            "ModelFeatureCount": len(features), "MissingModelColumnCount": len(set(row_missing)),
            "MissingModelColumns": "|".join(sorted(set(row_missing))),
            "FailedFeatureCount": len(failed), "FailedFeatures": "|".join(sorted(failed)),
            "MaxNormalizedFeatureError": max_normalized,
            "ReferencePrediction": ref_pred, "CandidatePrediction": cand_pred,
            "PredictionAbsDifference": pred_diff, "PredictionTolerance": prediction_tolerance,
            "ParityStatus": status,
        })
    return pd.DataFrame(summaries), pd.DataFrame(details)
