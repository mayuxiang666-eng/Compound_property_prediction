"""Current-batch Step-aligned curve features for Mooney process experts."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.feature_engineering.batchdata_step_boundary_builder import (
    BoundaryResult,
    build_batch_step_boundaries,
)
from Mooney_Prediction_Pipeline.model_training.r10_curve_shape_features import (
    SIGNALS,
    clean_curve,
    decode_hex_curve,
    resample,
)


def _features(prefix: str, values: np.ndarray, points: int) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    result = {f"{prefix}__observed_fraction": float(len(valid) / max(1, len(values)))}
    names = ("mean", "std", "min", "max", "p10", "p50", "p90", "first", "last", "net", "slope", "integral")
    if not len(valid):
        result.update({f"{prefix}__{name}": np.nan for name in names})
        result.update({f"{prefix}__p{index:02d}": np.nan for index in range(points)})
        return result
    slope = float(np.polyfit(np.arange(len(valid)), valid, 1)[0]) if len(valid) >= 3 else np.nan
    result.update(
        {
            f"{prefix}__mean": float(np.mean(valid)),
            f"{prefix}__std": float(np.std(valid)),
            f"{prefix}__min": float(np.min(valid)),
            f"{prefix}__max": float(np.max(valid)),
            f"{prefix}__p10": float(np.quantile(valid, 0.10)),
            f"{prefix}__p50": float(np.quantile(valid, 0.50)),
            f"{prefix}__p90": float(np.quantile(valid, 0.90)),
            f"{prefix}__first": float(valid[0]),
            f"{prefix}__last": float(valid[-1]),
            f"{prefix}__net": float(valid[-1] - valid[0]),
            f"{prefix}__slope": slope,
            f"{prefix}__integral": float(np.trapezoid(valid)),
        }
    )
    for index, value in enumerate(resample(valid, points)):
        result[f"{prefix}__p{index:02d}"] = float(value)
    return result


def extract_step_aligned_curve_features(
    curve_row: pd.Series,
    batchdata: pd.DataFrame,
    phase_mapping: Mapping[tuple[str, int], str],
    *,
    points: int = 8,
    long_tail_threshold_s: float = 75.0,
) -> tuple[dict[str, object], BoundaryResult]:
    decoded, missing_rates = {}, {}
    for source, name in SIGNALS.items():
        raw = decode_hex_curve(curve_row[source])
        decoded[name], missing_rates[name] = clean_curve(raw)
    if not len(decoded["temp"]) or not len(decoded["power"]):
        raise ValueError("BLOCKED_INCOMPLETE_TEMP_OR_POWER_CURVE")
    curve_length = min(len(decoded["temp"]), len(decoded["power"]))
    boundary = build_batch_step_boundaries(
        batchdata,
        curve_length,
        phase_mapping,
        long_tail_threshold_s=long_tail_threshold_s,
    )
    aligned = {}
    for name, values in decoded.items():
        target = np.full(curve_length, np.nan, dtype=float)
        length = min(curve_length, len(values))
        target[:length] = values[:length]
        aligned[name] = target

    result: dict[str, object] = {
        "OrderID": str(curve_row["OrderID"]),
        "BatchNumber": str(curve_row["BatchNumber"]),
        "BoundaryContract": boundary.audit["BoundaryContract"],
        "BoundaryStatus": boundary.audit["BoundaryStatus"],
    }
    for key, value in boundary.audit.items():
        if key not in result:
            result[f"boundary__{key}"] = value
    for name, rate in missing_rates.items():
        result[f"boundary__{name}_missing_rate"] = rate
    for phase in boundary.phases.itertuples(index=False):
        start = max(0, min(curve_length, int(round(phase.CurveStart))))
        end = max(start, min(curve_length, int(round(phase.CurveEnd))))
        result[f"step_phase__{phase.Phase}__duration"] = float(phase.ActualDuration)
        result[f"step_phase__{phase.Phase}__curve_duration"] = float(end - start)
        for signal, values in aligned.items():
            result.update(_features(f"step_phase__{phase.Phase}__{signal}", values[start:end], points))
    return result, boundary


__all__ = ["extract_step_aligned_curve_features"]
