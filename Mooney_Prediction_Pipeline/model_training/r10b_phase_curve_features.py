"""Leakage-safe, phase-aligned curve features for V3.7.1 R10b.

Only information from the current physical batch is used.  Missing sensor
samples are interpolated within that signal; values are never borrowed from a
different batch.  The phase semantics match the existing V3.7 t1--t7
contract, while oil boundaries come from the current batch's BatchData-derived
fields already present in the frozen Phase snapshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.model_training.r10_curve_shape_features import (
    SIGNALS,
    clean_curve,
    decode_hex_curve,
    resample,
)


PHASES = (
    "Stage1_Loading",
    "Stage2_DryMixing",
    "Stage3_OilLoading",
    "Stage4_WetMixing",
    "Stage5_PID",
    "Transfer",
    "Stage6_BottomMixing",
)


def _first_ram_drop(ram: np.ndarray) -> int:
    if len(ram) < 10:
        return 0
    positions = np.flatnonzero(np.diff(ram) < -5.0)
    return int(positions[0] + 1) if len(positions) else 0


def _pid_start(temp: np.ndarray, target_temp: float, start: int) -> int:
    threshold = target_temp - 20.0
    start = max(0, min(int(start), len(temp) - 1))
    for index in range(start, len(temp)):
        if temp[index] < threshold:
            continue
        tail = temp[index : index + 5]
        if len(tail) < 5 or np.all(tail >= threshold - 10.0):
            return index
    return min(start + 10, len(temp) - 1)


def _top_discharge(temp: np.ndarray, start: int) -> int:
    total = len(temp)
    start = max(0, min(int(start), total - 1))
    for index in range(start, max(start, total - 15)):
        if temp[index] - temp[index + 5] >= 10.0 and temp[index] - temp[index + 15] >= 15.0:
            return index + 2
    difference = np.diff(temp)
    for threshold in (-10.0, -5.0):
        positions = np.flatnonzero(difference[start + 1 :] < threshold)
        if len(positions):
            return int(start + 1 + positions[0])
    return max(start + 1, total - 5)


def _bottom_start(temp: np.ndarray, discharge: int) -> int:
    total = len(temp)
    if discharge >= total - 3:
        return total - 1
    limit = min(discharge + 80, total)
    section = temp[discharge:limit]
    minimum = int(discharge + np.argmin(section))
    minimum_value = temp[minimum]
    for index in range(minimum + 1, min(minimum + 100, total)):
        if temp[index] >= minimum_value + 3.0:
            return max(discharge + 1, index)
    return max(discharge + 1, minimum)


def _bottom_end(temp: np.ndarray, power: np.ndarray, start: int) -> int:
    total = len(temp)
    if start >= total - 5:
        return total
    active = np.flatnonzero(power[start:total] >= 50.0)
    power_end = int(start + active[-1] + 1) if len(active) else total
    difference = np.diff(temp[start:total])
    drops = np.flatnonzero(difference < -5.0)
    temp_end = int(start + drops[np.argmin(difference[drops])] + 1) if len(drops) else total
    candidates = [value for value in (power_end, temp_end) if value > start + 5]
    return min(candidates) if candidates else total


def derive_phase_boundaries(row: pd.Series, curves: dict[str, np.ndarray]) -> dict[str, int]:
    """Derive current-batch t1--t7 boundaries without cross-batch filling."""

    temp = curves["temp"]
    power = curves["power"]
    if len(temp) < 20 or len(power) < 20:
        raise ValueError("BLOCKED_INCOMPLETE_TEMP_OR_POWER_CURVE")
    total = min(len(temp), len(power))
    temp, power = temp[:total], power[:total]
    ram = curves.get("ram", np.array([], dtype=float))
    t1 = _first_ram_drop(ram)
    if t1 < 25:
        t1 = 30

    oil_start = pd.to_numeric(pd.Series([row.get("PrevStepValue")]), errors="coerce").iloc[0]
    oil_end = pd.to_numeric(pd.Series([row.get("CurrentValue")]), errors="coerce").iloc[0]
    has_oil = bool(pd.notna(oil_start) and pd.notna(oil_end))
    if has_oil:
        t2 = int(round(oil_start))
        t3 = int(round(oil_end))
        if t2 < t1 + 10 or t2 >= total - 40:
            t2 = t1 + 45
        if t3 <= t2 or t3 >= total - 20:
            t3 = t2 + 40
    else:
        t2 = t1
        t3 = t1

    target = pd.to_numeric(pd.Series([row.get("Target_Temperature")]), errors="coerce").iloc[0]
    if pd.isna(target) or float(target) <= 0:
        raise ValueError("BLOCKED_MISSING_PROCESS_TARGET_TEMPERATURE")
    t4 = _pid_start(temp, float(target), t3 if has_oil else t1)
    t5 = _top_discharge(temp, t4)
    t6 = _bottom_start(temp, t5)
    t7 = _bottom_end(temp, power, t6)

    values = [0, t1, t2, t3, t4, t5, t6, t7, total]
    values = [max(0, min(int(value), total)) for value in values]
    for index in range(1, len(values)):
        values[index] = max(values[index], values[index - 1])
    _, t1, t2, t3, t4, t5, t6, t7, total = values
    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5, "t6": t6, "t7": t7, "total": total}


def _segment_features(prefix: str, values: np.ndarray, points: int) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    result = {f"{prefix}__observed_fraction": float(len(valid) / max(1, len(values)))}
    if not len(valid):
        for name in ("mean", "std", "min", "max", "p10", "p50", "p90", "first", "last", "net", "slope", "integral"):
            result[f"{prefix}__{name}"] = np.nan
        for index in range(points):
            result[f"{prefix}__p{index:02d}"] = np.nan
        return result
    difference_x = np.arange(len(valid), dtype=float)
    slope = float(np.polyfit(difference_x, valid, 1)[0]) if len(valid) >= 3 else np.nan
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
    sampled = resample(valid, points)
    for index, value in enumerate(sampled):
        result[f"{prefix}__p{index:02d}"] = float(value)
    return result


def extract_phase_curve_row(row: pd.Series, points: int = 8) -> dict[str, float | str]:
    raw, cleaned = {}, {}
    for source, name in SIGNALS.items():
        raw[name] = decode_hex_curve(row[source])
        cleaned[name], _ = clean_curve(raw[name])
    boundaries = derive_phase_boundaries(row, cleaned)
    total = boundaries["total"]
    aligned: dict[str, np.ndarray] = {}
    for name, values in cleaned.items():
        target = np.full(total, np.nan, dtype=float)
        length = min(len(values), total)
        target[:length] = values[:length]
        aligned[name] = target

    spans = {
        "Stage1_Loading": (0, boundaries["t1"]),
        "Stage2_DryMixing": (boundaries["t1"], boundaries["t2"] if boundaries["t2"] > boundaries["t1"] else boundaries["t4"]),
        "Stage3_OilLoading": (boundaries["t2"], boundaries["t3"]),
        "Stage4_WetMixing": (boundaries["t3"], boundaries["t4"]),
        "Stage5_PID": (boundaries["t4"], boundaries["t5"]),
        "Transfer": (boundaries["t5"], boundaries["t6"]),
        "Stage6_BottomMixing": (boundaries["t6"], boundaries["t7"]),
    }
    result: dict[str, float | str] = {
        "OrderID": str(row["OrderID"]),
        "BatchNumber": str(row["BatchNumber"]),
        "phase_curve__boundary_status": "CURRENT_BATCH_ONLY",
    }
    for key, value in boundaries.items():
        result[f"phase_curve__boundary__{key}"] = float(value)
    for phase, (start, end) in spans.items():
        result[f"phase_curve__{phase}__duration"] = float(max(0, end - start))
        for signal, values in aligned.items():
            result.update(
                _segment_features(
                    f"phase_curve__{phase}__{signal}", values[start:end], points
                )
            )
    return result


__all__ = ["PHASES", "derive_phase_boundaries", "extract_phase_curve_row"]
