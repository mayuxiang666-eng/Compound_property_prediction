"""Leakage-safe shape features for frozen RDS Active BatchCurve rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


SIGNALS = {
    "Curve1": "temp",
    "Curve2": "power",
    "Curve5": "torque",
    "Curve6": "rotor",
    "Curve7": "ram",
}


def decode_hex_curve(value) -> np.ndarray:
    """Decode MMS signed 16-bit samples with ``FFFF`` as the missing sentinel.

    Curve2/Curve5 can legitimately contain negative values encoded as two's
    complement (for example ``FFFA`` is -6). Treating every word above 10000
    as missing silently removed real power/torque observations. ``FFFF`` is
    retained as the source missing marker; other high words are decoded as
    signed 16-bit values and remain subject to downstream physical audits.
    """

    if pd.isna(value):
        return np.array([], dtype=float)
    text = str(value).strip().replace(" ", "").replace("\n", "").replace("\r", "")
    if not text or len(text) % 4:
        return np.array([], dtype=float)
    try:
        unsigned = np.fromiter(
            (int(text[index : index + 4], 16) for index in range(0, len(text), 4)),
            dtype=float,
            count=len(text) // 4,
        )
    except ValueError:
        return np.array([], dtype=float)
    missing = unsigned == 0xFFFF
    values = unsigned.copy()
    signed = values >= 0x8000
    values[signed] -= 0x10000
    values[missing] = np.nan
    return values


def clean_curve(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Interpolate missing sensor points without borrowing another batch."""

    if not len(values):
        return values, 1.0
    missing_rate = float(np.mean(~np.isfinite(values)))
    series = pd.Series(values, dtype=float).interpolate(
        method="linear", limit_direction="both"
    )
    if series.isna().all():
        return np.array([], dtype=float), 1.0
    return series.ffill().bfill().to_numpy(dtype=float), missing_rate


def resample(values: np.ndarray, points: int) -> np.ndarray:
    if not len(values):
        return np.full(points, np.nan, dtype=float)
    if len(values) == 1:
        return np.repeat(values[0], points).astype(float)
    return np.interp(
        np.linspace(0.0, 1.0, points),
        np.linspace(0.0, 1.0, len(values)),
        values,
    )


def _crossing_fraction(values: np.ndarray, threshold: float) -> float:
    crossed = np.flatnonzero(values >= threshold)
    if not len(crossed):
        return np.nan
    return float(crossed[0] / max(1, len(values) - 1))


def signal_features(name: str, values: np.ndarray, points: int = 32) -> dict[str, float]:
    clean, missing_rate = clean_curve(values)
    result = {
        f"curve__{name}__length": float(len(values)),
        f"curve__{name}__missing_rate": missing_rate,
    }
    if not len(clean):
        for key in (
            "mean", "std", "min", "max", "p05", "p50", "p95", "first",
            "last", "net_change", "max_rise", "max_drop", "mean_abs_diff",
            "integral_norm",
        ):
            result[f"curve__{name}__{key}"] = np.nan
        for index in range(points):
            result[f"curve__{name}__p{index:02d}"] = np.nan
        return result
    difference = np.diff(clean)
    result.update(
        {
            f"curve__{name}__mean": float(np.mean(clean)),
            f"curve__{name}__std": float(np.std(clean, ddof=0)),
            f"curve__{name}__min": float(np.min(clean)),
            f"curve__{name}__max": float(np.max(clean)),
            f"curve__{name}__p05": float(np.quantile(clean, 0.05)),
            f"curve__{name}__p50": float(np.quantile(clean, 0.50)),
            f"curve__{name}__p95": float(np.quantile(clean, 0.95)),
            f"curve__{name}__first": float(clean[0]),
            f"curve__{name}__last": float(clean[-1]),
            f"curve__{name}__net_change": float(clean[-1] - clean[0]),
            f"curve__{name}__max_rise": float(np.max(difference)) if len(difference) else 0.0,
            f"curve__{name}__max_drop": float(np.min(difference)) if len(difference) else 0.0,
            f"curve__{name}__mean_abs_diff": float(np.mean(np.abs(difference))) if len(difference) else 0.0,
            f"curve__{name}__integral_norm": float(np.trapezoid(clean) / max(1, len(clean) - 1)),
        }
    )
    sampled = resample(clean, points)
    for index, value in enumerate(sampled):
        result[f"curve__{name}__p{index:02d}"] = float(value)
    if name == "temp":
        result["curve__temp__time_to_100_frac"] = _crossing_fraction(clean, 100.0)
        result["curve__temp__time_to_120_frac"] = _crossing_fraction(clean, 120.0)
        tail = clean[-min(30, len(clean)) :]
        result["curve__temp__tail30_slope"] = (
            float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) >= 3 else np.nan
        )
    return result


def extract_curve_row(row, points: int = 32) -> dict[str, float | str]:
    result: dict[str, float | str] = {
        "OrderID": str(row["OrderID"]),
        "BatchNumber": str(row["BatchNumber"]),
    }
    lengths = []
    for source, name in SIGNALS.items():
        values = decode_hex_curve(row[source])
        lengths.append(len(values))
        result.update(signal_features(name, values, points=points))
    positive = [length for length in lengths if length > 0]
    result["curve__all_signals_present"] = float(len(positive) == len(SIGNALS))
    result["curve__length_mismatch"] = float(len(set(positive)) > 1) if positive else 1.0
    return result


__all__ = [
    "SIGNALS",
    "clean_curve",
    "decode_hex_curve",
    "extract_curve_row",
    "resample",
    "signal_features",
]
