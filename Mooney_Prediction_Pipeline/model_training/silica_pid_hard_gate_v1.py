"""Hard engineering gate for Silica PID top+bottom target time.

The temperature bands are supplied by process engineering.  The extraction
of top/bottom PID duration must be explicit for each recipe family; this
module intentionally refuses to infer it from all MSB steps.
"""

from __future__ import annotations

from typing import Any


SILICA_PID_TIME_BANDS = {
    165: {"min_s": 80.0, "max_s": 140.0},
    155: {"min_s": 130.0, "max_s": 230.0},
    142: {"min_s": 270.0, "max_s": 330.0},
}


def normalize_temperature(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    rounded = int(round(number))
    return rounded if abs(number - rounded) <= 0.25 else None


def validate_silica_pid_total_target_time(
    pid_target_temperature: Any,
    top_pid_duration_target_s: Any,
    bottom_pid_duration_target_s: Any,
) -> dict[str, Any]:
    """Validate a candidate vector against the engineering PID time band.

    ``top_pid_duration_target_s`` and ``bottom_pid_duration_target_s`` must
    already be extracted by an explicit recipe/phase contract.  No default
    mapping is applied here.
    """
    temperature = normalize_temperature(pid_target_temperature)
    if temperature not in SILICA_PID_TIME_BANDS:
        return {
            "status": "BLOCKED_UNSUPPORTED_SILICA_PID_TARGET_TEMPERATURE",
            "pid_target_temperature": temperature,
            "total_pid_target_time_s": None,
            "band_min_s": None,
            "band_max_s": None,
        }
    try:
        top = float(top_pid_duration_target_s)
        bottom = float(bottom_pid_duration_target_s)
    except (TypeError, ValueError):
        return {
            "status": "BLOCKED_MISSING_SILICA_PID_TOP_BOTTOM_TARGET_TIME",
            "pid_target_temperature": temperature,
            "total_pid_target_time_s": None,
            "band_min_s": SILICA_PID_TIME_BANDS[temperature]["min_s"],
            "band_max_s": SILICA_PID_TIME_BANDS[temperature]["max_s"],
        }
    if top < 0 or bottom < 0:
        return {
            "status": "REJECTED_NEGATIVE_SILICA_PID_TARGET_TIME",
            "pid_target_temperature": temperature,
            "total_pid_target_time_s": top + bottom,
            "band_min_s": SILICA_PID_TIME_BANDS[temperature]["min_s"],
            "band_max_s": SILICA_PID_TIME_BANDS[temperature]["max_s"],
        }
    total = top + bottom
    band = SILICA_PID_TIME_BANDS[temperature]
    status = "PASS_SILICA_PID_TOTAL_TARGET_TIME" if band["min_s"] <= total <= band["max_s"] else "REJECTED_SILICA_PID_TOTAL_TARGET_TIME_BAND"
    return {
        "status": status,
        "pid_target_temperature": temperature,
        "top_pid_duration_target_s": top,
        "bottom_pid_duration_target_s": bottom,
        "total_pid_target_time_s": total,
        "band_min_s": band["min_s"],
        "band_max_s": band["max_s"],
    }


__all__ = ["SILICA_PID_TIME_BANDS", "normalize_temperature", "validate_silica_pid_total_target_time"]
