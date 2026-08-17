"""M1 relative-Mooney improvement research contracts.

All modules are snapshot-only and fail closed.  They never read local MMS
Archive, write recipes, or enable Stage3 calibration/automatic control.
"""

from .contracts import (
    ReductionRecommendation,
    ReductionTargetRequest,
    evaluate_dual_pilot_gate,
)

__all__ = [
    "ReductionRecommendation",
    "ReductionTargetRequest",
    "evaluate_dual_pilot_gate",
]
