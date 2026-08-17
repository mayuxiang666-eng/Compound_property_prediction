"""Final local compatibility fixes over cached exact cross-database data."""

from __future__ import annotations

from Mooney_Prediction_Pipeline.optimization import (
    extract_crossdb_exact_pilot_step_history_v1 as base,
)
from Mooney_Prediction_Pipeline.optimization import (
    extract_crossdb_exact_pilot_step_history_v1_exact as exact,
)
from Mooney_Prediction_Pipeline.optimization import (
    extract_crossdb_exact_pilot_step_history_v1_fast as fast,
)
from Mooney_Prediction_Pipeline.optimization import (
    extract_crossdb_exact_pilot_step_history_v1_fast_v2 as v2,
)


_ORIGINAL_STEP_READINESS = base._step_readiness


def _step_readiness_compatible(mapped):
    result = _ORIGINAL_STEP_READINESS(mapped)
    return result.rename(columns={"RecipeID": "CompoundDescription"})


def main() -> None:
    base._batch_template = v2._batch_template_from_group_key
    base._step_readiness = _step_readiness_compatible
    exact._query_exact_pairs = fast._query_exact_pairs_fast
    exact.main()


if __name__ == "__main__":
    main()
