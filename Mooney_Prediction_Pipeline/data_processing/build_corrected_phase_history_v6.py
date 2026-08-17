"""V6 facade: preserve every non-empty MMS varchar OrderID, including year prefixes."""

from __future__ import annotations

from Mooney_Prediction_Pipeline.data_processing import build_corrected_phase_history_v2 as v2
from Mooney_Prediction_Pipeline.data_processing import build_corrected_phase_history_v5 as v5  # noqa: F401


def _all_varchar_orders(order_ids: list[str]) -> tuple[list[str], list[str]]:
    accepted, rejected = [], []
    for value in order_ids:
        text = str(value).strip()
        if text and text.upper() not in {"NAN", "NONE", "NULL", "<NA>"}:
            accepted.append(text)
        else:
            rejected.append(text)
    return sorted(set(accepted)), sorted(set(rejected))


v2._numeric_orders = _all_varchar_orders


if __name__ == "__main__":
    v2.main()
