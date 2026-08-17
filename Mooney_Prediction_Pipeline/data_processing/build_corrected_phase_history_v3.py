"""V3 facade: bind digit-only MMS OrderIDs as strings for varchar indexes."""

from __future__ import annotations

from Mooney_Prediction_Pipeline.data_processing import build_corrected_phase_history_v2 as v2


def _varchar_order_scope(order_ids: list[str]) -> tuple[list[str], list[str]]:
    numeric, rejected = [], []
    for value in order_ids:
        text = str(value).strip()
        (numeric if text.isdigit() else rejected).append(text)
    return sorted(set(numeric)), sorted(set(rejected))


v2._numeric_orders = _varchar_order_scope


if __name__ == "__main__":
    v2.main()
