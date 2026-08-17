"""Run the strict Step -> Phase -> V3.7 audit with legacy OrderID formats."""

from __future__ import annotations

from Mooney_Prediction_Pipeline.optimization import run_t09170_strict_step_phase_v37_audit as audit


def _pair_clause(pairs: list[tuple[str, int]]) -> str:
    terms = []
    for order_id, batch_number in pairs:
        safe_order_id = str(order_id).replace("'", "''")
        terms.append(f"(bd.OrderID = '{safe_order_id}' AND bd.BatchNumber = {int(batch_number)})")
    return " OR ".join(terms) or "1 = 0"


audit._pair_clause = _pair_clause

if __name__ == "__main__":
    audit.main()
