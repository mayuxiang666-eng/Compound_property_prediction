"""Local template-key fix over the cached fast exact-pair extraction."""

from __future__ import annotations

import hashlib

import pandas as pd

from Mooney_Prediction_Pipeline.optimization import (
    extract_crossdb_exact_pilot_step_history_v1 as base,
)
from Mooney_Prediction_Pipeline.optimization import (
    extract_crossdb_exact_pilot_step_history_v1_exact as exact,
)
from Mooney_Prediction_Pipeline.optimization import (
    extract_crossdb_exact_pilot_step_history_v1_fast as fast,
)


def _batch_template_from_group_key(batch: pd.DataFrame) -> pd.Series:
    recipe, _compound, material, mixer, _order_id, _batch_number = batch.name
    rows: list[str] = []
    route: list[str] = []
    for row in batch.sort_values(["GroupName", "StepNo"], kind="stable").itertuples():
        controls = "".join([
            "D" if bool(row.is_time_settable) else "-",
            "T" if bool(row.is_temp_settable) else "-",
            "R" if bool(row.is_rotor_settable) else "-",
            "P" if bool(row.is_ram_settable) else "-",
        ])
        rows.append(f"{row.GroupName}:S{int(row.StepNo)}:{row.mapped_phase}:{controls}")
        route.append(f"{row.GroupName}:{row.EquipmentID}")
    signature = "|".join(rows)
    route_signature = "|".join(dict.fromkeys(route))
    digest = hashlib.sha256(
        f"{recipe}|{material}|{mixer}|{route_signature}|{signature}".encode("utf-8")
    ).hexdigest()[:12].upper()
    return pd.Series({
        "TemplateID": digest,
        "TemplateSignature": signature,
        "RouteSignature": route_signature,
    })


def main() -> None:
    base._batch_template = _batch_template_from_group_key
    exact._query_exact_pairs = fast._query_exact_pairs_fast
    exact.main()


if __name__ == "__main__":
    main()
