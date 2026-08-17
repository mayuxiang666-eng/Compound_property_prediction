from __future__ import annotations

import pandas as pd

from Mooney_Prediction_Pipeline.model_training.run_v371_active_pilot_backtests import (
    pilot_cutoffs,
    split_all_orders,
)


def test_pilot_cutoffs_reserve_recent_groups_and_are_strict() -> None:
    rows = []
    for order in range(100):
        for pallet in range(2):
            rows.append({
                "OrderID": str(order),
                "OrderStartTime": pd.Timestamp("2025-01-01") + pd.Timedelta(days=order),
                "LabelGroupID": f"{order}|{pallet}",
                "CompoundName": "B00458",
                "GroupComplete": True,
                "MNYConflict": False,
            })
    groups = pd.DataFrame(rows)
    validation_start, test_start, counts = pilot_cutoffs(groups, "B00458")
    assert validation_start < test_start
    assert counts == {"total": 200, "train": 140, "validation": 30, "oot_test": 30}

    source = groups.assign(PalletID="P")
    splits = split_all_orders(source, validation_start, test_start)
    train_max = splits.loc[splits["Split"].eq("train"), "OrderTime"].max()
    test_min = splits.loc[splits["Split"].eq("oot_test"), "OrderTime"].min()
    assert train_max < validation_start <= test_start <= test_min
