from __future__ import annotations

import pandas as pd

from Mooney_Prediction_Pipeline.model_training.train_v371_active_candidates import (
    aggregate_groups,
    assign_order_time_splits,
)


def test_order_time_split_never_splits_an_order() -> None:
    rows = []
    for order in range(20):
        for pallet in range(2):
            rows.append({
                "OrderID": str(order),
                "OrderStartTime": pd.Timestamp("2025-01-01") + pd.Timedelta(days=order),
                "LabelGroupID": f"{order}|{pallet}",
            })
    split = assign_order_time_splits(pd.DataFrame(rows))
    assert split["OrderID"].is_unique
    assert set(split["Split"]) == {"train", "validation", "oot_test"}


def test_group_prediction_is_batch_mean_not_representative_batch() -> None:
    batch = pd.DataFrame({
        "OrderID": ["1", "1"],
        "BatchNumber": [1, 2],
        "PalletID": ["P", "P"],
        "LabelGroupID": ["1|P", "1|P"],
        "CompoundName": ["X", "X"],
        "OrderStartTime": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "Split": ["oot_test", "oot_test"],
        "MNY": [50.0, 50.0],
        "RawPrediction": [48.0, 52.0],
    })
    group = aggregate_groups(batch)
    assert len(group) == 1
    assert group.loc[0, "GroupRawPrediction"] == 50.0
    assert group.loc[0, "ActualMNY"] == 50.0
