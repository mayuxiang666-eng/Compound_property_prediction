import pandas as pd

from Mooney_Prediction_Pipeline.model_training.group_atomic_subsystems import (
    expanding_group_row_folds,
)


def test_group_atomic_folds_never_split_a_lab_group():
    ids = pd.Series([f"G{group}" for group in range(20) for _ in range(3)])
    folds, audit = expanding_group_row_folds(ids, n_splits=4, min_train_fraction=0.2)
    for train_rows, valid_rows in folds:
        train_groups = set(ids.iloc[train_rows])
        valid_groups = set(ids.iloc[valid_rows])
        assert not train_groups.intersection(valid_groups)
    assert sum(item["valid_groups"] for item in audit) == 16
