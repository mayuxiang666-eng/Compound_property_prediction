import pandas as pd

from Mooney_Prediction_Pipeline.model_training.order_atomic_trend_metrics import within_order_pairs


def test_pairs_never_cross_order_boundary():
    frame = pd.DataFrame(
        {
            "OrderID": ["A", "A", "B", "B"], "PalletID": [1, 2, 1, 2],
            "LabelGroupID": ["A1", "A2", "B1", "B2"],
            "production_start": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            "split": ["val"] * 4, "MNY": [50, 55, 80, 70], "Prediction": [51, 54, 78, 72],
        }
    )
    pairs = within_order_pairs(frame, "Prediction")
    assert len(pairs) == 2
    assert set(zip(pairs["from_group"], pairs["to_group"])) == {("A1", "A2"), ("B1", "B2")}
