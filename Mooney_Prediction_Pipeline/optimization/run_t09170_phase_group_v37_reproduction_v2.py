"""Corrected group reproduction runner; excludes Lab fields from model input."""

from __future__ import annotations

import pandas as pd

from Mooney_Prediction_Pipeline.optimization import run_t09170_phase_group_v37_reproduction as audit


def _predict_group_mean_features(work, predictor):
    numeric = work.groupby("_label_group_id", as_index=False)[predictor.feature_columns].mean()
    context = work.groupby("_label_group_id", as_index=False).agg(
        CompoundName=("CompoundName", "first"),
        material_system=("material_system", "first"),
    )
    group_features = context.merge(numeric, on="_label_group_id", how="inner", validate="one_to_one")
    prediction = predictor.predict_raw(group_features).final_prediction
    return pd.DataFrame({
        "_label_group_id": group_features["_label_group_id"],
        "prediction_from_mean_features": prediction,
    })


audit._predict_group_mean_features = _predict_group_mean_features

if __name__ == "__main__":
    audit.main()
