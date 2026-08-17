"""Metadata-complete entry point for safe group-state replay."""

from __future__ import annotations

import pandas as pd

from . import run_v371_safe_group_state_diagnostic as v1
from . import run_v371_r1_temporal_group as r1


def load_predictions() -> pd.DataFrame:
    group_frame, _ = r1.load_contract(r1.DEFAULT_FRAME, r1.DEFAULT_C3_METADATA)
    metadata_columns = [
        "LabelGroupID", "CompoundName", "RecipeID", "ProcessVersion", "MixerLine",
        "material_system", "phase_route", "TemplateKey", "MNY", "lab_time", "split",
    ]
    metadata = group_frame[metadata_columns].copy()
    outputs = []

    group_model = pd.read_csv(
        v1.R1_OUTPUT / "r1_selected_group_predictions.csv",
        usecols=["LabelGroupID", "RawPrediction"],
    )
    group_model["Model"] = "R1_GROUP_AGGREGATE_RAW"
    outputs.append(group_model.merge(metadata, on="LabelGroupID", how="left", validate="one_to_one"))

    batch_model = pd.read_csv(v1.R1B_OUTPUT / "r1b_group_predictions.csv", low_memory=False)
    batch_model = batch_model.loc[
        batch_model["Candidate"].eq("R1B_GROUP_ATOMIC_BATCH"),
        ["LabelGroupID", "RawPrediction"],
    ]
    batch_model["Model"] = "R1B_BATCH_TO_GROUP_RAW"
    outputs.append(batch_model.merge(metadata, on="LabelGroupID", how="left", validate="one_to_one"))

    for filename in ("c3_val_predictions.csv", "c3_oot_predictions.csv"):
        c3 = pd.read_csv(r1.DEFAULT_BASE / filename, usecols=["LabelGroupID", "C3Prediction"])
        c3 = c3.rename(columns={"C3Prediction": "RawPrediction"})
        c3["Model"] = "C3_RAW"
        outputs.append(c3.merge(metadata, on="LabelGroupID", how="left", validate="one_to_one"))
    return pd.concat(outputs, ignore_index=True)


def main() -> None:
    v1.load_predictions = load_predictions
    v1.main()


if __name__ == "__main__":
    main()
