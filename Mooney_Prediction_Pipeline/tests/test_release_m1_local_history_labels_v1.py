import pandas as pd

from Mooney_Prediction_Pipeline.model_training.release_m1_local_history_labels_v1 import (
    normalize_disposition,
    release_labels,
)


def _quarantine():
    return pd.DataFrame([
        {
            "OrderID": "100", "BatchNumber": "1", "PalletID": "10",
            "PhysicalBatchKey": "100|1", "LabelGroupID": "100|10",
            "CompoundName": "M1-T15760----", "RecipeID": "R_PV01_001",
            "ProcessVersion": 1, "MixerLine": "MB02",
            "OrderStartTime": "2024-01-01", "historical_lab_candidate_value": 50.0,
            "MNY": 50.0, "MNY target": 55.0, "test_result_start_time": "2024-01-02",
        },
        {
            "OrderID": "101", "BatchNumber": "1", "PalletID": "11",
            "PhysicalBatchKey": "101|1", "LabelGroupID": "101|11",
            "CompoundName": "M1-T15760----", "RecipeID": "R_PV01_001",
            "ProcessVersion": 1, "MixerLine": "MB02",
            "OrderStartTime": "2025-06-01", "historical_lab_candidate_value": 60.0,
            "MNY": 60.0, "MNY target": 55.0, "test_result_start_time": "2025-06-02",
        },
    ])


def test_disposition_normalisation():
    assert normalize_disposition(1) == "001"
    assert normalize_disposition("11") == "011"


def test_release_uses_canonical_target_and_exact_recipe_cutoff():
    lab = pd.DataFrame([
        {
            "order_id": 100, "pallet_id": 10, "test_result": 54.0,
            "test_result_start_time": "2024-01-03", "disposition_code": 1,
            "status_valid": True, "retest_number_numeric": 1,
            "last_modified_timestamp": "2024-01-03", "datamart_pk": "a",
        },
        {
            "order_id": 101, "pallet_id": 11, "test_result": 61.0,
            "test_result_start_time": "2025-06-03", "disposition_code": 11,
            "status_valid": True, "retest_number_numeric": 0,
            "last_modified_timestamp": "2025-06-03", "datamart_pk": "b",
        },
    ])
    current_batch = pd.DataFrame([
        {"LabelGroupID": "200|20", "RecipeID": "R_PV01_001", "BatchStartTime": "2025-01-01"},
    ])
    current_group = pd.DataFrame([
        {"LabelGroupID": "200|20", "split": "validation"},
    ])
    verified, safe_groups, safe_process, manifest = release_labels(
        _quarantine(), lab, current_batch, current_group
    )
    first = verified.loc[verified["LabelGroupID"].eq("100|10")].iloc[0]
    assert first["MNY"] == 54.0
    assert first["legacy_mny_abs_difference"] == 4.0
    assert set(safe_groups["LabelGroupID"]) == {"100|10"}
    assert set(safe_process["PhysicalBatchKey"]) == {"100|1"}
    assert "MNY" not in safe_process.columns
    assert manifest["embedded_legacy_mny_used_as_target"] is False
