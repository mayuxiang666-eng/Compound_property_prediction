from __future__ import annotations

import numpy as np
import pandas as pd

from services.mooney_predict_service.phase_group_inference import (
    predict_batches_and_groups,
    prepare_phase_batches,
)


class FakeV37Model:
    def predict(self, frame: pd.DataFrame, cluster_col: str):
        assert cluster_col == "material_system"
        final = frame["Target_Temperature"].to_numpy(dtype=float)
        return final, final - 1.0, np.ones(len(frame)), np.zeros(len(frame))


def _frame() -> pd.DataFrame:
    rows = []
    for batch, pallet, target in [(1, "P1", 50.0), (2, "P1", 54.0), (3, "P2", 60.0)]:
        rows.append({
            "OrderID": "O1", "BatchNumber": batch, "PalletID": pallet,
            "CompoundName": "M1-A00458", "OrderStartTime": "2026-01-01 08:00:00",
            "Target_Temperature": target, "weight_pct_silica": np.nan,
            "is_oil_loading_present": 1.0, "MNY": 61.0,
            "env_temp_mean": 20.0, "env_temp_min": 15.0, "env_temp_max": 25.0,
            "env_humidity_mean": 50.0, "env_humidity_min": 40.0, "env_humidity_max": 60.0,
            "Stage5_PID_Duration": 10.0, "Stage5_PID_temp_Mean": 150.0,
            "Stage5_PID_temp_Std": 1.0, "Stage5_PID_power_Mean": 100.0,
            "phys_power_stability_pid": 0.1, "Stage5_PID_Specific_Energy": 2.0,
            "Stage5_PID_Torque_Integral": 3.0, "Stage2_DryMixing_Duration": 20.0,
            "Stage2_DryMixing_power_Mean": 100.0, "Stage2_DryMixing_Torque_Mean": 50.0,
            "Stage6_BottomMixing_Torque_Mean": 40.0,
            "supplier_carbon_black_structure_avg": 110.0,
            "supplier_carbon_black_surface_area_avg": 85.0,
            "Stage2_power_decay_slope": 0.0, "Stage2_DryMixing_Specific_Energy": 2.0,
            "Stage4_WetMixing_Specific_Energy": 2.0,
        })
    return pd.DataFrame(rows)


def test_batch_first_then_order_pallet_group_mean_and_lab_removed():
    schema = {
        "s1_features": ["Target_Temperature", "weight_pct_silica", "env_temp_mean"],
        "s2_features": ["env_temp_min", "env_temp_max", "env_humidity_mean", "env_humidity_min", "env_humidity_max"],
    }
    prepared = prepare_phase_batches(_frame(), schema)
    assert prepared.quality["FeatureQualityStatus"].eq("MODEL_INPUT_OK").all()
    assert prepared.quality["RouteCompatibleZeroFillCount"].eq(1).all()
    assert prepared.quality["LabColumnsRemoved"].eq("MNY").all()
    assert "MNY" not in prepared.features.columns

    result = predict_batches_and_groups(prepared, FakeV37Model(), "test-v37")
    group = result.group_predictions.set_index("PalletID")
    assert group.loc["P1", "BatchCount"] == 2
    assert group.loc["P1", "GroupRawPrediction"] == 52.0
    assert group.loc["P2", "GroupRawPrediction"] == 60.0


def test_non_route_missing_qm_is_blocked_not_zero_filled():
    frame = _frame().iloc[[0]].copy()
    frame["supplier_rubber_viscosity_avg"] = np.nan
    schema = {
        "s1_features": ["Target_Temperature", "supplier_rubber_viscosity_avg"],
        "s2_features": list((
            "env_temp_mean", "env_temp_min", "env_temp_max",
            "env_humidity_mean", "env_humidity_min", "env_humidity_max",
        )),
    }
    prepared = prepare_phase_batches(frame, schema)
    assert prepared.features.empty
    row = prepared.quality.iloc[0]
    assert row["FeatureQualityStatus"] == "BLOCKED_FEATURE_DATA_MISSING"
    assert row["MissingFeatureCells"] == "supplier_rubber_viscosity_avg"
