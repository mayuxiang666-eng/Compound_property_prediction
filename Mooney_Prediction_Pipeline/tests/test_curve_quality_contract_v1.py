import pandas as pd

from Mooney_Prediction_Pipeline.model_training.curve_quality_contract_v1 import build_quality_audit


def _frame():
    rows = []
    for i in range(60):
        rows.append({
            "OrderID": str(1000 + i), "BatchNumber": 1, "LabelGroupID": f"G{i}",
            "material_system": "Silica", "phase_route": "oil_wet", "split": "train",
            "BatchCompleteMsgReceived": 1, "pid_data_valid_flag": 1,
            "idx_total_length": 100, "time_total_mixing": 100,
            "Stage1_Loading_Duration": 20, "Stage2_DryMixing_Duration": 50,
            "Stage5_PID_Duration": 30, "phys_init_temp": 40,
            "phys_max_temp": 165, "phys_t_max_temp": 80,
            "Stage1_Loading_temp_Mean": 50 + (i % 3),
            "Stage2_DryMixing_temp_Mean": 100 + (i % 3),
            "Stage5_PID_temp_Mean": 160 + (i % 3),
            "phys_discharge_temp": 160 + (i % 3),
            "phys_power_integral": 1000 + i,
            "MNY": 50 + i,  # Must not affect the detector.
        })
    rows[0]["Stage2_DryMixing_Duration"] = -1
    rows[-1]["Stage2_DryMixing_temp_Mean"] = 10000
    return pd.DataFrame(rows)


def test_physical_invalid_is_excluded_and_statistical_outlier_is_retained():
    batch = _frame()
    groups = batch[["LabelGroupID", "split"]]
    audit, group_audit, manifest = build_quality_audit(batch, groups)
    invalid = audit.loc[audit["LabelGroupID"].eq("G0")].iloc[0]
    assert invalid["curve_quality_status"] == "CURVE_PHYSICAL_INVALID"
    assert bool(invalid["curve_model_include"]) is False
    assert int(manifest["physical_invalid_batches"]) == 1
    assert int(manifest["statistical_outlier_batches"]) >= 1
    assert bool(audit.loc[audit["LabelGroupID"].eq("G59"), "curve_model_include"].iloc[0]) is True
    assert bool(group_audit.loc[group_audit["LabelGroupID"].eq("G59"), "curve_quality_risk_flag"].iloc[0]) is True


def test_mny_is_not_a_detector_feature():
    batch = _frame()
    groups = batch[["LabelGroupID", "split"]]
    first, _, _ = build_quality_audit(batch, groups)
    batch["MNY"] = batch["MNY"] * 100000
    second, _, _ = build_quality_audit(batch, groups)
    cols = ["curve_quality_status", "curve_physical_valid", "curve_statistical_outlier"]
    assert first[cols].reset_index(drop=True).equals(second[cols].reset_index(drop=True))
