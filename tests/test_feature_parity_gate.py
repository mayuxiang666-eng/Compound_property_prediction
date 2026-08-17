import pandas as pd

from services.mooney_predict_service.feature_parity_gate import evaluate_feature_parity


def test_parity_requires_all_features_and_prediction_within_point_one():
    reference = pd.DataFrame([{
        "OrderID": "1", "BatchNumber": 1, "Stage2_DryMixing_Duration": 100.0,
        "Stage4_WetMixing_temp_Mean": 150.0, "RawPrediction": 60.0,
    }])
    candidate = reference.copy()
    candidate["Stage2_DryMixing_Duration"] = 100.5
    candidate["Stage4_WetMixing_temp_Mean"] = 150.2
    candidate["RawPrediction"] = 60.09
    summary, detail = evaluate_feature_parity(
        reference, candidate, ["Stage2_DryMixing_Duration", "Stage4_WetMixing_temp_Mean"],
    )
    assert summary.iloc[0]["ParityStatus"] == "PARITY_PASS"
    assert detail["FeatureParityStatus"].eq("PASS").all()


def test_feature_drift_blocks_even_if_prediction_matches():
    reference = pd.DataFrame([{
        "OrderID": "1", "BatchNumber": 1, "Stage6_BottomMixing_WayofRam_Mean": 340.0,
        "RawPrediction": 60.0,
    }])
    candidate = reference.copy()
    candidate["Stage6_BottomMixing_WayofRam_Mean"] = 0.0
    summary, _ = evaluate_feature_parity(
        reference, candidate, ["Stage6_BottomMixing_WayofRam_Mean"],
    )
    assert summary.iloc[0]["ParityStatus"] == "BLOCKED_FEATURE_DRIFT"
    assert summary.iloc[0]["FailedFeatureCount"] == 1


def test_missing_model_column_is_a_hard_block():
    reference = pd.DataFrame([{
        "OrderID": "1", "BatchNumber": 1, "feature_a": 1.0, "RawPrediction": 60.0,
    }])
    candidate = reference.drop(columns=["feature_a"])
    summary, _ = evaluate_feature_parity(reference, candidate, ["feature_a"])
    assert summary.iloc[0]["ParityStatus"] == "BLOCKED_MISSING_FEATURE"
    assert summary.iloc[0]["MissingModelColumnCount"] == 1
