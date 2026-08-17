import pandas as pd

from Mooney_Prediction_Pipeline.feature_engineering.batchdata_step_boundary_builder import (
    TIME_VARIABLE,
    build_batch_step_boundaries,
)


MAPPING = {
    ("AVR_MST", 1): "Stage1_Loading",
    ("AVR_MST", 2): "Stage1_Loading",
    ("AVR_MST", 3): "Stage2_DryMixing",
    ("AVR_MST", 4): "Stage3_OilLoading",
    ("AVR_MST", 5): "Stage4_WetMixing",
    ("AVR_MST", 10): "Stage5_PID_Controlled",
    ("AVR_MST", 11): "Stage5_PID_Controlled",
    ("AVR_MSB", 1): "Stage6_MSB_PID_Step1",
    ("AVR_MSB", 2): "Stage6_MSB_PID_Step2",
    ("AVR_MSB", 3): "Stage6_MSB_PID_Step3",
}


def _rows(msb=(75.0, 90.0, 105.0), tail=203.0):
    top = [13.0, 28.0, 75.0, 95.0, 111.0, 112.0, 150.0, 160.0, 180.0, 200.0, tail]
    rows = []
    for number, value in enumerate(top, 1):
        rows.append({"GroupName": "AVR_MST", "StepNo": number, "VariablePath": TIME_VARIABLE, "Target": value, "Value": value})
    for number, value in enumerate(msb, 1):
        rows.append({"GroupName": "AVR_MSB", "StepNo": number, "VariablePath": TIME_VARIABLE, "Target": value, "Value": value})
    return pd.DataFrame(rows)


def test_msb_is_anchored_from_curve_end_and_transfer_is_explicit():
    result = build_batch_step_boundaries(_rows(), 416, MAPPING)
    assert result.audit["BoundaryStatus"] == "NORMAL_COMPLETE_ROUTE"
    assert result.audit["MSBCurveStart"] == 311.0
    assert result.audit["TransferDuration"] == 108.0
    transfer = result.phases.set_index("Phase").loc["Transfer"]
    assert transfer["CurveStart"] == 203.0
    assert transfer["CurveEnd"] == 311.0
    assert result.phases.set_index("Phase").loc["Stage6_MSB_PID_Step3", "CurveEnd"] == 416.0


def test_step11_is_separate_tail_and_missing_msb_is_upset():
    result = build_batch_step_boundaries(_rows(msb=(0.0, 0.0, 0.0), tail=561.0), 567, MAPPING)
    assert result.audit["BoundaryStatus"] == "PROCESS_UPSET_MSB_NOT_EXECUTED"
    assert result.audit["PostPIDTailDuration"] == 361.0
    assert result.audit["LongPostPIDTail"] is True
    assert "PostPID_Tail" in set(result.steps["MappedPhase"])
    assert not result.phases["Phase"].str.startswith("Stage6").any()
