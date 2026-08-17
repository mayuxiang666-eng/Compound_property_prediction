import pandas as pd

from Mooney_Prediction_Pipeline.model_training.silica_pid_duration_contract_v2 import build_pid_intervals


def test_cumulative_step_time_is_differenced_and_pid_start_is_dynamic():
    rows = []
    for group, steps in [("AVR_MST", [1, 2, 3]), ("AVR_MSB", [1, 2])]:
        for step in steps:
            rows.append({
                "LabelGroupID": "g1", "GroupName": group, "StepNo": step, "kind": "duration",
                "target_mean": {("AVR_MST", 1): 100, ("AVR_MST", 2): 130, ("AVR_MST", 3): 150, ("AVR_MSB", 1): 80, ("AVR_MSB", 2): 100}[(group, step)],
                "actual_value_mean": {("AVR_MST", 1): 100, ("AVR_MST", 2): 130, ("AVR_MST", 3): 150, ("AVR_MSB", 1): 80, ("AVR_MSB", 2): 100}[(group, step)],
                "Target_Temperature": 165,
            })
    for step, actual_temp in [(1, 150), (2, 162), (3, 165)]:
        rows.append({"LabelGroupID": "g1", "GroupName": "AVR_MST", "StepNo": step, "kind": "temperature", "target_mean": actual_temp, "actual_value_mean": actual_temp, "Target_Temperature": 165})
    result = build_pid_intervals(pd.DataFrame(rows))
    row = result.iloc[0]
    assert row["pid_start_step"] == 2
    assert row["target_pid_duration_s_TOP_MST"] == 50
    assert row["target_pid_duration_s_BOTTOM_MSB"] == 100
    assert row["target_pid_total_duration_s"] == 150
