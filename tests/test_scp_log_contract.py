import pandas as pd

from Mooney_Prediction_Pipeline.data_processing.scp_log_contract import (
    parse_scp_log_contract,
)


def test_legacy_mask_parses_duration_temperature_and():
    source = pd.DataFrame(
        [
            ["R1", 2, "MI.1.MSB.02.SCPAND#_SCP-Log-AND-LBits", "B00000101"],
            ["R1", 2, "MI.1.MSB.02.SCPOR#_SCP-Log-OR-LBits", "B00000000"],
            ["R1", 2, "MI.1.MSB.02.SCPANDOR#_SCP-Log-AND/OR-LBits", "B00000000"],
        ],
        columns=["RecipeID", "StepNumber", "ParameterID", "ParameterValue"],
    )
    parsed = parse_scp_log_contract(source).iloc[0]
    assert parsed["duration_enabled"]
    assert parsed["temperature_enabled"]
    assert parsed["control_relation"] == "AND"


def test_individual_bits_parse_duration_temperature_or():
    source = pd.DataFrame(
        [
            ["R2", 8, "MI.1.MST.08.SCPOR#_SCP-Log-OR-LBit0", 1],
            ["R2", 8, "MI.1.MST.08.SCPOR#_SCP-Log-OR-LBit2", 1],
        ],
        columns=["RecipeID", "StepNumber", "ParameterID", "ParameterValue"],
    )
    parsed = parse_scp_log_contract(source).iloc[0]
    assert parsed["control_relation"] == "OR"
    assert parsed["logic_status"] == "CONTROL_LOGIC_OK"


def test_andor_mask_fails_closed():
    source = pd.DataFrame(
        [["R3", 4, "MI.1.MST.04.SCPANDOR#_SCP-Log-AND/OR-LBits", "B00000101"]],
        columns=["RecipeID", "StepNumber", "ParameterID", "ParameterValue"],
    )
    parsed = parse_scp_log_contract(source).iloc[0]
    assert parsed["control_relation"] == "AMBIGUOUS_ANDOR"
    assert parsed["logic_status"] == "BLOCKED_AMBIGUOUS_SCP_LOG"

