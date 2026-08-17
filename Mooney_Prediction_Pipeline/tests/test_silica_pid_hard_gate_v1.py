from Mooney_Prediction_Pipeline.model_training.silica_pid_hard_gate_v1 import validate_silica_pid_total_target_time


def test_165_band_passes_at_boundary():
    result = validate_silica_pid_total_target_time(165, 60, 80)
    assert result["status"] == "PASS_SILICA_PID_TOTAL_TARGET_TIME"
    assert result["total_pid_target_time_s"] == 140


def test_155_band_rejects_outside_range():
    result = validate_silica_pid_total_target_time(155, 100, 150)
    assert result["status"] == "REJECTED_SILICA_PID_TOTAL_TARGET_TIME_BAND"


def test_unknown_temperature_does_not_fall_back_to_a_band():
    result = validate_silica_pid_total_target_time(163, 50, 50)
    assert result["status"] == "BLOCKED_UNSUPPORTED_SILICA_PID_TARGET_TEMPERATURE"


def test_missing_top_or_bottom_is_blocked():
    result = validate_silica_pid_total_target_time(142, None, 200)
    assert result["status"] == "BLOCKED_MISSING_SILICA_PID_TOP_BOTTOM_TARGET_TIME"
