from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from Mooney_Prediction_Pipeline.m1_relative_improvement.contracts import (
    ReductionTargetRequest,
    evaluate_dual_pilot_gate,
)
from Mooney_Prediction_Pipeline.m1_relative_improvement.gate0 import (
    audit_action_support,
    build_group_action_frame,
)
from Mooney_Prediction_Pipeline.m1_relative_improvement.integrated_data import (
    attach_authentic_trailing_weather,
)
from Mooney_Prediction_Pipeline.m1_relative_improvement.reduction_optimizer import (
    RelativeModelReadiness,
    optimize_reduction_candidates,
)


def test_reduction_request_is_manual_quality_first_only():
    ReductionTargetRequest("recipe", 2.0).validate()
    with pytest.raises(ValueError):
        ReductionTargetRequest("recipe", 0.0).validate()
    with pytest.raises(ValueError):
        ReductionTargetRequest("recipe", 2.0, recommendation_mode="AUTO").validate()


def test_dual_pilot_gate_requires_both_b6_passes():
    blocked = evaluate_dual_pilot_gate({"T15760": "B6_GATE_PASSED", "B00458": "B6_NOT_STARTED"})
    assert not blocked["dual_pilot_gate_passed"]
    passed = evaluate_dual_pilot_gate({"T15760": "B6_GATE_PASSED", "B00458": "B6_GATE_PASSED"})
    assert passed["micro_trial_status"] == "HUMAN_MICRO_TRIAL_ELIGIBLE"


def _step_rows(levels, *, step_no=5):
    rows = []
    time = pd.Timestamp("2026-01-01", tz="UTC")
    for index, level in enumerate(levels):
        rows.append({
            "compound": "M1-X", "recipe_id": "R", "process_version": "1",
            "mixer_line": "M1", "phase_route": "oil_wet",
            "structural_template_key": "STRUCT", "label_group_id": f"G{index}",
            "order_id": f"O{index // 4}", "event_time": time + timedelta(days=int(index)),
            "group_name": "AVR_MST", "step_no": step_no, "kind": "duration",
            "target": level, "actual": level + 1, "phase": "Stage4_WetMixing",
        })
    return pd.DataFrame(rows)


def test_fixed_action_is_not_causal_support():
    grouped = build_group_action_frame(_step_rows([10.0] * 40))
    support, transitions = audit_action_support(grouped)
    assert support.iloc[0]["causal_support_status"] == "FIXED_IN_HISTORY"
    assert transitions.empty


def test_loading_steps_remain_locked_even_with_variation():
    grouped = build_group_action_frame(_step_rows([10.0, 20.0, 30.0] * 15, step_no=2))
    support, _ = audit_action_support(grouped)
    assert support.iloc[0]["causal_support_status"] == "LOCKED_EXPERT_RULE"


def test_action_support_requires_three_major_levels_and_transitions():
    levels = np.tile([10.0, 20.0, 30.0], 20)
    grouped = build_group_action_frame(_step_rows(levels))
    support, _ = audit_action_support(grouped)
    # Each control level occurs across multiple months and has enough groups.
    assert support.iloc[0]["major_control_levels"] == 3
    assert support.iloc[0]["independent_transitions"] >= 30
    assert support.iloc[0]["causal_support_status"] == "CAUSAL_ACTION_SUPPORTED"


def test_target_variation_without_detectable_actual_response_is_not_causal():
    rows = _step_rows(np.tile([10.0, 20.0, 30.0], 20))
    rows["actual"] = 15.0
    grouped = build_group_action_frame(rows)
    support, _ = audit_action_support(grouped)
    assert support.iloc[0]["causal_support_status"] == "CONTROL_TRANSFER_UNAVAILABLE"


def test_unmapped_phase_blocks_action_support():
    rows = _step_rows(np.tile([10.0, 20.0, 30.0], 20))
    rows["phase"] = "UNMAPPED_REQUIRES_EXPERT_CONFIRMATION"
    grouped = build_group_action_frame(rows)
    support, _ = audit_action_support(grouped)
    assert support.iloc[0]["causal_support_status"] == "BLOCKED_UNMAPPED_PHASE"


def test_b3_failure_blocks_all_reduction_candidates():
    candidate = pd.DataFrame([{
        "candidate_id": "C1", "reduction_p10": 1.0, "reduction_p50": 2.0,
        "reduction_p90": 3.0, "probability_meet_target": 0.8,
        "max_supported_reduction": 3.0, "model_response_delta": -2.0,
        "distance_to_optimal_before": 2.0, "distance_to_optimal_after": 1.0,
        "action_cost": 1.0, "changed_step_count": 1,
        "model_path_status": "MODEL_PATH_OK", "control_transfer_status": "CONTROL_TRANSFER_AVAILABLE",
        "causal_support_status": "CAUSAL_ACTION_SUPPORTED",
        "confounding_status": "NO_DOMINANT_MEASURED_CONFOUNDING", "safety_status": "PASSED",
        "thermal_budget_status": "PASSED", "execution_capability_status": "PASSED",
        "ood_status": "IN_DISTRIBUTION",
    }])
    readiness = RelativeModelReadiness(
        direction_magnitude_status="B2_GATE_PASSED",
        process_response_status="CONTROL_TRANSFER_UNAVAILABLE",
        exact_template_status="EXACT_OR_APPROVED_POOLED_TEMPLATE",
        recipe_version_status="CURRENT_VERSION_BOUND",
    )
    result = optimize_reduction_candidates(candidate, ReductionTargetRequest("R", 2.0), readiness)
    assert result.recommendation.status == "BLOCKED_MODEL_READINESS"
    assert "CONTROL_TRANSFER_UNAVAILABLE" in result.recommendation.rejection_codes


def test_weather_uses_only_trailing_window_and_never_future_observation():
    hourly = pd.DataFrame({
        "ObservationTime": pd.date_range("2026-01-01", periods=26, freq="h"),
        "Temperature": np.r_[np.arange(24, dtype=float), 999.0, 1000.0],
        "Humidity": np.arange(26, dtype=float) + 40.0,
    })
    steps = pd.DataFrame({
        "BatchKey": ["B", "B"],
        "BatchStartTime": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-02")],
        "env_temp_mean": [0.0, 0.0],
        "env_humidity_mean": [0.0, 0.0],
        "WeatherStatus": ["LEGACY", "LEGACY"],
    })
    result = attach_authentic_trailing_weather(steps, hourly)
    assert result["WeatherStatus"].eq("AUTHENTIC_LOCAL_24H").all()
    assert result["authentic_weather_points"].eq(24).all()
    assert result["env_temp_mean"].eq(np.mean(np.arange(24, dtype=float))).all()
    assert not result["env_temp_mean"].isin([999.0, 1000.0]).any()
