import pandas as pd

from Mooney_Prediction_Pipeline.optimization.dual_target_optimizer import (
    DualModelReadiness,
    DualTargetRequest,
    optimize_dual_target,
)


def _candidates():
    return pd.DataFrame([
        {
            "candidate_id": "C1",
            "mooney_p10": 49.0,
            "mooney_p50": 50.0,
            "mooney_p90": 51.0,
            "energy_kwh_p10": 37.0,
            "energy_kwh_p50": 39.0,
            "energy_kwh_p90": 41.0,
            "model_path_status": "MODEL_PATH_OK",
            "strict_actual_phase_response_status": "ACTUAL_PHASE_RESPONSE_OK",
            "safety_status": "PASSED",
            "thermal_budget_status": "PASSED",
            "execution_capability_status": "PASSED",
            "ood_status": "IN_DISTRIBUTION",
            "action_cost": 10.0,
            "changed_step_count": 1,
            "model_response_delta": -1.0,
        },
        {
            "candidate_id": "C2",
            "mooney_p10": 49.5,
            "mooney_p50": 50.5,
            "mooney_p90": 51.5,
            "energy_kwh_p10": 35.0,
            "energy_kwh_p50": 36.0,
            "energy_kwh_p90": 37.0,
            "model_path_status": "MODEL_PATH_OK",
            "strict_actual_phase_response_status": "ACTUAL_PHASE_RESPONSE_OK",
            "safety_status": "PASSED",
            "thermal_budget_status": "PASSED",
            "execution_capability_status": "PASSED",
            "ood_status": "IN_DISTRIBUTION",
            "action_cost": 15.0,
            "changed_step_count": 2,
            "model_response_delta": -0.5,
        },
    ])


def _ready():
    return DualModelReadiness(
        mooney_model_status="RELEASE_GATES_PASSED",
        mooney_model_id="V371",
        stage3_enabled=False,
        process_response_status="CONTROL_TRANSFER_APPROVED",
        strict_actual_phase_response_status="ACTUAL_PHASE_RESPONSE_OK",
        energy_model_status="SHADOW_RANKING_APPROVED",
        energy_model_id="V41_MODE_B",
        energy_route_status="ROUTE_GATE_PASSED",
        template_status="EXACT_OR_APPROVED_POOLED_TEMPLATE",
        recipe_version_status="CURRENT_VERSION_BOUND",
    )


def _request(strategy="QUALITY_FIRST"):
    return DualTargetRequest(
        recipe_id="R",
        target_mooney=50.0,
        target_energy_kwh=36.0,
        mooney_spec_low=45.0,
        mooney_spec_high=55.0,
        strategy=strategy,
    )


def test_quality_first_prioritizes_mooney_target_then_energy():
    result = optimize_dual_target(_candidates(), _request(), _ready())
    assert result.status == "MANUAL_REVIEW_ONLY"
    assert result.selected_candidate_id == "C1"


def test_balanced_target_can_choose_better_joint_target():
    result = optimize_dual_target(_candidates(), _request("BALANCED_TARGET"), _ready())
    assert result.status == "MANUAL_REVIEW_ONLY"
    assert result.selected_candidate_id == "C2"


def test_model_readiness_blocks_all_candidates():
    readiness = DualModelReadiness(
        mooney_model_status="NOT_RELEASED",
        mooney_model_id="R3",
        stage3_enabled=False,
        process_response_status="CONTROL_TRANSFER_UNAVAILABLE",
        strict_actual_phase_response_status="BLOCKED",
        energy_model_status="SHADOW_RANKING_APPROVED",
        energy_model_id="V41",
        energy_route_status="ROUTE_GATE_PASSED",
        template_status="EXACT_OR_APPROVED_POOLED_TEMPLATE",
        recipe_version_status="CURRENT_VERSION_BOUND",
    )
    result = optimize_dual_target(_candidates(), _request(), readiness)
    assert result.status == "BLOCKED_MODEL_READINESS"
    assert result.selected_candidate_id is None
    assert "BLOCKED_MOONEY_MODEL_NOT_RELEASED" in result.reason_codes
    assert "BLOCKED_ENDPOINT_TO_PHASE_SUBSTITUTION" in result.reason_codes


def test_mooney_interval_outside_spec_is_rejected():
    candidates = _candidates()
    candidates["mooney_p90"] = 60.0
    result = optimize_dual_target(candidates, _request(), _ready())
    assert result.status == "NO_VALID_CANDIDATE"
    assert "REJECTED_MOONEY_INTERVAL_ABOVE_SPEC" in result.reason_codes
