import pandas as pd

from Mooney_Prediction_Pipeline.optimization.next_batch_request_contract import (
    CompletedBatchContext,
    CurrentBatchContext,
    canonical_target_vector_hash,
    prepare_next_batch_request,
)


def _targets():
    return pd.DataFrame(
        [
            {"GroupName": "MST", "StepNo": 3, "duration_SP": 40, "rotor_SP": 35, "temp_SP": 105, "control_logic": "AND"},
            {"GroupName": "MST", "StepNo": 4, "duration_SP": 35, "rotor_SP": 30, "temp_SP": 115, "control_logic": "OR"},
        ]
    )


def _current(**changes):
    values = dict(
        order_id="8000001",
        batch_number=12,
        recipe_id="9922A15760AA_PV46_004",
        process_version="PV46",
        mixer_line="MST",
        phase_route="OilWet",
        template_key="template-current",
        target_vector_hash=canonical_target_vector_hash(_targets()),
        batch_started=False,
        relevant_steps_started=False,
    )
    values.update(changes)
    return CurrentBatchContext(**values)


def _predecessor(**changes):
    values = dict(
        order_id="8000001",
        batch_number=11,
        recipe_id="9922A15760AA_PV46_004",
        process_version="PV46",
        mixer_line="MST",
        template_key="template-current",
        batch_complete=True,
        curve_complete=True,
    )
    values.update(changes)
    return CompletedBatchContext(**values)


_DEFAULT_PREDECESSOR = object()


def _prepare(current=None, predecessor=_DEFAULT_PREDECESSOR, **changes):
    values = dict(
        requested_recipe_id="9922A15760AA_PV46_004",
        target_mooney=62.0,
        target_energy_kwh=190.0,
        mooney_spec_low=58.0,
        mooney_spec_high=66.0,
        current=current or _current(),
        step_targets=_targets(),
        predecessor=_predecessor() if predecessor is _DEFAULT_PREDECESSOR else predecessor,
    )
    values.update(changes)
    return prepare_next_batch_request(**values)


def test_ready_request_binds_current_version_target_vector():
    result = _prepare()
    assert result.status == "READY_FOR_MODEL_GATES"
    assert result.predecessor_context_status == "APPLICABLE_SAME_TEMPLATE"
    assert result.optimization_request.target_mooney == 62.0
    assert result.recommendation_mode == "MANUAL_REVIEW_ONLY"


def test_started_batch_and_actual_features_are_blocked():
    result = _prepare(current=_current(batch_started=True, current_batch_actual_feature_count=2))
    assert result.status == "BLOCKED_INPUT_CONTEXT"
    assert "BLOCKED_NEXT_BATCH_ALREADY_STARTED" in result.reason_codes
    assert "BLOCKED_CURRENT_OR_FUTURE_ACTUAL_FEATURE_LEAKAGE" in result.reason_codes


def test_changed_process_version_disables_predecessor_context_but_is_not_itself_blocked():
    result = _prepare(predecessor=_predecessor(process_version="PV41", template_key="old"))
    assert result.status == "READY_FOR_MODEL_GATES"
    assert result.predecessor_context_status == "NOT_APPLICABLE_VERSION_OR_TEMPLATE_CHANGE"


def test_target_vector_hash_mismatch_is_blocked():
    result = _prepare(current=_current(target_vector_hash="wrong"))
    assert "BLOCKED_TARGET_VECTOR_VERSION_MISMATCH" in result.reason_codes


def test_non_rds_source_and_missing_predecessor_are_blocked():
    result = _prepare(current=_current(source_status="LOCAL_MMS"), predecessor=None)
    assert "BLOCKED_UNAPPROVED_DATA_SOURCE" in result.reason_codes
    assert "BLOCKED_NO_COMPLETED_PREDECESSOR" in result.reason_codes
