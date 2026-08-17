"""Leakage-safe request contract for a currently producing recipe.

The first deployment mode is not live within-batch control.  It uses the most
recent completed batch only as optional same-template context and prepares a
recommendation for a later batch whose controllable Steps have not started.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .dual_target_optimizer import DualTargetRequest


TARGET_VECTOR_COLUMNS = (
    "GroupName",
    "StepNo",
    "duration_SP",
    "rotor_SP",
    "temp_SP",
    "control_logic",
)


@dataclass(frozen=True)
class CurrentBatchContext:
    order_id: str
    batch_number: int
    recipe_id: str
    process_version: str
    mixer_line: str
    phase_route: str
    template_key: str
    target_vector_hash: str
    batch_started: bool
    relevant_steps_started: bool
    current_batch_actual_feature_count: int = 0
    source_status: str = "RDS_ACTIVE_ONLY"


@dataclass(frozen=True)
class CompletedBatchContext:
    order_id: str
    batch_number: int
    recipe_id: str
    process_version: str
    mixer_line: str
    template_key: str
    batch_complete: bool
    curve_complete: bool


@dataclass(frozen=True)
class PreparedNextBatchRequest:
    status: str
    reason_codes: tuple[str, ...]
    optimization_request: DualTargetRequest | None
    current_context: CurrentBatchContext
    predecessor_context_status: str
    target_vector_hash: str | None
    inference_mode: str = "NEXT_BATCH_AFTER_COMPLETED_PREDECESSOR"
    recommendation_mode: str = "MANUAL_REVIEW_ONLY"
    archive_status: str = "BLOCKED_ARCHIVE_NOT_REPLICATED"

    def audit_record(self) -> dict:
        record = asdict(self)
        request = record.pop("optimization_request")
        if request is not None:
            record.update({f"request_{key}": value for key, value in request.items()})
        record["reason_codes"] = json.dumps(list(self.reason_codes))
        return record


def canonical_target_vector_hash(step_targets: pd.DataFrame) -> str:
    """Hash the authoritative current-version Step target vector."""

    missing = [column for column in TARGET_VECTOR_COLUMNS if column not in step_targets]
    if missing:
        raise ValueError(f"Current Step target vector is missing columns: {missing}")
    frame = step_targets.loc[:, TARGET_VECTOR_COLUMNS].copy()
    if frame.empty:
        raise ValueError("Current Step target vector is empty")
    frame["GroupName"] = frame["GroupName"].astype(str).str.strip()
    frame["control_logic"] = frame["control_logic"].astype(str).str.strip().str.upper()
    for column in ("StepNo", "duration_SP", "rotor_SP", "temp_SP"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(TARGET_VECTOR_COLUMNS)].isna().any().any():
        raise ValueError("Current Step target vector contains missing/non-numeric values")
    if not np.isfinite(frame[["StepNo", "duration_SP", "rotor_SP", "temp_SP"]]).all().all():
        raise ValueError("Current Step target vector contains non-finite values")
    if (frame[["duration_SP", "rotor_SP", "temp_SP"]] < 0).any().any():
        raise ValueError("Current Step target vector contains negative setpoints")
    if not (frame[["duration_SP", "rotor_SP", "temp_SP"]] > 0).any().any():
        raise ValueError("Current Step target vector has no existing controllable target")
    if frame[["GroupName", "StepNo"]].duplicated().any():
        raise ValueError("Current Step target vector has duplicate GroupName + StepNo")
    frame = frame.sort_values(["GroupName", "StepNo"], kind="stable")
    payload = frame.to_json(orient="records", double_precision=10)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_next_batch_request(
    *,
    requested_recipe_id: str,
    target_mooney: float,
    target_energy_kwh: float,
    mooney_spec_low: float,
    mooney_spec_high: float,
    current: CurrentBatchContext,
    step_targets: pd.DataFrame,
    predecessor: CompletedBatchContext | None,
    strategy: str = "QUALITY_FIRST",
) -> PreparedNextBatchRequest:
    """Validate production timing and bind targets to the current recipe version."""

    blockers: list[str] = []
    requested_recipe = str(requested_recipe_id).strip()
    if requested_recipe != str(current.recipe_id).strip():
        blockers.append("BLOCKED_REQUEST_RECIPE_MISMATCH")
    if current.source_status != "RDS_ACTIVE_ONLY":
        blockers.append("BLOCKED_UNAPPROVED_DATA_SOURCE")
    if not str(current.process_version).strip():
        blockers.append("BLOCKED_MISSING_PROCESS_VERSION")
    if not str(current.template_key).strip():
        blockers.append("BLOCKED_MISSING_TEMPLATE_KEY")
    if current.batch_started or current.relevant_steps_started:
        blockers.append("BLOCKED_NEXT_BATCH_ALREADY_STARTED")
    if int(current.current_batch_actual_feature_count) != 0:
        blockers.append("BLOCKED_CURRENT_OR_FUTURE_ACTUAL_FEATURE_LEAKAGE")

    computed_hash: str | None = None
    try:
        computed_hash = canonical_target_vector_hash(step_targets)
    except ValueError:
        blockers.append("BLOCKED_INVALID_CURRENT_STEP_VECTOR")
    if computed_hash is not None and computed_hash != str(current.target_vector_hash):
        blockers.append("BLOCKED_TARGET_VECTOR_VERSION_MISMATCH")

    predecessor_status = "MISSING"
    if predecessor is None:
        blockers.append("BLOCKED_NO_COMPLETED_PREDECESSOR")
    else:
        if not predecessor.batch_complete or not predecessor.curve_complete:
            blockers.append("BLOCKED_PREDECESSOR_INCOMPLETE")
        if str(predecessor.order_id) != str(current.order_id):
            blockers.append("BLOCKED_PREDECESSOR_ORDER_MISMATCH")
        if int(predecessor.batch_number) >= int(current.batch_number):
            blockers.append("BLOCKED_PREDECESSOR_TIME_ORDER")
        same_template = (
            str(predecessor.recipe_id) == str(current.recipe_id)
            and str(predecessor.process_version) == str(current.process_version)
            and str(predecessor.mixer_line) == str(current.mixer_line)
            and str(predecessor.template_key) == str(current.template_key)
        )
        predecessor_status = (
            "APPLICABLE_SAME_TEMPLATE"
            if same_template
            else "NOT_APPLICABLE_VERSION_OR_TEMPLATE_CHANGE"
        )

    request: DualTargetRequest | None = None
    try:
        request = DualTargetRequest(
            recipe_id=requested_recipe,
            target_mooney=target_mooney,
            target_energy_kwh=target_energy_kwh,
            mooney_spec_low=mooney_spec_low,
            mooney_spec_high=mooney_spec_high,
            strategy=strategy,
        )
        request.validate()
    except ValueError:
        blockers.append("BLOCKED_INVALID_USER_TARGET")
        request = None

    blockers = list(dict.fromkeys(blockers))
    return PreparedNextBatchRequest(
        status="BLOCKED_INPUT_CONTEXT" if blockers else "READY_FOR_MODEL_GATES",
        reason_codes=tuple(blockers),
        optimization_request=request,
        current_context=current,
        predecessor_context_status=predecessor_status,
        target_vector_hash=computed_hash,
    )


__all__ = [
    "TARGET_VECTOR_COLUMNS",
    "CurrentBatchContext",
    "CompletedBatchContext",
    "PreparedNextBatchRequest",
    "canonical_target_vector_hash",
    "prepare_next_batch_request",
]
