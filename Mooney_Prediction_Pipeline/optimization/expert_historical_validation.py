"""History-derived bounds, execution audits, and automatic Silica pilot selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.optimization.expert_shadow_policy import TRACKING_GATES, tracking_summary


def attach_rotor_history_bounds(steps: pd.DataFrame, history_steps: pd.DataFrame, *, low: float = 0.05, high: float = 0.95, minimum_samples: int = 5) -> pd.DataFrame:
    """Attach exact equipment/group/step rotor P5-P95 bounds, fail-closed when absent."""
    required = {"EquipmentID", "GroupName", "StepNo", "rotor_SP"}
    if required - set(history_steps.columns):
        raise ValueError("Rotor history must contain EquipmentID, GroupName, StepNo, rotor_SP.")
    history = history_steps.copy()
    history["rotor_SP"] = pd.to_numeric(history["rotor_SP"], errors="coerce")
    history = history[history["rotor_SP"].gt(0)].dropna(subset=["rotor_SP"])
    grouped = history.groupby(["EquipmentID", "GroupName", "StepNo"])["rotor_SP"].agg(
        step_rotor_min=lambda values: values.quantile(low),
        step_rotor_max=lambda values: values.quantile(high),
        step_rotor_nominal="median",
        step_rotor_sample_count="count",
    ).reset_index()
    grouped = grouped[grouped["step_rotor_sample_count"] >= minimum_samples]
    out = steps.merge(grouped, on=["EquipmentID", "GroupName", "StepNo"], how="left")
    return out


def discharge_temperature_tracking_audit(batch_links: pd.DataFrame) -> dict[str, float | bool]:
    """Audit cross-step MSB discharge Target-versus-final-actual tracking."""
    required = {"discharge_temp_target", "discharge_temp_actual"}
    if required - set(batch_links.columns):
        raise ValueError(f"Missing discharge tracking columns: {sorted(required - set(batch_links.columns))}")
    errors = pd.to_numeric(batch_links["discharge_temp_actual"], errors="coerce") - pd.to_numeric(batch_links["discharge_temp_target"], errors="coerce")
    gate = TRACKING_GATES["temperature"]
    result = tracking_summary(errors, limit=gate["p95"])
    return {**result, "sample_count": int(errors.notna().sum()), "control_tracking": "PASS" if result["passed"] else "FAIL"}


def select_silica_pilot(validation_summary: pd.DataFrame) -> pd.DataFrame:
    """Return eligible Silica templates in deterministic readiness order.

    The caller supplies one row per Compound/Equipment/Route/Template after
    historical extraction.  This separation keeps selection auditable and
    avoids silently mixing route or template populations.
    """
    required = {
        "CompoundName", "material_system", "EquipmentID", "route_key", "template_key",
        "n_labeled_batches", "template_consistency", "stage6_duration_levels",
        "discharge_temperature_levels", "temperature_tracking_pass", "endpoint_complete_rate",
    }
    missing = required - set(validation_summary.columns)
    if missing:
        raise ValueError(f"Missing Silica selection columns: {sorted(missing)}")
    candidates = validation_summary.copy()
    eligible = (
        candidates["material_system"].eq("Silica")
        & candidates["n_labeled_batches"].ge(150)
        & candidates["template_consistency"].ge(0.90)
        & candidates["stage6_duration_levels"].ge(3)
        & candidates["discharge_temperature_levels"].ge(3)
        & candidates["temperature_tracking_pass"].fillna(False).astype(bool)
        & candidates["endpoint_complete_rate"].ge(0.98)
    )
    out = candidates.loc[eligible].copy()
    out["pilot_readiness"] = "ELIGIBLE"
    out = out.sort_values(
        ["n_labeled_batches", "template_consistency", "endpoint_complete_rate", "CompoundName"],
        ascending=[False, False, False, True], kind="stable",
    )
    return out.reset_index(drop=True)
