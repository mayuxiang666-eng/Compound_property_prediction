"""Parity v4 contract with isolated evidence output directory."""

from __future__ import annotations

from Mooney_Prediction_Pipeline.feature_engineering.material_dependent_features import recompute_material_dependent_features
from Mooney_Prediction_Pipeline.optimization.audit_corrected_operational_parity_v2 import add_derived as add_pid_cb_features
from Mooney_Prediction_Pipeline.optimization import audit_corrected_operational_parity_v3 as audit


def corrected_add_derived(frame):
    return add_pid_cb_features(recompute_material_dependent_features(frame))


audit.add_derived = corrected_add_derived
audit.OUT = audit.OUT / "recomputed_material_v4"


if __name__ == "__main__":
    audit.main()
