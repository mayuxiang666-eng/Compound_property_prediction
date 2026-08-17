"""Final corrected Phase contract parity: explicit Stage6 ram N/A=0."""

from __future__ import annotations

from pathlib import Path

from Mooney_Prediction_Pipeline.feature_engineering.corrected_phase_contract import apply_corrected_phase_contract
from Mooney_Prediction_Pipeline.optimization import audit_corrected_operational_parity_v3 as audit


audit.add_derived = apply_corrected_phase_contract
audit.OUT = Path(__file__).resolve().parents[2] / "reports" / "phase_parity_v5"


if __name__ == "__main__":
    audit.main()
