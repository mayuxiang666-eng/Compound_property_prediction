from __future__ import annotations

import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.v371_challenger.qm_contract import (
    aggregate_batch_qm_with_status,
    charge_key,
    qm_category,
)


def test_charge_key_normalizes_separators_and_leading_zeroes():
    key = charge_key(" 00-ab_12 ")
    assert key.exact == "00-AB_12"
    assert key.compact == "00AB12"
    assert key.no_leading_zero == "AB12"


def test_rubber_parameter_variants_are_recognized():
    assert qm_category("CE123", "UML 1+4") == "supplier_rubber_viscosity_avg"
    assert qm_category("CN100", "Mooney Viscosity ML 1+4") == "supplier_rubber_viscosity_avg"


def test_three_state_qm_contract_and_weighted_aggregation():
    charges = pd.DataFrame([
        {"OrderID": "1", "BatchNumber": 1, "MaterialCode": "CE100", "ChargeID": "A-1", "ActualWeight": 3.0},
        {"OrderID": "1", "BatchNumber": 1, "MaterialCode": "CE101", "ChargeID": "A2", "ActualWeight": 1.0},
        {"OrderID": "1", "BatchNumber": 1, "MaterialCode": "CS1056", "ChargeID": "S1", "ActualWeight": 2.0},
    ])
    qm = pd.DataFrame([
        {"batch": "A1", "material": "CE100", "parameter": "UML 1+4", "mean_value": 40.0},
        {"batch": "A2", "material": "CE101", "parameter": "UML 1+4", "mean_value": 60.0},
    ])
    result = aggregate_batch_qm_with_status(charges, qm).iloc[0]
    assert np.isclose(result["supplier_rubber_viscosity_avg"], 45.0)
    assert result["supplier_rubber_viscosity_avg__status"] == "OBSERVED"
    assert result["supplier_silica_moisture_avg__status"] == "SOURCE_DATA_MISSING"
    assert result["supplier_carbon_black_structure_avg__status"] == "ROUTE_NOT_APPLICABLE"

