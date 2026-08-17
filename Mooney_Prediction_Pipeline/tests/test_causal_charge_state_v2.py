from Mooney_Prediction_Pipeline.feature_engineering.causal_charge_state_v2 import normalize_charge_lot


def test_storage_prefix_is_removed_from_supplier_lot():
    assert normalize_charge_lot("S040340332") == "0340332"
    assert normalize_charge_lot("S060340332") == "0340332"
    assert normalize_charge_lot("9000345405") == "9000345405"
