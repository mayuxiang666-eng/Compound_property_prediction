import pandas as pd

from Mooney_Prediction_Pipeline.model_training.r5_order_state_features import (
    add_prior_order_state,
    build_adjacent_order_pairs,
    build_group_target_features,
    parse_target_vector,
)


def test_target_vector_parser_preserves_numeric_step_values():
    raw = (
        "AVR_MST:3:Rotor-Speed-1/min-Rpm:40"
        "|AVR_MST:3:SCP-3-Temperature-C-F:110"
    )
    parsed = parse_target_vector(raw)
    assert parsed["AVR_MST:3:Rotor-Speed-1/min-Rpm"] == 40
    assert parsed["AVR_MST:3:SCP-3-Temperature-C-F"] == 110


def test_group_target_builder_blocks_inconsistent_vectors():
    source = pd.DataFrame(
        [
            {
                "LabelGroupID": "G1", "TargetVectorRaw": "A:1", "TargetVectorHash": "H1",
                "StepPatternHash": "S", "TemplateKey": "T",
                "BatchStartTime": "2026-01-01 00:00", "BatchEndTime": "2026-01-01 00:10",
            },
            {
                "LabelGroupID": "G1", "TargetVectorRaw": "A:2", "TargetVectorHash": "H2",
                "StepPatternHash": "S", "TemplateKey": "T",
                "BatchStartTime": "2026-01-01 00:11", "BatchEndTime": "2026-01-01 00:20",
            },
        ]
    )
    try:
        build_group_target_features(source)
    except ValueError as error:
        assert "INCONSISTENT" in str(error)
    else:
        raise AssertionError("Expected inconsistent Target vectors to fail")


def test_prior_state_requires_previous_group_to_be_complete():
    groups = pd.DataFrame(
        [
            {
                "LabelGroupID": "G1", "OrderID": "O1",
                "GroupProductionStart": "2026-01-01 00:00",
                "GroupProductionComplete": "2026-01-01 01:00", "x": 10,
            },
            {
                "LabelGroupID": "G2", "OrderID": "O1",
                "GroupProductionStart": "2026-01-01 01:10",
                "GroupProductionComplete": "2026-01-01 02:00", "x": 13,
            },
        ]
    )
    result, features = add_prior_order_state(groups, state_features=["x"])
    second = result.loc[result.LabelGroupID.eq("G2")].iloc[0]
    assert second["prior_order_state_available"] == 1
    assert second["prior__x"] == 10
    assert second["delta_prior__x"] == 3
    assert "delta_prior__x" in features


def test_delta_pairs_refuse_non_atomic_orders():
    groups = pd.DataFrame(
        [
            {
                "LabelGroupID": "G1", "OrderID": "O1", "split": "train",
                "GroupProductionStart": "2026-01-01", "MNY": 50,
                "LevelPrediction": 51, "x": 1,
            },
            {
                "LabelGroupID": "G2", "OrderID": "O1", "split": "val",
                "GroupProductionStart": "2026-01-02", "MNY": 52,
                "LevelPrediction": 52, "x": 2,
            },
        ]
    )
    try:
        build_adjacent_order_pairs(groups, feature_columns=["x"])
    except ValueError as error:
        assert "order-atomic" in str(error)
    else:
        raise AssertionError("Expected non-atomic pair source to fail")
