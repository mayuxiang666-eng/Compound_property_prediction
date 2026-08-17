"""Dependency-free runner for environments where pytest is not installed."""

from test_feature_parity_gate import (
    test_feature_drift_blocks_even_if_prediction_matches,
    test_missing_model_column_is_a_hard_block,
    test_parity_requires_all_features_and_prediction_within_point_one,
)
from test_phase_group_contract import (
    test_batch_first_then_order_pallet_group_mean_and_lab_removed,
    test_non_route_missing_qm_is_blocked_not_zero_filled,
)


def main() -> None:
    checks = [
        test_batch_first_then_order_pallet_group_mean_and_lab_removed,
        test_non_route_missing_qm_is_blocked_not_zero_filled,
        test_parity_requires_all_features_and_prediction_within_point_one,
        test_feature_drift_blocks_even_if_prediction_matches,
        test_missing_model_column_is_a_hard_block,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")


if __name__ == "__main__":
    main()
