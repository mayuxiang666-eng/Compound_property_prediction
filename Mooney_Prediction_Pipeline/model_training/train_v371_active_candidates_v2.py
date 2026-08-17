"""V2 runner: preserve temporal-coverage blockers and continue other candidates."""

from __future__ import annotations

import json

from Mooney_Prediction_Pipeline.model_training import train_v371_active_candidates as v1


_original_train_candidate = v1.train_candidate


def _train_candidate_or_record_blocker(name, source, schema, order_splits, output_dir):
    try:
        return _original_train_candidate(name, source, schema, order_splits, output_dir)
    except ValueError as exc:
        if "empty chronological partition" not in str(exc):
            raise
        candidate_dir = output_dir / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        features, quality, groups = v1.build_candidate_frame(source, schema, order_splits)
        quality.to_csv(candidate_dir / "feature_quality_audit.csv", index=False, encoding="utf-8-sig")
        groups.to_csv(candidate_dir / "group_eligibility_and_split.csv", index=False, encoding="utf-8-sig")
        partition_names = ("train", "validation", "oot_test")
        blocked = {
            "candidate": name,
            "status": "BLOCKED_INSUFFICIENT_TEMPORAL_PARTITION_COVERAGE",
            "blocking_reason": "At least one chronological partition has zero complete label groups",
            "data_contract": "CORRECTED_ACTIVE_PHASE_NO_CROSS_BATCH_CURVE_FILL",
            "label_contract": "BATCH_PREDICT_THEN_ORDERID_PALLETID_MEAN",
            "split_contract": "ORDER_ATOMIC_CHRONOLOGICAL_70_15_15_BY_GROUP_CUMULATIVE_COUNT",
            "dropped_features": schema.get("dropped_features", []),
            "accepted_batches": int(len(features)),
            "partition_batches": {
                key: int((features["Split"] == key).sum()) for key in partition_names
            },
            "partition_groups": {
                key: int(features.loc[features["Split"].eq(key), "LabelGroupID"].nunique())
                for key in partition_names
            },
            "lab_used_as_feature": False,
            "automatic_control": "DISABLED",
            "recommendation_status": "BLOCKED_NO_RECENT_OOT",
        }
        (candidate_dir / "summary.json").write_text(
            json.dumps(blocked, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return blocked


def main() -> None:
    v1.train_candidate = _train_candidate_or_record_blocker
    v1.main()


if __name__ == "__main__":
    main()
