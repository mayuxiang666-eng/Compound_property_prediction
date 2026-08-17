"""Audit legacy-versus-corrected feature drift on matched Active batches."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = ROOT / "Mooney_Prediction_Pipeline"
for path in (ROOT, PIPELINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from services.mooney_predict_service.phase_group_contract import _add_v37_derived_features


BASE = ROOT / "reports" / "corrected_phase_history"
CORRECTED = BASE / "phase_corrected_v371_active.csv"
LEGACY = Path(r"\\10.246.97.159\d\MNY master batch prediction model pipline\stage_statistics_enriched_all_features_weather_v4.csv")
SCHEMA = PIPELINE_ROOT / "models" / "v37_production_model_package" / "feature_metadata.json"
OUTPUT = BASE / "v371_active_feature_drift_v1"


def norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    feature_cols = list(dict.fromkeys(schema["s1_features"] + schema["s2_features"]))

    corrected = pd.read_csv(CORRECTED, low_memory=False)
    corrected["LegacyOrderID"] = norm_id(corrected["OriginalOrderID"])
    corrected["BatchNumber"] = pd.to_numeric(corrected["BatchNumber"], errors="raise").astype(int)
    corrected = _add_v37_derived_features(corrected)

    legacy = pd.read_csv(LEGACY, low_memory=False)
    legacy["LegacyOrderID"] = norm_id(legacy["OrderID"])
    legacy["BatchNumber"] = pd.to_numeric(legacy["BatchNumber"], errors="coerce").astype("Int64")
    legacy = _add_v37_derived_features(legacy)

    identity = corrected[[
        "LegacyOrderID", "BatchNumber", "OrderID", "PalletID", "CompoundName", "material_system"
    ]].copy()
    old_cols = ["LegacyOrderID", "BatchNumber"] + feature_cols
    new_cols = ["LegacyOrderID", "BatchNumber"] + feature_cols
    matched = identity.merge(
        legacy[old_cols], on=["LegacyOrderID", "BatchNumber"], how="left", validate="one_to_one"
    ).merge(
        corrected[new_cols], on=["LegacyOrderID", "BatchNumber"], how="left", validate="one_to_one",
        suffixes=("_legacy", "_corrected"),
    )

    rows = []
    for feature in feature_cols:
        old = pd.to_numeric(matched[f"{feature}_legacy"], errors="coerce")
        new = pd.to_numeric(matched[f"{feature}_corrected"], errors="coerce")
        paired = old.notna() & new.notna()
        difference = (new[paired] - old[paired]).abs()
        old_std = float(old[paired].std()) if paired.any() else np.nan
        rows.append({
            "feature": feature,
            "stage": "stage1" if feature in schema["s1_features"] else "stage2",
            "paired_batches": int(paired.sum()),
            "legacy_missing_batches": int(old.isna().sum()),
            "corrected_missing_batches": int(new.isna().sum()),
            "mean_abs_diff": float(difference.mean()) if len(difference) else None,
            "median_abs_diff": float(difference.median()) if len(difference) else None,
            "p95_abs_diff": float(difference.quantile(0.95)) if len(difference) else None,
            "legacy_std": old_std if np.isfinite(old_std) else None,
            "normalized_mae_by_legacy_std": (
                float(difference.mean() / old_std) if len(difference) and np.isfinite(old_std) and old_std > 1e-12 else None
            ),
            "pearson": float(old[paired].corr(new[paired])) if paired.sum() > 1 else None,
            "exact_match_rate": float(np.isclose(old[paired], new[paired], equal_nan=True).mean()) if paired.any() else None,
            "legacy_nonzero_corrected_zero": int(((old.abs() > 1e-12) & (new.abs() <= 1e-12)).sum()),
            "legacy_zero_corrected_nonzero": int(((old.abs() <= 1e-12) & (new.abs() > 1e-12)).sum()),
        })
    audit = pd.DataFrame(rows).sort_values(
        ["normalized_mae_by_legacy_std", "mean_abs_diff"], ascending=False, na_position="last"
    )
    audit.to_csv(OUTPUT / "feature_drift_all_active_batches.csv", index=False, encoding="utf-8-sig")

    phase_durations = [
        "Stage1_Loading_Duration", "Stage2_DryMixing_Duration", "Stage3_OilLoading_Duration",
        "Stage4_WetMixing_Duration", "Stage5_PID_Duration", "Stage6_BottomMixing_Duration",
    ]
    phase_zero_transitions = {}
    for feature in phase_durations:
        old = pd.to_numeric(matched[f"{feature}_legacy"], errors="coerce")
        new = pd.to_numeric(matched[f"{feature}_corrected"], errors="coerce")
        phase_zero_transitions[feature] = {
            "legacy_present_corrected_zero": int(((old > 0) & (new <= 0)).sum()),
            "legacy_zero_corrected_present": int(((old <= 0) & (new > 0)).sum()),
            "both_present": int(((old > 0) & (new > 0)).sum()),
            "both_zero": int(((old <= 0) & (new <= 0)).sum()),
        }

    top = audit.head(30).replace({np.nan: None}).to_dict("records")
    summary = {
        "contract": "LEGACY_V37_VS_CORRECTED_V371_MATCHED_ACTIVE_FEATURE_DRIFT_V1",
        "corrected_batches": int(len(corrected)),
        "legacy_matched_batches": int(matched[f"{feature_cols[0]}_legacy"].notna().sum()),
        "feature_count": int(len(feature_cols)),
        "phase_duration_presence_transitions": phase_zero_transitions,
        "top_30_normalized_feature_drifts": top,
        "interpretation_guard": "A corrected zero is valid only for an absent/non-applicable phase; this audit does not impute missing values.",
        "automatic_control": "DISABLED",
    }
    (OUTPUT / "feature_drift_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
