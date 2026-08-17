"""Complete the Active feature-drift audit with independent Phase presence checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history"
CORRECTED = BASE / "phase_corrected_v371_active.csv"
LEGACY = Path(r"\\10.246.97.159\d\MNY master batch prediction model pipline\stage_statistics_enriched_all_features_weather_v4.csv")
OUTPUT = BASE / "v371_active_feature_drift_v1"
DETAIL = OUTPUT / "feature_drift_all_active_batches.csv"


def norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def main() -> None:
    phase_durations = [
        "Stage1_Loading_Duration", "Stage2_DryMixing_Duration", "Stage3_OilLoading_Duration",
        "Stage4_WetMixing_Duration", "Stage5_PID_Duration", "Stage6_BottomMixing_Duration",
    ]
    corrected = pd.read_csv(
        CORRECTED,
        usecols=["OriginalOrderID", "BatchNumber", "CompoundName", *phase_durations],
        low_memory=False,
    )
    corrected["LegacyOrderID"] = norm_id(corrected["OriginalOrderID"])
    corrected["BatchNumber"] = pd.to_numeric(corrected["BatchNumber"], errors="raise").astype(int)
    legacy = pd.read_csv(
        LEGACY,
        usecols=["OrderID", "BatchNumber", *phase_durations],
        low_memory=False,
    )
    legacy["LegacyOrderID"] = norm_id(legacy["OrderID"])
    legacy["BatchNumber"] = pd.to_numeric(legacy["BatchNumber"], errors="coerce").astype("Int64")
    matched = corrected.merge(
        legacy[["LegacyOrderID", "BatchNumber", *phase_durations]],
        on=["LegacyOrderID", "BatchNumber"], how="left", validate="one_to_one",
        suffixes=("_corrected", "_legacy"), indicator=True,
    )

    transitions = {}
    for feature in phase_durations:
        old = pd.to_numeric(matched[f"{feature}_legacy"], errors="coerce")
        new = pd.to_numeric(matched[f"{feature}_corrected"], errors="coerce")
        transitions[feature] = {
            "legacy_present_corrected_zero": int(((old > 0) & (new <= 0)).sum()),
            "legacy_zero_corrected_present": int(((old <= 0) & (new > 0)).sum()),
            "both_present": int(((old > 0) & (new > 0)).sum()),
            "both_zero": int(((old <= 0) & (new <= 0)).sum()),
            "legacy_missing": int(old.isna().sum()),
            "corrected_missing": int(new.isna().sum()),
        }

    detail = pd.read_csv(DETAIL, low_memory=False)
    top = detail.head(30).replace({np.nan: None}).to_dict("records")
    summary = {
        "contract": "LEGACY_V37_VS_CORRECTED_V371_MATCHED_ACTIVE_FEATURE_DRIFT_V2",
        "corrected_batches": int(len(corrected)),
        "legacy_matched_batches": int(matched["_merge"].eq("both").sum()),
        "feature_count": int(len(detail)),
        "phase_duration_presence_transitions": transitions,
        "top_30_normalized_feature_drifts": top,
        "interpretation_guard": "A corrected zero is valid only for an absent/non-applicable phase; no missing value is imputed here.",
        "automatic_control": "DISABLED",
    }
    (OUTPUT / "feature_drift_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
