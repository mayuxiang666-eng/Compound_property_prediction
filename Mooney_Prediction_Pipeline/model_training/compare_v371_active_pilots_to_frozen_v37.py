"""Compare corrected Active pilot OOT results with frozen V3.7 on common batches.

The frozen model is scored with its legacy preprocessing contract on the old
Phase table, but only for batches belonging to the two corrected Active OOT
pilot cohorts.  Candidate artifacts are also reloaded and checked against the
persisted prediction CSVs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.mooney_predict_service.phase_group_contract import _add_v37_derived_features
from services.mooney_predict_service.phase_group_inference import prepare_phase_batches


BASE = ROOT / "reports" / "corrected_phase_history"
SOURCE = BASE / "phase_corrected_v371_active.csv"
BACKTEST = BASE / "v371_active_pilot_backtests_v1"
OUTPUT = BASE / "v371_active_pilot_common_baseline_v1"
LEGACY = Path(r"\\10.246.97.159\d\MNY master batch prediction model pipline\stage_statistics_enriched_all_features_weather_v4.csv")
FROZEN_DIR = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v37_production_model_package"

PILOTS = {"b00458": "B00458", "t09170": "T09170"}


def norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def metrics(groups: pd.DataFrame, prediction: str) -> dict[str, object]:
    if groups.empty:
        return {"groups": 0, "mae": None, "rmse": None, "bias": None, "r2": None}
    y = groups["ActualMNY"].to_numpy(float)
    p = groups[prediction].to_numpy(float)
    error = p - y
    return {
        "groups": int(len(groups)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "bias": float(error.mean()),
        "r2": float(r2_score(y, p)) if len(groups) > 1 else None,
        "within_2_mny_rate": float(np.mean(np.abs(error) <= 2.0)),
        "within_3_mny_rate": float(np.mean(np.abs(error) <= 3.0)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, low_memory=False)
    source["OrderID"] = norm_id(source["OrderID"])
    source["OriginalOrderID"] = norm_id(source["OriginalOrderID"])
    source["PalletID"] = norm_id(source["PalletID"])
    source["BatchNumber"] = pd.to_numeric(source["BatchNumber"], errors="raise").astype(int)
    source["LabelGroupID"] = source["OrderID"] + "|" + source["PalletID"]

    schema_frozen = json.loads((FROZEN_DIR / "feature_metadata.json").read_text(encoding="utf-8"))
    schema_core = dict(schema_frozen)
    schema_core["s1_features"] = [
        c for c in schema_frozen["s1_features"] if c != "supplier_rubber_viscosity_avg"
    ]
    prepared_core = prepare_phase_batches(source, schema_core)
    corrected_features = prepared_core.features

    legacy = pd.read_csv(LEGACY, low_memory=False)
    legacy["LegacyOrderID"] = norm_id(legacy["OrderID"])
    legacy["BatchNumber"] = pd.to_numeric(legacy["BatchNumber"], errors="coerce").astype("Int64")
    frozen_model = joblib.load(FROZEN_DIR / "hybrid_model.joblib")

    summaries = {}
    for name, pattern in PILOTS.items():
        candidate_dir = BACKTEST / f"{name}_operational_core"
        stored_batch = pd.read_csv(candidate_dir / "batch_predictions.csv", low_memory=False)
        stored_batch["OrderID"] = norm_id(stored_batch["OrderID"])
        stored_batch["BatchNumber"] = pd.to_numeric(stored_batch["BatchNumber"], errors="raise").astype(int)
        stored_batch["LabelGroupID"] = stored_batch["LabelGroupID"].astype(str)
        pilot_oot = stored_batch.loc[
            stored_batch["Split"].eq("oot_test")
            & stored_batch["CompoundName"].astype(str).str.upper().str.contains(pattern, regex=False)
        ].copy()
        group_ids = set(pilot_oot["LabelGroupID"])
        cohort = source.loc[source["LabelGroupID"].isin(group_ids)].copy()

        candidate_model_path = candidate_dir / "hybrid_model_shadow.joblib"
        candidate_model = joblib.load(candidate_model_path)
        candidate_input = corrected_features.loc[corrected_features["LabelGroupID"].isin(group_ids)].copy()
        candidate_reload, _, _, _ = candidate_model.predict(candidate_input, cluster_col="material_system")
        reload_rows = candidate_input[["OrderID", "BatchNumber"]].copy()
        reload_rows["ReloadedPrediction"] = candidate_reload
        deterministic = pilot_oot.merge(
            reload_rows, on=["OrderID", "BatchNumber"], how="left", validate="one_to_one"
        )
        deterministic_delta = (
            deterministic["RawPrediction"] - deterministic["ReloadedPrediction"]
        ).abs()

        key_map = cohort[[
            "OriginalOrderID", "BatchNumber", "LabelGroupID", "MNY", "CompoundName"
        ]].rename(columns={"OriginalOrderID": "LegacyOrderID", "MNY": "CorrectedMNY"})
        matched = legacy.merge(
            key_map, on=["LegacyOrderID", "BatchNumber"], how="inner", validate="one_to_one",
            suffixes=("", "_corrected"),
        )
        expected_batches = cohort.groupby("LabelGroupID").size()
        matched_batches = matched.groupby("LabelGroupID").size()
        complete_ids = set(expected_batches.index[expected_batches.eq(matched_batches.reindex(expected_batches.index).fillna(0))])
        matched = matched.loc[matched["LabelGroupID"].isin(complete_ids)].copy()

        legacy_features = _add_v37_derived_features(matched)
        feature_cols = list(dict.fromkeys(schema_frozen["s1_features"] + schema_frozen["s2_features"]))
        for column in feature_cols:
            if column not in legacy_features:
                legacy_features[column] = 0.0
            legacy_features[column] = pd.to_numeric(legacy_features[column], errors="coerce").fillna(0.0)
        frozen_prediction, _, _, _ = frozen_model.predict(legacy_features, cluster_col="material_system")
        matched["FrozenLegacyPrediction"] = frozen_prediction
        frozen_groups = matched.groupby("LabelGroupID", as_index=False).agg(
            ActualMNY=("CorrectedMNY", "median"),
            FrozenLegacyPrediction=("FrozenLegacyPrediction", "mean"),
            FrozenBatchCount=("BatchNumber", "size"),
        )

        new_groups = pilot_oot.groupby("LabelGroupID", as_index=False).agg(
            ActualMNY=("MNY", "median"),
            CandidatePrediction=("RawPrediction", "mean"),
            CandidateBatchCount=("BatchNumber", "size"),
        )
        common = new_groups.merge(
            frozen_groups, on=["LabelGroupID", "ActualMNY"], how="inner", validate="one_to_one"
        )
        common.to_csv(OUTPUT / f"{name}_common_oot_group_predictions.csv", index=False, encoding="utf-8-sig")
        summaries[name] = {
            "expected_oot_groups": int(len(group_ids)),
            "expected_oot_batches": int(len(cohort)),
            "candidate_reloaded_batches": int(deterministic["ReloadedPrediction"].notna().sum()),
            "candidate_reload_max_abs_delta": float(deterministic_delta.max()) if len(deterministic_delta) else None,
            "legacy_matched_complete_groups": int(len(complete_ids)),
            "common_groups": int(len(common)),
            "common_group_coverage": float(len(common) / max(len(group_ids), 1)),
            "frozen_v37_legacy_metrics": metrics(common, "FrozenLegacyPrediction"),
            "v371_operational_core_metrics": metrics(common, "CandidatePrediction"),
            "candidate_model_sha256": sha256(candidate_model_path),
        }

    result = {
        "experiment": "V3.7-FROZEN-LEGACY-VS-V3.7.1-CORRECTED-ACTIVE-COMMON-OOT-V1",
        "comparison_contract": "SAME_ACTIVE_PILOT_OOT_GROUPS; EACH_MODEL_USES_ITS_OWN_DATA_CONTRACT",
        "pilots": summaries,
        "frozen_model_sha256": sha256(FROZEN_DIR / "hybrid_model.joblib"),
        "database_writes": False,
        "automatic_control": "DISABLED",
    }
    (OUTPUT / "comparison_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
