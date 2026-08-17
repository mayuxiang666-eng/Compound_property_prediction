"""Checkpointed R11E: corrected Step/Phase level model plus within-order delta model.

This runner intentionally uses a compact, deterministic, single-thread search.  It
exists because the broader R11C search did not finish within the execution window.
All trend metrics and delta training pairs are strictly within OrderID.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training import run_v371_r11c_order_metric_reduced as r11c
from Mooney_Prediction_Pipeline.model_training import run_v371_r11_step_gated_delta as core
from Mooney_Prediction_Pipeline.model_training.order_atomic_trend_metrics import candidate_score, metric_record


OUTPUT = core.BASE / "v371_r11e_checkpointed_order_model"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r11e_checkpointed_order_model"


def estimator(objective: str, leaves: int, min_child: int, trees: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective=objective,
        n_estimators=trees,
        learning_rate=0.025,
        num_leaves=leaves,
        min_child_samples=min_child,
        colsample_bytree=0.72,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=4.0,
        reg_lambda=18.0,
        random_state=42,
        verbosity=-1,
        n_jobs=1,
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    groups, usable, mandatory = r11c.prepare()
    train = groups.loc[groups["split"].eq("train")]
    center = float(train["MNY"].mean())
    features = core.select_features(train, usable, "MNY", 60, mandatory)
    print(f"PREPARED groups={len(groups)} features={len(features)} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    candidates: dict[str, np.ndarray] = {}
    bundles: dict[str, dict] = {}
    rows: list[dict] = []
    for objective, leaves, min_child in (("regression_l1", 3, 18), ("huber", 5, 25)):
        model = estimator(objective, leaves, min_child, 180)
        model.fit(train[features], train["MNY"])
        raw = model.predict(groups[features])
        for amplitude in (1.0, 1.25, 1.50, 1.75):
            prediction = center + amplitude * (raw - center)
            name = f"R11E_{objective.upper()}_L{leaves}_A{int(amplitude*100):03d}"
            score, level, trend = candidate_score(groups, prediction)
            candidates[name] = prediction
            bundles[name] = {"model": model, "features": features, "amplitude": amplitude, "center": center}
            rows.append({"candidate": name, "score": score, **level, **trend})
        pd.DataFrame(rows).to_csv(OUTPUT / "r11e_level_validation_search.csv", index=False)
        print(f"LEVEL_FIT objective={objective} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    search = pd.DataFrame(rows).sort_values(["score", "mae", "candidate"], kind="stable")
    level_name = str(search.iloc[0]["candidate"])
    groups["LevelPrediction"] = candidates[level_name]
    groups.to_pickle(OUTPUT / "r11e_level_checkpoint.pkl")
    joblib.dump(
        {"contract": "V3.7.1-R11E-LEVEL-CHECKPOINT-V1", "selected": level_name, "bundle": bundles[level_name], "trend_evaluation": "WITHIN_ORDERID_ONLY"},
        MODEL_OUTPUT / "r11e_level_checkpoint.joblib",
    )
    print(f"LEVEL_CHECKPOINT selected={level_name} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    pairs = core.build_pairs(groups, features)
    train_pairs = pairs.loc[pairs["split"].eq("train")]
    delta_pool = [column for column in pairs if column.startswith("delta__")]
    selected_delta = core.select_features(train_pairs, delta_pool, "ActualDelta", 20, [])
    delta_model = estimator("regression_l1", 3, 5, 160)
    delta_model.fit(train_pairs[selected_delta], train_pairs["ActualDelta"])
    pair_prediction = pairs[["from_group", "to_group", "OrderID", "split", "ActualDelta"]].copy()
    pair_prediction["DeltaPrediction"] = delta_model.predict(pairs[selected_delta])

    delta_rows: list[dict] = []
    delta_candidates: dict[str, np.ndarray] = {}
    for beta in (0.0, 0.25, 0.50, 0.75, 1.0):
        prediction = core.recursive_prediction(groups, groups["LevelPrediction"].to_numpy(), pair_prediction, beta)
        name = f"R11E_DELTA_B{int(beta*100):03d}"
        score, level, trend = candidate_score(groups, prediction)
        delta_candidates[name] = prediction
        delta_rows.append({"candidate": name, "score": score, **level, **trend})
    delta_search = pd.DataFrame(delta_rows).sort_values(["score", "mae", "candidate"], kind="stable")
    delta_name = str(delta_search.iloc[0]["candidate"])
    groups["FinalPrediction"] = delta_candidates[delta_name]

    records: list[dict] = []
    for prediction_col, candidate in (("LevelPrediction", "R11E_LEVEL"), ("FinalPrediction", "R11E_FINAL")):
        for split_name in ("val", "oot"):
            records.append(metric_record(groups, prediction_col, split_name, candidate))
            for regime in ("NORMAL_COMPLETE_ROUTE", "PROCESS_UPSET"):
                subset = groups.loc[groups["regime"].eq(regime)]
                if subset.loc[subset["split"].eq(split_name)].empty:
                    continue
                record = metric_record(subset, prediction_col, split_name, f"{candidate}_{regime}")
                record["regime"] = regime
                records.append(record)
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "r11e_metrics.csv", index=False)
    delta_search.to_csv(OUTPUT / "r11e_delta_validation_search.csv", index=False)
    pair_prediction.to_csv(OUTPUT / "r11e_pair_predictions.csv", index=False)
    groups[["LabelGroupID", "OrderID", "PalletID", "RecipeID", "split", "regime", "MNY", "LevelPrediction", "FinalPrediction", "production_start", "lab_time"]].to_csv(OUTPUT / "r11e_group_predictions.csv", index=False)

    bundle = {
        "contract": "V3.7.1-R11E-CHECKPOINTED-STEP-ORDER-DELTA-V1",
        "level_candidate": level_name,
        "level": bundles[level_name],
        "delta_candidate": delta_name,
        "delta": {"model": delta_model, "features": selected_delta, "beta": float(delta_name.rsplit("B", 1)[1]) / 100.0},
        "trend_evaluation": "WITHIN_ORDERID_ONLY",
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
    }
    model_path = MODEL_OUTPUT / "r11e_step_order_delta_shadow.joblib"
    joblib.dump(bundle, model_path)
    val = metrics.loc[(metrics["candidate"] == "R11E_FINAL") & (metrics["split"] == "val")].iloc[0]
    oot = metrics.loc[(metrics["candidate"] == "R11E_FINAL") & (metrics["split"] == "oot")].iloc[0]
    summary = {
        "contract": bundle["contract"],
        "groups": len(groups),
        "selected_level": level_name,
        "selected_delta": delta_name,
        "pair_counts_by_split": pairs["split"].value_counts().to_dict(),
        "validation": val.to_dict(),
        "oot": oot.to_dict(),
        "target_mae_mny": 2.5,
        "target_trend": {"delta_pearson_min": 0.5, "direction_accuracy_min": 0.65, "delta_std_ratio_range": [0.7, 1.3]},
        "release_status": "NOT_RELEASED_GATES_FAILED_OR_BLIND_WINDOW_PENDING",
        "database_access_during_run": False,
        "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "elapsed_seconds": time.perf_counter() - started,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUTPUT / "r11e_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
