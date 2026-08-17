"""Re-run only the R11E delta stage with type-stable OrderID recursion."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training import run_v371_r11_step_gated_delta as core
from Mooney_Prediction_Pipeline.model_training.order_atomic_trend_metrics import candidate_score, metric_record
from Mooney_Prediction_Pipeline.model_training.order_delta_recursion import recursive_order_prediction
from Mooney_Prediction_Pipeline.model_training.run_v371_r11e_checkpointed_order_model import estimator


SOURCE = core.BASE / "v371_r11e_checkpointed_order_model"
LEVEL_MODEL = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r11e_checkpointed_order_model" / "r11e_level_checkpoint.joblib"
OUTPUT = core.BASE / "v371_r11f_fixed_order_delta"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r11f_fixed_order_delta"


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    groups = pd.read_pickle(SOURCE / "r11e_level_checkpoint.pkl")
    level_bundle = joblib.load(LEVEL_MODEL)
    features = level_bundle["bundle"]["features"]
    pairs = core.build_pairs(groups, features)
    train_pairs = pairs.loc[pairs["split"].eq("train")]
    delta_pool = [column for column in pairs if column.startswith("delta__")]
    selected = core.select_features(train_pairs, delta_pool, "ActualDelta", 20, [])
    print(f"PAIRS total={len(pairs)} train={len(train_pairs)} selected={len(selected)}", flush=True)
    model = estimator("regression_l1", 3, 5, 160)
    model.fit(train_pairs[selected], train_pairs["ActualDelta"])
    pair_prediction = pairs[["from_group", "to_group", "OrderID", "split", "ActualDelta"]].copy()
    pair_prediction["DeltaPrediction"] = model.predict(pairs[selected])

    candidates = {}
    rows = []
    for beta in (0.0, 0.25, 0.50, 0.75, 1.0):
        prediction = recursive_order_prediction(groups, groups["LevelPrediction"].to_numpy(), pair_prediction, beta)
        name = f"R11F_FIXED_DELTA_B{int(beta*100):03d}"
        score, level, trend = candidate_score(groups, prediction)
        candidates[name] = prediction
        rows.append({"candidate": name, "score": score, **level, **trend})
    search = pd.DataFrame(rows).sort_values(["score", "mae", "candidate"], kind="stable")
    selected_name = str(search.iloc[0]["candidate"])
    groups = groups.copy()
    groups["FinalPrediction"] = candidates[selected_name]

    records = []
    for split_name in ("val", "oot"):
        records.append(metric_record(groups, "LevelPrediction", split_name, "R11E_LEVEL"))
        records.append(metric_record(groups, "FinalPrediction", split_name, "R11F_FINAL"))
    metrics = pd.DataFrame(records)
    search.to_csv(OUTPUT / "r11f_delta_validation_search.csv", index=False)
    metrics.to_csv(OUTPUT / "r11f_metrics.csv", index=False)
    pair_prediction.to_csv(OUTPUT / "r11f_pair_predictions.csv", index=False)
    pd.DataFrame({"feature": selected}).to_csv(OUTPUT / "r11f_delta_features.csv", index=False)
    groups[["LabelGroupID", "OrderID", "PalletID", "RecipeID", "split", "regime", "MNY", "LevelPrediction", "FinalPrediction", "production_start", "lab_time"]].to_csv(OUTPUT / "r11f_group_predictions.csv", index=False)
    beta = float(selected_name.rsplit("B", 1)[1]) / 100.0
    bundle = {
        "contract": "V3.7.1-R11F-FIXED-ORDER-DELTA-V1",
        "level": level_bundle,
        "delta": {"model": model, "features": selected, "beta": beta},
        "trend_evaluation": "WITHIN_ORDERID_ONLY",
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
    }
    joblib.dump(bundle, MODEL_OUTPUT / "r11f_fixed_order_delta_shadow.joblib")
    val = metrics.loc[(metrics["candidate"] == "R11F_FINAL") & (metrics["split"] == "val")].iloc[0]
    oot = metrics.loc[(metrics["candidate"] == "R11F_FINAL") & (metrics["split"] == "oot")].iloc[0]
    summary = {
        "contract": bundle["contract"],
        "selected_delta": selected_name,
        "selected_beta": beta,
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
    (OUTPUT / "r11f_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
