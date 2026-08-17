"""R11b causal Charge-state residual ablation on the locked R11 predictions."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.run_v371_r8_batch_group_mean import BASE, candidate_score, metric_record, model


R11_SOURCE = BASE / "v371_r11_step_gated_delta" / "r11_group_predictions.csv"
CHARGE_SOURCE = BASE / "t15760_causal_charge_state_v2" / "t15760_causal_charge_features_v2.csv"
STEP_SOURCE = BASE / "t15760_within_order_contrast" / "step_execution" / "t15760_group_step_execution_features.csv"
OUTPUT = BASE / "v371_r11b_charge_ablation"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r11b_charge_ablation"


def select(train: pd.DataFrame, pool: list[str], top_k: int) -> list[str]:
    rows = []
    for feature in pool:
        values = pd.to_numeric(train[feature], errors="coerce")
        valid = values.notna() & train["Residual"].notna()
        if int(valid.sum()) < 35 or values.loc[valid].nunique() < 3:
            continue
        corr = values.loc[valid].corr(train.loc[valid, "Residual"], method="spearman")
        rows.append((abs(float(corr)) if pd.notna(corr) else 0.0, feature))
    return [feature for _, feature in sorted(rows, key=lambda item: (-item[0], item[1]))[:top_k]]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(R11_SOURCE, low_memory=False)
    charge = pd.read_csv(CHARGE_SOURCE, low_memory=False)
    step = pd.read_csv(STEP_SOURCE, low_memory=False)
    charge_features = [column for column in charge.columns if column.startswith("charge_state__")]
    process_candidates = [
        column for column in step.columns
        if column.startswith("step_exec__") and any(token in column for token in (
            "S3__temp", "S4__temp", "S5__time_increment", "S5__temp",
            "S7__time_increment", "S7__energy", "S7__temp",
            "S8__time", "S9__time", "S10__time", "MSB__S1__time",
        ))
    ]
    step = step[["LabelGroupID"] + process_candidates]
    frame = frame.merge(charge, on="LabelGroupID", how="left", validate="one_to_one").merge(step, on="LabelGroupID", how="left", validate="one_to_one")
    pool = charge_features + process_candidates
    for feature in pool:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame["Residual"] = frame["MNY"] - frame["FinalPrediction"]
    train = frame.loc[frame["split"].eq("train")]
    candidates, bundles, rows = {}, {}, []
    for top_k in (8, 15, 25, 40):
        features = select(train, pool, top_k)
        for objective in ("regression_l1", "huber"):
            for leaves, min_child in ((3, 20), (5, 30)):
                fitted = model(objective, leaves, min_child)
                fitted.fit(train[features], train["Residual"])
                correction = fitted.predict(frame[features])
                for weight in (0.25, 0.50, 0.75, 1.00):
                    prediction = frame["FinalPrediction"].to_numpy() + weight * correction
                    candidate = f"R11B_K{top_k}_{objective.upper()}_L{leaves}_W{int(weight * 100):03d}"
                    score, level, trend = candidate_score(frame, prediction)
                    candidates[candidate] = prediction
                    bundles[candidate] = {"model": fitted, "features": features, "weight": weight}
                    rows.append({"candidate": candidate, "score": score, **level, **trend})
    search = pd.DataFrame(rows)
    selected = search.sort_values(["score", "mae", "candidate"], kind="stable").iloc[0]["candidate"]
    frame["R11bPrediction"] = candidates[selected]
    metrics = []
    for column, name in (("FinalPrediction", "R11_BASELINE"), ("R11bPrediction", "R11B_CHARGE")):
        for split in ("val", "oot"):
            metrics.append(metric_record(frame, column, split, name))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUTPUT / "r11b_metrics.csv", index=False)
    search.to_csv(OUTPUT / "r11b_validation_search.csv", index=False)
    frame[["LabelGroupID", "OrderID", "PalletID", "split", "regime", "MNY", "FinalPrediction", "R11bPrediction"]].to_csv(OUTPUT / "r11b_group_predictions.csv", index=False)
    bundle = {"contract": "V3.7.1-R11B-CAUSAL-CHARGE-ABLATION-V1", "candidate": selected, **bundles[selected], "stage3_calibration_enabled": False, "automatic_control": "DISABLED"}
    joblib.dump(bundle, MODEL_OUTPUT / "r11b_charge_ablation.joblib")
    val = metrics.loc[metrics["candidate"].eq("R11B_CHARGE") & metrics["split"].eq("val")].iloc[0]
    oot = metrics.loc[metrics["candidate"].eq("R11B_CHARGE") & metrics["split"].eq("oot")].iloc[0]
    base_oot = metrics.loc[metrics["candidate"].eq("R11_BASELINE") & metrics["split"].eq("oot")].iloc[0]
    summary = {
        "contract": bundle["contract"], "selected": selected,
        "selected_features": bundles[selected]["features"],
        "validation": val.to_dict(), "oot": oot.to_dict(),
        "baseline_oot_mae": float(base_oot["mae"]),
        "oot_mae_delta_vs_r11": float(oot["mae"] - base_oot["mae"]),
        "target_mae_mny": 2.5,
        "release_status": "ABLATION_ONLY_NOT_RELEASED",
        "stage3_calibration_enabled": False, "automatic_control": "DISABLED",
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "r11b_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
