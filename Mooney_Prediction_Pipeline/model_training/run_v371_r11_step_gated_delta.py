"""R11 T15760 Step-boundary, regime-gated level + within-order delta model."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.run_v371_r8_batch_group_mean import (
    BASE, GROUP_SOURCE, METADATA_SOURCE, R6_PREDICTIONS, T15760_NAME,
    candidate_score, metric_record,
)


CORRECTED_SOURCE = BASE / "t15760_step_aligned_curve_v2" / "t15760_step_aligned_curve_features_v2.csv"
STEP_GROUP_SOURCE = BASE / "t15760_within_order_contrast" / "step_execution" / "t15760_group_step_execution_features.csv"
OUTPUT = BASE / "v371_r11_step_gated_delta"
MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r11_step_gated_delta"


def estimator(objective="regression_l1", leaves=5, min_child=20):
    return LGBMRegressor(
        objective=objective, n_estimators=450, learning_rate=0.015,
        num_leaves=leaves, min_child_samples=min_child,
        colsample_bytree=0.72, subsample=0.85, subsample_freq=1,
        reg_alpha=4.0, reg_lambda=18.0, random_state=42,
        verbosity=-1, n_jobs=-1,
    )


def aggregate_corrected(frame: pd.DataFrame) -> pd.DataFrame:
    id_columns = {"OrderID", "BatchNumber", "LabelGroupID", "BoundaryContract", "BoundaryStatus", "StepPatternHash", "TargetVectorHash", "TemplateKey", "BatchStartTime"}
    numeric = []
    for column in frame.columns:
        if column in id_columns or column == "ProcessVersion":
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() >= 30 and converted.nunique(dropna=True) >= 2:
            frame[column] = converted
            numeric.append(column)
    mean = frame.groupby("LabelGroupID")[numeric].mean().add_suffix("__group_mean")
    std = frame.groupby("LabelGroupID")[numeric].std(ddof=0).add_suffix("__group_std")
    result = mean.join(std, how="outer").reset_index()
    status = frame.groupby("LabelGroupID").agg(
        corrected_batch_count=("BatchNumber", "nunique"),
        blocked_batch_count=("BoundaryStatus", lambda s: int(s.astype(str).str.startswith("BLOCKED").sum())),
        upset_batch_count=("BoundaryStatus", lambda s: int(s.astype(str).str.startswith("PROCESS_UPSET").sum())),
        production_start=("BatchStartTime", "min"),
    ).reset_index()
    status["regime"] = np.select(
        [status["blocked_batch_count"].gt(0), status["upset_batch_count"].gt(0)],
        ["DATA_INCOMPLETE", "PROCESS_UPSET"], default="NORMAL_COMPLETE_ROUTE",
    )
    return status.merge(result, on="LabelGroupID", how="left", validate="one_to_one")


def select_features(train: pd.DataFrame, pool: list[str], target: str, top_k: int, mandatory: list[str]) -> list[str]:
    scored = []
    for feature in pool:
        values = pd.to_numeric(train[feature], errors="coerce")
        valid = values.notna() & train[target].notna()
        if int(valid.sum()) < 35 or values.loc[valid].nunique() < 3:
            continue
        corr = values.loc[valid].corr(train.loc[valid, target], method="spearman")
        scored.append((abs(float(corr)) if pd.notna(corr) else 0.0, feature))
    ranked = [feature for _, feature in sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]]
    return list(dict.fromkeys([feature for feature in mandatory if feature in pool] + ranked))


def predict_gated(models: dict[str, object], frame: pd.DataFrame, features: list[str], fallback: float) -> np.ndarray:
    output = np.full(len(frame), fallback, dtype=float)
    for regime, model in models.items():
        mask = frame["regime"].eq(regime).to_numpy()
        if mask.any():
            output[mask] = model.predict(frame.loc[mask, features])
    return output


def build_pairs(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    ordered = frame.sort_values(["OrderID", "production_start", "PalletID"], kind="stable")
    for _, order in ordered.groupby("OrderID", sort=False):
        if len(order) < 2:
            continue
        previous = None
        for _, current in order.iterrows():
            if previous is None:
                previous = current
                continue
            row = {
                "from_group": previous["LabelGroupID"], "to_group": current["LabelGroupID"],
                "OrderID": current["OrderID"], "split": current["split"],
                "ActualDelta": current["MNY"] - previous["MNY"],
            }
            for feature in features:
                row[f"delta__{feature}"] = current[feature] - previous[feature]
            rows.append(row)
            previous = current
    return pd.DataFrame(rows)


def recursive_prediction(frame: pd.DataFrame, level: np.ndarray, pair_prediction: pd.DataFrame, beta: float) -> np.ndarray:
    result = pd.Series(level, index=frame["LabelGroupID"].astype(str), dtype=float)
    pair_map = pair_prediction.set_index("to_group")["DeltaPrediction"].to_dict()
    ordered = frame.sort_values(["OrderID", "production_start", "PalletID"], kind="stable")
    for _, order in ordered.groupby("OrderID", sort=False):
        previous_id = None
        for current_id in order["LabelGroupID"].astype(str):
            if previous_id is not None and current_id in pair_map:
                path = result.loc[previous_id] + pair_map[current_id]
                result.loc[current_id] = beta * result.loc[current_id] + (1.0 - beta) * path
            previous_id = current_id
    return frame["LabelGroupID"].astype(str).map(result).to_numpy(dtype=float)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(METADATA_SOURCE.read_text(encoding="utf-8"))
    groups_source = pd.read_csv(GROUP_SOURCE, low_memory=False)
    split = pd.read_csv(R6_PREDICTIONS, usecols=["LabelGroupID", "split"], low_memory=False)
    groups = groups_source.loc[groups_source["CompoundName"].astype(str).eq(T15760_NAME)].copy()
    groups = groups.drop(columns=["split"], errors="ignore").merge(split, on="LabelGroupID", how="inner", validate="one_to_one")

    corrected_batch = pd.read_csv(CORRECTED_SOURCE, low_memory=False)
    corrected_group = aggregate_corrected(corrected_batch)
    step_group = pd.read_csv(STEP_GROUP_SOURCE, low_memory=False)
    step_columns = [column for column in step_group.columns if column.startswith("step_exec__") or column in {
        "LabelGroupID", "physical_batch_count", "abnormal_batch_count", "abnormal_batch_rate",
        "max_tail_step11_duration", "missing_bottom_batch_count",
    }]
    step_group = step_group[step_columns]
    groups = groups.merge(corrected_group, on="LabelGroupID", how="inner", validate="one_to_one").merge(step_group, on="LabelGroupID", how="left", validate="one_to_one")
    groups["production_start"] = pd.to_datetime(groups["production_start"], errors="coerce")
    groups["lab_time"] = pd.to_datetime(groups["lab_time"], errors="coerce")
    groups["ProcessVersion"] = pd.to_numeric(groups["ProcessVersion"], errors="coerce")
    groups["regime_upset"] = groups["regime"].eq("PROCESS_UPSET").astype(float)
    groups["regime_incomplete"] = groups["regime"].eq("DATA_INCOMPLETE").astype(float)

    static = [feature for feature in metadata["s1_features"] if feature in groups.columns]
    corrected_features = [column for column in corrected_group.columns if column not in {"LabelGroupID", "regime", "production_start"}]
    step_features = [column for column in step_group.columns if column != "LabelGroupID"]
    pool = list(dict.fromkeys(["ProcessVersion", "regime_upset", "regime_incomplete"] + static + corrected_features + step_features))
    usable = []
    for feature in pool:
        groups[feature] = pd.to_numeric(groups[feature], errors="coerce")
        if groups[feature].notna().sum() >= 35 and groups[feature].nunique(dropna=True) >= 2:
            usable.append(feature)
    mandatory = [
        "ProcessVersion", "regime_upset", "regime_incomplete", "physical_batch_count",
        "abnormal_batch_rate", "missing_bottom_batch_count", "max_tail_step11_duration",
        "supplier_rubber_viscosity_avg", "supplier_silica_moisture_avg", "supplier_silica_surface_area_avg",
    ]
    train = groups.loc[groups["split"].eq("train")]
    train_mean = float(train["MNY"].mean())
    candidates, bundles, search_rows = {}, {}, []
    for top_k in (30, 60, 100, 160, 240):
        features = select_features(train, usable, "MNY", top_k, mandatory)
        for objective in ("regression_l1", "huber"):
            for leaves, min_child in ((3, 18), (5, 25), (8, 35)):
                single = estimator(objective, leaves, min_child)
                single.fit(train[features], train["MNY"])
                raw_single = single.predict(groups[features])
                name = f"R11_SINGLE_K{top_k}_{objective.upper()}_L{leaves}"
                for amplitude in (0.75, 1.00, 1.25, 1.50):
                    prediction = train_mean + amplitude * (raw_single - train_mean)
                    candidate = f"{name}_A{int(amplitude * 100):03d}"
                    score, level, trend = candidate_score(groups, prediction)
                    candidates[candidate] = prediction
                    bundles[candidate] = {"kind": "single", "model": single, "features": features, "amplitude": amplitude, "center": train_mean}
                    search_rows.append({"candidate": candidate, "score": score, **level, **trend})

                regime_models, regime_centers = {}, {}
                for regime in ("NORMAL_COMPLETE_ROUTE", "PROCESS_UPSET"):
                    part = train.loc[train["regime"].eq(regime)]
                    if len(part) < 30:
                        continue
                    fitted = estimator(objective, leaves, min_child)
                    fitted.fit(part[features], part["MNY"])
                    regime_models[regime] = fitted
                    regime_centers[regime] = float(part["MNY"].mean())
                if "NORMAL_COMPLETE_ROUTE" in regime_models and "PROCESS_UPSET" in regime_models:
                    raw_gated = predict_gated(regime_models, groups, features, train_mean)
                    center = groups["regime"].map(regime_centers).fillna(train_mean).to_numpy(dtype=float)
                    name = f"R11_GATED_K{top_k}_{objective.upper()}_L{leaves}"
                    for amplitude in (0.75, 1.00, 1.25, 1.50):
                        prediction = center + amplitude * (raw_gated - center)
                        candidate = f"{name}_A{int(amplitude * 100):03d}"
                        score, level, trend = candidate_score(groups, prediction)
                        candidates[candidate] = prediction
                        bundles[candidate] = {"kind": "gated", "models": regime_models, "features": features, "amplitude": amplitude, "centers": regime_centers, "fallback": train_mean}
                        search_rows.append({"candidate": candidate, "score": score, **level, **trend})

    search = pd.DataFrame(search_rows)
    selected_level = search.sort_values(["score", "mae", "candidate"], kind="stable").iloc[0]["candidate"]
    groups["LevelPrediction"] = candidates[selected_level]
    level_features = bundles[selected_level]["features"]

    pairs = build_pairs(groups, level_features)
    delta_pool = [column for column in pairs.columns if column.startswith("delta__")]
    pair_train = pairs.loc[pairs["split"].eq("train")].copy()
    delta_candidates, delta_bundles, delta_rows = {}, {}, []
    for top_k in (15, 30, 60):
        selected_delta = select_features(pair_train, delta_pool, "ActualDelta", top_k, [])
        for leaves, min_child in ((3, 5), (5, 8)):
            delta_model = estimator("regression_l1", leaves, min_child)
            delta_model.set_params(n_estimators=250, learning_rate=0.02)
            delta_model.fit(pair_train[selected_delta], pair_train["ActualDelta"])
            pair_prediction = pairs[["from_group", "to_group", "OrderID", "split", "ActualDelta"]].copy()
            pair_prediction["DeltaPrediction"] = delta_model.predict(pairs[selected_delta])
            for beta in (0.25, 0.50, 0.75, 1.00):
                prediction = recursive_prediction(groups, groups["LevelPrediction"].to_numpy(), pair_prediction, beta)
                candidate = f"R11_DELTA_K{top_k}_L{leaves}_B{int(beta * 100):03d}"
                score, level, trend = candidate_score(groups, prediction)
                delta_candidates[candidate] = prediction
                delta_bundles[candidate] = {"model": delta_model, "features": selected_delta, "beta": beta, "pair_predictions": pair_prediction}
                delta_rows.append({"candidate": candidate, "score": score, **level, **trend})
    delta_search = pd.DataFrame(delta_rows)
    selected_delta = delta_search.sort_values(["score", "mae", "candidate"], kind="stable").iloc[0]["candidate"]
    groups["FinalPrediction"] = delta_candidates[selected_delta]

    metrics = []
    for column, candidate in (("LevelPrediction", "R11_LEVEL"), ("FinalPrediction", "R11_FINAL")):
        for split_name in ("val", "oot"):
            metrics.append(metric_record(groups, column, split_name, candidate))
            for regime in ("NORMAL_COMPLETE_ROUTE", "PROCESS_UPSET"):
                part = groups.loc[groups["regime"].eq(regime)].copy()
                record = metric_record(part, column, split_name, f"{candidate}_{regime}")
                record["regime"] = regime
                metrics.append(record)
    metrics = pd.DataFrame(metrics)
    search.to_csv(OUTPUT / "r11_level_validation_search.csv", index=False)
    delta_search.to_csv(OUTPUT / "r11_delta_validation_search.csv", index=False)
    metrics.to_csv(OUTPUT / "r11_metrics.csv", index=False)
    groups[["LabelGroupID", "OrderID", "PalletID", "split", "regime", "MNY", "LevelPrediction", "FinalPrediction", "production_start", "lab_time"]].to_csv(OUTPUT / "r11_group_predictions.csv", index=False)
    delta_bundles[selected_delta]["pair_predictions"].to_csv(OUTPUT / "r11_pair_predictions.csv", index=False)
    bundle = {
        "contract": "V3.7.1-R11-STEP-BOUNDARY-GATED-DELTA-V1",
        "level_candidate": selected_level,
        "level": bundles[selected_level],
        "delta_candidate": selected_delta,
        "delta": {key: value for key, value in delta_bundles[selected_delta].items() if key != "pair_predictions"},
        "evaluation": "ONE_PREDICTION_PER_ORDERID_PALLETID",
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
    }
    joblib.dump(bundle, MODEL_OUTPUT / "r11_step_gated_delta_shadow.joblib")
    val = metrics.loc[metrics["candidate"].eq("R11_FINAL") & metrics["split"].eq("val")].iloc[0]
    oot = metrics.loc[metrics["candidate"].eq("R11_FINAL") & metrics["split"].eq("oot")].iloc[0]
    summary = {
        "contract": bundle["contract"], "groups": len(groups),
        "regime_counts": groups["regime"].value_counts().to_dict(),
        "selected_level": selected_level, "selected_delta": selected_delta,
        "validation": val.to_dict(), "oot": oot.to_dict(),
        "target_mae_mny": 2.5,
        "target_trend": {"delta_pearson_min": 0.50, "direction_accuracy_min": 0.65, "delta_std_ratio_range": [0.70, 1.30]},
        "release_status": "NOT_RELEASED_GATES_FAILED_OR_BLIND_WINDOW_PENDING",
        "stage3_calibration_enabled": False, "automatic_control": "DISABLED",
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "r11_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
