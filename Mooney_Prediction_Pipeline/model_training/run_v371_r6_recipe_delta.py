"""Train R6: purged-time four-route base plus recipe level/delta expert.

This is an offline shadow experiment.  It reads frozen RDS Active snapshots,
does not connect to any database, keeps Stage3 disabled, and never emits a
control recommendation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.purged_time_validation import (
    assign_purged_global_time_split,
    audit_purged_time_split,
)
from Mooney_Prediction_Pipeline.model_training.r5_order_state_features import (
    add_prior_order_state,
    build_group_target_features,
)
from Mooney_Prediction_Pipeline.model_training.v371_purged_hybrid import (
    V371PurgedTimeGroupModel,
)


BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
GROUP_SOURCE = BASE / "pallet_group_training_frame_c2_prepared.csv"
PHASE_SOURCE = BASE / "phase_batch_features_active_v1.csv"
METADATA_SOURCE = (
    ROOT
    / "Mooney_Prediction_Pipeline"
    / "models"
    / "v371_challenger_r3_expert_ablation"
    / "feature_metadata.json"
)
OUTPUT = BASE / "v371_r6_recipe_delta"
MODEL_OUTPUT = (
    ROOT
    / "Mooney_Prediction_Pipeline"
    / "models"
    / "v371_challenger_r6_recipe_delta"
)
T15760_NAME = "M1-T15760---- 06 002"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scope_mask(frame: pd.DataFrame, scope: str) -> pd.Series:
    compound = frame["CompoundName"].astype(str).str.upper()
    if scope == "global":
        return pd.Series(True, index=frame.index)
    if scope == "T15760":
        return compound.eq(T15760_NAME.upper())
    if scope == "T15760_PV46":
        return compound.eq(T15760_NAME.upper()) & pd.to_numeric(
            frame["ProcessVersion"], errors="coerce"
        ).eq(46)
    if scope == "B00458":
        return compound.str.contains("B00458", regex=False)
    raise ValueError(scope)


def level_metrics(frame: pd.DataFrame, prediction_col: str) -> dict[str, float | int]:
    actual = pd.to_numeric(frame["MNY"], errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(frame[prediction_col], errors="coerce").to_numpy(
        dtype=float
    )
    error = predicted - actual
    actual_std = np.std(actual, ddof=1) if len(actual) > 1 else np.nan
    prediction_std = np.std(predicted, ddof=1) if len(predicted) > 1 else np.nan
    return {
        "groups": int(len(frame)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "bias": float(np.mean(error)),
        "r2": float(r2_score(actual, predicted)) if len(frame) > 1 else np.nan,
        "prediction_to_actual_std_ratio": (
            float(prediction_std / actual_std)
            if len(frame) > 1 and actual_std > 0
            else np.nan
        ),
        "pearson": (
            float(np.corrcoef(actual, predicted)[0, 1])
            if len(frame) > 1
            and np.std(actual) > 0
            and np.std(predicted) > 0
            else np.nan
        ),
    }


def recipe_sequence_pairs(
    frame: pd.DataFrame,
    prediction_col: str,
    *,
    significant_delta: float = 2.0,
) -> pd.DataFrame:
    rows = []
    ordered = frame.sort_values(
        ["RecipeID", "lab_time", "OrderID", "LabelGroupID"], kind="stable"
    )
    for recipe, part in ordered.groupby("RecipeID", sort=False):
        if len(part) < 2:
            continue
        previous = None
        for current in part.itertuples(index=False):
            if previous is not None:
                actual_delta = float(current.MNY) - float(previous.MNY)
                prediction_delta = float(getattr(current, prediction_col)) - float(
                    getattr(previous, prediction_col)
                )
                rows.append(
                    {
                        "RecipeID": recipe,
                        "CurrentLabelGroupID": current.LabelGroupID,
                        "PreviousLabelGroupID": previous.LabelGroupID,
                        "CurrentSplit": current.split,
                        "PreviousSplit": previous.split,
                        "actual_delta": actual_delta,
                        "prediction_delta": prediction_delta,
                        "significant": abs(actual_delta) >= significant_delta,
                    }
                )
            previous = current
    return pd.DataFrame(rows)


def trend_metrics(frame: pd.DataFrame, prediction_col: str) -> dict[str, float | int]:
    pairs = recipe_sequence_pairs(frame, prediction_col)
    if pairs.empty:
        return {
            "pairs": 0,
            "significant_pairs": 0,
            "delta_pearson": np.nan,
            "direction_accuracy": np.nan,
            "delta_std_ratio": np.nan,
        }
    actual = pairs["actual_delta"].to_numpy(dtype=float)
    predicted = pairs["prediction_delta"].to_numpy(dtype=float)
    significant = pairs["significant"].to_numpy(dtype=bool)
    return {
        "pairs": int(len(pairs)),
        "significant_pairs": int(significant.sum()),
        "delta_pearson": (
            float(np.corrcoef(actual, predicted)[0, 1])
            if np.std(actual) > 0 and np.std(predicted) > 0
            else np.nan
        ),
        "direction_accuracy": (
            float(np.mean(np.sign(actual[significant]) == np.sign(predicted[significant])))
            if significant.any()
            else np.nan
        ),
        "delta_std_ratio": (
            float(np.std(predicted, ddof=1) / np.std(actual, ddof=1))
            if len(actual) > 1 and np.std(actual, ddof=1) > 0
            else np.nan
        ),
    }


def metric_rows(
    frame: pd.DataFrame,
    prediction_col: str,
    candidate: str,
    stage: str,
) -> list[dict[str, object]]:
    records = []
    for split in ("val", "oot"):
        split_frame = frame.loc[frame["split"].eq(split)]
        for scope in ("global", "T15760", "T15760_PV46", "B00458"):
            subset = split_frame.loc[scope_mask(split_frame, scope)]
            if subset.empty:
                continue
            records.append(
                {
                    "stage": stage,
                    "candidate": candidate,
                    "split": split,
                    "scope": scope,
                    **level_metrics(subset, prediction_col),
                    **trend_metrics(subset, prediction_col),
                }
            )
    return records


def t15760_validation_score(metrics: pd.DataFrame, candidate: str, stage: str) -> float:
    selected = metrics.loc[
        metrics["stage"].eq(stage)
        & metrics["candidate"].eq(candidate)
        & metrics["split"].eq("val")
        & metrics["scope"].eq("T15760")
    ]
    if selected.empty:
        return float("inf")
    row = selected.iloc[0]
    score = float(row["mae"])
    correlation = float(row["delta_pearson"]) if pd.notna(row["delta_pearson"]) else -1.0
    direction = float(row["direction_accuracy"]) if pd.notna(row["direction_accuracy"]) else 0.0
    ratio = float(row["delta_std_ratio"]) if pd.notna(row["delta_std_ratio"]) else 0.0
    score += 1.5 * max(0.0, 0.50 - correlation)
    score += 1.5 * max(0.0, 0.70 - direction)
    score += 0.75 * max(0.0, 0.70 - ratio)
    score += 0.75 * max(0.0, ratio - 1.30)
    return score


def select_numeric_features(
    train: pd.DataFrame,
    pool: list[str],
    target: pd.Series,
    top_k: int,
) -> list[str]:
    scores = []
    for feature in pool:
        values = pd.to_numeric(train[feature], errors="coerce")
        valid = values.notna() & target.notna()
        if int(valid.sum()) < max(30, int(0.20 * len(train))):
            continue
        if values.loc[valid].nunique() < 3:
            continue
        correlation = values.loc[valid].corr(target.loc[valid], method="spearman")
        scores.append((abs(float(correlation)) if pd.notna(correlation) else 0.0, feature))
    return [name for _, name in sorted(scores, key=lambda item: (-item[0], item[1]))[:top_k]]


def new_model(objective: str, leaves: int, min_child: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective=objective,
        n_estimators=350,
        learning_rate=0.02,
        num_leaves=leaves,
        min_child_samples=min_child,
        max_depth=-1,
        colsample_bytree=0.70,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=3.0,
        reg_lambda=12.0,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


def prepare_frame():
    metadata = json.loads(METADATA_SOURCE.read_text(encoding="utf-8"))
    groups = pd.read_csv(GROUP_SOURCE, low_memory=False)
    phase = pd.read_csv(
        PHASE_SOURCE,
        usecols=[
            "LabelGroupID",
            "TargetVectorRaw",
            "TargetVectorHash",
            "StepPatternHash",
            "TemplateKey",
            "BatchStartTime",
            "BatchEndTime",
        ],
        low_memory=False,
    )
    target_contract = build_group_target_features(phase)
    groups = groups.drop(
        columns=[
            column
            for column in ("TargetVectorHash", "StepPatternHash", "TemplateKey")
            if column in groups.columns
        ]
    ).merge(target_contract.frame, on="LabelGroupID", how="inner", validate="one_to_one")
    groups["split"] = assign_purged_global_time_split(groups)
    split_audit = audit_purged_time_split(groups, "split")
    if not split_audit["passed"]:
        raise AssertionError("R6 strict outer time split audit failed")
    groups["ProcessVersion"] = pd.to_numeric(groups["ProcessVersion"], errors="coerce")
    state_features = [feature for feature in metadata["s2_features"] if feature in groups]
    groups, lag_features = add_prior_order_state(groups, state_features=state_features)
    return groups, metadata, target_contract.feature_map, lag_features, split_audit


def train_base(groups: pd.DataFrame, metadata: dict):
    train = groups.loc[groups["split"].eq("train")]
    models = {}
    predictions = {}
    records = []
    for objective in ("regression", "regression_l1"):
        for shrinkage in (5.0, 20.0):
            name = f"R6_BASE_{objective.upper()}_K{int(shrinkage)}"
            model = V371PurgedTimeGroupModel(
                shrinkage_k=shrinkage,
                use_cluster_experts=True,
                use_material_route_matrix=True,
                use_silica_subsystem=False,
                use_cb_subsystem=False,
                crossfit_splits=5,
                time_col="lab_time",
                objective=objective,
            ).fit(
                train,
                metadata["s1_features"],
                metadata["s2_features"],
                target_col="MNY",
                cluster_col="material_system",
            )
            prediction = model.predict(groups, cluster_col="material_system")[0]
            candidate_frame = groups.copy()
            candidate_frame["Prediction"] = prediction
            records.extend(metric_rows(candidate_frame, "Prediction", name, "base"))
            models[name] = model
            predictions[name] = prediction
    metrics = pd.DataFrame(records)
    candidates = []
    for name in sorted(models):
        global_row = metrics.loc[
            metrics["candidate"].eq(name)
            & metrics["split"].eq("val")
            & metrics["scope"].eq("global")
        ].iloc[0]
        candidates.append((float(global_row["mae"]), name))
    selected = min(candidates)[1]
    output = groups.copy()
    output["BasePrediction"] = predictions[selected]
    return models[selected], selected, output, records


def train_t15760_level(groups, metadata, target_features, lag_features):
    tmask = scope_mask(groups, "T15760")
    train = groups.loc[tmask & groups["split"].eq("train")].copy()
    dispersion = [
        f"{feature}{suffix}"
        for feature in metadata["s2_features"]
        for suffix in ("__group_std", "__group_min", "__group_max")
        if f"{feature}{suffix}" in groups.columns
    ]
    pool = list(
        dict.fromkeys(
            ["ProcessVersion", "PhaseBatchCount", "batch_count"]
            + metadata["s2_features"]
            + dispersion
            + target_features
            + lag_features
        )
    )
    pool = [feature for feature in pool if feature in groups.columns]
    models = {}
    feature_sets = {}
    corrections = {}
    records = []
    candidate_predictions = {}
    residual = train["MNY"] - train["BasePrediction"]
    for top_k in (15, 30, 60, 100):
        features = select_numeric_features(train, pool, residual, top_k)
        for objective in ("regression_l1", "huber"):
            for leaves, min_child in ((3, 20), (5, 25), (8, 35)):
                model_name = f"R6_T_LEVEL_K{top_k}_{objective.upper()}_L{leaves}"
                model = new_model(objective, leaves, min_child)
                model.fit(train[features], residual)
                correction = model.predict(groups[features])
                models[model_name] = model
                feature_sets[model_name] = features
                corrections[model_name] = correction
                for weight in (0.25, 0.50, 0.75, 1.00, 1.25):
                    candidate = f"{model_name}_W{int(weight * 100):03d}"
                    prediction = groups["BasePrediction"].to_numpy(copy=True)
                    prediction[tmask.to_numpy()] += weight * correction[tmask.to_numpy()]
                    candidate_frame = groups.copy()
                    candidate_frame["Prediction"] = prediction
                    records.extend(metric_rows(candidate_frame, "Prediction", candidate, "level"))
                    candidate_predictions[candidate] = prediction
    metrics = pd.DataFrame(records)
    selected = min(
        sorted(candidate_predictions),
        key=lambda name: t15760_validation_score(metrics, name, "level"),
    )
    model_name, weight_text = selected.rsplit("_W", 1)
    output = groups.copy()
    output["LevelPrediction"] = candidate_predictions[selected]
    bundle = {
        "model": models[model_name],
        "features": feature_sets[model_name],
        "weight": int(weight_text) / 100.0,
    }
    return bundle, selected, feature_sets[model_name], output, records


def build_t15760_delta_pairs(groups, feature_columns):
    source = groups.loc[scope_mask(groups, "T15760")].copy()
    source = source.sort_values(
        ["RecipeID", "lab_time", "OrderID", "LabelGroupID"], kind="stable"
    )
    rows = []
    for _, part in source.groupby("RecipeID", sort=False):
        previous = None
        for index, current in part.iterrows():
            if previous is None:
                previous = current
                continue
            current_split = str(current["split"])
            previous_split = str(previous["split"])
            if current_split == "purged" or previous_split == "purged":
                previous = current
                continue
            row = {
                "CurrentIndex": index,
                "CurrentLabelGroupID": current["LabelGroupID"],
                "PreviousLabelGroupID": previous["LabelGroupID"],
                "CurrentSplit": current_split,
                "PreviousSplit": previous_split,
                "ActualDelta": float(current["MNY"]) - float(previous["MNY"]),
                "LevelDelta": float(current["LevelPrediction"])
                - float(previous["LevelPrediction"]),
            }
            for feature in feature_columns:
                row[f"delta__{feature}"] = pd.to_numeric(
                    pd.Series([current[feature]]), errors="coerce"
                ).iloc[0] - pd.to_numeric(
                    pd.Series([previous[feature]]), errors="coerce"
                ).iloc[0]
            rows.append(row)
            previous = current
    return pd.DataFrame(rows)


def recursive_delta_prediction(groups, pair_predictions, alpha):
    output = groups["LevelPrediction"].to_numpy(copy=True)
    position = {index: pos for pos, index in enumerate(groups.index)}
    tsource = groups.loc[scope_mask(groups, "T15760")].sort_values(
        ["RecipeID", "lab_time", "OrderID", "LabelGroupID"], kind="stable"
    )
    delta_map = dict(zip(pair_predictions["CurrentIndex"], pair_predictions["DeltaPrediction"]))
    for _, part in tsource.groupby("RecipeID", sort=False):
        previous_index = None
        previous_split = None
        for index, row in part.iterrows():
            current_split = str(row["split"])
            if (
                previous_index is None
                or current_split == "purged"
                or current_split != previous_split
                or index not in delta_map
            ):
                previous_index = index
                previous_split = current_split
                continue
            anchor = output[position[previous_index]] + float(delta_map[index])
            base_level = float(row["LevelPrediction"])
            output[position[index]] = alpha * base_level + (1.0 - alpha) * anchor
            previous_index = index
            previous_split = current_split
    return output


def train_t15760_delta(groups, level_features):
    feature_columns = list(dict.fromkeys(level_features))
    pairs = build_t15760_delta_pairs(groups, feature_columns)
    delta_pool = [column for column in pairs.columns if column.startswith("delta__")]
    train = pairs.loc[
        pairs["CurrentSplit"].eq("train") & pairs["PreviousSplit"].eq("train")
    ].copy()
    train["DeltaResidual"] = train["ActualDelta"] - train["LevelDelta"]
    records = []
    candidates = {}
    models = {}
    feature_sets = {}
    for top_k in (10, 20, 40):
        selected_features = select_numeric_features(
            train, delta_pool, train["DeltaResidual"], top_k
        )
        features = ["LevelDelta"] + selected_features
        for leaves, min_child in ((3, 15), (5, 20)):
            model_name = f"R6_T_DELTA_K{top_k}_L{leaves}"
            model = new_model("regression_l1", leaves, min_child)
            model.fit(train[features], train["ActualDelta"])
            delta_prediction = model.predict(pairs[features])
            pair_predictions = pairs.copy()
            pair_predictions["DeltaPrediction"] = delta_prediction
            models[model_name] = model
            feature_sets[model_name] = features
            for alpha in (0.25, 0.50, 0.75, 1.00):
                candidate = f"{model_name}_A{int(alpha * 100):03d}"
                prediction = recursive_delta_prediction(groups, pair_predictions, alpha)
                candidate_frame = groups.copy()
                candidate_frame["Prediction"] = prediction
                records.extend(metric_rows(candidate_frame, "Prediction", candidate, "delta"))
                candidates[candidate] = (prediction, pair_predictions)
    metrics = pd.DataFrame(records)
    selected = min(
        sorted(candidates),
        key=lambda name: t15760_validation_score(metrics, name, "delta"),
    )
    model_name, alpha_text = selected.rsplit("_A", 1)
    output = groups.copy()
    output["FinalPrediction"] = candidates[selected][0]
    final_pairs = candidates[selected][1]
    bundle = {
        "model": models[model_name],
        "features": feature_sets[model_name],
        "alpha": int(alpha_text) / 100.0,
    }
    return bundle, selected, feature_sets[model_name], output, final_pairs, records


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    groups, metadata, target_feature_map, lag_features, split_audit = prepare_frame()
    target_features = sorted(target_feature_map)
    base_model, base_name, groups, base_records = train_base(groups, metadata)
    level_bundle, level_name, level_features, groups, level_records = train_t15760_level(
        groups, metadata, target_features, lag_features
    )
    delta_bundle, delta_name, delta_features, groups, pairs, delta_records = (
        train_t15760_delta(groups, level_features)
    )
    final_records = metric_rows(groups, "FinalPrediction", "R6_SELECTED", "final")
    metrics = pd.DataFrame(base_records + level_records + delta_records + final_records)

    prediction_columns = [
        "LabelGroupID",
        "OrderID",
        "PalletID",
        "CompoundName",
        "RecipeID",
        "ProcessVersion",
        "MixerLine",
        "material_system",
        "phase_route",
        "lab_time",
        "split",
        "MNY",
        "BasePrediction",
        "LevelPrediction",
        "FinalPrediction",
    ]
    groups.loc[:, prediction_columns].to_csv(OUTPUT / "r6_group_predictions.csv", index=False)
    pairs.to_csv(OUTPUT / "r6_t15760_delta_pairs.csv", index=False)
    metrics.to_csv(OUTPUT / "r6_metrics.csv", index=False)
    (OUTPUT / "target_feature_map.json").write_text(
        json.dumps(target_feature_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    bundle = {
        "contract": "V3.7.1-R6-PURGED-RECIPE-DELTA-V1",
        "base_model": base_model,
        "base_candidate": base_name,
        "t15760_level": level_bundle,
        "t15760_level_candidate": level_name,
        "t15760_delta": delta_bundle,
        "t15760_delta_candidate": delta_name,
        "target_feature_map": target_feature_map,
        "stage1_features": metadata["s1_features"],
        "stage2_features": metadata["s2_features"],
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
    }
    model_path = MODEL_OUTPUT / "r6_purged_recipe_delta_shadow.joblib"
    joblib.dump(bundle, model_path)

    final = pd.DataFrame(final_records)
    validation_t = final.loc[
        final["split"].eq("val") & final["scope"].eq("T15760")
    ].iloc[0]
    validation_global = final.loc[
        final["split"].eq("val") & final["scope"].eq("global")
    ].iloc[0]
    summary = {
        "contract": bundle["contract"],
        "selected_base": base_name,
        "selected_t15760_level": level_name,
        "selected_t15760_delta": delta_name,
        "outer_split_contract": "GLOBAL_TIME_70_15_15_ORDER_INTERVAL_PURGED",
        "outer_split_audit": split_audit,
        "base_crossfit_audit": base_model.crossfit_audit_,
        "target_feature_count": len(target_features),
        "t15760_level_feature_count": len(level_features),
        "t15760_delta_feature_count": len(delta_features),
        "validation_global_mae": float(validation_global["mae"]),
        "validation_t15760_mae": float(validation_t["mae"]),
        "validation_t15760_delta_pearson": (
            float(validation_t["delta_pearson"])
            if pd.notna(validation_t["delta_pearson"])
            else None
        ),
        "validation_t15760_direction_accuracy": (
            float(validation_t["direction_accuracy"])
            if pd.notna(validation_t["direction_accuracy"])
            else None
        ),
        "validation_t15760_delta_std_ratio": (
            float(validation_t["delta_std_ratio"])
            if pd.notna(validation_t["delta_std_ratio"])
            else None
        ),
        "target_mae_mny": 2.5,
        "release_status": "NOT_RELEASED_GATES_FAILED_OR_BLIND_WINDOW_PENDING",
        "stage3_calibration_enabled": False,
        "model_sha256": sha256(model_path),
        "group_source_sha256": sha256(GROUP_SOURCE),
        "phase_source_sha256": sha256(PHASE_SOURCE),
        "local_mms_access": False,
        "database_access_during_run": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "automatic_control": "DISABLED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "r6_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
