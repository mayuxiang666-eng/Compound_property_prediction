"""Train the R5 order-state level + within-order delta Mooney challenger."""

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

from Mooney_Prediction_Pipeline.model_training.order_atomic_validation import (
    assign_recipe_order_atomic_splits,
    audit_order_split,
)
from Mooney_Prediction_Pipeline.model_training.r5_order_state_features import (
    add_prior_order_state,
    build_adjacent_order_pairs,
    build_group_target_features,
)
from Mooney_Prediction_Pipeline.model_training.v371_order_atomic_hybrid import (
    V371OrderAtomicGroupModel,
)


BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
GROUP_SOURCE = BASE / "pallet_group_training_frame_c2_prepared.csv"
PHASE_SOURCE = BASE / "phase_batch_features_active_v1.csv"
METADATA_SOURCE = (
    ROOT / "Mooney_Prediction_Pipeline" / "models"
    / "v371_challenger_r3_expert_ablation" / "feature_metadata.json"
)
OUTPUT = BASE / "v371_r5_order_state_delta"
MODEL_OUTPUT = (
    ROOT / "Mooney_Prediction_Pipeline" / "models"
    / "v371_challenger_r5_order_state_delta"
)


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
        return compound.str.contains("T15760", regex=False)
    if scope == "T15760_PV46":
        return (
            compound.str.contains("T15760", regex=False)
            & pd.to_numeric(frame["ProcessVersion"], errors="coerce").eq(46)
        )
    if scope == "B00458":
        return compound.str.contains("B00458", regex=False)
    raise ValueError(scope)


def level_metrics(frame: pd.DataFrame, prediction_col: str) -> dict[str, float | int]:
    y = pd.to_numeric(frame["MNY"], errors="coerce").to_numpy(dtype=float)
    prediction = pd.to_numeric(frame[prediction_col], errors="coerce").to_numpy(dtype=float)
    error = prediction - y
    return {
        "groups": int(len(frame)),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "bias": float(np.mean(error)),
        "r2": float(r2_score(y, prediction)) if len(frame) > 1 else np.nan,
        "prediction_to_actual_std_ratio": (
            float(np.std(prediction, ddof=1) / np.std(y, ddof=1))
            if len(frame) > 1 and np.std(y, ddof=1) > 0 else np.nan
        ),
        "pearson": (
            float(np.corrcoef(y, prediction)[0, 1])
            if len(frame) > 1 and np.std(y) > 0 and np.std(prediction) > 0 else np.nan
        ),
    }


def within_order_trend_metrics(
    frame: pd.DataFrame,
    prediction_col: str,
    *,
    significant_delta: float = 2.0,
) -> dict[str, float | int]:
    deltas = []
    ordered = frame.sort_values(
        ["OrderID", "GroupProductionStart", "LabelGroupID"], kind="stable"
    )
    for _, order in ordered.groupby("OrderID", sort=False):
        if len(order) < 2:
            continue
        actual = pd.to_numeric(order["MNY"], errors="coerce").to_numpy(dtype=float)
        predicted = pd.to_numeric(
            order[prediction_col], errors="coerce"
        ).to_numpy(dtype=float)
        for index in range(1, len(order)):
            deltas.append(
                {
                    "actual_delta": actual[index] - actual[index - 1],
                    "prediction_delta": predicted[index] - predicted[index - 1],
                }
            )
    pairs = pd.DataFrame(deltas)
    if pairs.empty:
        return {
            "pairs": 0, "significant_pairs": 0, "delta_pearson": np.nan,
            "direction_accuracy": np.nan, "delta_std_ratio": np.nan,
        }
    actual = pairs["actual_delta"].to_numpy(dtype=float)
    predicted = pairs["prediction_delta"].to_numpy(dtype=float)
    significant = np.abs(actual) >= significant_delta
    return {
        "pairs": int(len(pairs)),
        "significant_pairs": int(significant.sum()),
        "delta_pearson": (
            float(np.corrcoef(actual, predicted)[0, 1])
            if np.std(actual) > 0 and np.std(predicted) > 0 else np.nan
        ),
        "direction_accuracy": (
            float(np.mean(np.sign(actual[significant]) == np.sign(predicted[significant])))
            if significant.any() else np.nan
        ),
        "delta_std_ratio": (
            float(np.std(predicted, ddof=1) / np.std(actual, ddof=1))
            if len(pairs) > 1 and np.std(actual, ddof=1) > 0 else np.nan
        ),
    }


def metric_rows(
    frame: pd.DataFrame,
    prediction_col: str,
    candidate: str,
    stage: str,
) -> list[dict[str, object]]:
    rows = []
    for split in ("val", "oot"):
        split_frame = frame.loc[frame["split"].eq(split)]
        for scope in ("global", "T15760", "T15760_PV46", "B00458"):
            subset = split_frame.loc[scope_mask(split_frame, scope)]
            if subset.empty:
                continue
            rows.append(
                {
                    "stage": stage,
                    "candidate": candidate,
                    "split": split,
                    "scope": scope,
                    **level_metrics(subset, prediction_col),
                    **within_order_trend_metrics(subset, prediction_col),
                }
            )
    return rows


def candidate_score(metrics: pd.DataFrame, candidate: str, stage: str) -> float:
    selected = metrics.loc[
        metrics["stage"].eq(stage)
        & metrics["candidate"].eq(candidate)
        & metrics["split"].eq("val")
    ].set_index("scope")
    if "global" not in selected.index:
        return float("inf")
    score = float(selected.loc["global", "mae"])
    if "T15760" in selected.index:
        score += 0.5 * float(selected.loc["T15760", "mae"])
    for scope, weight in (("global", 1.0), ("T15760", 0.5)):
        if scope not in selected.index:
            continue
        row = selected.loc[scope]
        if int(row["significant_pairs"]) < 5:
            continue
        correlation = float(row["delta_pearson"]) if pd.notna(row["delta_pearson"]) else -1.0
        direction = (
            float(row["direction_accuracy"])
            if pd.notna(row["direction_accuracy"]) else 0.0
        )
        ratio = float(row["delta_std_ratio"]) if pd.notna(row["delta_std_ratio"]) else 0.0
        score += weight * max(0.0, 0.50 - correlation)
        score += weight * max(0.0, 0.70 - direction)
        score += 0.5 * weight * max(0.0, 0.70 - ratio)
        score += 0.5 * weight * max(0.0, ratio - 1.30)
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
        if int(valid.sum()) < max(20, int(0.10 * len(train))):
            continue
        if values.loc[valid].nunique() < 2:
            continue
        correlation = values.loc[valid].corr(target.loc[valid], method="spearman")
        scores.append((abs(float(correlation)) if pd.notna(correlation) else 0.0, feature))
    return [feature for _, feature in sorted(scores, key=lambda item: (-item[0], item[1]))[:top_k]]


def new_residual_model(num_leaves: int, min_child_samples: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=250,
        learning_rate=0.025,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        colsample_bytree=0.70,
        reg_alpha=2.0,
        reg_lambda=8.0,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


def prepare_frame() -> tuple[pd.DataFrame, dict, dict[str, str], list[str]]:
    metadata = json.loads(METADATA_SOURCE.read_text(encoding="utf-8"))
    groups = pd.read_csv(GROUP_SOURCE, low_memory=False)
    phase_columns = [
        "LabelGroupID", "TargetVectorRaw", "TargetVectorHash", "StepPatternHash",
        "TemplateKey", "BatchStartTime", "BatchEndTime",
    ]
    phase = pd.read_csv(PHASE_SOURCE, usecols=phase_columns, low_memory=False)
    target_contract = build_group_target_features(phase)
    groups = groups.drop(
        columns=[
            column for column in ("TargetVectorHash", "StepPatternHash", "TemplateKey")
            if column in groups.columns
        ]
    ).merge(target_contract.frame, on="LabelGroupID", how="inner", validate="one_to_one")
    groups["split"] = assign_recipe_order_atomic_splits(groups)
    split_audit = audit_order_split(groups, "split")
    if not split_audit["passed"]:
        raise AssertionError("R5 outer OrderID split audit failed")
    state_features = [
        feature for feature in metadata["s2_features"] if feature in groups.columns
    ]
    groups, lag_features = add_prior_order_state(
        groups,
        state_features=state_features,
    )
    groups["ProcessVersion"] = pd.to_numeric(groups["ProcessVersion"], errors="coerce")
    return groups, metadata, target_contract.feature_map, lag_features


def train_base(
    groups: pd.DataFrame,
    metadata: dict,
) -> tuple[V371OrderAtomicGroupModel, str, pd.DataFrame, list[dict[str, object]]]:
    models = {}
    predictions = {}
    metric_records = []
    for objective in ("regression", "regression_l1"):
        for shrinkage in (5.0, 20.0):
            name = f"R5_BASE_{objective.upper()}_K{int(shrinkage)}"
            model = V371OrderAtomicGroupModel(
                shrinkage_k=shrinkage,
                use_cluster_experts=True,
                use_material_route_matrix=True,
                use_silica_subsystem=False,
                use_cb_subsystem=False,
                crossfit_splits=5,
                time_col="lab_time",
                objective=objective,
            ).fit(
                groups.loc[groups["split"].eq("train")],
                metadata["s1_features"],
                metadata["s2_features"],
                target_col="MNY",
                cluster_col="material_system",
            )
            prediction = model.predict(groups, cluster_col="material_system")[0]
            candidate_frame = groups.copy()
            candidate_frame["Prediction"] = prediction
            metric_records.extend(metric_rows(candidate_frame, "Prediction", name, "base"))
            models[name] = model
            predictions[name] = prediction
    metrics = pd.DataFrame(metric_records)
    candidates = sorted(models)
    selected = min(candidates, key=lambda name: candidate_score(metrics, name, "base"))
    output = groups.copy()
    output["BasePrediction"] = predictions[selected]
    return models[selected], selected, output, metric_records


def train_level_head(
    groups: pd.DataFrame,
    metadata: dict,
    target_features: list[str],
    lag_features: list[str],
) -> tuple[object, str, list[str], pd.DataFrame, list[dict[str, object]]]:
    dispersion = [
        f"{feature}{suffix}"
        for feature in metadata["s2_features"]
        for suffix in ("__group_std", "__group_min", "__group_max")
        if f"{feature}{suffix}" in groups.columns
    ]
    pool = list(dict.fromkeys(
        ["ProcessVersion", "PhaseBatchCountR5", "BasePrediction"]
        + target_features + dispersion + lag_features
    ))
    train = groups.loc[groups["split"].eq("train")].copy()
    residual = train["MNY"] - train["BasePrediction"]
    models = {}
    feature_sets = {}
    predictions = {}
    metric_records = []
    for top_k in (30, 60, 120):
        features = select_numeric_features(train, pool, residual, top_k)
        for leaves, min_child in ((5, 20), (8, 25)):
            model_name = f"R5_LEVEL_K{top_k}_L{leaves}"
            model = new_residual_model(leaves, min_child)
            model.fit(train[features], residual)
            correction = model.predict(groups[features])
            models[model_name] = model
            feature_sets[model_name] = features
            for weight in (0.25, 0.50, 0.75, 1.00):
                candidate = f"{model_name}_W{int(weight * 100):03d}"
                prediction = groups["BasePrediction"].to_numpy() + weight * correction
                candidate_frame = groups.copy()
                candidate_frame["Prediction"] = prediction
                metric_records.extend(
                    metric_rows(candidate_frame, "Prediction", candidate, "level")
                )
                predictions[candidate] = prediction
    metrics = pd.DataFrame(metric_records)
    selected = min(
        sorted(predictions),
        key=lambda name: candidate_score(metrics, name, "level"),
    )
    model_name, weight_text = selected.rsplit("_W", 1)
    output = groups.copy()
    output["LevelPrediction"] = predictions[selected]
    bundle = {
        "model": models[model_name],
        "features": feature_sets[model_name],
        "weight": int(weight_text) / 100.0,
    }
    return bundle, selected, feature_sets[model_name], output, metric_records


def train_delta_head(
    groups: pd.DataFrame,
    metadata: dict,
    target_features: list[str],
) -> tuple[object | None, str, list[str], pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    dispersion = [
        f"{feature}{suffix}"
        for feature in metadata["s2_features"]
        for suffix in ("__group_std",)
        if f"{feature}{suffix}" in groups.columns
    ]
    pair_source = list(dict.fromkeys(
        ["ProcessVersion"] + target_features + metadata["s2_features"] + dispersion
    ))
    pairs = build_adjacent_order_pairs(
        groups,
        feature_columns=pair_source,
        target_col="MNY",
        prediction_col="LevelPrediction",
    )
    pair_feature_pool = [
        column for column in pairs.columns if column.startswith("pair_delta__")
    ]
    pairs["DeltaResidual"] = pairs["ActualDelta"] - pairs["LevelDelta"]
    train = pairs.loc[pairs["split"].eq("train")].copy()
    models = {}
    feature_sets = {}
    corrections = {}
    metric_records = []
    candidate_predictions = {}
    for top_k in (20, 50, 100):
        features = ["LevelDelta"] + select_numeric_features(
            train, pair_feature_pool, train["DeltaResidual"], top_k
        )
        model_name = f"R5_DELTA_K{top_k}"
        model = new_residual_model(5, 12)
        model.fit(train[features], train["DeltaResidual"])
        correction = model.predict(pairs[features])
        models[model_name] = model
        feature_sets[model_name] = features
        corrections[model_name] = correction
        for weight in (0.25, 0.50, 0.75, 1.00):
            candidate = f"{model_name}_W{int(weight * 100):03d}"
            output = groups.copy()
            correction_map = dict(
                zip(pairs["CurrentLabelGroupID"], weight * correction)
            )
            output["Prediction"] = (
                output["LevelPrediction"]
                + output["LabelGroupID"].map(correction_map).fillna(0.0)
            )
            metric_records.extend(metric_rows(output, "Prediction", candidate, "delta"))
            candidate_predictions[candidate] = output["Prediction"].to_numpy()
    if not candidate_predictions:
        output = groups.copy()
        output["FinalPrediction"] = output["LevelPrediction"]
        return None, "NO_DELTA_MODEL", [], output, pairs, metric_records
    metrics = pd.DataFrame(metric_records)
    selected = min(
        sorted(candidate_predictions),
        key=lambda name: candidate_score(metrics, name, "delta"),
    )
    model_name, weight_text = selected.rsplit("_W", 1)
    output = groups.copy()
    output["FinalPrediction"] = candidate_predictions[selected]
    pairs["SelectedDeltaCorrection"] = corrections[model_name]
    bundle = {
        "model": models[model_name],
        "features": feature_sets[model_name],
        "weight": int(weight_text) / 100.0,
    }
    return bundle, selected, feature_sets[model_name], output, pairs, metric_records


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    groups, metadata, target_feature_map, lag_features = prepare_frame()
    target_features = sorted(target_feature_map)

    base_model, base_name, groups, base_metrics = train_base(groups, metadata)
    level_bundle, level_name, level_features, groups, level_metrics_rows = train_level_head(
        groups, metadata, target_features, lag_features
    )
    delta_bundle, delta_name, delta_features, groups, pairs, delta_metrics_rows = (
        train_delta_head(groups, metadata, target_features)
    )
    all_metrics = pd.DataFrame(
        base_metrics + level_metrics_rows + delta_metrics_rows
    )
    final_metrics = metric_rows(groups, "FinalPrediction", "R5_SELECTED", "final")
    all_metrics = pd.concat(
        [all_metrics, pd.DataFrame(final_metrics)], ignore_index=True
    )

    split_audit = audit_order_split(groups, "split")
    if not split_audit["passed"]:
        raise AssertionError("Final R5 split contains OrderID overlap")
    repeat_base = base_model.predict(groups.loc[groups["split"].eq("val")])[0]
    repeat_base_2 = base_model.predict(groups.loc[groups["split"].eq("val")])[0]
    repeat_delta = float(np.max(np.abs(repeat_base - repeat_base_2)))

    group_columns = [
        "LabelGroupID", "OrderID", "PalletID", "CompoundName", "RecipeID",
        "ProcessVersion", "MixerLine", "material_system", "phase_route",
        "TargetVectorHash", "StepPatternHash", "TemplateKey", "lab_time",
        "GroupProductionStart", "GroupProductionComplete", "split", "MNY",
        "BasePrediction", "LevelPrediction", "FinalPrediction",
        "prior_order_state_available", "prior_group_gap_hours", "order_group_index",
    ]
    groups.loc[:, group_columns].to_csv(
        OUTPUT / "r5_group_predictions.csv", index=False
    )
    pairs.to_csv(OUTPUT / "r5_order_delta_pairs.csv", index=False)
    all_metrics.to_csv(OUTPUT / "r5_metrics.csv", index=False)
    (OUTPUT / "target_feature_map.json").write_text(
        json.dumps(target_feature_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    bundle = {
        "contract": "V3.7.1-R5-ORDER-STATE-DELTA-V1",
        "base_model": base_model,
        "base_candidate": base_name,
        "level_head": level_bundle,
        "level_candidate": level_name,
        "delta_head": delta_bundle,
        "delta_candidate": delta_name,
        "target_feature_map": target_feature_map,
        "stage1_features": metadata["s1_features"],
        "stage2_features": metadata["s2_features"],
        "stage3_calibration_enabled": False,
    }
    model_path = MODEL_OUTPUT / "r5_order_state_delta_shadow.joblib"
    joblib.dump(bundle, model_path)

    final_frame = pd.DataFrame(final_metrics)
    validation_global = final_frame.loc[
        final_frame["split"].eq("val") & final_frame["scope"].eq("global")
    ].iloc[0]
    validation_t15760 = final_frame.loc[
        final_frame["split"].eq("val") & final_frame["scope"].eq("T15760")
    ]
    summary = {
        "contract": bundle["contract"],
        "selected_base": base_name,
        "selected_level": level_name,
        "selected_delta": delta_name,
        "outer_split_contract": "RECIPE_TIME_ORDER_ATOMIC_70_15_15",
        "outer_split_audit": split_audit,
        "base_crossfit_audit": base_model.crossfit_audit_,
        "target_feature_count": len(target_features),
        "level_feature_count": len(level_features),
        "delta_feature_count": len(delta_features),
        "delta_pair_counts": pairs["split"].value_counts().to_dict(),
        "validation_global_mae": float(validation_global["mae"]),
        "validation_global_direction_accuracy": (
            float(validation_global["direction_accuracy"])
            if pd.notna(validation_global["direction_accuracy"]) else None
        ),
        "validation_t15760_mae": (
            float(validation_t15760.iloc[0]["mae"])
            if not validation_t15760.empty else None
        ),
        "target_mae_mny": 2.5,
        "release_status": "NOT_RELEASED_GATES_FAILED_OR_BLIND_WINDOW_PENDING",
        "same_input_repeat_max_abs_delta": repeat_delta,
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
    (OUTPUT / "r5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
