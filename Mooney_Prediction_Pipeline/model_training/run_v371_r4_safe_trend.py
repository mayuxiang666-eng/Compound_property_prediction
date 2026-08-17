"""Train/evaluate R4 prequential trend experts on the frozen RDS Active snapshot."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "Mooney_Prediction_Pipeline"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.model_training.safe_trend_expert import (
    SafeTrendResidualExpert,
    attach_safe_prior_lab_anchors,
)
from Mooney_Prediction_Pipeline.model_training.trend_metric_contract import (
    evaluate_level_and_trend,
)


BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUTPUT = BASE / "v371_r4_safe_trend"
MODEL_OUTPUT = PIPELINE / "models" / "v371_challenger_r4_safe_trend"
GROUP_SOURCE = BASE / "pallet_group_training_frame_c2_prepared.csv"
PHASE_SOURCE = BASE / "phase_batch_features_active_v1.csv"
R3_MODEL = PIPELINE / "models" / "v371_challenger_r3_expert_ablation" / "hybrid_model_shadow.joblib"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scope_mask(frame: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "global":
        return pd.Series(True, index=frame.index)
    compound = frame["CompoundName"].astype(str)
    if scope == "T15760":
        return compound.str.contains("T15760", na=False)
    if scope == "T15760_PV46":
        version = pd.to_numeric(frame["ProcessVersion"], errors="coerce")
        return compound.str.contains("T15760", na=False) & version.eq(46.0)
    if scope == "B00458":
        return compound.str.contains("B00458", na=False)
    raise ValueError(scope)


def metric_rows(frame: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    rows = []
    segments = ["RecipeID", "ProcessVersion", "MixerLine"]
    for candidate in candidates:
        prediction_col = f"prediction__{candidate}"
        for split in ("train", "val", "oot"):
            split_frame = frame.loc[frame["split"].eq(split)].copy()
            for scope in ("global", "T15760", "T15760_PV46", "B00458"):
                part = split_frame.loc[scope_mask(split_frame, scope)].copy()
                if part.empty:
                    continue
                metrics, _ = evaluate_level_and_trend(
                    part,
                    actual_col="MNY",
                    prediction_col=prediction_col,
                    time_col="lab_time",
                    segment_cols=segments,
                    significant_delta_mny=2.0,
                    max_gap_days=30.0,
                    level_mae_limit=2.5,
                    minimum_pairs=30,
                    minimum_significant_pairs=20,
                    minimum_delta_pearson=0.50,
                    minimum_direction_accuracy=0.70,
                    delta_std_ratio_low=0.70,
                    delta_std_ratio_high=1.30,
                )
                rows.append({"candidate": candidate, "split": split, "scope": scope, **metrics.as_dict()})
    return pd.DataFrame(rows)


def selection_score(metrics: pd.DataFrame, candidate: str) -> float:
    selected = metrics.loc[
        metrics["candidate"].eq(candidate)
        & metrics["split"].eq("val")
        & metrics["scope"].isin(["global", "T15760"])
    ].set_index("scope")
    if not {"global", "T15760"}.issubset(selected.index):
        return float("inf")
    global_row, t_row = selected.loc["global"], selected.loc["T15760"]
    score = float(global_row["level_mae"] + 0.50 * t_row["level_mae"])
    for row, weight in ((global_row, 2.0), (t_row, 1.0)):
        corr = float(row["delta_pearson"]) if pd.notna(row["delta_pearson"]) else -1.0
        direction = float(row["direction_accuracy"]) if pd.notna(row["direction_accuracy"]) else 0.0
        ratio = float(row["delta_std_ratio"]) if pd.notna(row["delta_std_ratio"]) else 0.0
        score += weight * max(0.0, 0.50 - corr)
        score += weight * max(0.0, 0.70 - direction)
        score += 0.5 * weight * max(0.0, 0.70 - ratio)
        score += 0.5 * weight * max(0.0, ratio - 1.30)
    return score


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    groups = pd.read_csv(GROUP_SOURCE, low_memory=False)
    phase = pd.read_csv(
        PHASE_SOURCE,
        usecols=["LabelGroupID", "BatchEndTime"],
        low_memory=False,
    )
    completion = (
        phase.assign(BatchEndTime=lambda frame: pd.to_datetime(frame["BatchEndTime"], errors="coerce"))
        .dropna(subset=["LabelGroupID", "BatchEndTime"])
        .groupby("LabelGroupID", as_index=False)["BatchEndTime"].max()
        .rename(columns={"BatchEndTime": "GroupProductionComplete"})
    )
    groups = groups.merge(completion, on="LabelGroupID", how="left", validate="one_to_one")
    if groups["GroupProductionComplete"].isna().any():
        raise ValueError("R4 cannot infer production completion for every Lab group")

    model = joblib.load(R3_MODEL)
    raw, stage1, stage1b, stage2 = model.predict(groups, cluster_col="material_system")
    groups["RawPrediction"] = raw
    groups["Stage1Prediction"] = stage1
    groups["Stage1bBias"] = stage1b
    groups["Stage2Residual"] = stage2
    process_features = tuple(model.feature_names_s2_)
    anchor_columns = list(dict.fromkeys([
        "LabelGroupID", "OrderID", "PalletID", "RecipeID", "ProcessVersion", "MixerLine",
        "CompoundName", "material_system", "phase_route", "TemplateKey", "MNY", "lab_time",
        "split", "GroupProductionComplete", "RawPrediction", *process_features,
    ]))
    anchored = attach_safe_prior_lab_anchors(
        groups.loc[:, anchor_columns],
        production_time_col="GroupProductionComplete",
    )
    for column in ("Stage1Prediction", "Stage1bBias", "Stage2Residual"):
        anchored[column] = groups.set_index("LabelGroupID").loc[anchored["LabelGroupID"], column].to_numpy()

    anchored["prediction__R3_RAW"] = anchored["RawPrediction"]
    candidate_names = ["R3_RAW"]
    anchor_residual = (
        pd.to_numeric(anchored["anchor__MNY"], errors="coerce")
        - pd.to_numeric(anchored["anchor__RawPrediction"], errors="coerce")
    ).fillna(0.0)
    for weight in (0.25, 0.50, 0.75, 1.00):
        name = f"R4_SAFE_STATE_{int(weight * 100):03d}"
        anchored[f"prediction__{name}"] = anchored["RawPrediction"] + weight * anchor_residual
        candidate_names.append(name)
        same_order_name = f"R4_SAME_ORDER_STATE_{int(weight * 100):03d}"
        same_order_residual = anchor_residual.where(
            anchored["anchor_scope"].eq("SAME_ORDER_RECIPE_VERSION_MIXER"), 0.0
        )
        anchored[f"prediction__{same_order_name}"] = (
            anchored["RawPrediction"] + weight * same_order_residual
        )
        candidate_names.append(same_order_name)

    fitted_experts = {}
    train = anchored.loc[anchored["split"].eq("train")].copy()
    for top_k in (10, 30, 60):
        for alpha in (1.0, 10.0, 100.0):
            name = f"R4_RIDGE_K{top_k}_A{int(alpha)}"
            expert = SafeTrendResidualExpert(
                alpha=alpha,
                top_k_process_deltas=top_k,
                process_features=process_features,
            ).fit(train)
            anchored[f"prediction__{name}"] = expert.predict(anchored)
            fitted_experts[name] = expert
            candidate_names.append(name)

    metrics = metric_rows(anchored, candidate_names)
    scores = pd.DataFrame([
        {"candidate": candidate, "selection_score": selection_score(metrics, candidate)}
        for candidate in candidate_names
    ]).sort_values(["selection_score", "candidate"], kind="stable")
    selected = str(scores.iloc[0]["candidate"])
    anchored["SelectedPrediction"] = anchored[f"prediction__{selected}"]

    audit_columns = [
        "LabelGroupID", "OrderID", "PalletID", "CompoundName", "RecipeID", "ProcessVersion",
        "MixerLine", "material_system", "phase_route", "TemplateKey", "MNY", "lab_time", "split",
        "GroupProductionComplete", "RawPrediction", "anchor_available", "anchor_scope",
        "anchor__LabelGroupID", "anchor__MNY", "anchor__RawPrediction", "anchor__lab_time",
        "anchor_lab_age_hours", "anchor_production_gap_hours", "anchor_is_same_template",
        *[f"prediction__{name}" for name in candidate_names], "SelectedPrediction",
    ]
    anchored.loc[:, audit_columns].to_csv(OUTPUT / "r4_group_predictions.csv", index=False)
    metrics.to_csv(OUTPUT / "r4_metrics.csv", index=False)
    scores.to_csv(OUTPUT / "r4_selection_scores.csv", index=False)

    selected_object = fitted_experts.get(selected)
    bundle = {
        "contract": "V3.7.1-R4-SAFE-PREQUENTIAL-TREND-V1",
        "selected_candidate": selected,
        "base_model_path": str(R3_MODEL),
        "base_model_sha256": sha256(R3_MODEL),
        "trend_expert": selected_object,
        "state_weight": (
            float(selected.rsplit("_", 1)[-1]) / 100.0
            if selected.startswith("R4_SAFE_STATE_") else None
        ),
        "production_time_contract": "ANCHOR_LAB_TIME_LE_CURRENT_GROUP_PRODUCTION_COMPLETE",
        "stage3_calibration_enabled": False,
    }
    model_path = MODEL_OUTPUT / "safe_trend_expert_shadow.joblib"
    joblib.dump(bundle, model_path)

    key_metrics = metrics.loc[
        metrics["candidate"].eq(selected)
        & metrics["split"].isin(["val", "oot"])
        & metrics["scope"].isin(["global", "T15760", "T15760_PV46", "B00458"])
    ].copy()
    val_global = key_metrics.loc[
        key_metrics["split"].eq("val") & key_metrics["scope"].eq("global")
    ].iloc[0]
    release_status = "NOT_RELEASED_LEVEL_AND_TREND_GATES_FAILED"
    summary = {
        "contract": bundle["contract"],
        "selected_candidate": selected,
        "selection_rule": "VAL_GLOBAL_PLUS_T15760_LEVEL_AND_TREND_PENALTY",
        "validation_global_mae": float(val_global["level_mae"]),
        "validation_global_trend_passed": bool(val_global["trend_gate_passed"]),
        "target_mae_mny": 2.5,
        "release_status": release_status,
        "anchor_availability_rate": float(anchored["anchor_available"].mean()),
        "same_order_anchor_availability_rate": float(
            anchored["anchor_scope"].eq("SAME_ORDER_RECIPE_VERSION_MIXER").mean()
        ),
        "future_lab_anchor_violations": int((
            pd.to_datetime(anchored["anchor__lab_time"], errors="coerce", utc=True)
            > pd.to_datetime(anchored["GroupProductionComplete"], errors="coerce", utc=True)
        ).sum()),
        "model_sha256": sha256(model_path),
        "source_sha256": sha256(GROUP_SOURCE),
        "local_mms_access": False,
        "database_access_during_run": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "r4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**summary, "key_metrics": key_metrics.to_dict("records")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
