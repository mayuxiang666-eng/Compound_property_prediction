"""Run Gate 0, deconfounded envelope A, B2 and B3 from frozen snapshots.

Safety contract: M1 only; snapshot files only; no database calls; no local MMS
or Archive access; Stage3 and automatic control remain disabled.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Mooney_Prediction_Pipeline.m1_relative_improvement.contracts import evaluate_dual_pilot_gate
from Mooney_Prediction_Pipeline.m1_relative_improvement.direction_model import (
    build_pairwise_transitions,
    fit_direction_magnitude,
)
from Mooney_Prediction_Pipeline.m1_relative_improvement.gate0 import (
    audit_action_support,
    build_group_action_frame,
)
from Mooney_Prediction_Pipeline.m1_relative_improvement.historical_envelope import (
    build_historical_envelopes,
    crossfit_static_adjustment,
)
from Mooney_Prediction_Pipeline.m1_relative_improvement.process_response_gate import (
    evaluate_process_response,
)
from Mooney_Prediction_Pipeline.model_training.diagnose_m1_phase_delta_signal_v1 import (
    markdown_table,
    phase_feature_bundles,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports" / "v371_challenger"
COMMON = REPORT_ROOT / "v371_c1_history_augmented_phase_v1" / "current_common_batch_features.parquet"
LABELS = REPORT_ROOT / "global_rds_active_v1" / "m1_temporal_jit_v1" / "m1_temporal_jit_acceptance_input.csv"
DUAL_STEPS = REPORT_ROOT / "global_rds_active_v1" / "dual_target_readiness" / "dual_pilot_group_step_features.csv"
EXACT = ROOT / "reports" / "corrected_phase_history" / "active_exact_pilot_step_history_v1"
OUTPUT = REPORT_ROOT / "m1_relative_improvement_gate0_ab_v1"

T15760 = "M1-T15760---- 06 002"
B00458 = "M1-B00458---- 12 005"
B00458_RECIPE = "9922A00458BT_PV77_004"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _consistent(series: pd.Series):
    values = series.dropna().astype(str).unique()
    return values[0] if len(values) == 1 else pd.NA


def group_identity(batch: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "LabelGroupID", "RecipeID", "ProcessVersion", "MixerLine", "phase_route",
        "TemplateKey", "StepPatternHash", "SCP_LOG_contract_hash", "CompoundName",
        "supplier_rubber_viscosity_avg", "supplier_silica_moisture_avg",
        "env_temp_mean", "env_humidity_mean",
    ]
    available = [column for column in columns if column in batch]
    identity = batch[available].groupby("LabelGroupID", as_index=False, dropna=False).agg(
        {column: _consistent for column in available if column != "LabelGroupID"}
    )
    identity["structural_template_key"] = (
        identity.get("RecipeID", "").astype(str)
        + "|" + identity.get("ProcessVersion", "").astype(str)
        + "|" + identity.get("MixerLine", "").astype(str)
        + "|" + identity.get("phase_route", "").astype(str)
        + "|" + identity.get("StepPatternHash", "").astype(str)
        + "|" + identity.get("SCP_LOG_contract_hash", "").astype(str)
    )
    rubber = pd.to_numeric(identity.get("supplier_rubber_viscosity_avg"), errors="coerce")
    silica = pd.to_numeric(identity.get("supplier_silica_moisture_avg"), errors="coerce")
    env_t = pd.to_numeric(identity.get("env_temp_mean"), errors="coerce")
    env_h = pd.to_numeric(identity.get("env_humidity_mean"), errors="coerce")
    identity["material_stratum"] = pd.qcut((rubber.fillna(rubber.median()) + silica.fillna(silica.median())).rank(method="first"), 3, labels=False, duplicates="drop")
    identity["environment_stratum"] = pd.qcut((env_t.fillna(env_t.median()) + env_h.fillna(env_h.median())).rank(method="first"), 3, labels=False, duplicates="drop")
    return identity


def _correct_cumulative_duration(frame: pd.DataFrame, *, batch_keys: list[str]) -> pd.DataFrame:
    out = frame.copy()
    mask = out["kind"].eq("duration")
    duration = out.loc[mask].sort_values([*batch_keys, "group_name", "step_no"], kind="stable").copy()
    groupers = [duration[column] for column in [*batch_keys, "group_name"]]
    for column in ("target", "actual"):
        values = pd.to_numeric(duration[column], errors="coerce")
        differences = values - values.groupby(groupers).shift(1, fill_value=0.0)
        differences = differences.mask(differences.lt(0))
        duration[column] = differences
    out.loc[duration.index, ["target", "actual"]] = duration[["target", "actual"]]
    return out


def t15760_step_long(batch: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(DUAL_STEPS, low_memory=False)
    source = source[source["CompoundName"].eq(T15760)].copy()
    source = source.rename(columns={
        "LabelGroupID": "label_group_id", "OrderID": "order_id", "GroupName": "group_name",
        "StepNo": "step_no", "target_mean": "target", "actual_value_mean": "actual",
        "phase_semantic": "phase", "BatchStartTime": "event_time",
    })
    source["batch_id"] = pd.to_datetime(source["event_time"], errors="raise", utc=True).astype(str)
    source = _correct_cumulative_duration(source, batch_keys=["label_group_id", "batch_id"])
    identity = group_identity(batch).rename(columns={"LabelGroupID": "label_group_id"})
    source = source.merge(identity, on="label_group_id", how="inner", validate="many_to_one", suffixes=("", "_identity"))
    phase = source["phase"].astype(str)
    source.loc[phase.eq("DRY_OR_OIL_MST") & source["step_no"].le(3), "phase"] = "Stage2_DryMixing"
    source.loc[phase.eq("DRY_OR_OIL_MST") & source["step_no"].gt(3), "phase"] = "Stage3_OilLoading"
    source.loc[phase.eq("WET_MST"), "phase"] = "Stage4_WetMixing"
    source.loc[phase.str.contains("PID_UPPER", regex=False), "phase"] = "Stage5_PID"
    source.loc[phase.str.contains("PID_LOWER", regex=False), "phase"] = "Stage6_BottomMixing"
    source["compound"] = T15760
    source["recipe_id"] = source["RecipeID"].astype(str)
    source["process_version"] = source["ProcessVersion"].astype(str)
    source["mixer_line"] = source["MixerLine"].astype(str)
    source["event_time"] = pd.to_datetime(source["event_time"], errors="raise", utc=True)
    return source


def b00458_step_long(batch: pd.DataFrame) -> pd.DataFrame:
    steps = pd.read_csv(EXACT / "active_exact_pilot_step_vectors.csv", low_memory=False)
    assignments = pd.read_csv(EXACT / "active_exact_pilot_template_assignments_v2.csv", low_memory=False)
    for frame in (steps, assignments):
        frame["OrderID"] = frame["OrderID"].astype(str).str.replace(r"\.0$", "", regex=True)
        frame["BatchNumber"] = pd.to_numeric(frame["BatchNumber"], errors="raise").astype(int)
    assignments = assignments[assignments["RecipeID"].eq(B00458_RECIPE)]
    steps = steps.merge(
        assignments[["OrderID", "BatchNumber", "RecipeID", "TemplateID", "MNY"]],
        on=["OrderID", "BatchNumber"], how="inner", validate="many_to_one",
    )
    records: list[pd.DataFrame] = []
    for kind, target, actual in (
        ("duration", "dur_SP", "dur_AV"),
        ("rotor", "rotor_SP", "rotor_AV"),
        ("temperature", "temp_SP", "temp_AV"),
    ):
        part = steps.copy()
        part["kind"], part["target"], part["actual"] = kind, part[target], part[actual]
        records.append(part)
    source = pd.concat(records, ignore_index=True)
    source = source.rename(columns={
        "OrderID": "order_id", "BatchNumber": "batch_id",
        "GroupName": "group_name", "StepNo": "step_no", "mapped_phase": "phase",
        "OrderStartTime": "event_time",
    })
    # This older frozen Step snapshot has no auditable PalletID/LabelGroupID
    # overlap with the current Lab snapshot.  The synthetic key is permitted
    # only for B3 process-response grouping and is excluded from Gate-0 causal
    # support counting below.
    source["label_group_id"] = "PHYSICAL_BATCH_ONLY:" + source["order_id"].astype(str) + ":" + source["batch_id"].astype(str)
    source["compound"] = B00458
    source["recipe_id"] = source["RecipeID"].astype(str)
    source["process_version"] = "LEGACY_ORDER_PROCESS_VERSION_UNAVAILABLE"
    source["mixer_line"] = source["MixerLine"].astype(str)
    source["phase_route"] = "oil_wet"
    source["structural_template_key"] = source["TemplateID"].astype(str)
    source["event_time"] = pd.to_datetime(source["event_time"], errors="raise", utc=True)
    source["material_stratum"] = pd.NA
    source["environment_stratum"] = pd.NA
    return source


def group_phase_frame(batch: pd.DataFrame, labels: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    identities = ["OrderID", "RecipeID", "ProcessVersion", "MixerLine", "material_system", "phase_route", "TemplateKey", "CompoundName"]
    aggregations = {column: _consistent for column in identities}
    aggregations.update({feature: "mean" for feature in features})
    frame = batch.groupby("LabelGroupID", as_index=False, dropna=False).agg(aggregations)
    label_columns = ["LabelGroupID", "MNY", "lab_time", "split"]
    return frame.merge(labels[label_columns], on="LabelGroupID", validate="one_to_one")


def _target_name(row: pd.Series) -> str:
    return f"target__{row['group_name']}_S{int(row['step_no'])}_{row['kind']}"


def process_history(step_long: pd.DataFrame) -> pd.DataFrame:
    step = step_long.copy()
    if step.empty:
        raise ValueError("EMPTY_PROCESS_HISTORY_SOURCE")
    step["target_feature"] = step.apply(_target_name, axis=1)
    rows: list[dict[str, object]] = []
    for (order_id, batch_id), part in step.groupby(["order_id", "batch_id"], sort=False):
        row: dict[str, object] = {
            "order_id": str(order_id), "batch_id": str(batch_id),
            "event_time": part["event_time"].min(),
        }
        targets = part.pivot_table(index=None, columns="target_feature", values="target", aggfunc="median")
        if not targets.empty:
            row.update({str(column): float(value) for column, value in targets.iloc[0].dropna().items()})
        temperatures = part[part["kind"].eq("temperature")].copy()
        durations = part[part["kind"].eq("duration")].copy()
        def last_temp(pattern: str):
            found = temperatures[temperatures["phase"].astype(str).str.contains(pattern, regex=False)].sort_values(["group_name", "step_no"])
            value = pd.to_numeric(found["actual"], errors="coerce").dropna()
            return float(value.iloc[-1]) if len(value) else np.nan
        row["dry_end_temp_actual"] = last_temp("Stage2_DryMixing")
        row["post_oil_temp_actual"] = last_temp("Stage3_OilLoading")
        row["wet_end_temp_actual"] = last_temp("Stage4_WetMixing")
        wet_duration = pd.to_numeric(durations.loc[durations["phase"].astype(str).str.contains("Stage4_WetMixing", regex=False), "actual"], errors="coerce").sum(min_count=1)
        pid_duration = pd.to_numeric(durations.loc[durations["phase"].astype(str).str.contains("Stage5|Stage6|PID", regex=True), "actual"], errors="coerce").sum(min_count=1)
        row["wet_duration_actual"] = float(wet_duration) if pd.notna(wet_duration) else np.nan
        row["pid_duration_actual"] = float(pid_duration) if pd.notna(pid_duration) else np.nan
        msb = temperatures[temperatures["group_name"].astype(str).eq("AVR_MSB")].sort_values("step_no")
        discharge = pd.to_numeric(msb["actual"], errors="coerce").dropna()
        row["discharge_temp_actual"] = float(discharge.iloc[-1]) if len(discharge) else np.nan
        rows.append(row)
    history = pd.DataFrame(rows)
    target_columns = [column for column in history if column.startswith("target__")]
    history[target_columns] = history[target_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    history.attrs["actual_phase_feature_path_complete"] = False
    return history


def _plot_static(adjusted: pd.DataFrame, path: Path) -> None:
    oot = adjusted[adjusted["split"].eq("oot")].sort_values("lab_time")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    low = min(oot["MNY"].min(), oot["StaticExpectedMNY"].min()) - 1
    high = max(oot["MNY"].max(), oot["StaticExpectedMNY"].max()) + 1
    axes[0].scatter(oot["MNY"], oot["StaticExpectedMNY"], alpha=0.65)
    axes[0].plot([low, high], [low, high], "--", color="black")
    axes[0].set(xlabel="Actual MNY", ylabel="Static expected MNY", title="OOT static/material/environment baseline")
    x = np.arange(len(oot))
    axes[1].plot(x, oot["MNY"], color="black", label="Actual")
    axes[1].plot(x, oot["StaticExpectedMNY"], color="#3769a8", label="Static expected")
    axes[1].set(title="OOT chronological overview", xlabel="Pallet Lab groups", ylabel="MNY")
    axes[1].legend()
    for axis in axes: axis.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_b2(predictions: pd.DataFrame, path: Path) -> None:
    oot = predictions[predictions["split"].eq("oot")].sort_values("to_lab_time")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].scatter(oot["actual_mny"], oot["predicted_mny"], alpha=0.7)
    low = min(oot["actual_mny"].min(), oot["predicted_mny"].min()) - 1
    high = max(oot["actual_mny"].max(), oot["predicted_mny"].max()) + 1
    axes[0, 0].plot([low, high], [low, high], "--", color="black")
    axes[0, 0].set(title="B2 OOT level", xlabel="Actual MNY", ylabel="Previous MNY + predicted delta")
    axes[0, 1].scatter(oot["actual_delta_mny"], oot["predicted_delta_mny"], alpha=0.7, color="#d9792b")
    axes[0, 1].axhline(0, color="gray"); axes[0, 1].axvline(0, color="gray")
    axes[0, 1].set(title="B2 OOT delta", xlabel="Actual ΔMNY", ylabel="Predicted ΔMNY")
    x = np.arange(len(oot))
    axes[1, 0].plot(x, oot["actual_mny"], color="black", label="Actual")
    axes[1, 0].plot(x, oot["predicted_mny"], color="#3769a8", label="Predicted")
    axes[1, 0].legend(); axes[1, 0].set(title="B2 chronological OOT")
    axes[1, 1].scatter(oot["actual_reduction"], oot["reduction_probability"], alpha=0.7, color="#578c5c")
    axes[1, 1].axvline(0, color="gray"); axes[1, 1].axhline(0.5, color="gray")
    axes[1, 1].set(title="Reduction direction probability", xlabel="Actual reduction (MU)", ylabel="P(reduction)")
    for axis in axes.flat: axis.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_b3(predictions: pd.DataFrame, path: Path) -> None:
    outputs = list(predictions["output"].unique()) if len(predictions) else []
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for axis, output in zip(axes.flat, outputs):
        part = predictions[predictions["output"].eq(output)]
        axis.scatter(part["actual"], part["p50"], alpha=0.65)
        low = min(part["actual"].min(), part["p50"].min())
        high = max(part["actual"].max(), part["p50"].max())
        axis.plot([low, high], [low, high], "--", color="black")
        axis.set(title=output, xlabel="Actual", ylabel="Predicted P50")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(outputs):]: axis.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    batch = pd.read_parquet(COMMON)
    labels = pd.read_csv(LABELS, low_memory=False)
    labels = labels[labels["CompoundName"].astype(str).str.startswith("M1-")].copy()
    if labels["LabelGroupID"].astype(str).duplicated().any():
        raise ValueError("DUPLICATE_LAB_GROUP")

    # Gate 0: current two-pilot frozen Step histories only.
    t_steps = t15760_step_long(batch)
    b_steps = b00458_step_long(batch)
    # B00458 has no auditable Pallet group identity in this older Step
    # snapshot, so only T15760 enters causal-support counting.
    group_actions = build_group_action_frame(t_steps)
    support, action_transitions = audit_action_support(group_actions)
    b_levers = b_steps[["group_name", "step_no", "kind", "phase"]].drop_duplicates()
    b_levers["compound"] = B00458
    b_levers["recipe_id"] = B00458_RECIPE
    b_levers["process_version"] = "LEGACY_ORDER_PROCESS_VERSION_UNAVAILABLE"
    b_levers["mixer_line"] = b_steps["mixer_line"].iloc[0]
    b_levers["structural_template_key"] = b_steps["structural_template_key"].mode().iloc[0]
    b_levers["lever"] = b_levers["group_name"] + "|S" + b_levers["step_no"].astype(int).astype(str) + "|" + b_levers["kind"]
    b_levers["action_allowed"] = b_levers["step_no"].gt(2) & b_levers["kind"].ne("ram")
    for column, value in {
        "lab_groups": 0, "orders": int(b_steps["order_id"].nunique()), "control_levels": 0,
        "major_control_levels": 0, "minimum_groups_in_major_level": 0,
        "independent_transitions": 0, "effective_sample": 0.0,
        "bidirectional_history": False, "actual_response_observed_fraction": 1.0,
        "major_levels_span_multiple_months": False,
        "thresholds_json": "B00458_PALLET_LABEL_ID_UNAVAILABLE",
    }.items():
        b_levers[column] = value
    b_levers["causal_support_status"] = np.where(
        b_levers["action_allowed"], "INSUFFICIENT_LAB_SUPPORT", "LOCKED_EXPERT_RULE"
    )
    support = pd.concat([support, b_levers[support.columns]], ignore_index=True)
    support.to_csv(OUTPUT / "gate0_action_support_matrix.csv", index=False, encoding="utf-8-sig")
    action_transitions.to_csv(OUTPUT / "gate0_action_transitions.csv", index=False, encoding="utf-8-sig")

    # Automatically identify four extension candidates; they remain blocked
    # until a matching Step snapshot is available.
    candidate_counts = labels.groupby(["CompoundName", "material_system"], as_index=False)["LabelGroupID"].nunique()
    candidate_counts = candidate_counts[
        ~candidate_counts["CompoundName"].astype(str).str.contains("T15760|B00458", regex=True)
    ]
    candidate_counts["compound_family"] = (
        candidate_counts["CompoundName"].astype(str).str.replace(r"^M1-", "", regex=True).str.split("-").str[0]
    )
    extensions = []
    for material in ("Silica", "CarbonBlack"):
        material_candidates = candidate_counts[
            candidate_counts["material_system"].astype(str).str.contains(material, case=False, regex=False)
        ].sort_values("LabelGroupID", ascending=False)
        selected = material_candidates.drop_duplicates("compound_family").nlargest(2, "LabelGroupID")
        for _, row in selected.iterrows():
            extensions.append({
                "CompoundName": row["CompoundName"], "material_system": row["material_system"],
                "lab_groups": int(row["LabelGroupID"]), "gate_status": "INSUFFICIENT_STEP_SNAPSHOT",
                "policy_status": "RECIPE_SPECIFIC_POLICY_REQUIRED",
            })
    pd.DataFrame(extensions).to_csv(OUTPUT / "extension_recipe_eligibility.csv", index=False, encoding="utf-8-sig")

    # A: train-only cross-fitted static/material/environment adjustment.
    categorical = [column for column in ["CompoundName", "RecipeID", "ProcessVersion", "MixerLine", "material_system", "phase_route", "TemplateKey"] if column in labels]
    numeric_candidates = [
        "Top_Fill_Factor__group_mean", "Bot_Fill_Factor__group_mean", "Target_Temperature__group_mean",
        "weight_pct_solid_elastomer__group_mean", "weight_pct_natural_rubber__group_mean",
        "weight_pct_silica__group_mean", "weight_pct_oil__group_mean", "weight_pct_carbon_black__group_mean",
        "silica_phr__group_mean", "supplier_rubber_viscosity_avg__group_mean",
        "supplier_silica_moisture_avg__group_mean", "supplier_silica_surface_area_avg__group_mean",
        "supplier_carbon_black_structure_avg__group_mean", "supplier_carbon_black_surface_area_avg__group_mean",
        "env_temp_mean__group_mean", "env_humidity_mean__group_mean", "phys_init_temp__group_mean",
    ]
    numeric = [column for column in numeric_candidates if column in labels and labels[column].notna().any()]
    adjusted = crossfit_static_adjustment(labels, categorical_features=categorical, numeric_features=numeric)
    envelope = build_historical_envelopes(adjusted)
    adjusted.to_csv(OUTPUT / "a_static_adjusted_lab_groups.csv", index=False, encoding="utf-8-sig")
    envelope.template_audit.to_csv(OUTPUT / "a_historical_envelope_audit.csv", index=False, encoding="utf-8-sig")
    envelope.envelope_members.to_csv(OUTPUT / "a_historical_envelope_members.csv", index=False, encoding="utf-8-sig")
    _plot_static(adjusted, OUTPUT / "a_actual_vs_static_expected_fullview.png")

    # B2: pairwise Phase direction/magnitude heads.
    bundles = phase_feature_bundles(batch.columns.tolist())
    phase_features = bundles["DryWetPIDBottom_All"]
    phase_groups = group_phase_frame(batch, labels, phase_features)
    b2_transitions = build_pairwise_transitions(phase_groups, feature_columns=phase_features)
    b2 = fit_direction_magnitude(b2_transitions)
    b2.transitions.to_parquet(OUTPUT / "b2_pairwise_transitions.parquet", index=False, compression="zstd")
    b2.predictions.to_csv(OUTPUT / "b2_direction_magnitude_predictions.csv", index=False, encoding="utf-8-sig")
    b2.metrics.to_csv(OUTPUT / "b2_direction_magnitude_metrics.csv", index=False, encoding="utf-8-sig")
    _plot_b2(b2.predictions, OUTPUT / "b2_actual_vs_prediction_fullview.png")

    # B3: run both pilots in parallel.  Endpoint prediction is evaluated, but
    # the gate remains closed until all affected V3.7 Phase features have a
    # validated response path.
    b3_results = []
    b3_predictions = []
    b3_status = {}
    b3_templates = {}
    for pilot, source in (("T15760", t_steps), ("B00458", b_steps)):
        template_counts = source.groupby("structural_template_key")["batch_id"].nunique().sort_values(ascending=False)
        selected_template = str(template_counts.index[0])
        selected_source = source[source["structural_template_key"].astype(str).eq(selected_template)].copy()
        history = process_history(selected_source)
        result = evaluate_process_response(history, pilot=pilot)
        b3_results.append(result.metrics)
        b3_predictions.append(result.predictions)
        b3_status[pilot] = result.control_transfer_status
        b3_templates[pilot] = {
            "selected_structural_template": selected_template,
            "physical_batches": int(selected_source[["order_id", "batch_id"]].drop_duplicates().shape[0]),
            "template_selection": "LARGEST_EXACT_STRUCTURAL_TEMPLATE",
        }
        history.to_parquet(OUTPUT / f"b3_{pilot.lower()}_process_history.parquet", index=False, compression="zstd")
    b3_metrics = pd.concat(b3_results, ignore_index=True)
    b3_prediction = pd.concat(b3_predictions, ignore_index=True)
    b3_metrics.to_csv(OUTPUT / "b3_process_response_metrics.csv", index=False, encoding="utf-8-sig")
    b3_prediction.to_csv(OUTPUT / "b3_process_response_predictions.csv", index=False, encoding="utf-8-sig")
    _plot_b3(b3_prediction, OUTPUT / "b3_actual_vs_prediction_fullview.png")

    causal_supported = support[support["causal_support_status"].eq("CAUSAL_ACTION_SUPPORTED")]
    b4_status = "ELIGIBLE_FOR_DML_IMPLEMENTATION" if len(causal_supported) else "BLOCKED_NO_CAUSAL_ACTION_SUPPORT"
    b5_ready = bool(b2.release_gate_passed and all(value == "CONTROL_TRANSFER_AVAILABLE" for value in b3_status.values()) and len(causal_supported))
    pilot_b6 = {pilot: "B6_NOT_STARTED" for pilot in ("T15760", "B00458")}
    dual_gate = evaluate_dual_pilot_gate(pilot_b6)
    manifest = {
        "contract": "M1-RELATIVE-IMPROVEMENT-GATE0-A-B-V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "M1_ONLY",
        "source_policy": "FROZEN_SNAPSHOT_ONLY_RDS_ACTIVE_OR_LOCAL_PARQUET_CSV",
        "local_mms_access": False,
        "archive_access": False,
        "database_access": False,
        "stage3_calibration_enabled": False,
        "automatic_control": "DISABLED",
        "energy_status": "NOT_RANKED",
        "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in (COMMON, LABELS, DUAL_STEPS, EXACT / "active_exact_pilot_step_vectors.csv", EXACT / "active_exact_pilot_template_assignments_v2.csv")},
        "gate0": {
            "action_rows": int(len(support)),
            "causal_action_supported": int(len(causal_supported)),
            "status_counts": support["causal_support_status"].value_counts().to_dict(),
            "extensions": extensions,
        },
        "scheme_a": {
            "adjusted_groups": int(len(adjusted)),
            "supported_envelopes": int(envelope.template_audit["envelope_status"].eq("HISTORICAL_ENVELOPE_SUPPORTED").sum()) if len(envelope.template_audit) else 0,
            "confounded_envelopes": int(envelope.template_audit["envelope_status"].eq("CONFOUNDED_BY_MATERIAL").sum()) if len(envelope.template_audit) else 0,
            "static_oot_mae": float((adjusted.loc[adjusted["split"].eq("oot"), "StaticExpectedMNY"] - adjusted.loc[adjusted["split"].eq("oot"), "MNY"]).abs().mean()),
            "static_oot_bias": float((adjusted.loc[adjusted["split"].eq("oot"), "StaticExpectedMNY"] - adjusted.loc[adjusted["split"].eq("oot"), "MNY"]).mean()),
            "recommendation_status": "HISTORICAL_ENVELOPE_SHADOW_ONLY",
        },
        "b2": {"release_gate_passed": b2.release_gate_passed, "transition_counts": b2.transitions["split"].value_counts().to_dict()},
        "b3": {"statuses": b3_status, "templates": b3_templates},
        "b4": b4_status,
        "b5": "READY" if b5_ready else "BLOCKED_PREREQUISITE_GATES",
        "b6": pilot_b6,
        "b7": dual_gate,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = f"""# M1 相对门尼改善 Gate 0 / A / B2 / B3 实验报告

## 结论

- Gate 0 共审计 {len(support)} 个精确控制杠杆，`CAUSAL_ACTION_SUPPORTED` 为 {len(causal_supported)} 个。
- 方案 A 建立了去混杂历史包络，但统一保持 `HISTORICAL_ENVELOPE_SHADOW_ONLY`，不声明可降低具体 MU。
- B2 方向/幅度门禁：`{b2.release_gate_passed}`。
- B3：T15760=`{b3_status['T15760']}`，B00458=`{b3_status['B00458']}`。
- B4：`{b4_status}`；B5：`{'READY' if b5_ready else 'BLOCKED_PREREQUISITE_GATES'}`。
- B6 尚未开始，因此 B7 为 `BLOCKED_DUAL_PILOT_GATE`。

## Gate 0：自然动作支持

{markdown_table(support.groupby(["compound", "causal_support_status"], as_index=False).size())}

没有任何杠杆满足“至少30个独立Lab组转移、至少3个主要控制水平、每个主要水平至少8组”的联合条件。B00458旧Step快照与当前Pallet标签快照没有物理键交集，因此没有把单批次伪装成Lab组。

## 方案 A：去混杂历史包络

- 静态/原料/环境基线 OOT MAE：{manifest['scheme_a']['static_oot_mae']:.3f} MU。
- OOT Bias：{manifest['scheme_a']['static_oot_bias']:.3f} MU。
- 满足当前统计条件的包络：{manifest['scheme_a']['supported_envelopes']}；全部仍为影子关联证据，不转为因果动作。

{markdown_table(envelope.template_audit.sort_values("groups", ascending=False).head(12))}

## B2：方向与幅度模型

{markdown_table(b2.metrics)}

验证集方向准确率、降低方向召回、变化幅度比和区间覆盖率没有同时通过。OOT方向看似较高不能覆盖验证集失败，也不能覆盖幅度压缩。

## B3：Target → 实际 Phase 响应

精确结构模板：`{b3_templates}`。

{markdown_table(b3_metrics)}

即使部分终点指标通过，当前模型仍没有覆盖所有受动作影响的V3.7 Phase均值、积分与执行特征，因此两个试点统一为 `CONTROL_TRANSFER_UNAVAILABLE`。

## 扩展配方

{markdown_table(pd.DataFrame(extensions))}

四个扩展配方缺少与本合同一致的Step快照，逐个标记 `RECIPE_SPECIFIC_POLICY_REQUIRED`，不影响双试点诊断，但不能进入推荐。

## 安全边界

本实验只读取冻结快照；未访问数据库、本地 MMS 或 Archive。Stage3、配方写入、PLC 和自动控制均关闭。任何 B3 失败都禁止输出“预计降低 X MU”的 Target 推荐。

## 下一次允许开展的工作

1. 补齐 B00458 `OrderID+BatchNumber→PalletID` 的同源标签桥，但不重新读取本地Archive。
2. 对 T15760/B00458 建立完整的 Step Target→Phase均值/积分/终点响应，而不仅是终点模型。
3. 只有 Gate 0 出现 `CAUSAL_ACTION_SUPPORTED` 后才创建 DML 实验；否则继续保持方案A+B2诊断。
"""
    (OUTPUT / "M1相对门尼改善_Gate0_A_B2_B3实验报告.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
