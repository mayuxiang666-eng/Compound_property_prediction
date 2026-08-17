"""Evaluate Step/parameter readiness inside corrected structural templates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history" / "crossdb_exact_pilot_step_history_v1"
SOURCE = BASE / "structural_template_contract_v2" / "mapped_step_vectors_structural_v2.csv"
OUTPUT = ROOT / "reports" / "corrected_phase_history" / "structural_template_readiness_v2"

PARAMETERS = {
    "duration_s": ("dur_SP", "dur_AV", "is_time_settable"),
    "temperature_c": ("temp_SP", "temp_AV", "is_temp_settable"),
    "rotor_rpm": ("rotor_SP", "rotor_AV", "is_rotor_settable"),
    "ram_percent": ("ram_SP", "ram_AV", "is_ram_settable"),
}


def _canonical_phases(mapped: pd.DataFrame) -> pd.DataFrame:
    counts = mapped.groupby(
        ["StructuralTemplateID", "GroupName", "StepNo", "mapped_phase"],
        as_index=False,
    ).size().rename(columns={"size": "PhaseBatchRows"})
    counts = counts.sort_values(
        ["StructuralTemplateID", "GroupName", "StepNo", "PhaseBatchRows", "mapped_phase"],
        ascending=[True, True, True, False, True], kind="stable",
    )
    dominant = counts.drop_duplicates(
        ["StructuralTemplateID", "GroupName", "StepNo"], keep="first"
    ).rename(columns={"mapped_phase": "CanonicalPhase"})
    totals = counts.groupby(["StructuralTemplateID", "GroupName", "StepNo"])["PhaseBatchRows"].transform("sum")
    counts["PhaseShare"] = counts["PhaseBatchRows"] / totals
    return dominant, counts


def _policy_status(material: str, group: str, step_no: int, phase: str, parameter: str) -> str:
    if group == "AVR_MST" and step_no in (1, 2):
        return "LOCKED_LOADING_STEP_1_2"
    if parameter == "ram_percent":
        return "LOCKED_RAM"
    if phase == "Stage2_DryMixing":
        return "POLICY_ALLOWED" if parameter in {"duration_s", "temperature_c", "rotor_rpm"} else "LOCKED_POLICY"
    if phase == "Stage4_WetMixing_Ramp":
        return "POLICY_ALLOWED" if parameter in {"duration_s", "temperature_c", "rotor_rpm"} else "LOCKED_POLICY"
    if material == "CarbonBlack" and (
        phase == "Stage5_MST_PID_Silanization" or phase.startswith("Stage6_MSB_PID")
    ):
        return "POLICY_ALLOWED" if parameter in {"duration_s", "temperature_c"} else "LOCKED_POLICY"
    if material == "Silica" and phase.startswith("Stage6_MSB_PID"):
        return "POLICY_ALLOWED" if parameter in {"duration_s", "temperature_c"} else "LOCKED_POLICY"
    return "LOCKED_FIRST_ROUND_SCOPE"


def _tracking_gate(parameter: str, errors: pd.Series) -> str:
    values = pd.to_numeric(errors, errors="coerce").abs().dropna()
    if values.empty:
        return "BLOCKED_NO_TRACKING_DATA"
    mae = float(values.mean())
    p95 = float(np.percentile(values, 95))
    excursion = float((values > 5.0).mean())
    if parameter == "temperature_c":
        return "PASS" if mae <= 2.0 and p95 <= 5.0 and excursion <= 0.05 else "FAIL"
    if parameter == "duration_s":
        return "PASS" if mae <= 5.0 and p95 <= 10.0 else "FAIL"
    if parameter == "rotor_rpm":
        return "PASS" if mae <= 2.0 and p95 <= 5.0 and excursion <= 0.05 else "FAIL"
    return "DIAGNOSTIC_ONLY"


def build_readiness(mapped: pd.DataFrame, dominant: pd.DataFrame) -> pd.DataFrame:
    phase_lookup = dominant.set_index(
        ["StructuralTemplateID", "GroupName", "StepNo"]
    )["CanonicalPhase"].to_dict()
    rows: list[dict[str, object]] = []
    group_keys = [
        "CompoundDescription", "CompoundName", "material_system", "MixerLine",
        "StructuralTemplateID", "GroupName", "StepNo",
    ]
    for keys, frame in mapped.groupby(group_keys, sort=False, dropna=False):
        recipe, compound, material, mixer, template, group, step_no = keys
        phase = phase_lookup[(template, group, step_no)]
        template_batches = int(
            mapped.loc[mapped["StructuralTemplateID"].eq(template), ["OrderID", "BatchNumber"]]
            .drop_duplicates().shape[0]
        )
        for parameter, (sp_column, av_column, flag_column) in PARAMETERS.items():
            eligible = frame.loc[frame[flag_column].fillna(False).astype(bool)].copy()
            sp = pd.to_numeric(eligible[sp_column], errors="coerce").dropna()
            av = pd.to_numeric(eligible[av_column], errors="coerce")
            errors = av - pd.to_numeric(eligible[sp_column], errors="coerce")
            levels = sp.round(3).value_counts().sort_index()
            policy = _policy_status(str(material), str(group), int(step_no), str(phase), parameter)
            tracking = _tracking_gate(parameter, errors)
            min_level_n = int(levels.min()) if len(levels) else 0
            support = (
                "PASS" if len(levels) >= 3 and min_level_n >= 10
                else "BLOCKED_FEWER_THAN_3_LEVELS" if len(levels) < 3
                else "BLOCKED_SPARSE_LEVEL_SUPPORT"
            )
            readiness = (
                "READY_FOR_RESPONSE_MODEL"
                if policy == "POLICY_ALLOWED" and len(eligible) > 0 and tracking == "PASS" and support == "PASS"
                else "NOT_SETTABLE" if eligible.empty
                else policy if policy != "POLICY_ALLOWED"
                else "BLOCKED_TRACKING_FAILURE" if tracking == "FAIL"
                else support
            )
            rows.append({
                "RecipeID": recipe, "CompoundName": compound, "material_system": material,
                "MixerLine": mixer, "StructuralTemplateID": template,
                "TemplateBatchCount": template_batches, "GroupName": group,
                "StepNo": int(step_no), "CanonicalPhase": phase, "parameter": parameter,
                "settable_batches": int(eligible[["OrderID", "BatchNumber"]].drop_duplicates().shape[0]),
                "target_level_count": int(len(levels)),
                "target_level_counts_json": json.dumps(
                    {str(k): int(v) for k, v in levels.items()}, ensure_ascii=False
                ),
                "target_min": float(sp.min()) if len(sp) else None,
                "target_median": float(sp.median()) if len(sp) else None,
                "target_max": float(sp.max()) if len(sp) else None,
                "min_level_batch_count": min_level_n,
                "tracking_n": int(errors.notna().sum()),
                "tracking_mae": float(errors.abs().mean()) if errors.notna().any() else None,
                "tracking_p95_abs": float(np.percentile(errors.abs().dropna(), 95)) if errors.notna().any() else None,
                "tracking_excursion_gt5_rate": float((errors.abs().dropna() > 5).mean()) if errors.notna().any() else None,
                "policy_status": policy, "tracking_status": tracking,
                "historical_support_status": support, "readiness_status": readiness,
            })
    return pd.DataFrame(rows)


def build_discharge(mapped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    keys = ["CompoundDescription", "StructuralTemplateID", "OrderID", "BatchNumber"]
    for key, batch in mapped.groupby(keys, sort=False):
        recipe, template, order_id, batch_number = key
        msb = batch.loc[batch["GroupName"].eq("AVR_MSB")].sort_values("StepNo", kind="stable")
        controls = msb.loc[pd.to_numeric(msb["temp_SP"], errors="coerce").fillna(0).gt(0)]
        if msb.empty or controls.empty:
            continue
        control = controls.iloc[-1]
        final = msb.iloc[-1]
        actual = pd.to_numeric(pd.Series([final["temp_AV"]]), errors="coerce").iloc[0]
        target = float(control["temp_SP"])
        rows.append({
            "RecipeID": recipe, "StructuralTemplateID": template,
            "OrderID": order_id, "BatchNumber": int(batch_number),
            "control_step_no": int(control["StepNo"]), "final_step_no": int(final["StepNo"]),
            "discharge_target": target,
            "discharge_actual": float(actual) if pd.notna(actual) else None,
            "tracking_error": float(actual - target) if pd.notna(actual) else None,
            "trailing_no_target_step": bool(int(final["StepNo"]) > int(control["StepNo"])),
        })
    detail = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for (recipe, template), frame in detail.groupby(["RecipeID", "StructuralTemplateID"]):
        errors = pd.to_numeric(frame["tracking_error"], errors="coerce").abs().dropna()
        mae = float(errors.mean()) if len(errors) else None
        p95 = float(np.percentile(errors, 95)) if len(errors) else None
        excursion = float((errors > 5).mean()) if len(errors) else None
        summaries.append({
            "RecipeID": recipe, "StructuralTemplateID": template,
            "BatchCount": int(len(frame)), "MAE": mae, "P95AbsError": p95,
            "ExcursionGt5Rate": excursion,
            "Passed": bool(len(errors) and mae <= 2 and p95 <= 5 and excursion <= 0.05),
            "TrailingNoTargetRate": float(frame["trailing_no_target_step"].mean()),
        })
    return detail, pd.DataFrame(summaries)


def run(source: Path, output_dir: Path) -> dict[str, object]:
    mapped = pd.read_csv(source, low_memory=False)
    dominant, phase_variants = _canonical_phases(mapped)
    readiness = build_readiness(mapped, dominant)
    discharge, discharge_summary = build_discharge(mapped)
    output_dir.mkdir(parents=True, exist_ok=True)
    dominant.to_csv(output_dir / "canonical_phase_by_template_step.csv", index=False, encoding="utf-8-sig")
    phase_variants.to_csv(output_dir / "phase_assignment_variants.csv", index=False, encoding="utf-8-sig")
    readiness.to_csv(output_dir / "step_parameter_readiness.csv", index=False, encoding="utf-8-sig")
    discharge.to_csv(output_dir / "discharge_tracking_detail.csv", index=False, encoding="utf-8-sig")
    discharge_summary.to_csv(output_dir / "discharge_tracking_summary.csv", index=False, encoding="utf-8-sig")
    summary = {
        "contract": "STRUCTURAL-TEMPLATE-READINESS-V2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "structural_templates": int(mapped["StructuralTemplateID"].nunique()),
        "templates_ge150": int(
            mapped.groupby("StructuralTemplateID")[["OrderID", "BatchNumber"]]
            .apply(lambda x: x.drop_duplicates().shape[0]).ge(150).sum()
        ),
        "ready_step_parameters": int(readiness["readiness_status"].eq("READY_FOR_RESPONSE_MODEL").sum()),
        "ready_by_recipe": {
            str(k): int(v) for k, v in readiness.loc[
                readiness["readiness_status"].eq("READY_FOR_RESPONSE_MODEL")
            ].groupby("RecipeID").size().to_dict().items()
        },
        "discharge_templates_passed": int(discharge_summary.get("Passed", pd.Series(dtype=bool)).sum()),
        "discharge_templates_total": int(len(discharge_summary)),
        "database_writes": False, "automatic_control": "DISABLED",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
