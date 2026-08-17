"""Template-level action support and discharge tracking for exact Active pilots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history" / "active_exact_pilot_step_history_v1"


def allowed(material: str, phase: str, parameter: str, step_no: int) -> bool:
    if step_no in {1, 2} or parameter == "ram":
        return False
    if phase == "Stage2_DryMixing":
        return parameter in {"duration", "rotor", "temperature"}
    if phase == "Stage4_WetMixing_Ramp":
        return parameter in {"duration", "rotor", "temperature"}
    if material == "CarbonBlack" and (
        phase == "Stage5_MST_PID_Silanization" or phase.startswith("Stage6_MSB_PID")
    ):
        return parameter in {"duration", "temperature"}
    if material == "Silica" and phase.startswith("Stage6_MSB_PID"):
        return parameter in {"duration", "temperature"}
    return False


def main() -> None:
    mapped = pd.read_csv(BASE / "active_exact_pilot_step_vectors.csv", low_memory=False)
    assignments = pd.read_csv(BASE / "active_exact_pilot_template_assignments_v2.csv", low_memory=False)
    for frame in (mapped, assignments):
        frame["OrderID"] = frame["OrderID"].astype(str).str.replace(r"\.0$", "", regex=True)
        frame["BatchNumber"] = pd.to_numeric(frame["BatchNumber"], errors="raise").astype(int)
    work = mapped.merge(
        assignments[["OrderID", "BatchNumber", "TemplateID"]],
        on=["OrderID", "BatchNumber"], how="inner", validate="many_to_one"
    )

    rows = []
    for keys, frame in work.groupby(
        ["CompoundDescription", "CompoundName", "material_system", "TemplateID", "GroupName", "StepNo", "mapped_phase"]
    ):
        recipe, compound, material, template, group, step_no, phase = keys
        for parameter, target, settable in (
            ("duration", "dur_SP", "is_time_settable"),
            ("temperature", "temp_SP", "is_temp_settable"),
            ("rotor", "rotor_SP", "is_rotor_settable"),
        ):
            if not allowed(str(material), str(phase), parameter, int(step_no)):
                continue
            values = pd.to_numeric(frame.loc[frame[settable].fillna(False).astype(bool), target], errors="coerce").dropna()
            levels = int(values.nunique())
            status = (
                "PASS_MIN_3_LEVELS" if levels >= 3 else
                "DIAGNOSTIC_TWO_LEVELS" if levels == 2 else
                "BLOCKED_NO_HISTORICAL_VARIATION" if levels == 1 else
                "BLOCKED_NO_EXISTING_TARGET"
            )
            rows.append({
                "RecipeID": recipe, "CompoundName": compound, "material_system": material,
                "TemplateID": template, "GroupName": group, "StepNo": int(step_no),
                "mapped_phase": phase, "parameter": parameter, "sample_count": int(len(values)),
                "target_level_count": levels, "target_min": float(values.min()) if len(values) else None,
                "target_median": float(values.median()) if len(values) else None,
                "target_max": float(values.max()) if len(values) else None,
                "response_identification_status": status,
            })
    support = pd.DataFrame(rows)

    discharge_rows = []
    for keys, batch in work.groupby(
        ["CompoundDescription", "CompoundName", "TemplateID", "OrderID", "BatchNumber"], sort=False
    ):
        recipe, compound, template, order_id, batch_number = keys
        msb = batch.loc[batch["GroupName"].eq("AVR_MSB")].sort_values("StepNo", kind="stable")
        controls = msb.loc[pd.to_numeric(msb["temp_SP"], errors="coerce").fillna(0).gt(0)]
        if msb.empty or controls.empty:
            continue
        control, final = controls.iloc[-1], msb.iloc[-1]
        target = pd.to_numeric(pd.Series([control["temp_SP"]]), errors="coerce").iloc[0]
        actual = pd.to_numeric(pd.Series([final["temp_AV"]]), errors="coerce").iloc[0]
        discharge_rows.append({
            "RecipeID": recipe, "CompoundName": compound, "TemplateID": template,
            "OrderID": order_id, "BatchNumber": int(batch_number),
            "control_step_no": int(control["StepNo"]), "final_step_no": int(final["StepNo"]),
            "target": target, "actual": actual,
            "absolute_error": abs(float(actual - target)) if pd.notna(actual) and pd.notna(target) else None,
        })
    discharge = pd.DataFrame(discharge_rows)
    tracking_rows = []
    for keys, frame in discharge.groupby(["RecipeID", "CompoundName", "TemplateID"]):
        error = pd.to_numeric(frame["absolute_error"], errors="coerce").dropna()
        mae = float(error.mean()) if len(error) else None
        p95 = float(np.percentile(error, 95)) if len(error) else None
        excursion = float((error > 5.0).mean()) if len(error) else None
        tracking_rows.append({
            "RecipeID": keys[0], "CompoundName": keys[1], "TemplateID": keys[2],
            "batches": int(len(frame)), "control_steps": "|".join(map(str, sorted(frame["control_step_no"].unique()))),
            "mae": mae, "p95_abs_error": p95, "excursion_gt5_rate": excursion,
            "passed": bool(len(error) and mae <= 2.0 and p95 <= 5.0 and excursion <= 0.05),
        })
    tracking = pd.DataFrame(tracking_rows)

    support.to_csv(BASE / "active_template_action_support_v1.csv", index=False, encoding="utf-8-sig")
    discharge.to_csv(BASE / "active_template_discharge_rows_v1.csv", index=False, encoding="utf-8-sig")
    tracking.to_csv(BASE / "active_template_discharge_tracking_v1.csv", index=False, encoding="utf-8-sig")

    summary = {"contract": "ACTIVE-TEMPLATE-ACTION-SUPPORT-V1", "templates": {}}
    for keys, frame in support.groupby(["RecipeID", "TemplateID"]):
        track = tracking.loc[(tracking["RecipeID"].eq(keys[0])) & (tracking["TemplateID"].eq(keys[1]))]
        summary["templates"][f"{keys[0]}|{keys[1]}"] = {
            "action_cells": int(len(frame)),
            "pass_min_3_levels": int(frame["response_identification_status"].eq("PASS_MIN_3_LEVELS").sum()),
            "two_level_diagnostic": int(frame["response_identification_status"].eq("DIAGNOSTIC_TWO_LEVELS").sum()),
            "blocked_no_variation_or_target": int(frame["response_identification_status"].str.startswith("BLOCKED").sum()),
            "discharge_tracking_pass": bool(not track.empty and track.iloc[0]["passed"]),
            "formal_response_model_status": "ELIGIBLE_PARTIAL_ACTIONS_ONLY" if frame["response_identification_status"].eq("PASS_MIN_3_LEVELS").any() else "BLOCKED_NO_IDENTIFIABLE_ACTION",
        }
    summary.update({
        "rule": "Only Step-parameter cells with at least three historical Target levels may identify a formal response; two levels remain diagnostic.",
        "database_writes": False, "automatic_control": "DISABLED",
        "recommendation_status": "NOT_GENERATED_SUPPORT_AUDIT_ONLY",
    })
    (BASE / "active_template_action_support_summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
