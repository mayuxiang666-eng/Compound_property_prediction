"""Fail-closed discharge tracking using recipe-defined MSB control Step."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUT = BASE / "v371_dual_process_response_v2"
TARGETS = ["M1-T15760---- 06 002", "M3B-B00460---- 01 002"]
TEMP_TOKEN = re.compile(r"AVR_MSB:(\d+):SCP-3-Temperature-C-F:([-+]?\d+(?:\.\d+)?)")


def expected_control(recipe_text: str) -> tuple[int | None, float | None]:
    found = []
    for step, target in TEMP_TOKEN.findall(str(recipe_text)):
        value = float(target)
        if value > 0:
            found.append((int(step), value))
    return found[-1] if found else (None, None)


def main():
    static = pd.read_csv(BASE / "batch_static_template_skeleton.csv", usecols=["LabelGroupID", "RecipeID", "CompoundName", "TargetVectorRaw"], low_memory=False)
    static = static[static["CompoundName"].isin(TARGETS)].drop_duplicates("LabelGroupID")
    expected = static["TargetVectorRaw"].map(expected_control)
    static["control_step_expected"] = expected.map(lambda x: x[0])
    static["control_target_expected"] = expected.map(lambda x: x[1])

    step = pd.read_csv(BASE / "dual_target_readiness" / "dual_pilot_group_step_features.csv", low_memory=False)
    step = step[step["CompoundName"].isin(TARGETS)].copy()
    msb = step[(step["GroupName"] == "AVR_MSB") & (step["kind"] == "temperature")].copy()
    msb["StepNo"] = pd.to_numeric(msb["StepNo"], errors="coerce")
    msb["actual_value_mean"] = pd.to_numeric(msb["actual_value_mean"], errors="coerce")
    observed = msb[["LabelGroupID", "StepNo", "actual_value_mean"]].rename(columns={"StepNo": "control_step_observed", "actual_value_mean": "control_step_actual_observed"})
    observed = observed.merge(static[["LabelGroupID", "control_step_expected"]], on="LabelGroupID", how="left")
    observed = observed[observed["control_step_observed"].eq(observed["control_step_expected"])].copy()
    # A group can contain only one observation for its expected control Step.
    observed = observed.drop_duplicates("LabelGroupID") if not observed.empty else observed.drop(columns=["control_step_expected"], errors="ignore")
    observed = observed.drop(columns=["control_step_expected"], errors="ignore")
    phase = pd.read_csv(BASE / "phase_batch_features_active_v1.csv", usecols=["LabelGroupID", "CompoundName", "phys_discharge_temp"], low_memory=False)
    phase = phase[phase["CompoundName"].isin(TARGETS)].copy()
    phase["phys_discharge_temp"] = pd.to_numeric(phase["phys_discharge_temp"], errors="coerce")
    phase = phase.groupby(["LabelGroupID", "CompoundName"], as_index=False).agg(
        discharge_actual_mean=("phys_discharge_temp", "mean"), discharge_batch_count=("phys_discharge_temp", "count"),
    )
    audit = static.merge(phase, on=["LabelGroupID", "CompoundName"], how="left", validate="one_to_one")
    audit = audit.merge(observed, on="LabelGroupID", how="left", validate="one_to_one")
    audit["control_path_status"] = audit.apply(
        lambda r: "BLOCKED_MISSING_EXPECTED_CONTROL_STEP" if pd.isna(r["control_step_expected"]) else (
            "BLOCKED_EXPECTED_CONTROL_STEP_OBSERVATION_MISSING" if pd.isna(r["control_step_observed"]) else (
                "BLOCKED_CONTROL_STEP_MISMATCH" if int(r["control_step_expected"]) != int(r["control_step_observed"]) else "PASS_CONTROL_STEP_PATH"
            )
        ), axis=1,
    )
    audit["tracking_error"] = audit["discharge_actual_mean"] - audit["control_target_expected"]
    audit["absolute_tracking_error"] = audit["tracking_error"].abs()
    valid = audit[audit["control_path_status"].eq("PASS_CONTROL_STEP_PATH")].copy()
    audit.to_csv(OUT / "dual_endpoint_discharge_tracking_recipe_contract_v3.csv", index=False, encoding="utf-8-sig")

    summary = {"contract": "V3.7.1-DUAL-DISCHARGE-TRACKING-RECIPE-CONTRACT-V3", "source_policy": "RDS Active snapshot only; local MMS/Archive not accessed", "recommendation_ready": False, "pilots": {}}
    for compound, g in audit.groupby("CompoundName"):
        v = valid[valid["CompoundName"].eq(compound)]
        err = v["absolute_tracking_error"].dropna()
        summary["pilots"][compound] = {
            "total_groups": int(len(g)), "valid_control_path_groups": int(len(v)),
            "path_status_counts": g["control_path_status"].value_counts().to_dict(),
            "expected_control_steps": sorted(map(int, g["control_step_expected"].dropna().unique())),
            "mae_valid_path": float(err.mean()) if len(err) else None,
            "p95_valid_path": float(err.quantile(.95)) if len(err) else None,
            "excursion_gt5_valid_path": float((err > 5).mean()) if len(err) else None,
            "formal_tracking_pass": bool(len(err) and err.mean() <= 2 and err.quantile(.95) <= 5 and (err > 5).mean() <= .05 and len(v) / len(g) >= .95),
        }
    (OUT / "dual_endpoint_discharge_tracking_recipe_contract_summary_v3.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
