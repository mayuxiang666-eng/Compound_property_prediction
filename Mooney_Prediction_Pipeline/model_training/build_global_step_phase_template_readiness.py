"""Build an auditable template-level Step→Phase readiness inventory.

No template is auto-approved from timing alone.  The inferred phase is a
candidate mapping derived from frozen BatchData cumulative Step time and the
existing PHASE-CORRECTED-V2 boundaries.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
PHASE_SOURCE = BASE / "phase_batch_features_active_v1.csv"
BATCHDATA_SOURCE = BASE / "batchdata_targets_values_dedup.csv"
LOGIC_SOURCE = BASE / "recipe_scp_log_step_contract.csv"
OUTPUT = BASE / "global_step_phase_template_readiness_v1"
TIME_VARIABLE = "SCP-1-Step-Time-rel-s"


def normalize(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def infer_mst_phase(end_time: float, row: pd.Series) -> str:
    if not np.isfinite(end_time):
        return "UNMAPPED"
    boundaries = [
        ("Stage1_Loading", row.get("idx_t1_Loading")),
        ("Stage2_DryMixing", row.get("idx_t2_DryMixing")),
        ("Stage3_OilLoading", row.get("idx_t3_OilLoading")),
        ("Stage4_WetMixing", row.get("idx_t4_PID_Start")),
        ("Stage5_PID_Controlled", row.get("idx_t5_Discharge")),
    ]
    for phase, boundary in boundaries:
        boundary = pd.to_numeric(pd.Series([boundary]), errors="coerce").iloc[0]
        if pd.notna(boundary) and end_time <= float(boundary) + 1.0:
            return phase
    return "PostPID_Tail"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    meta_columns = [
        "OrderID", "BatchNumber", "LabelGroupID", "RecipeID", "CompoundName", "MixerLine",
        "ProcessVersion", "material_system", "phase_route", "StepPatternHash", "TargetVectorHash",
        "SCP_LOG_contract_hash", "TemplateKey", "BatchStartTime", "idx_t1_Loading",
        "idx_t2_DryMixing", "idx_t3_OilLoading", "idx_t4_PID_Start", "idx_t5_Discharge",
    ]
    batch = pd.read_csv(PHASE_SOURCE, usecols=meta_columns, low_memory=False)
    batch["OrderID"], batch["BatchNumber"] = normalize(batch["OrderID"]), normalize(batch["BatchNumber"])
    batch = batch.drop_duplicates(["OrderID", "BatchNumber"])
    keys = batch[["OrderID", "BatchNumber", "LabelGroupID", "TemplateKey"]]

    data = pd.read_csv(BATCHDATA_SOURCE, low_memory=False)
    data["OrderID"], data["BatchNumber"] = normalize(data["OrderID"]), normalize(data["BatchNumber"])
    data = data.merge(keys, on=["OrderID", "BatchNumber"], how="inner", validate="many_to_one")
    data["StepNo"] = pd.to_numeric(data["StepNo"], errors="coerce").astype("Int64")
    data["Target"] = pd.to_numeric(data["Target"], errors="coerce")
    data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
    steps = data[["OrderID", "BatchNumber", "LabelGroupID", "TemplateKey", "GroupName", "StepNo"]].drop_duplicates()
    target = data.loc[data["Target"].notna() & data["Target"].ne(0)].groupby(
        ["OrderID", "BatchNumber", "GroupName", "StepNo", "VariablePath"], as_index=False
    )["Target"].last()
    target_flags = target.assign(has_target=1).pivot_table(
        index=["OrderID", "BatchNumber", "GroupName", "StepNo"], columns="VariablePath",
        values="has_target", aggfunc="max", fill_value=0,
    ).reset_index()
    target_flags.columns.name = None
    for source, output in (
        ("SCP-1-Step-Time-rel-s", "duration_target"),
        ("SCP-3-Temperature-C-F", "temperature_target"),
        ("Rotor-Speed-1/min-Rpm", "rotor_target"),
        ("RAM-Pressure-percent", "ram_target"),
    ):
        target_flags[output] = pd.to_numeric(target_flags[source], errors="coerce").fillna(0).astype(int) if source in target_flags else 0
    steps = steps.merge(target_flags[["OrderID", "BatchNumber", "GroupName", "StepNo", "duration_target", "temperature_target", "rotor_target", "ram_target"]], on=["OrderID", "BatchNumber", "GroupName", "StepNo"], how="left")
    steps[["duration_target", "temperature_target", "rotor_target", "ram_target"]] = steps[["duration_target", "temperature_target", "rotor_target", "ram_target"]].fillna(0).astype(int)

    times = data.loc[data["VariablePath"].eq(TIME_VARIABLE), ["OrderID", "BatchNumber", "GroupName", "StepNo", "Value"]].drop_duplicates(["OrderID", "BatchNumber", "GroupName", "StepNo"], keep="last").rename(columns={"Value": "step_end_time"})
    steps = steps.merge(times, on=["OrderID", "BatchNumber", "GroupName", "StepNo"], how="left", validate="one_to_one")
    boundaries = batch[["OrderID", "BatchNumber", "idx_t1_Loading", "idx_t2_DryMixing", "idx_t3_OilLoading", "idx_t4_PID_Start", "idx_t5_Discharge"]]
    steps = steps.merge(boundaries, on=["OrderID", "BatchNumber"], how="left", validate="many_to_one")
    steps["candidate_phase"] = [
        f"Stage6_MSB_PID_Step{int(step)}" if str(group).upper().endswith("MSB") else infer_mst_phase(float(end) if pd.notna(end) else np.nan, row)
        for group, step, end, (_, row) in zip(steps["GroupName"], steps["StepNo"], steps["step_end_time"], steps.iterrows())
    ]

    modal_rows = []
    for (template, group, step), part in steps.groupby(["TemplateKey", "GroupName", "StepNo"], dropna=False):
        counts = part["candidate_phase"].value_counts(dropna=False)
        modal = str(counts.index[0])
        confidence = float(counts.iloc[0] / len(part))
        modal_rows.append({
            "TemplateKey": template, "GroupName": group, "StepNo": int(step),
            "candidate_phase": modal, "mapping_consistency": confidence,
            "batches_observed": int(part[["OrderID", "BatchNumber"]].drop_duplicates().shape[0]),
            "duration_target_rate": float(part["duration_target"].mean()),
            "temperature_target_rate": float(part["temperature_target"].mean()),
            "rotor_target_rate": float(part["rotor_target"].mean()),
            "ram_target_rate": float(part["ram_target"].mean()),
        })
    mapping = pd.DataFrame(modal_rows)
    mapping.to_csv(OUTPUT / "template_step_candidate_mapping.csv", index=False)

    logic = pd.read_csv(LOGIC_SOURCE, low_memory=False)
    logic_summary = logic.groupby("RecipeID", as_index=False).agg(
        logic_rows=("StepNumber", "size"),
        logic_ok_rate=("logic_status", lambda s: float(s.astype(str).eq("CONTROL_LOGIC_OK").mean())),
        control_relations=("control_relation", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
    )
    template = batch.groupby("TemplateKey", as_index=False).agg(
        RecipeID=("RecipeID", "first"), CompoundName=("CompoundName", "first"),
        MixerLine=("MixerLine", "first"), ProcessVersion=("ProcessVersion", "first"),
        material_system=("material_system", "first"), phase_route=("phase_route", "first"),
        StepPatternHash=("StepPatternHash", "first"), TargetVectorHash=("TargetVectorHash", "first"),
        SCP_LOG_contract_hash=("SCP_LOG_contract_hash", "first"), physical_batches=("BatchNumber", "size"),
        label_groups=("LabelGroupID", "nunique"), first_production=("BatchStartTime", "min"), last_production=("BatchStartTime", "max"),
    ).merge(logic_summary, on="RecipeID", how="left")
    map_summary = mapping.groupby("TemplateKey", as_index=False).agg(
        mapped_steps=("StepNo", "size"), min_mapping_consistency=("mapping_consistency", "min"),
        low_consistency_steps=("mapping_consistency", lambda s: int((s < 0.90).sum())),
        adjustable_steps=("duration_target_rate", lambda s: 0),
    )
    adjustable = mapping.assign(adjustable=(mapping[["duration_target_rate", "temperature_target_rate", "rotor_target_rate"]].max(axis=1) > 0).astype(int)).groupby("TemplateKey", as_index=False)["adjustable"].sum().rename(columns={"adjustable": "adjustable_steps"})
    map_summary = map_summary.drop(columns=["adjustable_steps"]).merge(adjustable, on="TemplateKey", how="left")
    template = template.merge(map_summary, on="TemplateKey", how="left")
    template["expert_scope"] = np.select(
        [template["CompoundName"].astype(str).str.contains("T15760", regex=False), template["CompoundName"].astype(str).str.contains("B00458", regex=False)],
        ["T15760_EXPERT_MAPPING_AVAILABLE", "B00458_EXPERT_RULES_MAPPING_REVIEW"], default="GENERAL_TEMPLATE_REVIEW_REQUIRED",
    )
    template["readiness_status"] = np.select(
        [template["min_mapping_consistency"].fillna(0).lt(0.90), template["logic_ok_rate"].fillna(1).lt(1.0), template["label_groups"].lt(30)],
        ["BLOCKED_LOW_MAPPING_CONSISTENCY", "BLOCKED_SCP_LOG_CONTRACT", "BLOCKED_INSUFFICIENT_LABEL_GROUPS"],
        default="CANDIDATE_REQUIRES_EXPERT_CONFIRMATION",
    )
    template.to_csv(OUTPUT / "template_readiness.csv", index=False)
    pilot = template.loc[template["CompoundName"].astype(str).str.contains("T15760|B00458", regex=True)].copy()
    pilot.to_csv(OUTPUT / "pilot_template_readiness.csv", index=False)
    summary = {
        "contract": "GLOBAL-STEP-PHASE-TEMPLATE-READINESS-V1",
        "templates": int(len(template)), "candidate_mapping_rows": int(len(mapping)),
        "physical_batches": int(len(batch)), "label_groups": int(batch["LabelGroupID"].nunique()),
        "readiness_counts": template["readiness_status"].value_counts().to_dict(),
        "pilot_templates": int(len(pilot)),
        "auto_approved_templates": 0,
        "mapping_rule": "Timing-derived candidate only; expert confirmation required",
        "target_rule": "Non-zero BatchData Target marks existing adjustable controls; ram remains recommendation-locked",
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED", "automatic_control": "DISABLED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
