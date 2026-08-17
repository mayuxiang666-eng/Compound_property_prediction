"""Template-specific and SCP-contract-coverage audit for Active pilot Steps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history" / "active_exact_pilot_step_history_v1"
SOURCE = ROOT / "reports" / "corrected_phase_history" / "phase_corrected_v371_active.csv"


def signature(batch: pd.DataFrame) -> str:
    rows = []
    for row in batch.sort_values(["GroupName", "StepNo"], kind="stable").itertuples():
        controls = "".join([
            "D" if bool(row.is_time_settable) else "-",
            "T" if bool(row.is_temp_settable) else "-",
            "R" if bool(row.is_rotor_settable) else "-",
            "P" if bool(row.is_ram_settable) else "-",
        ])
        rows.append(f"{row.GroupName}:S{int(row.StepNo)}:{row.mapped_phase}:{controls}")
    return "|".join(rows)


def main() -> None:
    mapped = pd.read_csv(BASE / "active_exact_pilot_step_vectors.csv", low_memory=False)
    mapped["OrderID"] = mapped["OrderID"].astype(str).str.replace(r"\.0$", "", regex=True)
    mapped["BatchNumber"] = pd.to_numeric(mapped["BatchNumber"], errors="raise").astype(int)
    mapped["OrderStartTime"] = pd.to_datetime(mapped["OrderStartTime"], errors="raise")
    contracts = pd.read_csv(BASE / "active_exact_pilot_scp_logic.csv", low_memory=False)

    assignment_rows = []
    for keys, batch in mapped.groupby(
        ["CompoundDescription", "CompoundName", "OrderID", "BatchNumber"], sort=False
    ):
        value = signature(batch)
        assignment_rows.append({
            "RecipeID": keys[0], "CompoundName": keys[1], "OrderID": keys[2],
            "BatchNumber": keys[3], "OrderStartTime": batch["OrderStartTime"].max(),
            "TemplateID": hashlib.sha256(value.encode("utf-8")).hexdigest()[:12].upper(),
            "TemplateSignature": value,
        })
    assignments = pd.DataFrame(assignment_rows)
    template_summary = assignments.groupby(
        ["RecipeID", "CompoundName", "TemplateID", "TemplateSignature"], as_index=False
    ).agg(
        BatchCount=("BatchNumber", "size"),
        OrderCount=("OrderID", "nunique"),
        FirstBatchTime=("OrderStartTime", "min"),
        LastBatchTime=("OrderStartTime", "max"),
    )
    template_summary["TemplateShare"] = template_summary["BatchCount"] / template_summary.groupby(
        "RecipeID"
    )["BatchCount"].transform("sum")
    template_summary["ProcessResponseMin150"] = template_summary["BatchCount"].ge(150)

    controls = mapped.loc[
        mapped[["is_time_settable", "is_temp_settable"]].fillna(False).any(axis=1),
        ["CompoundDescription", "GroupName", "StepNo", "mapped_phase",
         "is_time_settable", "is_temp_settable", "is_rotor_settable"],
    ].drop_duplicates()
    controls = controls.rename(columns={"CompoundDescription": "recipe_id"})
    contract_keys = contracts[[
        "recipe_id", "GroupName", "StepNo", "scp_relation", "scp_active_conditions", "scp_logic_status"
    ]].drop_duplicates(["recipe_id", "GroupName", "StepNo"])
    coverage = controls.merge(
        contract_keys, on=["recipe_id", "GroupName", "StepNo"], how="left", validate="many_to_one"
    )
    coverage["contract_coverage_status"] = coverage["scp_logic_status"].fillna("MISSING_CONTRACT")

    scope = pd.read_csv(
        SOURCE, usecols=["OrderID", "BatchNumber", "CompoundDescription", "MNY"], low_memory=False
    )
    scope["OrderID"] = scope["OrderID"].astype(str).str.replace(r"\.0$", "", regex=True)
    scope["BatchNumber"] = pd.to_numeric(scope["BatchNumber"], errors="raise").astype(int)
    assignments = assignments.merge(
        scope, left_on=["OrderID", "BatchNumber", "RecipeID"],
        right_on=["OrderID", "BatchNumber", "CompoundDescription"],
        how="left", validate="one_to_one",
    ).drop(columns=["CompoundDescription"])

    BASE.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(BASE / "active_exact_pilot_template_assignments_v2.csv", index=False, encoding="utf-8-sig")
    template_summary.to_csv(BASE / "active_exact_pilot_template_summary_v2.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(BASE / "active_exact_pilot_scp_contract_coverage_v2.csv", index=False, encoding="utf-8-sig")

    recipes = {}
    for recipe, frame in template_summary.groupby("RecipeID"):
        contract = coverage.loc[coverage["recipe_id"].astype(str).eq(str(recipe))]
        recipes[str(recipe)] = {
            "template_count": int(len(frame)),
            "templates": [
                {
                    "template_id": row.TemplateID,
                    "batches": int(row.BatchCount),
                    "orders": int(row.OrderCount),
                    "share": float(row.TemplateShare),
                    "minimum_150_met": bool(row.ProcessResponseMin150),
                    "first": pd.Timestamp(row.FirstBatchTime).isoformat(),
                    "last": pd.Timestamp(row.LastBatchTime).isoformat(),
                }
                for row in frame.itertuples()
            ],
            "controllable_steps": int(len(contract)),
            "known_contract_steps": int(contract["contract_coverage_status"].eq("KNOWN").sum()),
            "missing_contract_steps": int(contract["contract_coverage_status"].eq("MISSING_CONTRACT").sum()),
            "contract_coverage_pass": bool(contract["contract_coverage_status"].eq("KNOWN").all()),
        }
    summary = {
        "contract": "ACTIVE-EXACT-PILOT-TEMPLATE-SCP-COVERAGE-V2",
        "recipes": recipes,
        "gate_rule": "Each process-response model is template-specific; every controllable Step requires a KNOWN SCP contract.",
        "database_writes": False,
        "automatic_control": "DISABLED",
        "recommendation_status": "NOT_GENERATED_TEMPLATE_AUDIT_ONLY",
    }
    (BASE / "active_exact_pilot_template_contract_summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
