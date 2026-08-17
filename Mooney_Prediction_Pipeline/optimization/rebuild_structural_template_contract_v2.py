"""Separate physical Step structure from data-dependent Phase assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history" / "crossdb_exact_pilot_step_history_v1"
SOURCE = BASE / "mapped_step_vectors.csv"
OUTPUT = BASE / "structural_template_contract_v2"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12].upper()


def _batch_contract(batch: pd.DataFrame) -> pd.Series:
    recipe, _compound, material, mixer, _order, _batch = batch.name
    structure: list[str] = []
    phases: list[str] = []
    route: list[str] = []
    for row in batch.sort_values(["GroupName", "StepNo"], kind="stable").itertuples():
        controls = "".join([
            "D" if bool(row.is_time_settable) else "-",
            "T" if bool(row.is_temp_settable) else "-",
            "R" if bool(row.is_rotor_settable) else "-",
            "P" if bool(row.is_ram_settable) else "-",
        ])
        structure.append(f"{row.GroupName}:S{int(row.StepNo)}:{controls}")
        phases.append(f"{row.GroupName}:S{int(row.StepNo)}:{row.mapped_phase}")
        route.append(f"{row.GroupName}:{row.EquipmentID}")
    structural_signature = "|".join(structure)
    phase_signature = "|".join(phases)
    route_signature = "|".join(dict.fromkeys(route))
    structural_id = _digest(
        f"{recipe}|{material}|{mixer}|{route_signature}|{structural_signature}"
    )
    phase_id = _digest(f"{structural_id}|{phase_signature}")
    return pd.Series({
        "StructuralTemplateID": structural_id,
        "StructuralSignature": structural_signature,
        "RouteSignature": route_signature,
        "PhaseMappingVariantID": phase_id,
        "PhaseMappingSignature": phase_signature,
    })


def run(source: Path, output_dir: Path) -> dict[str, object]:
    mapped = pd.read_csv(source, low_memory=False)
    keys = [
        "CompoundDescription", "CompoundName", "material_system", "MixerLine",
        "OrderID", "BatchNumber",
    ]
    batch_contracts = mapped.groupby(keys, sort=False).apply(
        _batch_contract, include_groups=False
    ).reset_index()
    if "TemplateID" in mapped:
        mapped = mapped.rename(columns={"TemplateID": "LegacyPhaseCoupledTemplateID"})
    mapped = mapped.merge(
        batch_contracts[[
            "OrderID", "BatchNumber", "StructuralTemplateID", "PhaseMappingVariantID",
        ]], on=["OrderID", "BatchNumber"], how="left", validate="many_to_one",
    )
    mapped["TemplateID"] = mapped["StructuralTemplateID"]

    structural = batch_contracts.groupby([
        "CompoundDescription", "CompoundName", "material_system", "MixerLine",
        "RouteSignature", "StructuralTemplateID", "StructuralSignature",
    ], as_index=False).size().rename(columns={"size": "BatchCount"})
    totals = structural.groupby(["CompoundDescription", "MixerLine"])["BatchCount"].transform("sum")
    structural["TemplateShareWithinRecipeRoute"] = structural["BatchCount"] / totals
    structural["EligibleForResponseModel150"] = structural["BatchCount"].ge(150)

    variants = batch_contracts.groupby([
        "CompoundDescription", "MixerLine", "StructuralTemplateID",
        "PhaseMappingVariantID", "PhaseMappingSignature",
    ], as_index=False).size().rename(columns={"size": "BatchCount"})
    variant_totals = variants.groupby("StructuralTemplateID")["BatchCount"].transform("sum")
    variants["ShareWithinStructuralTemplate"] = variants["BatchCount"] / variant_totals
    variants["IsDominantPhaseMapping"] = variants["BatchCount"].eq(
        variants.groupby("StructuralTemplateID")["BatchCount"].transform("max")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_contracts.to_csv(output_dir / "batch_structural_templates.csv", index=False, encoding="utf-8-sig")
    structural.to_csv(output_dir / "structural_template_summary.csv", index=False, encoding="utf-8-sig")
    variants.to_csv(output_dir / "phase_mapping_variant_summary.csv", index=False, encoding="utf-8-sig")
    mapped.to_csv(output_dir / "mapped_step_vectors_structural_v2.csv", index=False, encoding="utf-8-sig")

    recipes: dict[str, object] = {}
    for recipe, frame in structural.groupby("CompoundDescription"):
        recipe_variants = variants.loc[variants["CompoundDescription"].eq(recipe)]
        recipes[str(recipe)] = {
            "batch_count": int(frame["BatchCount"].sum()),
            "structural_template_count": int(len(frame)),
            "eligible_150_template_count": int(frame["EligibleForResponseModel150"].sum()),
            "eligible_150_batch_coverage": float(
                frame.loc[frame["EligibleForResponseModel150"], "BatchCount"].sum()
                / frame["BatchCount"].sum()
            ),
            "phase_mapping_variant_count": int(len(recipe_variants)),
        }
    summary = {
        "contract": "STEP-STRUCTURAL-TEMPLATE-V2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "structural_template_excludes": ["mapped_phase", "Target_values", "AV_values"],
        "phase_mapping_retained_separately": True,
        "recipes": recipes,
        "database_writes": False,
        "automatic_control": "DISABLED",
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
