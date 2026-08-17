"""Build corrected, time-safe Phase history for V3.7.1 experiments.

The builder is read-only against MMS/Mustang. It replaces legacy material
fields with the canonical taxonomy and real ChargeID-linked supplier QM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.data_processing.canonical_material_contract import (
    aggregate_batch_qm,
    aggregate_order_material_percentages,
    corrected_silica_phr,
    deduplicate_cross_database_rows,
)
from Mooney_Prediction_Pipeline.data_processing.canonical_mms_source import (
    query_mms,
    query_recipe_materials,
    query_supplier_qm,
)
from Mooney_Prediction_Pipeline.feature_engineering.corrected_phase_contract import (
    CONTRACT_VERSION as PHASE_CONTRACT_VERSION,
    apply_corrected_phase_contract,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "reports" / "corrected_phase_history" / "phase_curve_timesafe_v1.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "corrected_phase_history" / "phase_corrected_v371_training.csv"
DEFAULT_AUDIT = ROOT / "reports" / "corrected_phase_history" / "phase_corrected_v371_training_audit.json"

MATERIAL_COLUMNS = [
    "weight_pct_solid_elastomer", "weight_pct_natural_rubber",
    "weight_pct_silica", "weight_pct_oil", "weight_pct_silian",
    "weight_pct_carbon_black",
]
QM_COLUMNS = [
    "supplier_rubber_viscosity_avg", "supplier_silica_moisture_avg",
    "supplier_silica_surface_area_avg", "supplier_carbon_black_structure_avg",
    "supplier_carbon_black_surface_area_avg", "supplier_carbon_black_moisture_avg",
]


def _chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _query_orders(order_ids: list[str], template: str, chunk_size: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in _chunks(order_ids, chunk_size):
        placeholders = ",".join("?" for _ in chunk)
        frames.append(query_mms(template.format(placeholders=placeholders), [*chunk, *chunk]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def query_order_materials_bulk(order_ids: list[str], chunk_size: int) -> pd.DataFrame:
    return _query_orders(order_ids, """
    SELECT N'SFEPLANT' SourceDatabase, CAST(o.OrderID AS VARCHAR(64)) OrderID,
           o.CompoundDescription RecipeID, om.MaterialCode, om.MaterialName,
           TRY_CAST(om.BatchWeight AS FLOAT) MaterialWeight
      FROM SFEPLANT.dbo.Orders o WITH (NOLOCK)
      JOIN SFEPLANT.dbo.OrderMaterials om WITH (NOLOCK) ON om.OrderID=o.OrderID
     WHERE CAST(o.OrderID AS VARCHAR(64)) IN ({placeholders})
    UNION ALL
    SELECT N'SFEPLANT_ARCHIVE', CAST(o.OrderID AS VARCHAR(64)),
           o.CompoundDescription, om.MaterialCode, om.MaterialName,
           TRY_CAST(om.BatchWeight AS FLOAT)
      FROM SFEPLANT_ARCHIVE.dbo.Orders o WITH (NOLOCK)
      JOIN SFEPLANT_ARCHIVE.dbo.OrderMaterials om WITH (NOLOCK) ON om.OrderID=o.OrderID
     WHERE CAST(o.OrderID AS VARCHAR(64)) IN ({placeholders})
    """, chunk_size)


def query_batch_headers_bulk(order_ids: list[str], chunk_size: int) -> pd.DataFrame:
    raw = _query_orders(order_ids, """
    SELECT N'SFEPLANT' SourceDatabase, CAST(OrderID AS VARCHAR(64)) OrderID,
           BatchNumber, CAST(PalletID AS VARCHAR(128)) PalletID,
           TRY_CAST(BatchWeight AS FLOAT) BatchWeight, EquipmentID
      FROM SFEPLANT.dbo.BatchHeader WITH (NOLOCK)
     WHERE CAST(OrderID AS VARCHAR(64)) IN ({placeholders})
    UNION ALL
    SELECT N'SFEPLANT_ARCHIVE', CAST(OrderID AS VARCHAR(64)), BatchNumber,
           CAST(PalletID AS VARCHAR(128)), TRY_CAST(BatchWeight AS FLOAT), EquipmentID
      FROM SFEPLANT_ARCHIVE.dbo.BatchHeader WITH (NOLOCK)
     WHERE CAST(OrderID AS VARCHAR(64)) IN ({placeholders})
    """, chunk_size)
    return raw if raw.empty else deduplicate_cross_database_rows(raw, ["OrderID", "BatchNumber"])


def query_batch_charges_bulk(order_ids: list[str], chunk_size: int) -> pd.DataFrame:
    raw = _query_orders(order_ids, """
    SELECT N'SFEPLANT' SourceDatabase, CAST(OrderID AS VARCHAR(64)) OrderID,
           BatchNumber, MaterialCode, MaterialName, CAST(ChargeID AS VARCHAR(256)) ChargeID,
           TRY_CAST(ActualWeight AS FLOAT) ActualWeight
      FROM SFEPLANT.dbo.BatchMaterials WITH (NOLOCK)
     WHERE CAST(OrderID AS VARCHAR(64)) IN ({placeholders}) AND ChargeID IS NOT NULL
    UNION ALL
    SELECT N'SFEPLANT_ARCHIVE', CAST(OrderID AS VARCHAR(64)), BatchNumber,
           MaterialCode, MaterialName, CAST(ChargeID AS VARCHAR(256)),
           TRY_CAST(ActualWeight AS FLOAT)
      FROM SFEPLANT_ARCHIVE.dbo.BatchMaterials WITH (NOLOCK)
     WHERE CAST(OrderID AS VARCHAR(64)) IN ({placeholders}) AND ChargeID IS NOT NULL
    """, chunk_size)
    if raw.empty:
        return raw
    raw = raw.loc[raw["ChargeID"].astype(str).str.strip().str.upper().ne("UNKNOWN LOT")]
    return deduplicate_cross_database_rows(raw, ["OrderID", "BatchNumber", "MaterialCode", "ChargeID"])


def query_qm_bulk(charges: pd.DataFrame, chunk_size: int = 4000) -> pd.DataFrame:
    if charges.empty:
        return pd.DataFrame()
    ids = charges["ChargeID"].astype(str).str.strip().str.upper().drop_duplicates().tolist()
    frames = [query_supplier_qm(pd.DataFrame({"ChargeID": chunk})) for chunk in _chunks(ids, chunk_size)]
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["OrderID"] = work["OrderID"].astype(str).str.strip()
    work["BatchNumber"] = pd.to_numeric(work["BatchNumber"], errors="coerce")
    work = work.dropna(subset=["OrderID", "BatchNumber"])
    work["BatchNumber"] = work["BatchNumber"].astype(int)
    return work


def build(input_path: Path, output_path: Path, audit_path: Path, chunk_size: int) -> dict[str, object]:
    phase = _normalise_keys(pd.read_csv(input_path, low_memory=False))
    if phase.duplicated(["OrderID", "BatchNumber"]).any():
        raise ValueError("Input Phase history contains duplicate OrderID+BatchNumber rows")
    keys = phase[["OrderID", "BatchNumber"]]
    order_ids = sorted(phase["OrderID"].drop_duplicates().tolist())

    materials = query_order_materials_bulk(order_ids, chunk_size)
    materials["OrderID"] = materials["OrderID"].astype(str).str.strip()
    headers = _normalise_keys(query_batch_headers_bulk(order_ids, chunk_size))
    headers = headers.merge(keys, on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one")
    composition = aggregate_order_material_percentages(materials, headers)

    charges = _normalise_keys(query_batch_charges_bulk(order_ids, chunk_size))
    charges = charges.merge(keys, on=["OrderID", "BatchNumber"], how="inner", validate="many_to_one")
    qm_raw = query_qm_bulk(charges)
    qm = aggregate_batch_qm(charges, qm_raw) if not charges.empty and not qm_raw.empty else keys.iloc[0:0].copy()

    recipe_map = materials[["OrderID", "RecipeID"]].dropna().drop_duplicates("OrderID")
    recipe_ids = sorted(recipe_map["RecipeID"].astype(str).drop_duplicates().tolist())
    recipe_frames = [query_recipe_materials(chunk) for chunk in _chunks(recipe_ids, chunk_size)]
    recipe_raw = pd.concat(recipe_frames, ignore_index=True) if recipe_frames else pd.DataFrame()
    phr = corrected_silica_phr(recipe_raw) if not recipe_raw.empty else pd.DataFrame(columns=["RecipeID", "silica_phr"])

    static = composition.merge(qm, on=["OrderID", "BatchNumber"], how="left", validate="one_to_one")
    static = static.merge(recipe_map, on="OrderID", how="left", validate="many_to_one")
    static = static.merge(phr, on="RecipeID", how="left", validate="many_to_one")

    replace = [*MATERIAL_COLUMNS, *QM_COLUMNS, "silica_phr"]
    corrected = phase.drop(columns=replace, errors="ignore").merge(
        static[["OrderID", "BatchNumber", *replace]],
        on=["OrderID", "BatchNumber"], how="left", validate="one_to_one",
    )
    silica_present = pd.to_numeric(corrected["weight_pct_silica"], errors="coerce").fillna(0).gt(0)
    corrected.loc[~silica_present, "silica_phr"] = 0.0
    corrected = apply_corrected_phase_contract(corrected)
    corrected["MaterialContractVersion"] = "MATERIAL-TAXONOMY-HF-V2"
    corrected["CurveContractVersion"] = "PHASE-CURVE-TIMESAFE-V1"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit = {
        "input": str(input_path), "output": str(output_path),
        "input_rows": int(len(phase)), "output_rows": int(len(corrected)),
        "unique_orders_requested": int(len(order_ids)),
        "orders_with_materials": int(materials["OrderID"].nunique()),
        "batch_headers_matched": int(len(headers)), "batch_charge_rows": int(len(charges)),
        "unique_charge_ids": int(charges["ChargeID"].nunique()) if not charges.empty else 0,
        "supplier_qm_rows": int(len(qm_raw)), "phase_contract_version": PHASE_CONTRACT_VERSION,
        "material_contract_version": "MATERIAL-TAXONOMY-HF-V2",
        "curve_contract_version": "PHASE-CURVE-TIMESAFE-V1",
        "missing_by_corrected_static_column": {column: int(corrected[column].isna().sum()) for column in replace},
        "lab_used_as_feature": False, "cross_batch_curve_imputation": False,
        "credentials_written_to_output": False, "automatic_control": "DISABLED",
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--chunk-size", type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output, args.audit, args.chunk_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
