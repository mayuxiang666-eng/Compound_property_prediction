"""Persist strict V3.7 feature-quality results for corrected full Phase history."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from Mooney_Prediction_Pipeline.model_training.train_v371_active_candidates import load_schema
from services.mooney_predict_service.phase_group_inference import prepare_phase_batches


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history"
SOURCE = BASE / "phase_curve_timesafe_physical_v3.csv"
SCHEMA = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v37_production_model_package" / "feature_metadata.json"
OUTPUT = BASE / "full_phase_feature_contract_v2"
PILOTS = ["9922A00458BT_PV77_004", "9922A09170AX_PV50_004"]


def run(source_path: Path, schema_path: Path, output_dir: Path) -> dict[str, object]:
    source = pd.read_csv(source_path, low_memory=False)
    source["OrderID"] = source["OrderID"].astype(str).str.strip()
    source["BatchNumber"] = pd.to_numeric(source["BatchNumber"], errors="raise").astype(int)
    source["PalletID"] = source["PalletID"].astype(str).str.strip()
    source["LabelGroupID"] = source["OrderID"] + "|" + source["PalletID"]
    schema = load_schema(schema_path, ("supplier_rubber_viscosity_avg",))
    prepared = prepare_phase_batches(source, schema)
    quality = prepared.quality
    output_dir.mkdir(parents=True, exist_ok=True)
    quality.to_csv(output_dir / "batch_feature_quality.csv", index=False, encoding="utf-8-sig")
    missing_cells: dict[str, int] = {}
    for value in quality.loc[
        quality["MissingFeatureCellCount"].gt(0), "MissingFeatureCells"
    ].astype(str):
        for cell in filter(None, value.split("|")):
            missing_cells[cell] = missing_cells.get(cell, 0) + 1
    pilots: dict[str, object] = {}
    for recipe in PILOTS:
        indexes = source["CompoundDescription"].astype(str).eq(recipe).to_numpy()
        selected = quality.loc[indexes]
        pilots[recipe] = {
            "batches": int(len(selected)),
            "model_input_ok": int(selected["FeatureQualityStatus"].eq("MODEL_INPUT_OK").sum()),
            "blocked": int(selected["FeatureQualityStatus"].ne("MODEL_INPUT_OK").sum()),
        }
    summary = {
        "contract": "FULL-PHASE-V37-FEATURE-QUALITY-V2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path), "schema": str(schema_path),
        "source_batches": int(len(source)),
        "model_input_ok": int(quality["FeatureQualityStatus"].eq("MODEL_INPUT_OK").sum()),
        "blocked": int(quality["FeatureQualityStatus"].ne("MODEL_INPUT_OK").sum()),
        "accepted_feature_rows": int(len(prepared.features)),
        "missing_feature_cells": dict(sorted(missing_cells.items(), key=lambda item: item[1], reverse=True)),
        "pilots": pilots,
        "zero_fill_contract": "ONLY_ROUTE_OR_MATERIAL_NONAPPLICABILITY",
        "database_writes": False, "automatic_control": "DISABLED",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--schema", type=Path, default=SCHEMA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.schema, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
