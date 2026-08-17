"""Build T15760 causal Charge state after supplier-lot normalization."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.feature_engineering.causal_charge_state_v2 import (
    build_causal_charge_features, material_family, normalize_charge_lot,
)
from Mooney_Prediction_Pipeline.model_training.run_v371_r8_batch_group_mean import BASE, GROUP_SOURCE, PHASE_SOURCE, T15760_NAME


MATERIAL_SOURCE = BASE / "batch_materials_active.csv"
OUTPUT = BASE / "t15760_causal_charge_state_v2"


def normalize(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    phase = pd.read_csv(PHASE_SOURCE, usecols=["OrderID", "BatchNumber", "LabelGroupID", "CompoundName", "BatchStartTime"], low_memory=False)
    phase = phase.loc[phase["CompoundName"].astype(str).eq(T15760_NAME)].copy()
    phase["OrderID"], phase["BatchNumber"] = normalize(phase["OrderID"]), normalize(phase["BatchNumber"])
    keys = phase[["OrderID", "BatchNumber", "LabelGroupID"]].drop_duplicates(["OrderID", "BatchNumber"])
    orders = set(keys["OrderID"])
    parts = []
    for chunk in pd.read_csv(MATERIAL_SOURCE, chunksize=250_000, low_memory=False):
        chunk["OrderID"] = normalize(chunk["OrderID"])
        selected = chunk.loc[chunk["OrderID"].isin(orders)].copy()
        if len(selected):
            selected["BatchNumber"] = normalize(selected["BatchNumber"])
            parts.append(selected)
    materials = pd.concat(parts, ignore_index=True).merge(keys, on=["OrderID", "BatchNumber"], how="inner", validate="many_to_one")
    materials["MaterialFamily"] = materials["MaterialCode"].map(material_family)
    materials["ChargeLotID"] = materials["ChargeID"].map(normalize_charge_lot)
    materials["ActualWeight"] = pd.to_numeric(materials["ActualWeight"], errors="coerce").fillna(0.0)
    charges = materials.groupby(["LabelGroupID", "MaterialFamily", "MaterialCode", "MaterialName", "ChargeID", "ChargeLotID"], dropna=False, as_index=False)["ActualWeight"].sum()
    charges.to_csv(OUTPUT / "t15760_group_charge_lot_composition.csv", index=False)

    groups = pd.read_csv(GROUP_SOURCE, usecols=["LabelGroupID", "CompoundName", "lab_time", "MNY"], low_memory=False)
    groups = groups.loc[groups["CompoundName"].astype(str).eq(T15760_NAME)].copy()
    start = phase.groupby("LabelGroupID", as_index=False)["BatchStartTime"].min().rename(columns={"BatchStartTime": "production_start"})
    groups = groups.merge(start, on="LabelGroupID", how="inner", validate="one_to_one")
    state_input = charges.groupby(["LabelGroupID", "MaterialFamily", "ChargeLotID"], dropna=False, as_index=False)["ActualWeight"].sum().rename(columns={"ChargeLotID": "ChargeID"})
    state = build_causal_charge_features(groups, state_input, shrinkage=10.0)
    state.to_csv(OUTPUT / "t15760_causal_charge_features_v2.csv", index=False)
    features = [column for column in state.columns if column != "LabelGroupID"]
    summary = {
        "contract": "T15760-CAUSAL-CHARGE-STATE-V2",
        "groups": len(groups), "charge_rows": len(charges),
        "material_family_lot_counts": charges.groupby("MaterialFamily")["ChargeLotID"].nunique().to_dict(),
        "feature_nonzero_rates": {column: float(pd.to_numeric(state[column], errors="coerce").fillna(0).ne(0).mean()) for column in features},
        "normalization": "S + two storage digits removed; supplier lot body retained",
        "causal_rule": "Only Lab events with lab_time < current production_start; current/future Lab excluded",
        "shrinkage_k": 10.0,
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
