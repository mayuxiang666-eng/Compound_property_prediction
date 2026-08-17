"""Build strictly aligned corrected Phase and MMS feature frames."""

from __future__ import annotations

import pandas as pd

from Mooney_Prediction_Pipeline.data_processing.canonical_mms_source_v2 import build_corrected_static_features
from Mooney_Prediction_Pipeline.feature_engineering.corrected_phase_contract import apply_corrected_phase_contract
from Mooney_Prediction_Pipeline.model_analysis.predict_latest_sfe_order import extract_features_for_row
from Mooney_Prediction_Pipeline.model_analysis.predict_order_v3 import query_order_rows
from Mooney_Prediction_Pipeline.optimization.audit_corrected_operational_parity_v2 import PHASE, WEATHER


def build_corrected_parity_frames(order_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase = pd.read_csv(PHASE, low_memory=False)
    phase = phase.loc[phase["OrderID"].astype(str).eq(str(order_id))].copy()
    phase["OrderID"] = phase["OrderID"].astype(str)
    phase["BatchNumber"] = pd.to_numeric(phase["BatchNumber"], errors="coerce").astype(int)
    keys = phase[["OrderID", "BatchNumber"]].drop_duplicates()

    static = build_corrected_static_features(str(order_id))
    static["OrderID"] = static["OrderID"].astype(str)
    static["BatchNumber"] = pd.to_numeric(static["BatchNumber"], errors="coerce").astype(int)
    source = query_order_rows(str(order_id)).copy()
    source["OrderID"] = source["OrderID"].astype(str)
    source["BatchNumber"] = pd.to_numeric(source["BatchNumber"], errors="coerce").astype(int)
    source = source.merge(keys, on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one")
    corrected_columns = [column for column in static.columns if column not in {"OrderID", "BatchNumber"}]
    source = source.drop(columns=[column for column in corrected_columns if column in source], errors="ignore")
    source = source.merge(static, on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one")

    rows = []
    for _, row in source.iterrows():
        feature = extract_features_for_row(row)
        static_row = static.loc[static["BatchNumber"].eq(int(row["BatchNumber"]))].iloc[0]
        feature.update(static_row.to_dict())
        rows.append(feature)
    operational = pd.DataFrame(rows)
    operational["OrderID"] = operational["OrderID"].astype(str)
    operational["BatchNumber"] = pd.to_numeric(operational["BatchNumber"], errors="coerce").astype(int)
    operational = operational.merge(
        phase[["OrderID", "BatchNumber", "PalletID", *WEATHER]],
        on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one",
    ).sort_values(["OrderID", "BatchNumber"]).reset_index(drop=True)

    reference = phase.merge(
        static, on=["OrderID", "BatchNumber"], how="inner", suffixes=("", "__corrected"),
        validate="one_to_one",
    ).sort_values(["OrderID", "BatchNumber"]).reset_index(drop=True)
    for column in corrected_columns:
        corrected = f"{column}__corrected"
        if corrected in reference:
            reference[column] = reference[corrected]
            reference.drop(columns=[corrected], inplace=True)
    return apply_corrected_phase_contract(reference), apply_corrected_phase_contract(operational)
