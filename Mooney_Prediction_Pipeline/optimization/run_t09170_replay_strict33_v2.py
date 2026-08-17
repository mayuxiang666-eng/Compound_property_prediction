"""Strict33 replay with raw-material lot stratification included in the audit."""

from __future__ import annotations

from Mooney_Prediction_Pipeline.optimization import silica_stage6_replay as replay
from Mooney_Prediction_Pipeline.optimization import run_t09170_replay_strict33 as runner


_original_build_strata = replay.build_strata


def _build_strata_with_lot(frame):
    out = _original_build_strata(frame)
    if "raw_material_lot_key" in frame.columns:
        lot = frame["raw_material_lot_key"].astype(str).replace({"nan": "MISSING_RAW_MATERIAL_LOT"})
        out["raw_material_lot_stratum"] = lot.to_numpy()
        out["stratum_key"] = out[
            ["time_stratum", "equipment_stratum", "route_stratum", "raw_material_lot_stratum", "initial_temperature_stratum", "recipe_version_stratum"]
        ].astype(str).agg("|".join, axis=1)
    return out


replay.build_strata = _build_strata_with_lot

if __name__ == "__main__":
    runner.main()
