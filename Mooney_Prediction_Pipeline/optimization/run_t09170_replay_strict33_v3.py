"""Strict33 replay with corrected raw-material stratum count in summary."""

from __future__ import annotations

from Mooney_Prediction_Pipeline.optimization import silica_stage6_replay as replay
from Mooney_Prediction_Pipeline.optimization import run_t09170_replay_strict33_v2 as runner


_original_run = replay.run_bidirectional_replay


def _run_with_lot_summary(history, *args, **kwargs):
    audit, summary = _original_run(history, *args, **kwargs)
    if "raw_material_lot_key" in history.columns:
        summary.setdefault("strata", {})["raw_material_lot"] = int(history["raw_material_lot_key"].astype(str).nunique())
    return audit, summary


replay.run_bidirectional_replay = _run_with_lot_summary

if __name__ == "__main__":
    runner.runner.main()
