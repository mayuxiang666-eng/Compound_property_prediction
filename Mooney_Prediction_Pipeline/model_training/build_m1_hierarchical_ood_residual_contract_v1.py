"""Build a conservative hierarchical OOD contract for M1 transition experts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    os.environ.get(
        "M1_OOD_CONTRACT_SOURCE",
        str(
            ROOT
            / "reports"
            / "v371_challenger"
            / "global_rds_active_v1"
            / "m1_order_delta_residual_v1"
            / "m1_order_delta_residual_predictions.csv"
        ),
    )
)
OUTPUT = Path(
    os.environ.get(
        "M1_OOD_CONTRACT_OUTPUT",
        str(
            ROOT
            / "reports"
            / "v371_challenger"
            / "global_rds_active_v1"
            / "m1_hierarchical_ood_residual_contract_v1"
        ),
    )
)

# These are eligibility thresholds, not model tuning parameters.  They are
# intentionally conservative and are applied using only earlier transitions.
MIN_HISTORY = {
    "RECIPE_TEMPLATE_MIXER_ROUTE": 15,
    "RECIPE": 20,
    "MATERIAL_ROUTE_MIXER": 30,
    "MATERIAL_SYSTEM": 80,
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(query: pd.Series, history: pd.DataFrame) -> dict[str, int]:
    return {
        "RECIPE_TEMPLATE_MIXER_ROUTE": int(
            (
                history["RecipeID"].eq(query["RecipeID"])
                & history["TemplateKey"].eq(query["TemplateKey"])
                & history["MixerLine"].eq(query["MixerLine"])
                & history["phase_route"].eq(query["phase_route"])
            ).sum()
        ),
        "RECIPE": int(history["RecipeID"].eq(query["RecipeID"]).sum()),
        "MATERIAL_ROUTE_MIXER": int(
            (
                history["material_system"].eq(query["material_system"])
                & history["phase_route"].eq(query["phase_route"])
                & history["MixerLine"].eq(query["MixerLine"])
            ).sum()
        ),
        "MATERIAL_SYSTEM": int(history["material_system"].eq(query["material_system"]).sum()),
    }


def _assign_scope(counts: dict[str, int]) -> tuple[str, int]:
    for scope in (
        "RECIPE_TEMPLATE_MIXER_ROUTE",
        "RECIPE",
        "MATERIAL_ROUTE_MIXER",
        "MATERIAL_SYSTEM",
    ):
        if counts[scope] >= MIN_HISTORY[scope]:
            return scope, counts[scope]
    return "BLOCKED_TRANSITION_HISTORY_SPARSE", max(counts.values(), default=0)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE, low_memory=False)
    frame = frame.loc[frame["CompoundName"].astype(str).str.startswith("M1-")].copy()
    frame["lab_time"] = pd.to_datetime(frame["lab_time"], errors="coerce", utc=True)
    frame = frame.sort_values(["lab_time", "LabelGroupID"], kind="stable").reset_index(drop=True)
    transition_history = frame.loc[frame["transition_available"]].copy()
    rows: list[dict[str, object]] = []
    for _, query in frame.iterrows():
        base = {
            "LabelGroupID": query["LabelGroupID"],
            "OrderID": query["OrderID"],
            "CompoundName": query["CompoundName"],
            "RecipeID": query["RecipeID"],
            "ProcessVersion": query["ProcessVersion"],
            "MixerLine": query["MixerLine"],
            "material_system": query["material_system"],
            "phase_route": query["phase_route"],
            "TemplateKey": query["TemplateKey"],
            "split": query["split"],
            "lab_time": query["lab_time"],
            "MNY": query["MNY"],
            "previous_MNY": query["previous_MNY"],
            "DeltaJITPrediction": query["DeltaJITPrediction"],
            "transition_available": bool(query["transition_available"]),
        }
        if not bool(query["transition_available"]):
            base.update(
                {
                    "ood_scope": "BLOCKED_NO_PRIOR_SAME_ORDER_LAB",
                    "ood_history_count": 0,
                    "ood_eligible": False,
                    "recommendation_gate": "BLOCKED_NO_PRIOR_SAME_ORDER_LAB",
                }
            )
            rows.append(base)
            continue
        history = transition_history.loc[
            transition_history["lab_time"].lt(query["lab_time"])
        ]
        counts = _counts(query, history)
        scope, history_count = _assign_scope(counts)
        base.update(
            {
                "ood_scope": scope,
                "ood_history_count": history_count,
                "ood_eligible": scope != "BLOCKED_TRANSITION_HISTORY_SPARSE",
                "recommendation_gate": (
                    "BLOCKED_MODEL_NOT_PASSED"
                    if scope != "BLOCKED_TRANSITION_HISTORY_SPARSE"
                    else "BLOCKED_TRANSITION_HISTORY_SPARSE"
                ),
                **{f"history_count__{key}": value for key, value in counts.items()},
            }
        )
        rows.append(base)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "m1_hierarchical_ood_contract.csv", index=False)
    coverage = (
        result.groupby(["split", "ood_scope"], dropna=False)
        .size()
        .rename("groups")
        .reset_index()
    )
    coverage.to_csv(OUTPUT / "m1_hierarchical_ood_coverage.csv", index=False)
    metadata = {
        "contract": "M1_HIERARCHICAL_OOD_RESIDUAL_CONTRACT_V1",
        "scope": "M1_ONLY",
        "source": str(SOURCE),
        "source_sha256": _hash_file(SOURCE),
        "minimum_history_by_scope": MIN_HISTORY,
        "history_rule": "only transition rows with lab_time strictly earlier than current group",
        "uses_current_lab_value": False,
        "uses_future_group": False,
        "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "stage3_calibration_enabled": False,
        "automatic_control": False,
        "all_groups": int(len(result)),
        "transition_groups": int(result["transition_available"].sum()),
        "ood_eligible_groups": int(result["ood_eligible"].sum()),
        "recommendation_gate_open_groups": 0,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (OUTPUT / "m1_hierarchical_ood_metadata_v1.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(coverage.to_string(index=False))
    print("\nTotals:")
    print(json.dumps({key: metadata[key] for key in ("all_groups", "transition_groups", "ood_eligible_groups", "recommendation_gate_open_groups")}, indent=2))


if __name__ == "__main__":
    main()
