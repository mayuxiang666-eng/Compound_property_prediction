"""Audit historical Target-vector revisions inside an exact RecipeID.

Structural templates and parameter revisions are deliberately separated.
The order-level modal vector represents the likely engineering recipe state;
within-order batch deviations are reported as overrides instead of silently
creating recipe revisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "corrected_phase_history" / "crossdb_exact_pilot_step_history_v1"
SOURCE = BASE / "mapped_step_vectors.csv"

PARAMETERS = {
    "duration_s": ("dur_SP", "is_time_settable", 1.0),
    "temperature_c": ("temp_SP", "is_temp_settable", 0.5),
    "rotor_rpm": ("rotor_SP", "is_rotor_settable", 0.5),
    "ram_percent": ("ram_SP", "is_ram_settable", 1.0),
}


def _quantize(value: object, quantum: float) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or float(numeric) <= 0:
        return None
    return round(round(float(numeric) / quantum) * quantum, 6)


def _vector_for_batch(batch: pd.DataFrame) -> tuple[str, str, int]:
    vector: dict[str, float] = {}
    for row in batch.sort_values(["GroupName", "StepNo"], kind="stable").itertuples():
        for parameter, (value_column, flag_column, quantum) in PARAMETERS.items():
            if not bool(getattr(row, flag_column)):
                continue
            value = _quantize(getattr(row, value_column), quantum)
            if value is not None:
                vector[f"{row.GroupName}|S{int(row.StepNo)}|{parameter}"] = value
    payload = json.dumps(vector, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12].upper()
    return signature, payload, len(vector)


def build_batch_vectors(mapped: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "CompoundDescription", "CompoundName", "material_system", "MixerLine",
        "TemplateID", "OrderID", "BatchNumber", "OrderStartTime", "PalletID",
    ]
    available = [column for column in keys if column in mapped]
    rows: list[dict[str, object]] = []
    for key, frame in mapped.groupby(available, sort=False, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        signature, payload, target_count = _vector_for_batch(frame)
        rows.append({
            **dict(zip(available, values)),
            "TargetVectorID": signature,
            "TargetVectorJSON": payload,
            "TargetFieldCount": target_count,
        })
    result = pd.DataFrame(rows)
    result["OrderStartTime"] = pd.to_datetime(result["OrderStartTime"], errors="coerce")
    return result


def build_order_vectors(batch_vectors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_keys = [
        "CompoundDescription", "CompoundName", "material_system", "MixerLine",
        "TemplateID", "OrderID",
    ]
    orders: list[dict[str, object]] = []
    overrides: list[pd.DataFrame] = []
    for key, frame in batch_vectors.groupby(group_keys, sort=False, dropna=False):
        counts = frame["TargetVectorID"].value_counts()
        max_count = int(counts.max())
        modal_ids = sorted(counts.loc[counts.eq(max_count)].index.astype(str))
        modal_id = modal_ids[0]
        modal_json = frame.loc[frame["TargetVectorID"].eq(modal_id), "TargetVectorJSON"].iloc[0]
        batch_count = int(len(frame))
        order_record = {
            **dict(zip(group_keys, key if isinstance(key, tuple) else (key,))),
            "OrderStartTime": frame["OrderStartTime"].min(),
            "OrderTargetVectorID": modal_id,
            "OrderTargetVectorJSON": modal_json,
            "BatchCount": batch_count,
            "LabGroupCount": int(frame[["OrderID", "PalletID"]].drop_duplicates().shape[0]),
            "ModalVectorBatchCount": max_count,
            "ModalVectorShare": max_count / batch_count,
            "WithinOrderVectorCount": int(frame["TargetVectorID"].nunique()),
            "HasWithinOrderOverride": bool(frame["TargetVectorID"].nunique() > 1),
            "AmbiguousOrderMode": bool(len(modal_ids) > 1),
        }
        orders.append(order_record)
        deviating = frame.loc[frame["TargetVectorID"].ne(modal_id)].copy()
        if not deviating.empty:
            deviating["OrderTargetVectorID"] = modal_id
            overrides.append(deviating)
    return pd.DataFrame(orders), pd.concat(overrides, ignore_index=True) if overrides else pd.DataFrame()


def _dict_diff(before_json: str, after_json: str) -> list[dict[str, object]]:
    before = json.loads(before_json)
    after = json.loads(after_json)
    rows: list[dict[str, object]] = []
    for field in sorted(set(before) | set(after)):
        old = before.get(field)
        new = after.get(field)
        if old == new:
            continue
        delta = None if old is None or new is None else float(new) - float(old)
        group, step, parameter = field.split("|", 2)
        rows.append({
            "TargetField": field, "GroupName": group,
            "StepNo": int(step.removeprefix("S")), "Parameter": parameter,
            "BeforeTarget": old, "AfterTarget": new, "Delta": delta,
            "ChangeType": "ADDED" if old is None else "REMOVED" if new is None else "CHANGED",
        })
    return rows


def build_revision_segments(order_vectors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    segment_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    group_keys = ["CompoundDescription", "MixerLine", "TemplateID"]
    for key, frame in order_vectors.groupby(group_keys, sort=False, dropna=False):
        frame = frame.sort_values(["OrderStartTime", "OrderID"], kind="stable").reset_index(drop=True)
        frame["RevisionSequence"] = frame["OrderTargetVectorID"].ne(
            frame["OrderTargetVectorID"].shift()
        ).cumsum().astype(int)
        previous = None
        for sequence, segment in frame.groupby("RevisionSequence", sort=True):
            current = segment.iloc[0]
            segment_id = f"{current.TemplateID}-R{int(sequence):03d}"
            segment_rows.append({
                "CompoundDescription": key[0], "MixerLine": key[1], "TemplateID": key[2],
                "RecipeRevisionSegment": segment_id,
                "RevisionSequence": int(sequence),
                "TargetVectorID": current.OrderTargetVectorID,
                "TargetVectorJSON": current.OrderTargetVectorJSON,
                "EffectiveFrom": segment["OrderStartTime"].min(),
                "EffectiveTo": segment["OrderStartTime"].max(),
                "OrderCount": int(segment["OrderID"].nunique()),
                "BatchCount": int(segment["BatchCount"].sum()),
                "LabGroupCount": int(segment["LabGroupCount"].sum()),
                "OrdersWithOverride": int(segment["HasWithinOrderOverride"].sum()),
                "MinModalVectorShare": float(segment["ModalVectorShare"].min()),
            })
            if previous is not None:
                for change in _dict_diff(previous.OrderTargetVectorJSON, current.OrderTargetVectorJSON):
                    event_rows.append({
                        "CompoundDescription": key[0], "MixerLine": key[1], "TemplateID": key[2],
                        "RecipeRevisionSegment": segment_id,
                        "EffectiveFrom": segment["OrderStartTime"].min(),
                        "PreviousTargetVectorID": previous.OrderTargetVectorID,
                        "NewTargetVectorID": current.OrderTargetVectorID,
                        **change,
                    })
            previous = current
    return pd.DataFrame(segment_rows), pd.DataFrame(event_rows)


def build_summary(
    batch_vectors: pd.DataFrame,
    order_vectors: pd.DataFrame,
    segments: pd.DataFrame,
    events: pd.DataFrame,
    overrides: pd.DataFrame,
) -> dict[str, object]:
    recipes: dict[str, object] = {}
    for recipe, batches in batch_vectors.groupby("CompoundDescription", sort=False):
        orders = order_vectors.loc[order_vectors["CompoundDescription"].eq(recipe)]
        recipe_segments = segments.loc[segments["CompoundDescription"].eq(recipe)]
        recipe_events = events.loc[events["CompoundDescription"].eq(recipe)] if not events.empty else events
        recipe_overrides = overrides.loc[overrides["CompoundDescription"].eq(recipe)] if not overrides.empty else overrides
        by_parameter = (
            recipe_events["Parameter"].value_counts().astype(int).to_dict()
            if not recipe_events.empty else {}
        )
        recipes[str(recipe)] = {
            "batch_count": int(len(batches)),
            "order_count": int(orders["OrderID"].nunique()),
            "structural_template_count": int(batches["TemplateID"].nunique()),
            "unique_batch_target_vectors": int(batches[["TemplateID", "TargetVectorID"]].drop_duplicates().shape[0]),
            "unique_order_target_vectors": int(orders[["TemplateID", "OrderTargetVectorID"]].drop_duplicates().shape[0]),
            "contiguous_revision_segments": int(len(recipe_segments)),
            "revision_transitions": int(max(len(recipe_segments) - batches["TemplateID"].nunique(), 0)),
            "parameter_change_events": int(len(recipe_events)),
            "parameter_change_events_by_type": {str(k): int(v) for k, v in by_parameter.items()},
            "orders_with_within_order_override": int(orders["HasWithinOrderOverride"].sum()),
            "override_batch_count": int(len(recipe_overrides)),
            "ambiguous_order_modes": int(orders["AmbiguousOrderMode"].sum()),
        }
    return {
        "contract": "RECIPE-TARGET-REVISION-AUDIT-V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "revision_unit": "ORDER_LEVEL_MODAL_TARGET_VECTOR_WITHIN_STRUCTURAL_TEMPLATE",
        "duration_quantum_s": 1.0,
        "temperature_quantum_c": 0.5,
        "rotor_quantum_rpm": 0.5,
        "ram_quantum_percent": 1.0,
        "recipes": recipes,
        "database_writes": False,
        "automatic_control": "DISABLED",
    }


def run(source: Path, output_dir: Path) -> dict[str, object]:
    mapped = pd.read_csv(source, low_memory=False)
    batch_vectors = build_batch_vectors(mapped)
    order_vectors, overrides = build_order_vectors(batch_vectors)
    segments, events = build_revision_segments(order_vectors)
    summary = build_summary(batch_vectors, order_vectors, segments, events, overrides)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "batch_target_vectors": batch_vectors,
        "order_target_vectors": order_vectors,
        "recipe_revision_segments": segments,
        "parameter_change_events": events,
        "within_order_target_overrides": overrides,
    }.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    (output_dir / "recipe_target_revision_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=BASE / "recipe_target_revisions_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output_dir), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
