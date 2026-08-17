"""Prospective holdout contract for V3.7.1 release evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BlindWindowContract:
    window_id: str
    cutoff_lab_time: str
    frozen_model_id: str
    frozen_model_sha256: str
    min_global_groups: int = 100
    min_t15760_groups: int = 50
    min_t15760_batches: int = 150
    min_b00458_groups: int = 30
    min_b00458_batches: int = 150
    source_status: str = "RDS_ACTIVE_ONLY"
    archive_status: str = "BLOCKED_ARCHIVE_NOT_REPLICATED"

    def cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(self.cutoff_lab_time)


def make_group_id(frame: pd.DataFrame) -> pd.Series:
    return frame["OrderID"].astype(str).str.strip() + "|" + frame["PalletID"].astype(str).str.strip()


def audit_blind_window(
    rows: pd.DataFrame,
    contract: BlindWindowContract,
    *,
    training_group_ids: set[str] | None = None,
) -> dict:
    required = {"OrderID", "PalletID", "BatchNumber", "lab_time", "CompoundName"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Blind-window rows missing columns: {missing}")
    frame = rows.copy()
    frame["LabelGroupID"] = make_group_id(frame)
    frame["lab_time"] = pd.to_datetime(frame["lab_time"], errors="coerce")
    if frame["lab_time"].isna().any():
        raise ValueError("Blind-window lab_time contains invalid values")
    if frame[["OrderID", "BatchNumber"]].astype(str).duplicated().any():
        raise ValueError("Blind-window physical batch key is not unique")
    before_or_at_cutoff = int(frame["lab_time"].le(contract.cutoff()).sum())
    train_ids = training_group_ids or set()
    overlap = sorted(set(frame["LabelGroupID"]).intersection(train_ids))
    future = frame.loc[frame["lab_time"].gt(contract.cutoff())].copy()

    def counts(compound: str | None) -> tuple[int, int]:
        part = future if compound is None else future.loc[future["CompoundName"].eq(compound)]
        return int(part["LabelGroupID"].nunique()), int(len(part))

    global_groups, global_batches = counts(None)
    t_groups, t_batches = counts("T15760")
    b_groups, b_batches = counts("B00458")
    blockers = []
    if contract.source_status != "RDS_ACTIVE_ONLY":
        blockers.append("BLOCKED_UNAPPROVED_DATA_SOURCE")
    if before_or_at_cutoff:
        blockers.append("BLOCKED_WINDOW_CONTAINS_PRE_CUTOFF_ROWS")
    if overlap:
        blockers.append("BLOCKED_TRAIN_HOLDOUT_GROUP_OVERLAP")
    if global_groups < contract.min_global_groups:
        blockers.append("WAITING_GLOBAL_BLIND_GROUPS")
    if t_groups < contract.min_t15760_groups or t_batches < contract.min_t15760_batches:
        blockers.append("WAITING_T15760_BLIND_SUPPORT")
    if b_groups < contract.min_b00458_groups or b_batches < contract.min_b00458_batches:
        blockers.append("WAITING_B00458_BLIND_SUPPORT")
    return {
        "contract": asdict(contract),
        "status": "READY_FOR_ONE_TIME_EVALUATION" if not blockers else "WAITING_NEW_BLIND_WINDOW",
        "reason_codes": blockers,
        "rows": int(len(future)),
        "global_groups": global_groups,
        "global_batches": global_batches,
        "t15760_groups": t_groups,
        "t15760_batches": t_batches,
        "b00458_groups": b_groups,
        "b00458_batches": b_batches,
        "training_overlap_groups": len(overlap),
        "pre_cutoff_rows": before_or_at_cutoff,
    }


def write_registry(path: Path, contract: BlindWindowContract, audit: dict) -> None:
    payload = {"contract": asdict(contract), "current_audit": audit}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    payload["registry_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["BlindWindowContract", "audit_blind_window", "make_group_id", "write_registry"]
