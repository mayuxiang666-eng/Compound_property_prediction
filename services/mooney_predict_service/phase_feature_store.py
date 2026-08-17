"""Read-only adapter for versioned incremental Phase feature files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .phase_group_contract import PhaseFeatureContractError


def _source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Phase feature store does not exist: {source}")
    files = sorted([*source.glob("*.parquet"), *source.glob("*.csv")])
    if not files:
        raise FileNotFoundError(f"No parquet/csv Phase partitions in: {source}")
    return files


def _read(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise PhaseFeatureContractError(f"Unsupported Phase store format: {path}")


def load_phase_feature_store(
    source: str | Path,
    *,
    order_id: str | None = None,
    compound_name: str | None = None,
    since: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load Phase partitions without querying the shortcut MMS extractor."""
    root = Path(source)
    files = _source_files(root)
    frames: list[pd.DataFrame] = []
    for path in files:
        part = _read(path)
        part["_phase_source_file"] = str(path)
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True, sort=False)
    source_rows = len(frame)

    if order_id is not None:
        if "OrderID" not in frame:
            raise PhaseFeatureContractError("OrderID is absent from Phase feature store")
        frame = frame.loc[frame["OrderID"].astype(str).str.strip() == str(order_id).strip()].copy()
    if compound_name is not None:
        if "CompoundName" not in frame:
            raise PhaseFeatureContractError("CompoundName is absent from Phase feature store")
        frame = frame.loc[
            frame["CompoundName"].astype(str).str.strip() == str(compound_name).strip()
        ].copy()
    if since is not None:
        if "OrderStartTime" not in frame:
            raise PhaseFeatureContractError("OrderStartTime is absent from Phase feature store")
        cutoff = pd.Timestamp(since)
        timestamps = pd.to_datetime(frame["OrderStartTime"], errors="coerce")
        frame = frame.loc[timestamps >= cutoff].copy()

    audit = {
        "source": str(root.resolve()),
        "source_files": [str(path.resolve()) for path in files],
        "source_rows_before_filter": int(source_rows),
        "source_rows_after_filter": int(len(frame)),
        "order_filter": order_id,
        "compound_filter": compound_name,
        "since_filter": since,
        "source_contract": "VERSIONED_PHASE_FEATURE_STORE",
    }
    return frame, audit
