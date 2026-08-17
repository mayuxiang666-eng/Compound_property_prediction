from __future__ import annotations

from typing import Mapping

from .db import connect


def write_prediction(record: Mapping[str, object]) -> str:
    """Insert one immutable prediction; database uniqueness provides idempotency."""
    sql = """
    INSERT INTO dbo.MooneyResult
    (OrderID, BatchID, SourceBatchNumber, PalletID, LabelGroupID,
     CompName, CompDescription, Mixer, MaterialSystem, PhaseRoute,
     ProductTime, ModelID, ModelVersion, Stage1Pred, Stage1bBias,
     Stage2Delta, RawPrediction, CalibrationOffset, FinalPrediction,
     LabStatus, FeatureMissingCount, CurvePointCount, FeatureBuilderVersion)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    values = [record.get(k) for k in (
        "OrderID", "BatchID", "SourceBatchNumber", "PalletID", "LabelGroupID",
        "CompName", "CompDescription", "Mixer", "MaterialSystem", "PhaseRoute",
        "ProductTime", "ModelID", "ModelVersion", "Stage1Pred", "Stage1bBias",
        "Stage2Delta", "RawPrediction", "CalibrationOffset", "FinalPrediction",
        "LabStatus", "FeatureMissingCount", "CurvePointCount", "FeatureBuilderVersion",
    )]
    try:
        with connect() as conn:
            conn.execute(sql, values)
            conn.commit()
        return "INSERTED"
    except Exception as exc:
        if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
            return "ALREADY_EXISTS"
        raise


def write_miss(record: Mapping[str, object]) -> None:
    sql = """
    INSERT INTO dbo.MooneyPredictMiss
    (OrderID, BatchID, PalletID, CompName, Mixer, ProductTime,
     MissReason, MissDetail, Retryable, RetryCount, ModelVersion)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    values = [record.get(k) for k in (
        "OrderID", "BatchID", "PalletID", "CompName", "Mixer", "ProductTime",
        "MissReason", "MissDetail", "Retryable", "RetryCount", "ModelVersion",
    )]
    with connect() as conn:
        conn.execute(sql, values)
        conn.commit()

