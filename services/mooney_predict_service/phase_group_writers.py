"""Transactional persistence for Phase V3.7 batch/group shadow evidence."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .db import connect


def write_quality_rows(quality: pd.DataFrame) -> int:
    sql = """
    INSERT dbo.MooneyFeatureQualityAudit
    (OrderID, BatchID, PalletID, LabelGroupID, FeatureQualityStatus,
     MissingModelColumnCount, MissingFeatureCellCount, MissingFeatureCells,
     RouteCompatibleZeroFillCount, RouteCompatibleZeroFillCells,
     LabColumnsRemoved, FeatureBuilderVersion)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    count = 0
    with connect() as conn:
        for row in quality.to_dict("records"):
            conn.execute(sql, (
                str(row["OrderID"]), str(row["BatchNumber"]), row.get("PalletID"),
                row.get("LabelGroupID"), row["FeatureQualityStatus"],
                int(row["MissingModelColumnCount"]), int(row["MissingFeatureCellCount"]),
                row.get("MissingFeatureCells"), int(row["RouteCompatibleZeroFillCount"]),
                row.get("RouteCompatibleZeroFillCells"), row.get("LabColumnsRemoved"),
                row["FeatureBuilderVersion"],
            ))
            count += 1
        conn.commit()
    return count


def write_batch_rows(batch: pd.DataFrame, quality: pd.DataFrame, model_id: int) -> int:
    sql = """
    IF NOT EXISTS
    (
        SELECT 1 FROM dbo.MooneyPhaseBatchResult WITH (UPDLOCK, HOLDLOCK)
        WHERE OrderID=? AND BatchID=? AND ModelVersion=? AND FeatureBuilderVersion=?
    )
    INSERT dbo.MooneyPhaseBatchResult
    (OrderID, BatchID, SourceBatchNumber, PalletID, LabelGroupID, CompName,
     ProductTime, ModelID, ModelVersion, Stage1Pred, Stage1bBias, Stage2Delta,
     RawPrediction, FeatureMissingCount, RouteCompatibleZeroFillCount,
     FeatureBuilderVersion, AutomaticControlStatus)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, N'DISABLED')
    """
    quality_map = quality.set_index(["OrderID", "BatchNumber"]).to_dict("index")
    inserted = 0
    with connect() as conn:
        for row in batch.to_dict("records"):
            order, batch_id = str(row["OrderID"]), str(int(row["BatchNumber"]))
            q = quality_map[(str(row["OrderID"]), int(row["BatchNumber"]))]
            before = conn.execute("SELECT @@ROWCOUNT").fetchone()[0]
            cursor = conn.execute(sql, (
                order, batch_id, row["ModelVersion"], row["FeatureBuilderVersion"],
                order, batch_id, str(row["BatchNumber"]), row["PalletID"],
                row["LabelGroupID"], row["CompoundName"], row["OrderStartTime"],
                int(model_id), row["ModelVersion"], float(row["Stage1Pred"]),
                float(row["Stage1bBias"]), float(row["Stage2Delta"]),
                float(row["RawPrediction"]), int(q["MissingFeatureCellCount"]),
                int(q["RouteCompatibleZeroFillCount"]), row["FeatureBuilderVersion"],
            ))
            inserted += max(int(cursor.rowcount), 0)
        conn.commit()
    return inserted


def upsert_group_rows(group: pd.DataFrame, model_id: int) -> int:
    """Update a growing pallet mean under serializable key locks."""
    sql = """
    SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
    UPDATE dbo.MooneyGroupResult WITH (UPDLOCK, HOLDLOCK)
       SET BatchCount=?, FirstBatchNumber=?, LastBatchNumber=?,
           GroupRawPrediction=?, BatchPredictionStd=?, ProductTime=?,
           GroupPredictionStatus=?, AutomaticControlStatus=N'DISABLED',
           Revision=Revision+1, UpdatedAt=SYSUTCDATETIME()
     WHERE LabelGroupID=? AND ModelVersion=? AND FeatureBuilderVersion=?;
    IF @@ROWCOUNT = 0
    BEGIN
        INSERT dbo.MooneyGroupResult
        (LabelGroupID, OrderID, PalletID, CompName, BatchCount,
         FirstBatchNumber, LastBatchNumber, GroupRawPrediction,
         BatchPredictionStd, ProductTime, ModelID, ModelVersion,
         FeatureBuilderVersion, GroupPredictionStatus, AutomaticControlStatus)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, N'DISABLED');
    END
    """
    count = 0
    with connect() as conn:
        for row in group.to_dict("records"):
            common = (
                int(row["BatchCount"]), int(row["FirstBatchNumber"]),
                int(row["LastBatchNumber"]), float(row["GroupRawPrediction"]),
                float(row["BatchPredictionStd"]), row["ProductTime"],
                row["GroupPredictionStatus"], row["LabelGroupID"],
                row["ModelVersion"], row["FeatureBuilderVersion"],
            )
            insert = (
                row["LabelGroupID"], row["OrderID"], row["PalletID"], row["CompoundName"],
                int(row["BatchCount"]), int(row["FirstBatchNumber"]),
                int(row["LastBatchNumber"]), float(row["GroupRawPrediction"]),
                float(row["BatchPredictionStd"]), row["ProductTime"], int(model_id),
                row["ModelVersion"], row["FeatureBuilderVersion"],
                row["GroupPredictionStatus"],
            )
            conn.execute(sql, (*common, *insert))
            count += 1
        conn.commit()
    return count
