/*
  V3.7 Phase-group shadow migration.
  This adds isolated V2 evidence tables; it does not alter existing predictions
  and it enables no calibration or automatic control.
*/
USE [MooneyPredictionV37Shadow];
GO

IF OBJECT_ID(N'dbo.MooneyPhaseBatchResult', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.MooneyPhaseBatchResult
    (
        PhaseBatchResultID BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT PK_MooneyPhaseBatchResult PRIMARY KEY,
        OrderID NVARCHAR(64) NOT NULL,
        BatchID NVARCHAR(64) NOT NULL,
        SourceBatchNumber NVARCHAR(128) NOT NULL,
        PalletID NVARCHAR(128) NOT NULL,
        LabelGroupID NVARCHAR(256) NOT NULL,
        CompName NVARCHAR(128) NOT NULL,
        ProductTime DATETIME2(3) NOT NULL,
        ModelID INT NOT NULL,
        ModelVersion NVARCHAR(64) NOT NULL,
        Stage1Pred FLOAT NOT NULL,
        Stage1bBias FLOAT NOT NULL,
        Stage2Delta FLOAT NOT NULL,
        RawPrediction FLOAT NOT NULL,
        FeatureMissingCount INT NOT NULL,
        RouteCompatibleZeroFillCount INT NOT NULL,
        FeatureBuilderVersion NVARCHAR(64) NOT NULL,
        AutomaticControlStatus NVARCHAR(32) NOT NULL CONSTRAINT DF_PhaseBatch_Auto DEFAULT(N'DISABLED'),
        CreatedAt DATETIME2(3) NOT NULL CONSTRAINT DF_PhaseBatch_Created DEFAULT(SYSUTCDATETIME()),
        CONSTRAINT FK_MooneyPhaseBatchResult_Model FOREIGN KEY(ModelID) REFERENCES dbo.MooneyModel(ModelID),
        CONSTRAINT UQ_MooneyPhaseBatchResult_Contract UNIQUE(OrderID, BatchID, ModelVersion, FeatureBuilderVersion),
        CONSTRAINT CK_MooneyPhaseBatchResult_Auto CHECK(AutomaticControlStatus = N'DISABLED')
    );
END;
GO

IF OBJECT_ID(N'dbo.MooneyGroupResult', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.MooneyGroupResult
    (
        GroupResultID BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT PK_MooneyGroupResult PRIMARY KEY,
        LabelGroupID NVARCHAR(256) NOT NULL,
        OrderID NVARCHAR(64) NOT NULL,
        PalletID NVARCHAR(128) NOT NULL,
        CompName NVARCHAR(128) NOT NULL,
        BatchCount INT NOT NULL,
        FirstBatchNumber INT NOT NULL,
        LastBatchNumber INT NOT NULL,
        GroupRawPrediction FLOAT NOT NULL,
        BatchPredictionStd FLOAT NOT NULL,
        ProductTime DATETIME2(3) NOT NULL,
        ModelID INT NOT NULL,
        ModelVersion NVARCHAR(64) NOT NULL,
        FeatureBuilderVersion NVARCHAR(64) NOT NULL,
        GroupPredictionStatus NVARCHAR(64) NOT NULL,
        AutomaticControlStatus NVARCHAR(32) NOT NULL CONSTRAINT DF_GroupResult_Auto DEFAULT(N'DISABLED'),
        Revision INT NOT NULL CONSTRAINT DF_GroupResult_Revision DEFAULT(1),
        CreatedAt DATETIME2(3) NOT NULL CONSTRAINT DF_GroupResult_Created DEFAULT(SYSUTCDATETIME()),
        UpdatedAt DATETIME2(3) NOT NULL CONSTRAINT DF_GroupResult_Updated DEFAULT(SYSUTCDATETIME()),
        CONSTRAINT FK_MooneyGroupResult_Model FOREIGN KEY(ModelID) REFERENCES dbo.MooneyModel(ModelID),
        CONSTRAINT UQ_MooneyGroupResult_Contract UNIQUE(LabelGroupID, ModelVersion, FeatureBuilderVersion),
        CONSTRAINT CK_MooneyGroupResult_Auto CHECK(AutomaticControlStatus = N'DISABLED')
    );
END;
GO

IF OBJECT_ID(N'dbo.MooneyFeatureQualityAudit', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.MooneyFeatureQualityAudit
    (
        QualityAuditID BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT PK_MooneyFeatureQualityAudit PRIMARY KEY,
        OrderID NVARCHAR(64) NOT NULL,
        BatchID NVARCHAR(64) NOT NULL,
        PalletID NVARCHAR(128) NULL,
        LabelGroupID NVARCHAR(256) NULL,
        FeatureQualityStatus NVARCHAR(64) NOT NULL,
        MissingModelColumnCount INT NOT NULL,
        MissingFeatureCellCount INT NOT NULL,
        MissingFeatureCells NVARCHAR(MAX) NULL,
        RouteCompatibleZeroFillCount INT NOT NULL,
        RouteCompatibleZeroFillCells NVARCHAR(MAX) NULL,
        LabColumnsRemoved NVARCHAR(512) NULL,
        FeatureBuilderVersion NVARCHAR(64) NOT NULL,
        EvaluatedAt DATETIME2(3) NOT NULL CONSTRAINT DF_FeatureQuality_Evaluated DEFAULT(SYSUTCDATETIME())
    );
    CREATE INDEX IX_MooneyFeatureQualityAudit_OrderBatch
        ON dbo.MooneyFeatureQualityAudit(OrderID, BatchID, EvaluatedAt DESC);
END;
GO

IF OBJECT_ID(N'dbo.MooneyFeatureParityAudit', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.MooneyFeatureParityAudit
    (
        ParityAuditID BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT PK_MooneyFeatureParityAudit PRIMARY KEY,
        OrderID NVARCHAR(64) NOT NULL,
        BatchID NVARCHAR(64) NOT NULL,
        ReferenceBuilderVersion NVARCHAR(64) NOT NULL,
        CandidateBuilderVersion NVARCHAR(64) NOT NULL,
        ModelFeatureCount INT NOT NULL,
        MissingModelColumnCount INT NOT NULL,
        FailedFeatureCount INT NOT NULL,
        MaxNormalizedFeatureError FLOAT NULL,
        ReferencePrediction FLOAT NULL,
        CandidatePrediction FLOAT NULL,
        PredictionAbsDifference FLOAT NULL,
        PredictionTolerance FLOAT NOT NULL CONSTRAINT DF_Parity_PredTol DEFAULT(0.1),
        FailedFeaturesJson NVARCHAR(MAX) NULL,
        ParityStatus NVARCHAR(64) NOT NULL,
        EvaluatedAt DATETIME2(3) NOT NULL CONSTRAINT DF_Parity_Evaluated DEFAULT(SYSUTCDATETIME()),
        CONSTRAINT CK_MooneyFeatureParityAudit_Status CHECK
            (ParityStatus IN (N'PARITY_PASS', N'BLOCKED_MISSING_FEATURE', N'BLOCKED_FEATURE_DRIFT', N'BLOCKED_PREDICTION_DRIFT'))
    );
    CREATE INDEX IX_MooneyFeatureParityAudit_OrderBatch
        ON dbo.MooneyFeatureParityAudit(OrderID, BatchID, EvaluatedAt DESC);
END;
GO
