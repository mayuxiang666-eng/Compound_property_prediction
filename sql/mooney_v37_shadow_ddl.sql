/*
  Mooney V3.7 Shadow database bootstrap
  Target: EFTMMS059\AIDATASERVER
  Database: MooneyPredictionV37Shadow

  This script is intentionally isolated from MixingCurveData and existing
  production databases. It creates V3.7 storage plus reserved V3.8 tables.
  Execute only with an authorized database deployment account.
*/

USE [master];
GO

IF DB_ID(N'MooneyPredictionV37Shadow') IS NULL
BEGIN
    CREATE DATABASE [MooneyPredictionV37Shadow];
END;
GO

USE [MooneyPredictionV37Shadow];
GO

IF OBJECT_ID(N'dbo.MooneyModel', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.MooneyModel
    (
        ModelID       INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_MooneyModel PRIMARY KEY,
        ModelCode     NVARCHAR(64) NOT NULL,
        ModelVersion  NVARCHAR(64) NOT NULL,
        DisplayName   NVARCHAR(128) NULL,
        PackagePath   NVARCHAR(512) NOT NULL,
        PackageHash   CHAR(64) NULL,
        FeatureSchemaHash CHAR(64) NULL,
        IsActive      BIT NOT NULL CONSTRAINT DF_MooneyModel_IsActive DEFAULT (0),
        ReleaseType   NVARCHAR(32) NOT NULL CONSTRAINT DF_MooneyModel_ReleaseType DEFAULT (N'SHADOW'),
        CreatedAt     DATETIME2(3) NOT NULL CONSTRAINT DF_MooneyModel_CreatedAt DEFAULT (SYSUTCDATETIME()),
        Notes         NVARCHAR(512) NULL,
        CONSTRAINT UQ_MooneyModel_CodeVersion UNIQUE (ModelCode, ModelVersion),
        CONSTRAINT CK_MooneyModel_ReleaseType CHECK (ReleaseType IN (N'SHADOW', N'PRODUCTION'))
    );
END;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'UX_MooneyModel_Active' AND object_id = OBJECT_ID(N'dbo.MooneyModel'))
BEGIN
    CREATE UNIQUE INDEX UX_MooneyModel_Active
        ON dbo.MooneyModel(IsActive)
        WHERE IsActive = 1;
END;
GO

IF OBJECT_ID(N'dbo.MooneyResult', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.MooneyResult
    (
        ResultID       BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT PK_MooneyResult PRIMARY KEY,
        OrderID        NVARCHAR(64) NOT NULL,
        BatchID        NVARCHAR(64) NOT NULL,
        SourceBatchNumber NVARCHAR(128) NULL,
        PalletID       NVARCHAR(128) NULL,
        LabelGroupID   NVARCHAR(256) NULL,
        CompName       NVARCHAR(128) NOT NULL,
        CompDescription NVARCHAR(256) NULL,
        Mixer          NVARCHAR(128) NULL,
        MaterialSystem NVARCHAR(32) NULL,
        PhaseRoute     NVARCHAR(32) NULL,
        ProductTime    DATETIME2(3) NULL,
        InputTime      DATETIME2(3) NOT NULL CONSTRAINT DF_MooneyResult_InputTime DEFAULT (SYSUTCDATETIME()),
        ModelID        INT NOT NULL,
        ModelVersion   NVARCHAR(64) NOT NULL,
        Stage1Pred     FLOAT NOT NULL,
        Stage1bBias    FLOAT NOT NULL,
        Stage2Delta    FLOAT NOT NULL,
        RawPrediction  FLOAT NOT NULL,
        CalibrationOffset FLOAT NOT NULL CONSTRAINT DF_MooneyResult_CalibrationOffset DEFAULT (0.0),
        FinalPrediction FLOAT NOT NULL,
        LabStatus      NVARCHAR(32) NOT NULL CONSTRAINT DF_MooneyResult_LabStatus DEFAULT (N'PENDING'),
        FeatureMissingCount INT NOT NULL CONSTRAINT DF_MooneyResult_FeatureMissingCount DEFAULT (0),
        CurvePointCount INT NULL,
        FeatureBuilderVersion NVARCHAR(64) NULL,
        CreatedAt      DATETIME2(3) NOT NULL CONSTRAINT DF_MooneyResult_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT FK_MooneyResult_Model FOREIGN KEY (ModelID) REFERENCES dbo.MooneyModel(ModelID),
        CONSTRAINT UQ_MooneyResult_OrderBatch UNIQUE (OrderID, BatchID),
        CONSTRAINT CK_MooneyResult_FinalEquation CHECK (ABS(FinalPrediction - (RawPrediction + CalibrationOffset)) < 0.001),
        CONSTRAINT CK_MooneyResult_LabStatus CHECK (LabStatus IN (N'PENDING', N'AVAILABLE', N'REVISED', N'REVOKED'))
    );
END;
GO

IF OBJECT_ID(N'dbo.MooneyPredictMiss', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.MooneyPredictMiss
    (
        MissID       BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT PK_MooneyPredictMiss PRIMARY KEY,
        OrderID      NVARCHAR(64) NOT NULL,
        BatchID      NVARCHAR(64) NOT NULL,
        PalletID     NVARCHAR(128) NULL,
        CompName     NVARCHAR(128) NULL,
        Mixer        NVARCHAR(128) NULL,
        ProductTime  DATETIME2(3) NULL,
        MissReason   NVARCHAR(64) NOT NULL,
        MissDetail   NVARCHAR(2000) NULL,
        Retryable    BIT NOT NULL CONSTRAINT DF_MooneyPredictMiss_Retryable DEFAULT (0),
        RetryCount   INT NOT NULL CONSTRAINT DF_MooneyPredictMiss_RetryCount DEFAULT (0),
        Status       NVARCHAR(32) NOT NULL CONSTRAINT DF_MooneyPredictMiss_Status DEFAULT (N'NEW'),
        ModelVersion NVARCHAR(64) NULL,
        DetectedAt   DATETIME2(3) NOT NULL CONSTRAINT DF_MooneyPredictMiss_DetectedAt DEFAULT (SYSUTCDATETIME()),
        LastRetryAt  DATETIME2(3) NULL,
        CONSTRAINT CK_MooneyPredictMiss_Status CHECK (Status IN (N'NEW', N'RETRYING', N'RESOLVED', N'PERMANENT_FAILURE'))
    );
END;
GO

/* Reserved for V3.8. No automatic calibration is enabled by this script. */
IF OBJECT_ID(N'dbo.MooneyCalibrationState', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.MooneyCalibrationState
    (
        CalibrationStateID BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT PK_MooneyCalibrationState PRIMARY KEY,
        ScopeKey NVARCHAR(256) NOT NULL,
        BiasOffset FLOAT NOT NULL CONSTRAINT DF_MooneyCalibrationState_BiasOffset DEFAULT (0.0),
        LastConsumedGroupRawError FLOAT NULL,
        Status NVARCHAR(32) NOT NULL CONSTRAINT DF_MooneyCalibrationState_Status DEFAULT (N'SHADOW'),
        UpdatedAt DATETIME2(3) NOT NULL CONSTRAINT DF_MooneyCalibrationState_UpdatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT UQ_MooneyCalibrationState_ScopeKey UNIQUE (ScopeKey)
    );
END;
GO

IF OBJECT_ID(N'dbo.MooneyLabFeedbackQueue', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.MooneyLabFeedbackQueue
    (
        FeedbackID BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT PK_MooneyLabFeedbackQueue PRIMARY KEY,
        LabSourceKey NVARCHAR(256) NOT NULL,
        LabelGroupID NVARCHAR(256) NOT NULL,
        LabActualMNY FLOAT NOT NULL,
        GroupRawError FLOAT NULL,
        GroupOnlineError FLOAT NULL,
        FeedbackStatus NVARCHAR(32) NOT NULL CONSTRAINT DF_MooneyLabFeedbackQueue_Status DEFAULT (N'NEW'),
        CreatedAt DATETIME2(3) NOT NULL CONSTRAINT DF_MooneyLabFeedbackQueue_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT UQ_MooneyLabFeedbackQueue_LabSourceKey UNIQUE (LabSourceKey),
        CONSTRAINT UQ_MooneyLabFeedbackQueue_LabelGroup UNIQUE (LabelGroupID)
    );
END;
GO

IF OBJECT_ID(N'dbo.MooneyCalibrationEventLog', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.MooneyCalibrationEventLog
    (
        EventID BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT PK_MooneyCalibrationEventLog PRIMARY KEY,
        ScopeKey NVARCHAR(256) NOT NULL,
        FeedbackID BIGINT NULL,
        PreviousBiasOffset FLOAT NOT NULL,
        NewBiasOffset FLOAT NOT NULL,
        AlphaUsed FLOAT NULL,
        ChangePointDetected BIT NOT NULL CONSTRAINT DF_MooneyCalibrationEventLog_ChangePoint DEFAULT (0),
        EventStatus NVARCHAR(32) NOT NULL CONSTRAINT DF_MooneyCalibrationEventLog_Status DEFAULT (N'SHADOW'),
        CreatedAt DATETIME2(3) NOT NULL CONSTRAINT DF_MooneyCalibrationEventLog_CreatedAt DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT FK_MooneyCalibrationEventLog_Feedback FOREIGN KEY (FeedbackID) REFERENCES dbo.MooneyLabFeedbackQueue(FeedbackID)
    );
END;
GO

/* Seed model metadata only after the package has been copied and hashed. */
-- INSERT dbo.MooneyModel (...)
-- VALUES (N'HybridUnified', N'V3.7.0', ..., N'SHADOW');

