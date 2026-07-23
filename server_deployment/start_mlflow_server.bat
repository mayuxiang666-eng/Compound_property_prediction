@echo off
TITLE MLflow Web Server (Port 9999)
cd /d "%~dp0.."
set ROOT_DIR=%cd%
set DB_PATH=%ROOT_DIR%\mlflow.db
set MLRUNS_PATH=%ROOT_DIR%\mlruns

echo ==============================================================================
echo Starting MLflow Web UI Server on Port 9999...
echo Root Directory: %ROOT_DIR%
echo Database Path: %DB_PATH%
echo Artifacts Path: %MLRUNS_PATH%
echo ==============================================================================

python -m mlflow server --host 0.0.0.0 --port 9999 --backend-store-uri "sqlite:///%DB_PATH%" --default-artifact-root "%MLRUNS_PATH%"
pause
