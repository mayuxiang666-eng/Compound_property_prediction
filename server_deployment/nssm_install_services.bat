@echo off
:: ==============================================================================
:: NSSM Windows Service Registration Script for MLflow Server (Port 9999)
:: Run this script as Administrator on Windows Server
:: ==============================================================================

cd /d "%~dp0.."
set ROOT_DIR=%cd%
set DB_PATH=%ROOT_DIR%\mlflow.db
set MLRUNS_PATH=%ROOT_DIR%\mlruns
set PYTHON_EXE=python.exe

echo ==============================================================================
echo Installing MLflow Web Server as a Windows Service via NSSM (Port 9999)...
echo Root Directory: %ROOT_DIR%
echo Database Path: %DB_PATH%
echo Artifacts Path: %MLRUNS_PATH%
echo ==============================================================================

:: Check if nssm is available
where nssm >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] NSSM.exe was not found in System PATH.
    echo Please download NSSM from https://nssm.cc/download and place nssm.exe into C:\Windows\System32 or in this directory.
    pause
    exit /b 1
)

:: Stop existing service if running
nssm stop MLflowWebServer >nul 2>nul
nssm remove MLflowWebServer confirm >nul 2>nul

:: Install MLflow Service on Port 9999 with absolute DB path
nssm install MLflowWebServer "%PYTHON_EXE%" "-m mlflow server --host 0.0.0.0 --port 9999 --backend-store-uri ""sqlite:///%DB_PATH%"" --default-artifact-root ""%MLRUNS_PATH%"""
nssm set MLflowWebServer AppDirectory "%ROOT_DIR%"
nssm set MLflowWebServer DisplayName "Mooney Viscosity MLflow UI (Port 9999)"
nssm set MLflowWebServer Description "MLflow Experiment Tracking and Model Registry Server for Mooney Viscosity Pipeline"
nssm set MLflowWebServer Start SERVICE_AUTO_START

:: Start the service
nssm start MLflowWebServer

echo ==============================================================================
echo MLflow Windows Service successfully installed and started!
echo Access MLflow UI at: http://localhost:9999
echo ==============================================================================
pause
