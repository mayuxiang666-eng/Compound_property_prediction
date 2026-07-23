@echo off
:: ==============================================================================
:: Windows Task Scheduler Registration Script for Daily Mooney Pipeline Runner
:: Schedule: Daily at 03:00 AM
:: ==============================================================================

cd /d "%~dp0.."
set ROOT_DIR=%cd%

echo Registering Windows Task Scheduler Job: MooneyDailyPipeline (Daily at 03:00 AM)...

schtasks /create /tn "MooneyDailyPipeline" /tr "python \"%ROOT_DIR%\server_deployment\daily_pipeline_runner.py\"" /sc daily /st 03:00 /f /ru "SYSTEM"

if %errorlevel% equ 0 (
    echo Task 'MooneyDailyPipeline' registered successfully!
) else (
    echo Failed to register task. Please run this script as Administrator.
)

pause
