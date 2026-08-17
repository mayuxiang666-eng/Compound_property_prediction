@echo off
setlocal

rem V3.7 Shadow Production Service Entrypoint
set "RELEASE_ROOT=D:\MNY master batch prediction model pipline\releases\V3.7.0-shadow-20260805\app"
set "MOONEY_RELEASE_TYPE=SHADOW"
set "MOONEY_PIPELINE_ROOT=D:\MNY master batch prediction model pipline"
set "MOONEY_MODEL_PACKAGE_DIR=%RELEASE_ROOT%\models\v37_production_model_package"
set "MOONEY_MMS_CREDENTIALS_PATH=D:\MNY master batch prediction model pipline\Mooney_Prediction_Pipeline\data_processing\credentials.json"
set "MOONEY_RESULT_CONNECTION_STRING=Driver={ODBC Driver 17 for SQL Server};Server=EFTMMS059\AIDATASERVER;Database=MooneyPredictionV37Shadow;Trusted_Connection=yes;Encrypt=yes;TrustServerCertificate=yes;"
set "PREDICT_POLL_SECONDS=5"
set "PYTHONPATH=%RELEASE_ROOT%;D:\MNY master batch prediction model pipline;%PYTHONPATH%"

python "%RELEASE_ROOT%\v37_shadow_runner.py" --write
exit /b %ERRORLEVEL%
