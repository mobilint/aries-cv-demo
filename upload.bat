@echo off
setlocal enabledelayedexpansion

set "WEAPON_ASSET_BUCKET_ID=%HF_WEAPON_ASSET_BUCKET_ID%"
if "%WEAPON_ASSET_BUCKET_ID%"=="" set "WEAPON_ASSET_BUCKET_ID=mobilint/aries-weapon-detection-demo-assets"

set "FIRE_ASSET_BUCKET_ID=%HF_FIRE_ASSET_BUCKET_ID%"
if "%FIRE_ASSET_BUCKET_ID%"=="" set "FIRE_ASSET_BUCKET_ID=mobilint/aries-fire-detection-demo-assets"

set "ULTRALYTICS_ASSET_BUCKET_ID=%HF_ULTRALYTICS_ASSET_BUCKET_ID%"
if "%ULTRALYTICS_ASSET_BUCKET_ID%"=="" set "ULTRALYTICS_ASSET_BUCKET_ID=mobilint/aries-ultralytics-demo-assets"

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%HF_UPLOAD_VENV_DIR%"
if "%VENV_DIR%"=="" set "VENV_DIR=%HF_DOWNLOAD_VENV_DIR%"
if "%VENV_DIR%"=="" set "VENV_DIR=%SCRIPT_DIR%.hf_venv"

set "WEAPON_LOCAL_DIR=%SCRIPT_DIR%assets\weapon"
set "FIRE_LOCAL_DIR=%SCRIPT_DIR%assets\fire"
set "ULTRALYTICS_LOCAL_DIR=%SCRIPT_DIR%assets\ultralytics"

where uv >nul 2>nul
if errorlevel 1 (
  echo uv not found. Please install uv before running this script.
  echo See: https://docs.astral.sh/uv/getting-started/installation/
  exit /b 1
)

echo Preparing Hugging Face upload environment: %VENV_DIR%
if not exist "%VENV_DIR%" (
  uv venv "%VENV_DIR%"
  if errorlevel 1 exit /b 1
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
  echo Cannot find venv python at %VENV_PYTHON%
  exit /b 1
)

uv pip install --python "%VENV_PYTHON%" huggingface-hub
if errorlevel 1 exit /b 1

set "HF_CLI=%VENV_DIR%\Scripts\hf.exe"
if not exist "%HF_CLI%" set "HF_CLI=%VENV_DIR%\Scripts\hf"
if not exist "%HF_CLI%" (
  echo Cannot find hf CLI in %VENV_DIR%\Scripts
  exit /b 1
)

call :upload_bucket "weapon" "%WEAPON_ASSET_BUCKET_ID%" "%WEAPON_LOCAL_DIR%"
if errorlevel 1 exit /b 1

call :upload_bucket "fire" "%FIRE_ASSET_BUCKET_ID%" "%FIRE_LOCAL_DIR%"
if errorlevel 1 exit /b 1

call :upload_bucket "ultralytics" "%ULTRALYTICS_ASSET_BUCKET_ID%" "%ULTRALYTICS_LOCAL_DIR%"
if errorlevel 1 exit /b 1

echo All CV assets uploaded and synchronized successfully.
endlocal
exit /b 0

:upload_bucket
set "LABEL=%~1"
set "BUCKET_ID=%~2"
set "LOCAL_DIR=%~3"
set "BUCKET_URI=hf://buckets/%BUCKET_ID%"

if not exist "%LOCAL_DIR%" (
  echo Local directory not found for %LABEL%: %LOCAL_DIR%
  exit /b 1
)

echo Uploading %LABEL% assets to Hugging Face bucket: %BUCKET_URI%
echo Remote files matching included asset patterns but missing locally will be deleted.
"%HF_CLI%" buckets sync "%LOCAL_DIR%" "%BUCKET_URI%" ^
  --delete ^
  --include "mxq/*" ^
  --include "video/*.mp4" ^
  --include "video/**/*.mp4" ^
  --exclude "mxq/.gitkeep" ^
  --exclude "video/.gitkeep"
if errorlevel 1 exit /b 1

echo %LABEL% assets uploaded from %LOCAL_DIR%
exit /b 0