@echo off
cd /d "%~dp0..\"

set "DIR=%~1"
if "%DIR%"=="" (
  echo Drag an image folder onto this .bat
  pause
  exit /b 1
)

set "OUTDIR=%cd%\outputs"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

for %%I in ("%DIR%") do set "BASENAME=%%~nI"
set "OUT=%OUTDIR%\preds_%BASENAME%.csv"

echo Input folder: %DIR%
echo Output CSV  : %OUT%

python -m Demo.infer_csv --input_dir "%DIR%" --output_csv "%OUT%"

echo.
echo Done. Output saved to:
echo %OUT%
explorer "%OUTDIR%"
pause