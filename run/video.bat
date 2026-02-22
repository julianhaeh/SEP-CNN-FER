@echo off
cd /d "%~dp0..\"

set "IN=%~1"
if "%IN%"=="" (
  echo Drag a video file onto this .bat
  pause
  exit /b 1
)

set "OUTDIR=%cd%\outputs"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set "OUT=%OUTDIR%\%~n1_out.mp4"

echo Input : %IN%
echo Output: %OUT%

python -m Demo.infer_video --input "%IN%" --output "%OUT%"

echo.
echo Done. Output saved to:
echo %OUT%
explorer "%OUTDIR%"
pause