@echo off
cd /d "%~dp0..\"
python -m Demo.infer_webcam --flip
pause