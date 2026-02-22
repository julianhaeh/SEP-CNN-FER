@echo off
cd /d "%~dp0..\"
python -m Demo.infer_webcam --weights ".\Experiments\Models\ReducedClassifier_Weighted_CE_Weighted_Acc_72.84_Model.pth" --flip
pause