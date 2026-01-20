@echo off
echo Starting VMAT DDPM Training on Windows
echo ========================================

cd /d C:\Users\Bill\vmat-diffusion-project

call C:\pinokio\bin\miniconda\Scripts\activate.bat vmat-win

echo Activated vmat-win environment
echo Python:
python --version

echo.
echo Starting training with data from I:\processed_npz
echo Output will be saved to C:\Users\Bill\vmat-diffusion-project\runs
echo.

python scripts\train_dose_ddpm_v2.py --data_dir I:\processed_npz --epochs 200

pause
