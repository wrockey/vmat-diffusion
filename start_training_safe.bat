@echo off
echo ========================================
echo VMAT DDPM Training - SAFE MODE
echo ========================================
echo.
echo Settings optimized for GPU stability:
echo   - Batch size: 1 (reduced from 2)
echo   - Base channels: 32 (reduced from 48)
echo   - Workers: 0 (Windows compatibility)
echo   - GPU cooling pauses enabled
echo.
echo Monitor GPU temps in GPU-Z - target under 80C
echo ========================================
echo.

cd /d C:\Users\Bill\vmat-diffusion-project

call C:\pinokio\bin\miniconda\Scripts\activate.bat vmat-win

echo Activated vmat-win environment
echo Python:
python --version

echo.
echo Starting training with SAFE settings...
echo Data: I:\processed_npz
echo Output: C:\Users\Bill\vmat-diffusion-project\runs
echo.

python scripts\train_dose_ddpm_v2.py ^
    --data_dir I:\processed_npz ^
    --epochs 200 ^
    --batch_size 1 ^
    --base_channels 32 ^
    --num_workers 0 ^
    --gpu_cooling ^
    --cooling_interval 10 ^
    --cooling_pause 0.5 ^
    --exp_name vmat_dose_ddpm_safe

echo.
echo ========================================
echo Training finished or interrupted
echo ========================================
pause
