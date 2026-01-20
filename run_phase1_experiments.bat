@echo off
REM Phase 1 DDPM Optimization Experiments - Windows Batch File
REM Run from: C:\Users\Bill\vmat-diffusion-project
REM Environment: vmat-win (Pinokio)

echo ============================================================
echo PHASE 1 DDPM OPTIMIZATION EXPERIMENTS
echo ============================================================
echo.

REM Activate the vmat-win conda environment
call C:\pinokio\bin\miniconda\Scripts\activate.bat vmat-win

REM Change to project directory
cd /d C:\Users\Bill\vmat-diffusion-project

echo Running Phase 1 experiments...
echo Checkpoint: runs\vmat_dose_ddpm\checkpoints\best-epoch=015-val\mae_gy=12.19.ckpt
echo Data: I:\processed_npz
echo Output: experiments\phase1_sampling
echo.

python scripts\run_phase1_experiments.py ^
    --checkpoint runs\vmat_dose_ddpm\checkpoints\best-epoch=015-val\mae_gy=12.19.ckpt ^
    --data_dir I:\processed_npz ^
    --output_dir experiments\phase1_sampling ^
    --steps "50,100,250,500,1000" ^
    --n_samples "1,3,5,10"

echo.
echo ============================================================
echo Phase 1 experiments complete!
echo Check experiments\phase1_sampling for results
echo ============================================================
pause
