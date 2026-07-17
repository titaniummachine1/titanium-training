@echo off
:: =============================================================================
:: TRAINING LAUNCHER
:: Trains the value NNUE. Auto-resumes from the latest checkpoint.
:: Uses 1 CPU thread (optimal for this small network on i7-4900MQ).
::
:: DATA PRIORITY:
::   1. teacher_dataset_extended  — if you have run extend_teacher_dataset.py
::      (adds zero.ink + self-play positions on top of the base dataset)
::   2. teacher_dataset           — original 2.28M position dataset (fallback)
::
:: To add new zero.ink / self-play positions to training:
::   python training\extend_teacher_dataset.py
::   (re-run any time; it is incremental — only adds positions not already there)
:: =============================================================================
set PATIENCE=200
set BATCH=512
set CKPT_STEPS=1000

:: =============================================================================
cd /d "%~dp0"
set PYTHONPATH=%~dp0training;%PYTHONPATH%

:: Pick best available dataset
set DATA=training\data\teacher_dataset
if exist "training\data\teacher_dataset_extended\manifest.json" (
    set DATA=training\data\teacher_dataset_extended
    echo Using extended dataset (zero.ink + self-play included^)
) else (
    echo Using base teacher_dataset. Run: python training\extend_teacher_dataset.py to add new data.
)

echo.
echo Training value NNUE — auto-resuming from training\runs\value_oracle
echo data=%DATA%  patience=%PATIENCE%  batch=%BATCH%
echo.

python -u training\titanium_training\training\trainer.py ^
    --data          %DATA% ^
    --max-samples   200000 ^
    --coverage-min  0.999 ^
    --resume ^
    --out-dir       training\runs\value_oracle ^
    --cpu ^
    --epochs        99999 ^
    --patience      %PATIENCE% ^
    --batch         %BATCH% ^
    --checkpoint-steps %CKPT_STEPS% ^
    --val-split     0.05 ^
    --min-val       64 ^
    --seed          0

echo.
echo Training finished (or stopped early by patience).
pause
