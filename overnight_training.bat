@echo off
:: =============================================================================
:: CONTINUOUS TRAINING POOL — 8 threads play non-stop until Ctrl+C
::
:: Each thread after every game:
::   games.db + labels.db (outcome labels) -> teacher parquet (value_i16)
::
:: Thread 0 also watches the batch counter; at --batch-games (default 512):
::   rebuild feature cache -> train 1 epoch -> deploy -> strength check -> resume
::
:: Positions touched 5+ times in training are retired (position_usage.py).
:: Pool stops if current net win rate vs previous drops below threshold.
:: =============================================================================
cd /d "%~dp0"
set PYTHONPATH=%~dp0training;%PYTHONPATH%
set RUSTFLAGS=-C target-cpu=native

echo.
echo [%date% %time%] Continuous training pool starting...
echo Log: training\data\overnight_logs\continuous_pool.log
echo Press Ctrl+C to stop after the current game / epoch step.
echo.

python -u training\continuous_pool.py --from-frozen --threads 8 --time 2.0 --batch-games 512 %*

if errorlevel 2 (
    echo.
    echo SATURATED — weights getting weaker vs previous checkpoint.
    goto done
)
if errorlevel 1 (
    echo.
    echo FAILED — see training\data\overnight_logs\continuous_pool.log
    goto done
)

echo.
echo Pool exited cleanly.

:done
pause
