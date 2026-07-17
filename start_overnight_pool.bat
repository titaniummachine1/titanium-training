@echo off
:: Launch clean database-first runtime: oracle_importer + local_game_pool + training_coordinator.
:: Each service enforces its own single-instance lock; if already running it exits cleanly.
:: Never launches legacy coupled continuous_pool.py.
cd /d "%~dp0"

echo [%date% %time%] Starting database-first runtime services...
echo   oracle_importer.py  - Oracle imports only
echo   local_game_pool.py  - local generation only
echo   training_coordinator.py - DB-triggered NNUE training/promotion

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0training\tools\start_oracle_importer_detached.ps1"
if errorlevel 1 echo   oracle_importer: already running or token missing

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0training\tools\start_local_game_pool_detached.ps1"
if errorlevel 1 echo   local_game_pool: already running

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0training\tools\start_training_coordinator_detached.ps1"
if errorlevel 1 echo   training_coordinator: already running

echo [%date% %time%] Done. Logs: training\data\overnight_logs\
exit /b 0
