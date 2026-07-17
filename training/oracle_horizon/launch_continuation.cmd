@echo off
setlocal
cd /d "%~dp0..\..\"
set TRAINING_PREP_ONLY=0
set PYTHONPATH=%CD%\training
set TITANIUM_BOOK_MODE=off
set TITANIUM_ENGINE_BIN=%CD%\engine\target-catv5-accepted-03856fe\release\titanium.exe
set RUSTFLAGS=-C target-cpu=native
py -3.12 training\oracle_horizon\run_continuation.py %*
exit /b %ERRORLEVEL%
