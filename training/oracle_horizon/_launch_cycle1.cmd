@echo off
setlocal
cd /d "c:\gitProjects\Quoridor best AI"
set PYTHONPATH=c:\gitProjects\Quoridor best AI\training
set TITANIUM_BOOK_MODE=off
set TRAINING_PREP_ONLY=0
set PYTHONUNBUFFERED=1
set RUSTFLAGS=-C target-cpu=native
echo CYCLE1_START %DATE% %TIME%
py -3.12 -u training\oracle_horizon\run_cycle1.py --out-dir training\runs\oracle_horizon_pilot_v1\cycle1 --games 100 --workers 4 --time-sec 0.8 --max-positions 10000 --max-cpu-hours 4.0 --skip-train --weights training\runs\v16\accepted\epoch_0003.bin --engine engine\target-catv5-accepted-03856fe\release\titanium.exe
echo CYCLE1_EXIT=%ERRORLEVEL%
endlocal
