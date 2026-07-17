@echo off
:: Reset to epoch-1 weights (22 Jun ~23:51) then run one overnight cycle.
cd /d "%~dp0"
call overnight_training.bat --revert-epoch1
