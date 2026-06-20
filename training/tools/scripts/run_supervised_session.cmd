@echo off
title Quoridor Supervised Training
cd /d "%~dp0.."
powershell -NoLogo -ExecutionPolicy Bypass -File "%~dp0run_supervised_session.ps1"
echo.
echo Session ended. Press any key to close.
pause >nul
