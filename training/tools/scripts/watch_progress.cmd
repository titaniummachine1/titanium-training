@echo off
title Quoridor Training Progress
cd /d "%~dp0.."
powershell -NoLogo -ExecutionPolicy Bypass -File "%~dp0watch_progress.ps1"
pause
