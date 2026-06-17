@echo off
title Stop Quoridor Training
cd /d "%~dp0.."
echo Stopping pool, supervisor, coordinator, titanium...
powershell -NoLogo -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'run_swiss_overnight|supervise\.py|overnight_batch|remote_game_worker|run_nnue_cycle|coordinator\.py|run_supervised_session|ka_teacher_worker|ka_value_label' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; cmd /c 'taskkill /F /IM titanium.exe /T >nul 2>nul'; Remove-Item -Force 'training/data/eval_batch.lock' -ErrorAction SilentlyContinue; Write-Host 'All training workers stopped.' -ForegroundColor Green"
