@echo off
title Quoridor Pool Watch
cd /d "%~dp0.."
powershell -NoLogo -ExecutionPolicy Bypass -File "%~dp0pool_watch.ps1"
