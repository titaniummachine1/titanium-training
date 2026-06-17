@echo off
title Import game from clipboard
cd /d "%~dp0.."
python training\import_clipboard_game.py %*
if errorlevel 1 pause
