@echo off
cd /d "%~dp0"
python "training\tools\analysis\cat_viz.py"
if errorlevel 1 pause
