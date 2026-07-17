@echo off
:: Rebuild website WASM from repo-root engine (embeds current net_weights.bin = epoch-1 live).
:: Does NOT touch net_weights_frozen.bin (v15-frozen / ace-v13 baselines).
cd /d "%~dp0"
set RUSTFLAGS=
echo Building WASM from engine\ (live weights baked in at compile time)...
cd site\web
call npm run build:wasm
if errorlevel 1 exit /b 1
echo.
echo WASM updated in site\web\src\wasm\titanium\ and site\web\src\wasm\titanium-v17\
echo Restart dev server (npm run dev) or rebuild pages (npm run build:pages) to refresh the menu.
echo Frozen baseline engines still use net_weights_frozen.bin in native builds only.
pause
