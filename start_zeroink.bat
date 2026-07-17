@echo off
:: =============================================================================
:: ZERO.INK GAME COLLECTOR
:: Collects self-play games from the zero.ink AlphaZero API.
:: Stops cleanly when rate-limited; run again to resume from checkpoint.
:: After collection, imports games into the canonical training DB automatically.
:: =============================================================================
:: Edit these values.

:: Parallel workers (each makes independent API calls).
:: zero.ink has a per-IP rate limit — start with 1-2, increase if no 429s.
:: Each worker: ~2.5s/request + ~45s/game → ~1 game/min per worker.
set WORKERS=2

:: Total games to collect this session (leave blank for unlimited until rate-limited).
:: Example: set GAMES=100
set GAMES=

:: =============================================================================
:: (do not edit below this line)
:: =============================================================================
cd /d "%~dp0"
echo.
echo Zero.ink collector: %WORKERS% worker(s)
if "%GAMES%"=="" (
    echo Games: unlimited (will stop on rate limit)
) else (
    echo Games: %GAMES%
)
echo Output: training\data\zeroink_games\
echo.

if "%GAMES%"=="" (
    python training\collect_zeroink.py --workers %WORKERS%
) else (
    python training\collect_zeroink.py --workers %WORKERS% --games %GAMES%
)

echo.
echo Done. Games are now in the canonical DB (training\data\canonical\).
pause
