@echo off
:: =============================================================================
:: SELF-PLAY LAUNCHER
:: Generates training games + verifies strength vs frozen baseline
:: =============================================================================
:: Edit these values, then double-click or run from cmd.

:: Parallel game instances.
:: Each instance = 2 engine processes (one per side).
:: i7-4900MQ (8 HT threads) with trainer running: use 3
:: Without trainer: use 4
set THREADS=4

:: Seconds the engine gets per move.
:: 2.0 = stronger play; 1.0 = faster games; 0.5 = light load
set TIME=2.0

:: Training games between each strength-verification game.
:: 3 = play 3 training games, then 1 verify game, repeat.
set VERIFY_RATIO=3

:: =============================================================================
:: (do not edit below this line)
:: =============================================================================
cd /d "%~dp0"
echo.
echo Self-play: %THREADS% thread(s), %TIME%s/move, verify every %VERIFY_RATIO%+1 games
echo Logs: training\data\self_play_1.log .. self_play_%THREADS%.log
echo Baseline: training\runs\value_oracle\net_weights_baseline.bin (previous best)
echo.
python training\run_selfplay.py --threads %THREADS% --time %TIME% --verify-ratio %VERIFY_RATIO% --baseline-weights training\runs\value_oracle\net_weights_baseline.bin
pause
