@echo off
:: Overnight: titanium-v15 vs ace-v13-ti-pure (JS v13 + O1 movegen baseline)
:: Runs until you close the window or press Ctrl+C.
:: Every completed game is saved immediately — no data lost on exit.
::
:: Results:  training\data\overnight.games  (raw GAME/RESULT lines)
::           training\data\all_games.db     (training DB, auto-updated per game)
:: Progress: check the window — one line per game with running score + Elo

setlocal
cd /d "%~dp0"

set GAMES=training\data\overnight.games
node site\self_match.js --games 9999 --time 5 --concurrency 4 ^
  --engine-a titanium-v15 --engine-b ace-v13-ti-pure ^
  --save-games "%GAMES%" --source-tag overnight-v15-vs-ti-pure %*
