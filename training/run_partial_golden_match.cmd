@echo off
:: 8 games @ 2s — current v15-frozen vs partial-iter golden (abe9ba5). Each game -> all_games.db
setlocal
cd /d "%~dp0.."

set GOLDEN=engine\worktrees\partial-golden\target\release\titanium.exe
set CURRENT=engine\target\release\titanium.exe

if not exist "%CURRENT%" cargo build --release -p titanium --manifest-path engine\Cargo.toml
if not exist "%GOLDEN%" (
  if not exist engine\worktrees\partial-golden git -C engine worktree add worktrees/partial-golden abe9ba5
  cargo build --release --manifest-path engine\worktrees\partial-golden\Cargo.toml
)

node site\self_match.js ^
  --engine-a titanium-v15-frozen ^
  --engine-b ace-v13-grafted ^
  --bin-a "%CURRENT%" ^
  --bin-b "%GOLDEN%" ^
  --games 8 ^
  --time 2 ^
  --concurrency 4 ^
  --no-ponder ^
  --standalone ^
  --save-games "%~dp0data\partial_golden_vs_frozen.games" ^
  --source-tag v15-frozen-vs-partial-golden-2s

endlocal
