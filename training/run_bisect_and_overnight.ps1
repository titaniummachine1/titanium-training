# Bisect search regression (abe9ba5 good .. HEAD bad), then overnight supervised training.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Log = Join-Path $Root "training/data/bisect_overnight.log"
function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Add-Content -Path $Log -Value $line
    Write-Host $line
}

Log "=== bisect + overnight pipeline ==="

$GoldenBin = Join-Path $Root "engine/worktrees/partial-golden/target/release/titanium.exe"
if (-not (Test-Path $GoldenBin)) {
    if (-not (Test-Path "$Root/engine/worktrees/partial-golden")) {
        git -C engine worktree add worktrees/partial-golden abe9ba5
    }
    cargo build --release --manifest-path engine/worktrees/partial-golden/Cargo.toml
}

Log "Rebuild current engine (TT store idx fix applied)..."
cargo build --release -p titanium --manifest-path engine/Cargo.toml

Log "Quick 8g @ 2s: v15-frozen vs golden..."
node site/self_match.js `
    --engine-a titanium-v15-frozen --engine-b ace-v13-grafted --bin-b $GoldenBin `
    --games 8 --time 2 --concurrency 4 --no-ponder --standalone `
    --source-tag v15-frozen-vs-golden-2s `
    --save-games training/data/partial_golden_vs_frozen.games `
    2>&1 | Tee-Object -FilePath training/data/quick_8game.log

Log "git bisect in engine/ (abe9ba5=good, HEAD=bad)..."
Push-Location engine
$stashName = "bisect-wip-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
git stash push -u -m $stashName 2>&1 | Tee-Object -Append $Log
git bisect reset 2>$null | Out-Null
git bisect start HEAD abe9ba5 2>&1 | Tee-Object -Append $Log
git bisect run python ../training/bisect_engine_step.py 2>&1 | Tee-Object -Append $Log
$culprit = git rev-parse HEAD 2>$null
git bisect log 2>&1 | Tee-Object -Append $Log | Out-Null
git bisect reset 2>&1 | Out-Null
git stash pop 2>&1 | Tee-Object -Append $Log
Pop-Location
Log "bisect finished; first bad near: $culprit"

Log "Rebuild after stash pop (TT fix + v15 WIP)..."
cargo build --release -p titanium --manifest-path engine/Cargo.toml 2>&1 | Tee-Object -Append $Log

Log "Post-fix verify 8g..."
node site/self_match.js `
    --engine-a titanium-v15-frozen --engine-b ace-v13-grafted --bin-b $GoldenBin `
    --games 8 --time 2 --concurrency 4 --no-ponder --standalone `
    --source-tag post-fix-vs-golden-2s `
    --save-games training/data/post_fix_verify.games `
    2>&1 | Tee-Object -Append $Log

Log "Launching overnight supervised session..."
Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoLogo", "-ExecutionPolicy", "Bypass",
    "-File", "`"$Root/training/run_supervised_session.ps1`""
) -WorkingDirectory $Root

Log "Done - log: $Log"
