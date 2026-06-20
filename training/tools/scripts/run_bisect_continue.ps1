# Continue bisect only (after quick 8g already done).
$ErrorActionPreference = "Continue"
$Root = "c:\gitProjects\Quoridor best AI"
$Log = "$Root/training/data/bisect_overnight.log"

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Add-Content -Path $Log -Value $line
    Write-Host $line
}

Log "=== resume bisect ==="
$env:RUSTFLAGS = "-C target-cpu=native"
Push-Location "$Root/engine"
git stash push -u -m "bisect-wip-resume" 2>&1 | Out-File -Append $Log
git bisect reset 2>&1 | Out-File -Append $Log
git bisect start HEAD abe9ba5 2>&1 | Out-File -Append $Log
git bisect run python ../training/bisect_engine_step.py 2>&1 | Out-File -Append $Log
$culprit = git rev-parse HEAD 2>$null
git bisect log 2>&1 | Out-File -Append $Log
git bisect reset 2>&1 | Out-File -Append $Log
git stash pop 2>&1 | Out-File -Append $Log
Pop-Location
Log "bisect culprit: $culprit"

Log "Rebuild post-stash-pop..."
Set-Location $Root
cargo build --release -p titanium --manifest-path engine/Cargo.toml 2>&1 | Out-File -Append $Log

Log "Post-fix verify 8g..."
node site/self_match.js `
    --engine-a titanium-v15-frozen --engine-b ace-v13-grafted `
    --bin-b "$Root/engine/worktrees/partial-golden/target/release/titanium.exe" `
    --games 8 --time 2 --concurrency 4 --no-ponder --standalone `
    --source-tag post-fix-vs-golden-2s `
    --save-games training/data/post_fix_verify.games `
    2>&1 | Out-File -Append $Log

Log "Starting overnight supervised session..."
Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoLogo", "-ExecutionPolicy", "Bypass",
    "-File", "`"$Root/training/run_supervised_session.ps1`""
) -WorkingDirectory $Root
Log "Done"
