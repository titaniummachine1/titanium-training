# Post-completion checklist for position_teacher_store friend import.
# Run only after the production upsert import reaches 20/20 shards and exits 0.
param(
    [string]$TeacherDb = "training/data/canonical/position_teacher_store.db",
    [string]$SidecarDir = "training/data/canonical/teacher_sidecars",
    [string]$FriendInput = "KaAiData/ANOTHER TRAINING DAT ASTUFF SUPER USEFULL/selfplay_iters_000001_000020"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$reportDir = "training/data/position_store_reports"
$backupDir = "training/data/archive/backups"
New-Item -ItemType Directory -Force -Path $reportDir, $backupDir | Out-Null

Write-Host "=== 1. Final audit ==="
python -m titanium_training.store.cli --teacher-db $TeacherDb audit-teacher-store `
| Tee-Object "$reportDir/teacher_audit_post_import_$stamp.json"

Write-Host "=== 2. Backup teacher store + sidecar manifest ==="
$backupDb = "$backupDir/position_teacher_store_post_friend_$stamp.db"
Copy-Item $TeacherDb $backupDb
Get-ChildItem $SidecarDir -Recurse -File | ForEach-Object {
    [PSCustomObject]@{ Path = $_.FullName; Bytes = $_.Length; Sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash }
} | ConvertTo-Json -Depth 3 | Set-Content "$reportDir/teacher_sidecars_manifest_$stamp.json"
Write-Host "Backed up DB -> $backupDb"

Write-Host "=== 3. Idempotence proof (no-op re-import) ==="
$before = python -c "from pathlib import Path; from titanium_training.store.lib import db_summary; import json; print(json.dumps(db_summary(Path('$TeacherDb'))))"
python -m titanium_training.store.cli --teacher-db $TeacherDb import-friend-rust --threads 8 2>&1 `
| Tee-Object "$reportDir/teacher_idempotence_rerun_$stamp.log"
$after = python -c "from pathlib import Path; from titanium_training.store.lib import db_summary; import json; print(json.dumps(db_summary(Path('$TeacherDb'))))"
if ($before -ne $after) { throw "Idempotence failed: db_summary changed" }
Write-Host "Idempotence OK: db_summary unchanged"

Write-Host "=== 4. Semantic checksum snapshot (canonical order) ==="
python -m titanium_training.store.cli teacher-semantic-checksum --teacher-db $TeacherDb `
| Tee-Object "$reportDir/teacher_semantic_baseline_$stamp.json"

Write-Host "Done. Keep $backupDb as correctness reference."
