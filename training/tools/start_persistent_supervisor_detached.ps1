param(
    [switch]$BootstrapLive
)

Write-Host "Deprecated: persistent_supervisor is not part of the database-first runtime."
Write-Host "Use start_overnight_pool.ps1 to start oracle_importer, local_game_pool, training_coordinator."
exit 0
