param(
    [int]$Workers = 4,
    [int]$EvalTimeoutSec = 900
)

Write-Host "Deprecated: safe_rebuild is optional cache acceleration, not a training prerequisite."
Write-Host "Database-first runtime does not start cache rebuilds."
exit 0
