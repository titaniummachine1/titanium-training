# Training tools

Supported operator and developer utilities. These are **not** the canonical production trainer — use `python training/nnue_cli.py` for value-NNUE workflows.

| Path | Purpose | Required on Oracle |
| ---- | ------- | ------------------ |
| `operations/` | Overnight pool, supervision, benchmarks | Optional |
| `datagen/` | Game ingest and eval-batch expansion | Optional |
| `engine_parity/` | Binary stamp, parity, preflight | Smoke/train |
| `dataset/` | Position-store CLI wrapper | Dataset admin only |
| `maintenance/` | Housekeeping, manifest, regression triage | No |
| `analysis/` | Profiling parse, field visualization | No |
| `scripts/` | PowerShell/cmd launchers | No |

Experimental LMR and feature probes live under `training/experiments/`, not here.
