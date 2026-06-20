# Technical debt (residual)

Honest list after the Oracle bookkeeping pass. Not zero debt.

| Item | Severity | Location | Why not fixed now | Recommended action |
| ---- | -------- | -------- | ----------------- | ------------------ |
| Teacher-value featurization in trainer | **Partial** — game_store prefix index + eval-batch; pathless friend rows still blocked | `training/titanium_training/data/teacher_value.py` | Engine `eval-packed-batch` or offline feature cache for full corpus |
| Duplicate training markdown | medium | `training/*.md` (legacy) | Consolidated into `docs/` | Delete remaining duplicates after link sweep |
| `coordinator/` submodule mapping missing | low | `.gitmodules` | Unrelated to Oracle value run | Fix submodule entry or remove empty coordinator checkout |
| Hard-coded Windows paths in legacy scripts | medium | various `training/*.ps1` | Local operator scripts | Parameterize or document as Windows-only |
| `nnue_cli.py train` uses game store only | medium | `training/nnue_cli.py` | Documented in ROADMAP | Extend when teacher-value loader lands |
| Inventory script heuristic classification | low | `scripts/maintenance/build_inventory.py` | First-pass automation | Refine with CI import graph later |
