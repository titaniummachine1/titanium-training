---
description: One-shot consolidated training status (processes, coordinator, accepted epochs, pool, Oracle importer, remote Oracle worker, strength gate)
---

Run this and report the output directly to the user, formatted clearly. Do not re-derive any of this information with separate tool calls — this single script already consolidates everything.

```bash
cd "C:\gitProjects\Quoridor best AI" && python training/tools/training_status.py
```

If a process shows DEAD, flag it and offer to restart it via the matching script in `training/tools/start_*_detached.ps1`. If the strength gate has crossed its `min_games` threshold, note whether the score passed or failed against `min_score`.
