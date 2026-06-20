# Architecture

Titanium combines a **fast HalfPW / NNUE-style evaluator** with **alpha-beta search** for Quoridor.

## Design principles

- The network provides a geometric prior and search-risk hints; search remains authoritative for tactics and legality.
- Field-plane HalfPW (11 planes, ws[14]=legal walls/128) is the frozen trunk contract.
- Dual-head roadmap: **value** first, then conservative **LMR / search-pressure** sidecars — not a monolithic Ka-style CNN.

## Full handoff

See sections above for NN + search design. Legacy handoff notes were consolidated into this document during the training root cleanup.

## Current training sequence

1. Train and validate **value NNUE** on audited teacher data (Oracle first run).
2. Freeze selected value weights + search configuration.
3. Generate **LMR supervision** only against the frozen environment.
4. Start with **+1-ply reduction prediction**; expand to magnitude only after validation.

See [ROADMAP.md](ROADMAP.md) for LMR status.
