# Titanium architecture (v1.0)

Frozen architectural laws for the Quoridor engine. Enforce on every change.

Training lives at **repo-root** `training/` — the engine crate must not depend on the trainer.

## Repository layout (navigability)

Two cleanups, both required:

1. **Code architecture** — `engine/src/titanium/{position,eval,endgame,search,…}`
2. **Repository architecture** — default folders show the *current* engine, not six months of Cargo target dirs

```text
repo/
  engine/           # permanent: Titanium only — src, benches, examples, scripts, docs, runs, Cargo.*
  engines/<name>/   # sibling engines (e.g. engines/ace/) — never under engine/src/
  training/         # sibling — not under engine
  docs/             # architecture.md, …
  research/         # index of historical experiments (READMEs → artifacts)
  artifacts/        # gitignored: old target-*, match logs, reports (moved, not deleted)
```

`engine/` is **Titanium only**. Other playing engines live under `engines/<name>/` as their own crates (e.g. `engines/ace/`). Do not put alternate engines under `engine/src/`.

Titanium NNUE / HalfPW weight blobs live at `engine/src/weights/` (not under `titanium/`).

`engine/` must stay findable in ~5 seconds. New alternate `CARGO_TARGET_DIR` trees go under `artifacts/targets/`, never as new `engine/target-*` siblings.

## Layers (depend only downward)

```text
Layer 4   UCI · Validation          (+ Training as a sibling tree, not under engine/)
Layer 3   Search
Layer 2   Eval · Endgame
Layer 1   Position
Layer 0   Core (movegen / pathfinding / cat / core)
```

## Data flow (debugging)

```text
UCI
  ↓
Search
  ↓
Eval ──┐
       ├──→ Position → Core
Endgame┘
```

Search alone decides (prune, aspirate, extend). Eval/Endgame only supply facts.

## Rules

### Rule #1 — One home per concept

Every concept has exactly one owning module. Two answers ⇒ drift.

| Concept | Owner |
|---------|--------|
| Race / jump-aware distance | `engine/src/titanium/endgame/` |
| ExactDP | `engine/src/titanium/endgame/exact_dp.rs` |
| NNUE | `engine/src/titanium/eval/` |
| NNUE weight blobs | `engine/src/weights/` |
| Board / game state | `engine/src/titanium/position/` |
| Time management | `engine/src/titanium/timeman/` |
| Opening book | `engine/src/titanium/opening/` |
| Play search (αβ / TT / LMR) | `engine/src/titanium/search/` |
| Historical αβ / CLI / perft TT | `engine/legacy/search/` |
| Historical crate-root opening book | `engine/legacy/opening/` |
| Canta fixtures | `engine/src/validation/canta/` |
| Dataset / selfplay / Claustro | `training/` (repo root) |

### Rule #2 — Own an abstraction, not an algorithm

No catch-all `distance/`. Race distance ≠ eval distance ≠ generic pathfinding.

### Rule #3 — Data flows downward, decisions flow upward

Lower layers compute **information**. Only Search makes **search decisions**.

```rust
let bound = endgame::race::…;  // fact
// search decides whether to prune

let score = eval::…;           // fact
// search decides aspiration
```

## ExactDP

- **Owns:** Endgame
- **Uses:** Validation / tests / benches
- **Search must not call** — exponential validation-only reference; would steal clock; not a production proof path

## Public API ≠ ownership

`titanium::RaceBound`, `titanium::GameState`, `titanium::net`, etc. stay stable via re-exports while implementation lives under façades.

## Research

`titanium/research/` holds under-investigation code (e.g. parked wall-ignore). Graduate or delete; do not treat as production.

## Validation never affects Elo

`engine/src/validation/` and ExactDP are not on the production search hot path.

## Repository hygiene (engine folder)

`engine/` is the **live crate only**. If it is not source, benches, examples, scripts, docs, or Cargo metadata, it does not live here.

```text
engine/
  src/ benches/ examples/ scripts/ docs/
  Cargo.toml Cargo.lock build.rs README.md LICENSE
  target/          # live cargo output ONLY
```

Everything else goes under repo-root:

| Kind | Where |
|------|--------|
| Old `target-*` Cargo dirs, match logs, baselines | `artifacts/` |
| Experiment notes (index into artifacts) | `research/` |
| Training | `training/` (sibling — engine must not know trainer) |

**Rule:** if you cannot find the engine in 5 seconds of browsing `engine/`, hygiene failed. Move, do not delete history.
