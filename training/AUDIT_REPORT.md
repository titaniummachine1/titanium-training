# Training pipeline audit report (2026-06-17)

Verification-only work per `WEAK_AI_TASKS.md`. No engine/search/training logic changed.

**Terminology:** `pbff_*` in the engine = **binary / bitboard flood fill** for path-to-goal wall-legality checks (not a custom NN/search layer). See [`ARCHITECTURE_HANDOFF.md`](ARCHITECTURE_HANDOFF.md) § _Terminology — pbff\__ / wall legality / flood fill\*.

---

## Task 1 — Doc consistency audit

**Scope:** `ARCHITECTURE_HANDOFF.md`, `README.md`, `zero_teacher/HANDOFF.md`, `zero_teacher/README.md`, `zero_teacher/REFERENCE.md`

### Agreements (consistent)

| Topic                                                 | Status                                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------------------------ |
| `ws[14] = legal_wall_count / 128`                     | All five agree (zero_teacher docs omit ws — scoped correctly)                  |
| `ws[15]` = opponent corridor width                    | ARCHITECTURE + README agree; code uses `width_opp` / `opponent_corridor_width` |
| Phase 3 pressure head **not** in Rust                 | ARCHITECTURE, zero_teacher HANDOFF, README search-pressure section             |
| Ka CNN / ResNet rejected for live eval                | ARCHITECTURE; REFERENCE rejects WDL distill + notes ResNet is external only    |
| DB = moves only; features from eval-batch             | ARCHITECTURE + README                                                          |
| No fallback `legal_wall_count = 128` in training path | ARCHITECTURE; code hard-fails in `datagen.py` / `halfpw.py`                    |
| `parity_check.py` required before training            | ARCHITECTURE + README pre-flight + `nnue_guards.pretrain_sanity_ok`            |
| Do not resume pre-ws[14] checkpoint                   | ARCHITECTURE + README; **implemented** in `train.py` (`TRAINING_SCHEMA`)       |

### Inconsistencies

| #   | Issue                                                                                                                                                | Files                                                                  | Suggested doc-only patch                                                                                                    |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Checkpoint resume on schema mismatch** docs say "refuse resume"; code **warns and starts fresh** from `net_weights.bin`                            | `train.py` L480–484 vs `ARCHITECTURE_HANDOFF.md` L354, `README.md` L86 | In ARCHITECTURE_HANDOFF + README: "schema mismatch → abort resume, re-init from `net_weights.bin` (no silent weight carry)" |
| 2   | **`ws[15]` name** audit checklist says `corridor_width_opp`; docs say "opponent corridor width"; JSON has `corridor_width0/1`, eval uses `width_opp` | `ARCHITECTURE_HANDOFF.md` L341                                         | Add one line: "`ws[15]` input = opponent corridor width (`width_opp` in engine JSON)"                                       |
| 3   | **`training/README.md` L139** uses path `training/ARCHITECTURE_HANDOFF.md` from inside `training/`                                                   | `README.md` L139                                                       | Change to `[ARCHITECTURE_HANDOFF.md](ARCHITECTURE_HANDOFF.md)`                                                              |
| 4   | **`REFERENCE.md` safe-use table** says "Full-game opponent (WDL only) **Maybe**" — softer than master "do not distill eval"                          | `zero_teacher/REFERENCE.md` L25                                        | Change Maybe → **No** for WDL into `train.py`; keep WDL-only games via bot-move as separate pool note if needed             |
| 5   | **`plan.md` outside audit set** still documents old `ws[14]=corridor_width_me` in historical section                                                 | `training/plan.md` L155                                                | Add banner at top: "Superseded by `ARCHITECTURE_HANDOFF.md` for ws[14]" or strike old line                                  |

### Not inconsistencies (OK)

- Zero-teacher docs don't repeat ws[14] contract — intentional scope
- `engine_identity.py` uses `legal_wall_count == 128` on **startpos** as schema smoke — not a training fallback

---

## Task 7 — Native build rules audit

| File                                                        | Command                             | RUSTFLAGS native?  | Risk                                    | Suggested change                                               |
| ----------------------------------------------------------- | ----------------------------------- | ------------------ | --------------------------------------- | -------------------------------------------------------------- |
| `.cursor/rules/titanium-native-build.mdc`                   | documented build                    | Yes                | —                                       | OK                                                             |
| `training/README.md` pre-flight                             | `cargo build --release`             | Yes (documented)   | —                                       | OK                                                             |
| `training/nnue_guards.py` `rebuild_titanium_release`        | `cargo build --release -p titanium` | Yes (`setdefault`) | —                                       | OK                                                             |
| `training/run_supervised_session.ps1`                       | cargo build                         | Yes                | —                                       | OK                                                             |
| **Root `README.md` L20**                                    | `cargo build --release`             | **No**             | Suboptimal binary for benchmarks        | Doc: add RUSTFLAGS note or link to `titanium-native-build.mdc` |
| **`training/run_bisect_and_overnight.ps1` L20,24,49**       | `cargo build --release`             | **No**             | Overnight/bisect may use scalar movegen | Add `$env:RUSTFLAGS='-C target-cpu=native'` before builds      |
| **`training/run_bisect_continue.ps1` L27**                  | same                                | **No**             | Same                                    | Same                                                           |
| **`training/run_partial_golden_match.cmd` L9,12**           | same                                | **No**             | Same                                    | Same                                                           |
| **`training/run_infinite_benchmark.py` L70**                | tells user `cargo build --release`  | **No** in message  | Misleading hint                         | Message: include RUSTFLAGS                                     |
| `training/probe_legal_wall_signal.py` / `compare_halfpw.py` | error hints only                    | N/A                | Low                                     | Optional: mention native in hint                               |
| `training/plan.md`                                          | documents native                    | Yes                | —                                       | OK                                                             |

---

## Task 4 — Infinite-search regression map (read-only)

### Default mode today

- **Pool / benchmarks:** `titanium-v15` via `manifest.CURRENT_ENGINE` (`training/manifest.py` L24)
- **CLI session:** `engine/src/main.rs` L75–78 routes ace-v13 family to `run_ace_session_stdio` with comment: **v15 uses standard warm session (`go TIME_SEC`); session_v15 infinite disabled**
- **No `session_v15` string** in current engine source tree (grep empty); infinite-session code removed or disabled in engine submodule commit `9dddde1` ("disable v15 infinite session" per `engine/.git/logs/HEAD`)

### Is session_v15 reachable?

- **Not by name** in current tree
- Ponder APIs exist on `AceSearch` (`set_pondering`, `migrate_root`, `is_pondering`) in `search.rs`
- `run_infinite_benchmark.py` = **infinite loop of match batches**, not engine infinite-search mode

### Commits (parent repo)

| Commit    | Summary                                                            |
| --------- | ------------------------------------------------------------------ |
| `98b703b` | v15 infrastructure: `run_infinite_benchmark.py`, manifest, datagen |
| `9a40c46` | Engine submodule bump: "titanium-v15 infinite-search session"      |

### Engine submodule (evidence from `.git/logs`)

| Commit message                                                         | Implication                  |
| ---------------------------------------------------------------------- | ---------------------------- |
| `Add titanium-v15 infinite-search session (two-thread daemon)`         | Introduced experimental path |
| `acev13: wire legal-wall NNUE fields and disable v15 infinite session` | Disabled default routing     |

### Risk points (code-evidenced)

| Risk                                                                                                                           | Evidence                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| TT/history behavior differs under ponder                                                                                       | `search.rs` L2319–2328 skips `tt_gen` advance + history decay when `is_pondering` |
| Warm session TT reuse                                                                                                          | `apply_move` keeps TT; comment L716 "next think reuses prior analysis"            |
| Root migration after ponder                                                                                                    | `migrate_root` L750+ decays history by surprise                                   |
| Partial-iter optional                                                                                                          | `set_partial_iter` off by default (L771)                                          |
| **Cannot confirm from tree:** `ponderhit` / `movemiss` / `always-restart` strings — **not found** in `site/` or `engine/` grep |

---

## Task 5 — Benchmark / probe inventory

| Script                           | What it tests                                   | Command                                                    | Needs `titanium.exe`          | Writes files                          | Safe before training          |
| -------------------------------- | ----------------------------------------------- | ---------------------------------------------------------- | ----------------------------- | ------------------------------------- | ----------------------------- |
| `parity_check.py`                | Python HalfPW == engine eval (6 positions)      | `python training/parity_check.py`                          | Yes                           | No                                    | **Yes — required**            |
| `engine_identity.py`             | Binary SHA stamp + startpos `legal_wall_count`  | `python training/engine_identity.py`                       | Yes                           | `data/engine_stamp.json` (--write)    | Yes                           |
| `regression_triage.py`           | parity + eval-batch schema + search smoke match | `python training/regression_triage.py`                     | Yes                           | No                                    | Yes                           |
| `validate_train_ready.py`        | parity + eval-batch (new draft)                 | `python training/validate_train_ready.py`                  | Yes                           | No                                    | **Yes — recommended**         |
| `plateau_probe.py`               | Eval drift + root move change vs deploy         | `python training/plateau_probe.py`                         | Yes                           | `data/nnue_eval_probe.json`           | Yes (after deploy)            |
| `probe_legal_wall_signal.py`     | ws[14] orthogonality correlations               | `python training/probe_legal_wall_signal.py`               | Yes                           | No (stdout)                           | Yes                           |
| `compare_halfpw.py`              | train vs frozen weights on probes               | `python training/compare_halfpw.py`                        | Yes                           | No                                    | Yes                           |
| `collect_search_importance.py`   | Shallow-vs-deep pressure labels                 | `python training/collect_search_importance.py --limit 200` | Yes                           | `data/search_pressure.jsonl`          | Yes (sidecar only)            |
| `zero_teacher/collect_budget.py` | Zero-ink MCTS labels                            | `python -m training.zero_teacher.collect_budget ...`       | No (HTTP)                     | `data/zero_teacher/labels/*.jsonl`    | Yes (sidecar only)            |
| `train_search_importance.py`     | Pressure head train                             | `python training/train_search_importance.py --data ...`    | No (eval-batch at train time) | `checkpoints/search_pressure_head.pt` | Yes (does not touch main net) |
| `run_benchmarks.py`              | A/B Elo + ingest games                          | `python training/run_benchmarks.py`                        | Yes                           | DB + `benchmarks_log.jsonl`           | Yes (long)                    |
| `run_infinite_benchmark.py`      | Endless v15 vs ti-pure batches                  | `python training/run_infinite_benchmark.py`                | Yes (via node)                | games + manifest                      | Yes (pool load)               |
| `run_swiss_overnight.py`         | Game pool + train                               | `python training/run_swiss_overnight.py`                   | Yes                           | DB, checkpoints                       | **No — trains**               |
| `run_nnue_cycle.py`              | Per-game micro-train                            | `python training/run_nnue_cycle.py --game-id N`            | Yes (via train)               | checkpoints                           | **No — trains**               |
| `nnue_guards.py`                 | Caps, deploy, rebuild                           | imported                                                   | Yes on deploy                 | guard state                           | Guard layer                   |

---

## Task 3 — Checkpoint resume guard proposal

### Current behavior (`train.py`)

| Item            | Location                                                                         |
| --------------- | -------------------------------------------------------------------------------- |
| Schema constant | L83 `TRAINING_SCHEMA = "halfpw-field11-ws14-legal-wall-v1"`                      |
| Save metadata   | L349–355 `save_checkpoint` writes `"schema"`                                     |
| Load guard      | L358–365 `load_checkpoint` raises if `schema != TRAINING_SCHEMA`                 |
| Resume entry    | L471–484 `--resume` / `--ckpt`; `run_nnue_cycle.py` L73 always passes `--resume` |
| On mismatch     | Prints WARN, **starts fresh** from `net_weights.bin` (does not exit)             |

### Gap vs docs

Docs imply hard refuse; code is **soft refuse** (fresh init). Old optimizer state discarded; model re-inits from deployed weights — **acceptable** but should be documented.

### Minimal safe enhancement (optional)

```python
# train.py — add flag, default unchanged
ap.add_argument("--allow-legacy-resume", action="store_true",
                help="Override schema mismatch (dangerous)")

# load_checkpoint: on mismatch
if schema != TRAINING_SCHEMA:
    if not args.allow_legacy_resume:
        raise RuntimeError(...)
```

**Risks:** `--allow-legacy-resume` could reload pre-ws[14] weights if someone forces load of old `.pt` without schema key — low if schema check stays default.

**Verdict:** Guard **already exists** for normal path. Optional: exit 1 on mismatch instead of silent fresh start; optional explicit override flag.

---

## Task 6 — Search-pressure Phase 3 gap report

| Question                                    | Answer                                                                                                                      |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Pressure head trained as sidecar only?      | **Yes** — `train_search_importance.py` saves `search_pressure_head.pt` separately (`kind: leaf_search_pressure_sidecar_v1`) |
| Exported into `net_weights.bin`?            | **No** — `HalfPW.save_weights` / `train.py` export only value trunk (L206–218)                                              |
| Rust computes `hidden_features` separately? | **No** named API — `evaluate()` fuses scalar ws + hid→w2 inline in `search.rs` (~L1384+)                                    |
| `search.rs` uses pressure for LMR/order?    | **No** — grep `search_pressure` / `pressure_head` in `engine/` = zero matches                                               |
| Files for Phase 3                           | See below                                                                                                                   |

### Phase 3 touch list (handoff only)

1. `training/train_search_importance.py` — export pressure weights (32+1 doubles) alongside or appended to blob
2. `training/train.py` or `extend_weights.py` — schema version bump for pressure tail
3. `engine/src/acev13/search.rs` — factor `hid[32]` once; add `pressure_head` dot + tanh; optional JSON field in `eval --json`
4. `training/halfpw.py` — parity for pressure if exposed
5. `training/parity_check.py` — only if pressure in eval JSON
6. `training/nnue_guards.py` — `assert_leaf_net_budget` for +33 weights
7. Phase 4 later: `search.rs` LMR block — **do not touch until Phase 3 validates**

---

## Task 2 — `validate_train_ready.py`

**Added:** `training/validate_train_ready.py` (~75 lines, standalone).

Not wired into hooks or `run_nnue_cycle.py` yet (additive draft only).

```powershell
python training/validate_train_ready.py
```
