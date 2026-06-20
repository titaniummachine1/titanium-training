# HalfPW NNUE retrain - pipeline

> **Canonical docs:** [docs/README.md](../docs/README.md) — Oracle deployment, dataset, training CLI.  
> **Architecture handoff:** [`ARCHITECTURE_HANDOFF.md`](ARCHITECTURE_HANDOFF.md)

Fine-tune the existing gen13 ACE HalfPW net. Do not retrain from scratch: tactical
knowledge is already in the weights; new inputs are zero-init and learned as residuals.

## Native search profiling

`profile_titanium.ps1` builds a symbolized native release in the isolated
`engine/target-profile` directory, then profiles start position, the c3h
midgame, and a tested wall-maze replay. SVG output goes to the ignored
`training/data/profiles/` directory and is summarized by
`parse_flamegraph.py` using exclusive samples.

Windows ETW collection requires an elevated shell; the repository and normal
training do not. Right-click PowerShell, select **Run as administrator**, then:

```powershell
Set-Location "C:\gitProjects\Quoridor best AI"
.\training\profile_titanium.ps1
```

Add `-IncludeScalar` to also profile the intentional non-BMI2/WASM-like build.
If the Cargo plugin is absent, install it once with
`cargo install flamegraph --locked`.

## Frozen Architecture (2026-06)

| Component         | Contract                                                                                                                   |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Field planes (11) | goal_inv, pawn_fwd, corridor_delta, path_cross, choke x2, contested per-player BFS                                         |
| Sparse embeds     | w1c (128 wall slots), po, px                                                                                               |
| ws[0-12]          | trained interaction terms; do not change semantics                                                                         |
| ws[13]            | fragile-lead formula (`pd * w_opp / 10`)                                                                                   |
| ws[14]            | `legal_wall_count / 128` (path-valid wall slots; counted via bitboard flood fill / `pbff_*`)                               |
| ws[15]            | opponent corridor width                                                                                                    |
| Search            | `titanium-v15` live NNUE (strongest); `titanium-v15-frozen` pinned baseline; warm session — **not** infinite `session_v15` |

Certificate/race-proof layer remains the only hard eval override.

## Data Pipeline

SQLite stores compact move sequences and the final winner. The normal path is:

```text
all_games.db (moves_bin + outcome)
    -> expand_games() / eval-batch  (feature authority)
    -> train.py                     (WDL loss on materialized records)
```

Every valid completed game is training data. Ka/JS/frozen/self-play games are all
trainable if the move list replays and the winner is known. Quoridor has no draw target.
Single-position Ka/CNN labels are not part of the default pipeline.

If `eval-batch` is correct, training is correct. There is no hidden dataset drift on the
`.db` path. Hard-fail if `legal_wall_count` is missing. Checkpoints must carry schema
`halfpw-field11-ws14-legal-wall-v1` to resume.

## Scripts

| Script                              | Role                                                                     |
| ----------------------------------- | ------------------------------------------------------------------------ |
| `validate_train_ready.py`           | Preflight: binary + parity 6/6 + eval-batch `legal_wall_count`           |
| `halfpw.py`                         | Python port of engine forward pass                                       |
| `REGRESSION_BISECT.md`              | Strength regression bisect runbook (evidence-based)                      |
| `AUDIT_REPORT.md`                   | Verification audits (docs, native build, Phase 3 gaps)                   |
| `WEAK_AI_TASKS.md`                  | Safe delegated task queue for weak agents                                |
| `parity_check.py`                   | `halfpw.forward` == `titanium eval ... --json` before train              |
| `engine_identity.py`                | SHA256 stamp for the single validated `titanium.exe`                     |
| `regression_triage.py`              | Classify strength drops: eval / search / rollout before blaming training |
| `nnue_guards.py`                    | Artifact caps, Elo snapshots, pre-train gates, deploy                    |
| `datagen.py`                        | Game ingest + `eval-batch` expansion                                     |
| `run_nnue_cycle.py`                 | Guarded micro/batch train from `all_games.db`                            |
| `run_swiss_overnight.py`            | Game pool + background NNUE (`--no-train` to disable)                    |
| `collect_search_importance.py`      | Build shallow-vs-deep search-pressure labels for scalar experiments      |
| `zero_teacher/`                     | External MCTS teacher — see `zero_teacher/HANDOFF.md`                    |
| `zero_teacher/collect_budget.py`    | MCTS attention labels from quoridor-zero.ink (50–400 visits)             |
| `train_search_importance.py`        | Train the sidecar search-pressure head                                   |
| `run_search_pressure_experiment.py` | Cloud/overnight wrapper for pressure-label collection + head training    |
| `collect_reduction_counterfactuals_v3.py` | Phase-3 grouped A/B collector (natural + hard-negative mining)   |
| `train_lmr_head_v3.py`             | Phase-3 LMR head trainer (P/PL/PL-NL8, phased: narrow→stability→holdout→shadow) |
| `position_store.py`                 | Canonical position graph DB: inventory, import, shard ingest, audits     |
| `probe_legal_wall_signal.py`        | Correlation probe for ws[14] ablation                                    |
| `plateau_probe.py`                  | Eval-drift / promotion gate                                              |

## Canonical Position Store (two databases)

Training data is split into **two physically separate SQLite databases**:

| Store | Default path | Used by |
|-------|--------------|---------|
| Game store | `training/data/canonical/game_store.db` | `train.py`, self-play shard ingest, pool imports, WDL training |
| Teacher store | `training/data/canonical/position_teacher_store.db` | Friend/zero/search/LMR labels — **explicit flags only** |

Both share the same position codec and canonical hash. They are not merged unless you run `export-mixed-training`.

The durable store lives behind `training/position_store.py`. The game store holds:

- canonical unique packed positions reachable by legal replay
- stable 1-byte move paths and graph edges
- WDL game records
- append-only shard imports

The teacher store holds pathless labeled positions and compact policy sidecars under `teacher_sidecars/`.

Read [`CANONICAL_DATASTORE.md`](CANONICAL_DATASTORE.md) first — it defines paths, legacy policy, and authoritative commands. Then read
[`POSITION_STORE_RUNBOOK.md`](POSITION_STORE_RUNBOOK.md) for import/migration details.

**For future agents:** do not recreate a single combined database; do not import friend shards into the game store.

## Pre-Flight

Native build only (`RUSTFLAGS="-C target-cpu=native"`):

```powershell
cd engine
$env:RUSTFLAGS="-C target-cpu=native"
cargo build --release
cd ..
python training/validate_train_ready.py
python training/engine_identity.py --write
python training/parity_check.py
python training/regression_triage.py
```

`validate_train_ready.py` is a fast gate before training or overnight retrain: confirms `titanium.exe` exists, parity is 6/6, and eval-batch emits `legal_wall_count`. It does **not** rebuild the engine and does **not** train or mutate weights — run a native rebuild first when the binary changed.

Restart the overnight pool after rebuild so match slots and eval-batch share the same binary.

For a bounded laptop smoke using only local opponents while draining successful
micro-trains before exit:

```powershell
python training/run_swiss_overnight.py --local-only --parallel 2 --games 2
```

The normal adaptive slot prefers zero-ink. It falls back to Ka only after
Titanium scores at most 4/16 in a complete color-balanced zero window. Never
feed Ka scalar evaluation labels into HalfPW; external MCTS attention belongs
only in the separate search-pressure dataset.

For unattended local-only volume with no remote engines:

```powershell
python training/run_swiss_overnight.py --local-only --parallel 2
```

Those slots use ti-pure@10s and v15 self-play@10s. Add `--games 20` for a
bounded run. Two slots are the sensible laptop setting; more mostly buys heat.

## Checkpoint resume

`run_nnue_cycle.py` always passes `--resume` to `train.py` for micro-trains. That is fine once you have at least one checkpoint stamped `halfpw-sparse-route5-ws14-v1`.

**Do not** expect useful resume from pre-ws[14 checkpoints whose optimizer state was trained when `ws[14]` meant `corridor_width_me`. `train.py` checks `TRAINING_SCHEMA` on load; on mismatch it prints a warning and **re-inits weights from deployed `net_weights.bin`** (optimizer state discarded). No manual `--ckpt` override exists today.

Fresh pool start after a schema or binary change: let the first micro-train create a new ws[14]-era checkpoint, or delete stale `ckpt_*.pt` if you want a clean optimizer state.

## Overnight + Training Guards

| Guard              | Behavior                                                                                                   |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| New games          | train after >=32 games since last epoch                                                                    |
| Interval           | min 10 min between epochs                                                                                  |
| Artifact soft cap  | 500 MB checkpoints -> prune old `ckpt_step*`                                                               |
| Artifact hard cap  | 1 GB -> refuse train                                                                                       |
| Pre-train win rate | skip batch if v15 vs ti-pure >70%; warn micro at >58%                                                      |
| Elo drop           | snapshot to `checkpoints/snapshots/` if ladder -12+ from peak                                              |
| Resume schema      | refuse checkpoints without `halfpw-sparse-route5-ws14-v1`; on mismatch re-init and reset deploy cadence   |
| Engine stamp       | block eval/self-play/train if `titanium.exe` hash changed                                                  |
| Parity/schema      | training blocked unless parity passes and `legal_wall_count` exists                                        |

Before search architecture experiments (`session_v15`, ponder, engine infinite search), run
`engine_identity.py --write` and `regression_triage.py`. **`run_infinite_benchmark.py`** loops match batches — it is **not** engine infinite-search mode. Do not assume a training problem
until eval/search/rollout smokes pass.

## Engine Commands

- `titanium eval <moves> --json` - raw inputs + eval for parity and training format
- `titanium eval-batch` - stdin: one move sequence per line; JSON per position
- `titanium match --a <eng> --b <eng> --games N --time S` - self-play strength

## Learned LMR Reduction Sidecar (Phase 3)

**Read [`PHASE3_LMRH_RUNBOOK.md`](PHASE3_LMRH_RUNBOOK.md) first.** This section is a brief pointer only.

The model predicts whether one additional search-depth reduction is safe and profitable for an already-LMR-eligible move (binary: safe +1 ply or not). It is **not** a depth extender — it only reduces, never increases search depth.

Runtime activation is **OFF** (shadow/candidate mode only). The eval trunk (`net_weights.bin`) is never modified by this pipeline.

**Phase-3 scripts (current):**

```powershell
# Natural collection (target 10 000 events, depth 8)
python training/collect_reduction_counterfactuals_v3.py `
  --natural-target 10000 --out-dir training/data/lmr_phase3 `
  --depth 8 --min-event-depth 6 --min-ply 11 --seed 777

# Hard-negative enrichment (separate pass)
python training/collect_reduction_counterfactuals_v3.py `
  --hard-negative-pass `
  --natural-file training/data/lmr_phase3/natural.jsonl `
  --out-dir training/data/lmr_phase3 --hard-negative-target 200

# Training phases: narrowing → stability → manifest → holdout → shadow
python training/train_lmr_head_v3.py `
  --natural training/data/lmr_phase3/natural.jsonl `
  --hard-negatives training/data/lmr_phase3/hard_negatives.jsonl `
  --out-dir training/checkpoints/lmr_v3 --phase narrowing
```

See [`PHASE3_LMRH_RUNBOOK.md`](PHASE3_LMRH_RUNBOOK.md) for the full sequence and smoke-run commands.

**Stage-2 legacy scripts** (`collect_reduction_counterfactuals.py`, `train_reduction_sidecar.py`, `train_reduction_sidecar_v2.py`) are preserved for reference. Do not use them for new collection.

Native Titanium A/B search is the label authority. Runtime activation remains disabled;
`titanium reduction-shadow` computes predictions without changing LMR or the search tree.

**Test suites:** `training/test_reduction_counterfactuals.py` (62 tests) and `training/test_lmr_head_v3.py` (69 tests). Run with `python -m pytest training/test_reduction_counterfactuals.py training/test_lmr_head_v3.py`.

## Historical Search-Pressure Labels

The leaf-local search scalar is collected as a sidecar dataset first:

```powershell
python training/run_search_pressure_experiment.py --labels 2000 --chunk 200 --time 2.0 --cpu
```

For a longer cloud run:

```powershell
python training/run_search_pressure_experiment.py --labels 20000 --chunk 500 --time 2.0 --epochs 50
```

The target is shallow-vs-deep search pressure. Read it as the parent node asking about the
child it just reached: how much do we trust this shallow evaluation, and does this node
deserve less or more budget than normal? `-1` means already saturated, `0` means normal,
`+1` means shallow search is likely unstable and this child deserves more budget. The
trained sidecar is not wired into live reductions until its labels and validation show it
helps rather than making search weaker.

Safe activation order:

1. Collect labels only; inspect distribution and examples.
2. Train the sidecar; require validation loss below a constant-mean baseline.
3. Add engine export/inference only as diagnostics.
4. Let pressure change LMR/extension by at most one ply, with mate/TT/forced-move overrides.
5. Run A/B matches before making it default.

The trainer supports three frozen linear feature taps for ablation:

- `--features hidden32`: legacy wall/pawn hidden layer only.
- `--features rich`: hidden layer plus cheap scalar and route summaries.
- `--features routefull`: hidden layer plus the existing full route vectors.

Checkpoints record the exact base-weight SHA-256 and cannot be considered
validated without grouped holdouts. Terminal and already-proven mate/race
positions are excluded because search overrides own those nodes.

### Zero-ink MCTS attention (optional teacher)

Treat paired zero-ink disagreement as auxiliary attention data only after it
correlates with native alpha-beta pressure on the same positions. Do not merge
the sources merely because both targets use `[-1,+1]`.

External AlphaZero MCTS from [quoridor-zero.ink](https://quoridor-zero.ink) — **not** main WDL
distill. Small rollouts (50–400 visits) → `visitFraction` / prior gaps → same sidecar head.

```powershell
python -m training.zero_teacher.collect_budget --from-db --limit 100 --shallow-visits 50 --deep-visits 400
python training/train_search_importance.py --data training/data/zero_teacher/labels/search_budget.jsonl
```

Docs: [`ARCHITECTURE_HANDOFF.md`](ARCHITECTURE_HANDOFF.md) (master), [`zero_teacher/HANDOFF.md`](zero_teacher/HANDOFF.md) (zero-ink).

Latest measured pressure results: [`SEARCH_PRESSURE_REPORT.md`](SEARCH_PRESSURE_REPORT.md).

## External AlphaZero Data

`KaAiData/ANOTHER TRAINING DAT ASTUFF SUPER USEFULL` contains per-position MCTS samples:
board state, sparse policy, side-to-move outcome, and root value. It is useful, but it is
not replayable game history, so it must not be imported into `all_games.db`.

Use it later as a separate streaming source for diagnostics or optional root-value/policy
experiments. Filter cutoff draws instead of forcing them into Quoridor WDL labels.

## Baseline (pre-retrain, 2026-06-15)

Grafted vs plain ace-v13 @ 2s: 54-58 / 112 = 48.2% (+/-9.3%) ~= -12 Elo (within noise).
