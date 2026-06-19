# HalfPW NNUE retrain - pipeline

> **Architecture handoff (read first):** [`ARCHITECTURE_HANDOFF.md`](ARCHITECTURE_HANDOFF.md) — project goal, dual-head design, training phases, do-not-do list.

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
| `collect_reduction_counterfactuals.py` | Complete-pipeline A/B labels for provisional +1 LMR                  |
| `train_reduction_sidecar.py`        | Frozen linear safe-and-beneficial reduction sidecar                      |
| `position_store.py`                 | Canonical position graph DB: inventory, import, shard ingest, audits     |
| `probe_legal_wall_signal.py`        | Correlation probe for ws[14] ablation                                    |
| `plateau_probe.py`                  | Eval-drift / promotion gate                                              |

## Canonical Position Store

The new durable training-data store lives behind `training/position_store.py`.
It stores:

- canonical unique packed positions
- stable 1-byte move paths
- parent/move/child graph edges
- versioned labels
- append-only shard imports

Read [`POSITION_STORE_RUNBOOK.md`](POSITION_STORE_RUNBOOK.md) before doing a
real migration. It includes the exact PowerShell commands, dry-run flow,
current reject signatures, and smoke-validated storage numbers.

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

## Reduction Sidecar

The generic pressure scalar below is historical. The operational experiment is now the
narrow binary question: can this already-LMR-eligible late wall take one additional
provisional reduction while preserving the native pipeline decision and saving enough
total nodes to matter?

```powershell
python training/collect_reduction_counterfactuals.py --positions 200 --samples-per-position 2 --depth 5
python training/train_reduction_sidecar.py --data training/data/reduction_counterfactuals.jsonl
```

The collector runs baseline and counterfactual searches from separate fresh fixed-TT
states. It stores safety and savings independently and retains failed comparisons as
`UNKNOWN`. The trainer excludes `UNKNOWN`, freezes the value network by consuming stored
hidden features only, uses natural runtime-eligible games for calibration/test, and writes
an independently hash-bound `search_reduction_head.bin`. Training never deploys it.

Zero-ink may rank candidate moves, but its visits/value/policy never define the label.
Native Titanium A/B search is the label authority. Runtime activation remains disabled;
`titanium reduction-shadow` computes predictions without changing LMR or the search tree.

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
