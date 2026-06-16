# HalfPW NNUE retrain — pipeline

Fine-tune the existing gen13 ACE HalfPW net. **Do not retrain from scratch** — tactical
knowledge is already in the weights; new inputs are zero-init and learned as residuals.

## Frozen architecture (2026-06)

| Component         | Contract                                                                                              |
| ----------------- | ----------------------------------------------------------------------------------------------------- |
| Field planes (11) | goal_inv, pawn_fwd, corridor_delta, path_cross, choke×2, contested (per-player BFS)                   |
| Sparse embeds     | w1c (128 wall slots), po, px                                                                          |
| ws[0–12]          | trained interaction terms — do not change semantics                                                   |
| ws[13]            | fragile-lead formula (`pd × w_opp / 10`)                                                              |
| ws[14]            | **`legal_wall_count / 128`** (path-valid wall slots, Ka “placable” scalar)                            |
| ws[15]            | opponent corridor width                                                                               |
| Search            | titanium-v15: full legal movegen, warm session (not production-routed through infinite `session_v15`) |

Certificate/race-proof layer remains the only hard eval override.

## Data pipeline (the real system)

SQLite stores **move sequences only** — no stored features.

```
games.db (moves + outcome)
    → expand_games() / eval-batch  (feature authority)
    → train.py                     (WDL loss on materialised records)
```

If `eval-batch` is correct, training is correct. There is no hidden dataset drift on the `.db` path.

Hard-fail if `legal_wall_count` is missing (no fallback). Checkpoints must carry schema
`halfpw-field11-ws14-legal-wall-v1` to resume.

## Scripts

| Script                       | Role                                                                     |
| ---------------------------- | ------------------------------------------------------------------------ |
| `halfpw.py`                  | Python port of engine forward pass                                       |
| `parity_check.py`            | `halfpw.forward` == `titanium eval … --json` (must be 6/6 before train)  |
| `engine_identity.py`         | SHA256 stamp for the single validated `titanium.exe`                     |
| `regression_triage.py`       | Classify strength drops: eval / search / rollout before blaming training |
| `nnue_guards.py`             | Artifact caps, Elo snapshots, pre-train gates, deploy                    |
| `datagen.py`                 | Self-play ingest + `eval-batch` expansion                                |
| `run_nnue_cycle.py`          | Guarded micro/batch train from `all_games.db`                            |
| `run_swiss_overnight.py`     | Game pool + background NNUE (`--no-train` to disable)                    |
| `probe_legal_wall_signal.py` | Correlation probe for ws[14] ablation                                    |
| `plateau_probe.py`           | Eval-drift / promotion gate                                              |

## Pre-flight (every engine change)

Native build only (`RUSTFLAGS="-C target-cpu=native"` — never suboptimal builds):

```powershell
cd engine
$env:RUSTFLAGS="-C target-cpu=native"
cargo build --release
cd ..
python training/engine_identity.py --write
python training/parity_check.py          # expect 6/6
python training/regression_triage.py     # optional strength smoke
```

Restart the overnight pool after rebuild so match slots and eval-batch share the same binary.

## Overnight + training guards

| Guard              | Behavior                                                          |
| ------------------ | ----------------------------------------------------------------- |
| New games          | train after ≥32 games since last epoch                            |
| Interval           | min 10 min between epochs                                         |
| Artifact soft cap  | 500 MB checkpoints → prune old `ckpt_step*`                       |
| Artifact hard cap  | 1 GB → refuse train                                               |
| Pre-train win rate | skip batch if v15 vs ti-pure >70%; warn micro at >58%             |
| Elo drop           | snapshot to `checkpoints/snapshots/` if ladder −12+ from peak     |
| Resume schema      | refuse checkpoints without `halfpw-field11-ws14-legal-wall-v1`    |
| Engine stamp       | block eval/self-play/train if `titanium.exe` hash changed         |
| Parity/schema      | training blocked unless parity 6/6 and `legal_wall_count` present |

Before search architecture experiments (`session_v15`, ponder, infinite search), run
`engine_identity.py --write` and `regression_triage.py`. Do not assume a training
problem until eval/search/rollout smokes pass.

## Engine commands

- `titanium eval <moves> --json` — raw inputs + eval (parity + training format)
- `titanium eval-batch` — stdin: one move sequence per line; JSON per position
- `titanium match --a <eng> --b <eng> --games N --time S` — self-play strength

## Baseline (pre-retrain, 2026-06-15)

grafted vs plain ace-v13 @ 2s: 54–58 / 112 = 48.2% (±9.3%) ≈ −12 Elo (within noise).
