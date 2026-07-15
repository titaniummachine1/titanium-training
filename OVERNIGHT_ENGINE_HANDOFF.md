# Titanium engine work status — 2026-07-15

This is the canonical resume note. Update it after every strength gate so
another Codex task can continue without repeating experiments.

## Current production decision

- `815a2ca fix(race): separate proof bounds from exact DTM` fixes the major race
  integration bug: proven outcome, approximate distance, and exact DTM are now
  separate. Approximate values never enter the exact `RACE_MATE - n` band.
- `164453b perf(cat): reuse corridor flood results` removes redundant CAT flood
  work without changing CAT values.
- One-wall race deduction is **accepted**: 200 games, randomized mirrored
  10-ply openings, 60 seconds per side, 4 local + 13 Oracle workers:
  `race1w 104 - 96 baseline` (0 draws, approximately +14 Elo). This is a small
  positive result, not statistically decisive, but project policy accepts small
  gains. The working tree makes it the default for `titanium-v17`.
- The incremental two-wall experiment is named `titanium-v17-race2w`. It means
  default race1 plus the two-wall layer; compare it against the new
  `titanium-v17` baseline, never against the old pre-race1 binary.

Clean race1 artifacts:

- local: `tools/binary_match/runs/race1w_gate_20260715/local/`
- Oracle resumed/fixed: `tools/binary_match/runs/race1w_gate_20260715/oracle_resume/`
- the first Oracle artifact contained 21 invalid `engine_dead`/`no_move` rows;
  those rows were excluded and rerun. Do not quote the contaminated raw score.

## Active gate

`race2w` versus the accepted race1 production baseline:

```text
A = titanium-v17-race2w
B = titanium-v17
games = 200, clock = 60s/side/game, opening = audited randomized 10-ply book
workers = 4 local (slots 0..3) + 13 Oracle (slots 4..16)
seed = 1337
run = tools/binary_match/runs/race2w_vs_race1w_20260715b/
```

The match harness rejects `engine_dead` and `no_move`, restarts both warm
sessions, and requeues the same game up to three times. Invalid attempts are
written separately and never score as wins.

## Ordered engine backlog

1. Finish the 200-game race2w incremental gate. Promote only if it beats the
   accepted race1 baseline under the project small-gain policy; otherwise keep
   the two-wall layer disabled.
2. Commit the accepted race1 default and the race2 decision with exact artifact
   paths and score in the commit message/notes.
3. Run one epoch after the CAT input normalization/reuse work, then strength-gate
   the resulting weights against the predecessor weights. Never mix a search
   change and new weights in the same A/B.
4. Audit `C:\Users\Terminatort8000\Downloads\search.rs` in this order:
   time-control-adaptive depth-4 RFP (smallest genuinely missing candidate),
   then WALLQ-TC dual-band leaf correction. Existing predictive stop, history,
   LMR, LMP, CMH/countermove, correction history, and aspiration must not be
   reimplemented.
5. Add reusable engine time management: reserve a minimum plausible remaining
   ply floor, spend more on unstable PV/close root alternatives, spend less on
   stable easy moves, and preserve an emergency move reserve. Tune with mirrored
   randomized self-play and verify zero time forfeits before strength claims.
6. Keep exact race math clean: bound-only semi-terminal deductions may prune only
   when they cross alpha/beta; exact DTM is lazy and only used for same-outcome
   finalists/UI. Add stubborn-loser behavior only as a policy after the exact
   solver result, never inside the proof.
7. Revisit walls-remaining immutable-route certificates only behind a flag and
   require zero false-positive winners against Canta/exhaustive/randomized
   opening, middlegame, and endgame counter-oracles before any strength gate.
8. Movegen/pathfinding ideas already tested must not be blindly repeated:
   Titanium already has BFF layers, topology gates, and free final-flood witness
   information. Any explicit route-witness/Lee-BFS addition must first beat the
   current non-TT perft(4) opening/middlegame/endgame battery with identical
   nodes and report both time and NPS.

## Separate website backlog

- Undo must remove the undone move's charged think time and restore the displayed
  per-move time.
- Applying settings after a time forfeit must clear the forfeit and reset the
  active clock bank without forcing a new game.
- Website time allocation should consume explicit engine telemetry (PV stability,
  score/root-alternative uncertainty) and must not display approximate race
  distance as exact `Win/Loss in N`.

---

# Historical overnight engine HCE loop — 2026-07-12

Parallel to flywheel labeling. Classical search only (no NN changes).

## Loop

- **Harness:** `tools/overnight_engine_improve.py`
- **State:** `training/data/overnight_logs/engine_improve_state.json`
- **Matches:** `tools/binary_match/runs/overnight_engine/`
- **Agent sentinel:** `AGENT_LOOP_WAKE_ENGINE` (30m heartbeat)

## Protocol

1. **d6 bench** — `search_bench` with `TITANIUM_BENCH_ENGINE=<flag>` on 6 canonical positions
2. **200-game A/B** — challenger (A) vs `titanium-v17` (B), clock 60s
   - **Local:** 4 workers, shard 0–3 of 17
   - **Oracle:** shards 4–16 via `launch_oracle_deployed.ps1` (pre-built binary on Oracle; ~few MB upload, not source tar)
3. **Verdict**
   - `PROMOTE` if unified pool score **> 0.5** after target games (any positive Elo, e.g. +3)
   - `DOUBLE` if score in **[0.48, 0.5]** (inconclusive) → 2× games (cap 1600)
   - `REJECT` if score **< 0.48**
   - Local shards 0–3 and Oracle 4–16 merge into **one** `status.json` at run root

## Candidate queue (initial)

| id      | engine_a             | note                                                |
| ------- | -------------------- | --------------------------------------------------- |
| rfp-ace | titanium-v17-rfp-ace | **PROMOTED** into default v17 (200g 101–99, +3 Elo) |
| probcut | titanium-v17-probcut | ProbCut verify                                      |

## Commands

```powershell
$env:RUSTFLAGS = '-C target-cpu=native'
py -3.12 tools/overnight_engine_improve.py build
py -3.12 tools/overnight_engine_improve.py status
py -3.12 tools/overnight_engine_improve.py tick

# Manual d6 baseline
$env:TITANIUM_BENCH_ENGINE = 'titanium-v17'
engine\target\release\search_bench.exe depth --depth 6 --position startpos --threads 1
```

## Promotion

Only candidates with `PROMOTE` after forensic match doubling get merged into default `titanium-v17` session flags (manual review + commit on `overnight/flywheel-cert-20260712` or new branch).

## Do not disturb

- Flywheel `phase-b-all` / Gen-0 pilot labeling (separate process)
- Frozen baseline until Wilson LB promotion
