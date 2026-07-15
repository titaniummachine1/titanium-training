# Titanium engine work status — 2026-07-15

This is the canonical resume note. Update it after every strength gate so
another Codex task can continue without repeating experiments.

For every candidate, record the raw W-D-L, score, approximate Elo, valid game
count, runtime/NPS or firing counters, exact artifact path, and whether it was
promoted. Preserve inconclusive measurements too; the owner may later promote a
small noisy gain using engineering judgment.

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
- Full-tree race2 is **accepted**: `105 - 95` over race1 in 200 games (0 draws,
  approximately +17 Elo). The working tree now promotes it into
  `titanium-v17`. Clean artifacts are under
  `tools/binary_match/runs/race2w_vs_race1w_20260715b/`.
- PV-only race2 is **accepted over full-tree race2**: local `26 - 22` plus
  Oracle `83 - 69`, combined `109 - 91` over 200 valid games (0 draws,
  approximately +31 Elo; score 54.5%). Commit `078ef78` makes race2 run only at
  PV/full-window nodes in production `titanium-v17`; explicit
  `titanium-v17-race2w` preserves the slower full-tree control. Artifacts:
  `tools/binary_match/runs/race2pv_vs_race2w_20260715/`.
- The current six-plane CAT network was already retrained for one epoch after
  the input-plane work. Accepted blob SHA-256 is
  `3d92ec1d1ed9aa935a7eb82ebe5d271373a7dc9500f0da4b5e9fa423b6bb0b86` and
  exactly matches `engine/src/titanium/net_weights.bin`. Epoch 1 used 8,192
  samples, reduced training loss from 0.53306 to 0.48723, validation loss was
  0.578592, and its strength gate was `104 - 96` (approximately +14 Elo,
  pair-sign p=0.298). It was accepted by explicit owner override; do not rerun
  the same epoch as though it were unfinished.

- Time-control-adaptive depth-4 RFP is **rejected**. Its first run was invalid
  because the candidate label accidentally disabled ACE RFP; the corrected,
  valid run was stopped on owner instruction at 153 games: local `18 - 16`,
  Oracle `45 - 74`, combined `63 - 90` (0 draws, score 41.2%, approximately
  -62 Elo). Never retry this depth-4 gate without a materially different idea.
  Artifacts: `tools/binary_match/runs/rfp_tc_d4_vs_v17_20260715b/`.
- One-wall PV-only is **rejected / no promotion**. Current all-node race1
  versus `titanium-v17-race1pv`: local `21 - 27`, Oracle `78 - 74`, combined
  `99 - 101` over 200 valid games (0 draws, score 49.5%, approximately -3.5
  Elo). Production retains all-node one-wall proof. Artifacts:
  `tools/binary_match/runs/race1pv_vs_race1all_20260715/`.

Clean race1 artifacts:

- local: `tools/binary_match/runs/race1w_gate_20260715/local/`
- Oracle resumed/fixed: `tools/binary_match/runs/race1w_gate_20260715/oracle_resume/`
- the first Oracle artifact contained 21 invalid `engine_dead`/`no_move` rows;
  those rows were excluded and rerun. Do not quote the contaminated raw score.

## Active gate

No strength gate is currently active. The strict immutable-path race projection
is parked, not deployed. Its initial implementation was reverted in engine
commit `cd494d7` after the full-wall minimax fixture did not confirm the claimed
terminal depth. Resume only with a standalone, exhaustive test harness. The
supplied two immutable-path position strings remain useful negative cases:
Black can still legally place an early path-intercepting wall, so neither may
certify.

The match harness rejects `engine_dead` and `no_move`, restarts both warm
sessions, and requeues the same game up to three times. Invalid attempts are
written separately and never score as wins.

## Ordered engine backlog

1. WALLQ-TC dual-band leaf correction from `Downloads/search.rs` was audited,
   not ported. Its effect is an exact one-wall interdiction correction at leaf
   nodes only inside a ±150cp alpha/beta relevance band, with isolated fast and
   long time-control TT namespaces. Titanium does not currently retain the
   required exact per-side one-wall damage feature; calculating it per leaf
   would reintroduce the expensive legal-wall/BFF work already removed from
   CAT evaluation. Treat it as a separate feature/data project, not a
   low-hanging search patch. Depth-4 time-control-adaptive RFP was measured and
   rejected. Existing predictive stop, history, LMR, LMP, CMH/countermove,
   correction history, and aspiration must not be reimplemented.
2. Add reusable engine time management: reserve a minimum plausible remaining
   ply floor, spend more on unstable PV/close root alternatives, spend less on
   stable easy moves, and preserve an emergency move reserve. Tune with mirrored
   randomized self-play and verify zero time forfeits before strength claims.
3. Keep exact race math clean: bound-only semi-terminal deductions may prune only
   when they cross alpha/beta; exact DTM is lazy and only used for same-outcome
   finalists/UI. Add stubborn-loser behavior only as a policy after the exact
   solver result, never inside the proof.
4. Revisit walls-remaining immutable-route certificates only behind a flag and
   require zero false-positive winners against Canta/exhaustive/randomized
   opening, middlegame, and endgame counter-oracles before any strength gate.
5. Movegen/pathfinding ideas already tested must not be blindly repeated:
   Titanium already has BFF layers, topology gates, and free final-flood witness
   information. Any explicit route-witness/Lee-BFS addition must first beat the
   current non-TT perft(4) opening/middlegame/endgame battery with identical
   nodes and report both time and NPS.
6. After the backlog above, test a new flagged race-certificate gate built from
   the current position's BFF data: reconstruct a concrete shortest edge path
   for each player, map every legal remaining wall to the path edges it can
   block, and admit the race deduction only when no legal wall can affect the
   certified winner's route for the required future pawn positions. A single
   current path-miss is not a whole-game proof; validate the temporal invariant
   with exhaustive/randomized Canta counter-oracles before measuring Elo.

## Separate website backlog

- Undo clock restoration and settings-after-flag recovery are fixed and pushed
  to website `main` as `63e33a3`. Unit tests cover canonical clock-log trimming
  and clearing a time-only result; a real browser test flagged a 0.25-second
  human clock, applied 60-second settings, and resumed at approximately 58s.
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
