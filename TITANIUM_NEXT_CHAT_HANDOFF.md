# Titanium v17 — fresh-chat handoff

Last updated: 2026-07-16

This file is the **only canonical source of truth** for resuming engine work.
Read it before running an experiment or changing a search flag. Update this
file in the same commit as every accepted, rejected, parked, or superseded
engine decision; do not create a parallel status log. It deliberately separates
accepted production work, rejected work, parked ideas, and uncommitted state.

## Repository state

- Workspace root: `C:\gitProjects\Quoridor best AI`
- Engine repository: `C:\gitProjects\Quoridor best AI\engine`
- Engine `HEAD`: `cd494d7` — revert of the unverified immutable-path PV
  projection. The prior committed candidate `df15485` is intentionally not
  active.
- Root branch: `codex/diversity-prep-only`.
- Do not create a hidden engine branch. Work directly from the engine baseline,
  using a small isolated commit only after a feature passes its required tests.
- Do not touch Adaptive TT. It is a known-good design and is out of scope.

### Dirty state that belongs to the user / other work

In `engine`, leave these tracked changes alone:

- `src/titanium/dist.rs`
- `src/titanium/net_weights.bin`

There are many generated `target-*`, `runs/`, and experiment-script files. Do
not clean, reset, delete, stage, or commit unrelated work. The external
reference files in `C:\Users\Terminatort8000\Downloads\` are not to be copied
into the engine or committed.

## Current production baseline — active and accepted

The named production engine is `titanium-v17`.

| Change | Commit(s) | Evidence / decision |
| --- | --- | --- |
| Correct race score semantics | `815a2ca` | **Active.** A proof bound, approximate distance, and exact DTM are distinct. Approximate values must never be encoded in the exact `RACE_MATE - n` band or displayed as “Win in N.” |
| CAT flood reuse | `164453b` | **Active.** Reuses CAT corridor flood results with identical CAT values; removes redundant work. |
| One-wall race proof | `2792020` | **Active, all eligible nodes.** 200 valid 60 s mirrored games: 104–96 vs prior baseline, about +14 Elo. |
| Two-wall race proof | `325d2fa`, `078ef78` | **Active only on PV/full-window nodes.** Full tree beat race1 105–95 (+17 Elo); PV-only then beat full tree 109–91 (+31 Elo). `titanium-v17-race2w` remains the slower full-tree control. |
| CAT six-plane one-epoch net | weights blob below | **Active, accepted owner decision.** 8,192 samples; train loss 0.53306 → 0.48723; validation loss 0.578592; 104–96 gate, roughly +14 Elo. Blob SHA-256: `3d92ec1d1ed9aa935a7eb82ebe5d271373a7dc9500f0da4b5e9fa423b6bb0b86`. |

The normal race rules remain:

- A semi-terminal proof may use a lower/upper bound only for alpha-beta
  cutoffs.
- Exact DTM is lazy: calculate it only for same-outcome finalists or an explicit
  UI request.
- Do not promote a lower bound or cheap distance estimate into exact DTM.
- A future “stubborn loser” policy belongs above exact game-theoretic solving;
  it must not change the proof result itself.

## Rejected, not enabled, or inconclusive

These are decisions, not invitations to repeat the same test.

| Candidate | Result | Decision |
| --- | --- | --- |
| Time-control-adaptive depth-4 RFP | Corrected valid gate stopped at 153 games: local 18–16 and Oracle 45–74; combined 63–90, about −62 Elo. An earlier run was invalid because the candidate label accidentally disabled ACE RFP. | **Rejected.** Do not retry depth-4 RFP without a materially different mechanism. |
| One-wall PV-only instead of all-node | 99–101 over 200 valid games, about −3.5 Elo. | **No promotion.** Production keeps all-node race1. |
| Two-way static-eval cache with the same capacity | Fixed-depth parity held, but timing was unreliable: the direct-cache midgame control alone ranged from 190k to 279k NPS. Candidate was never committed. | **Discarded as inconclusive.** Revisit only with pinned-core alternating A/B runs. |
| PV immutable-path projection | The full-wall minimax fixture did not confirm its claimed terminal depth. Candidate `df15485` was explicitly reverted by `cd494d7`. | **Parked; not deployed.** Need a standalone exhaustive verifier before any new implementation. |
| WALLQ-TC from `Downloads\search.rs` | Exact one-wall leaf correction within ±150 cp of alpha/beta requires an exact per-side one-wall damage feature Titanium does not retain. Adding it at every leaf would reintroduce legal-wall/BFF cost CAT removed. | **Not low-hanging; do not port now.** |

The old “rfp-ace promoted +3 Elo” line in the historical section of
`OVERNIGHT_ENGINE_HANDOFF.md` is superseded by the later, corrected
time-control depth-4 RFP result above. Do not treat that historical line as an
active promotion.

## CATv5 precise-witness experiment â€” current uncommitted work

CATv6 path-only is **rejected / discarded**. The initial implementation
replaced CATv5's broad Lee-wave propagation with a sparse four-path map. Its
200-game 100â€“100 result is invalid strength evidence because it tested that
wrong semantic change; a later corrected match was stopped at 126 games,
candidate 60â€“66, and is also not evidence. Preserve the artifacts for audit,
but do not restore, train on, or deploy CATv6 path-only fields or weights.

The owner-selected CATv5-precise candidate keeps CATv5's existing symmetric
BFF/Lee-wave propagation, heat LUT, distance bias, CAT search guidance, and
all existing LMR/move-order behavior. It replaces only the loose route sources
with up to four deterministic shortest paths per player. Subsequent paths may
reuse the pawn/current square and first ply, but squares from the second ply
onward are blocked. The path ranks are raw witness values `4, 3, 2, 1`, with
the higher rank taking overlap. Each precise path seeds the existing CATv5
Lee-wave propagation: this is deliberately not a sparse path-only heatmap.

NN inputs are exactly three CAT planes: raw White witness (0â€“4), raw Black
witness (0â€“4), and final combined propagated CATv5 heat, normalized `/256.0`
in the trainer exactly as in Rust. Do not add a raw combined witness plane: it
is a linear sum of the two raw inputs and adds no input information. Do not
alter CAT-dependent LMR/move ordering until this source and NN-input experiment
has independent evidence. The owner wants witnesses protected, not all
non-witness moves protected.

Uncommitted source is in `engine/src/cat/build.rs`, `titanium/search.rs`,
`titanium/net.rs`, `main.rs`, `search/v16_lmr.rs`, and the corresponding
streaming/training field modules. The failed CATv6 LMR/order experiment was
reverted to production behavior. Checks passed using isolated
`engine/target-catv5-precise-check` with `-C target-cpu=native`:

- `cargo check --bin titanium`;
- 18/18 `cat::build` tests, including deterministic witnesses and the
  assertion that witnesses seed propagation rather than form a sparse map;
- engine/Python parity against the isolated binary;
- release binary SHA-256
  `978059948daf16d57589fcaee13fbdc9686c1e6c3a736383896d6f254e51a6fe`.

`catv5_precise4_wave_comparison.png` is the corrected visualization. On its
legal 38-ply sample: baseline 57 active squares, precise CATv5 60, 34 changed,
L1 difference 2033. It is visualization only, not strength evidence.

### Fresh CATv5-precise epoch-1 result

A fresh direct-DB epoch completed, not resumed:

- start weights: accepted `training/runs/v16/accepted/epoch_0000.bin`, SHA-256
  `3a16efab1191ac163936b0abd8ae8c7c1a8061a031be9ce82fa1c01fb700073a`;
- run: `training/runs/catv5_precise_epoch1_20260715`;
- direct `labels.db` stream, no full-corpus RAM cache and `--no-usage-commit`;
  8,602 selected positions, 6,089 train / 2,513 validation, 4,096-position
  chunks, `RAYON_NUM_THREADS=8` for CAT extraction;
- train loss `0.562886 -> 0.519189`; validation loss `0.599743`;
  train 80.18 s, validation 32.63 s;
- candidate `net_weights_best.bin`, SHA-256
  `2517f9a56de136a0260bda662baca836ec17d340952927f162fbe4607dc52a0f`.

This candidate is **isolated and unaccepted**. Do not copy it into the
user-owned `engine/src/titanium/net_weights.bin`, deploy it, or claim a
strength result. It must first be built with precise CATv5 source, pass
parity/fixed-depth checks, then play the required 200-game gate against
accepted CATv5 if behavior differs.

**Superseding input-orientation correction (2026-07-16):** Titanium's NNUE
side-to-move mapping was found to reflect only the row (`8-row, col`) instead
of rotating the board 180 degrees. `NET_MIRC` and `NET_MIRS` now reverse both
row and column in Rust and every active Python featurization/training path.
This changes feature coordinates, so the CATv5-precise epoch-1 blob above is
stale and must not enter the 200-game gate. Retrain the epoch from the accepted
starting checkpoint using the corrected 180-degree featurizer, re-run
Rust/Python parity, then build and gate that new blob. Existing accepted
production weights remain paired with the old production binary until a
corrected-orientation candidate passes its own strength gate.

## Move generation / BFF facts already established

- 27/27 movegen tests passed and `perft_full_compare` passed.
- Prior measurements: perft(5) with TT/topology was about 12.7–15.7 s;
  non-TT perft(4) topology vs anchor was 0.999 s vs 15.9 s (same nodes,
  roughly 16x). The old no-TT perft(5) 20 s abort is expected.
- Titanium already has BFF layers, topology gates, and free final-flood witness
  information. Do not claim an external `bfs.rs` introduces BFF itself.
- Any proposed concrete route witness / Lee-style shortest-route reconstruction
  must first beat the existing identical-node non-TT perft(4) battery on
  opening, middlegame, and endgame. Report time **and** NPS.
- The supplied external `bfs.rs`, `movegen.rs`, and `search.rs` are reference
  material only. Do not copy them into the repository. Their useful ideas must
  be independently expressed and tested.

## Parked immutable-route idea — do not work on it now

The intended future idea is not “a wall misses the current path.” It would need
to prove that, for every future pawn position on the claimed winner route, no
remaining legal wall can delay the route at the time it could be placed.

Two supplied negative fixtures must reject because Black has an early legal
path-intercepting wall:

```text
d1 d9 d2 d8 d3 d7 d4 d6 d5 d4 d6 d6v d8v b8v b6v c6v d5 d3 c5 d2 d5 c2 d4v b4v c4v b2 d4 b3 d3 d2v b2v c1h

d1 d9 d2 d8 d3 d7 d4 d6 d5 d4 d6 d6v d8v b8v b6v c6v d5 d3 c5 d2 d5 c2 d4v b4v c4v b2 d4 b3 d3 d2v b2v c1h c3 b4 d3 b5 c3 b6 d3 b7 c3 b8 d3 b9 c3 a9 c2 b9 c3 c2h
```

Before returning to this idea: build an independent exhaustive full-wall
minimax harness, establish zero false-positive certificates across exhaustive,
randomized opening/middlegame/endgame cases, then gate it behind a flag. Only
then consider an Elo match.

## What the Downloads search file actually adds

`C:\Users\Terminatort8000\Downloads\search.rs` is a reference engine with many
features Titanium already has: predictive stop, history, LMR, LMP,
CMH/countermove, correction history, aspiration, and RFP variants. Do not
reimplement those by name.

The only clearly distinct feature found was the costly WALLQ-TC leaf correction
described above, which is parked. Therefore the correct immediate task is to
profile the current production source, then choose one small hotspot—not to
blindly copy a subsystem.

## Search-file review update (2026-07-16)

The earlier "only clearly distinct" conclusion needs this addendum. WALLQ-TC
is still parked: it needs exact one-wall damage at the leaf and would
reintroduce legal-wall/BFF work CAT removed.

Two additional bounded experiments were identified. Neither is implemented or
measured:

1. **LR-symmetric TT canonicalization (first search candidate).** Canonicalize
   a TT state against its left/right reflection. This is exact Quoridor
   symmetry and may increase effective TT reuse without changing search
   semantics. Keep it behind a flag; reflect cells as `(r, c) -> (r, 8-c)`,
   wall slots as `(r, c) -> (r, 7-c)`, preserve wall orientation, and
   unreflect a stored TT move before use. Do not touch Adaptive TT
   sizing/replacement. Require reflection-involution, legal-move bijection,
   equal-key, and fixed-depth parity tests before pinned alternating NPS
   measurement. It may lose to the extra key/mask work, so benchmark first.
2. **Conservative timed-search bank (second candidate).** Save unused
   per-move allocation in a per-game bank and spend more only in tense
   positions. This is strictly a clock-management experiment: never borrow
   future time; constrain allocation to 0.4xâ€“1.5x base;
   log base/allocated/spent/bank; test only in timed games, including
   time-loss counts. It must not alter fixed-depth search.

The external source's probable-wall/LMR and history variants overlap existing
CAT ordering/search work and are not current port candidates. Do not copy its
WALLQ-TC subsystem. The time-bank idea is from this downloaded alpha-beta
`search.rs`, not from Claustrophobia.

## Claustrophobia findings — training research, not a production port

`Plaaasma/Claustrophobia` is an AlphaZero-style batched-MCTS engine; tree
reuse, Gumbel root selection, and MCTS solver are a different engine design,
not a low-cost Titanium search optimization. Its encoder is nevertheless a
useful isolated model experiment. It implements the proposed 16-plane,
side-to-move canonical representation with genuine 180-degree rotation (cell
`80-c`, wall slot `63-s` for the second player), corroborating the Titanium
orientation correction.

Important detail: its threat planes are **not** based on a single
most-delaying wall. They take the pointwise maximum distance field over every
legal path-cutting wall. If explored, build an offline-only encoder/training
data pipeline; first ablate planes 0â€“13 and then 14â€“15; cache threat labels;
never compute them at alpha-beta leaves. This requires fresh model training
and must not be mixed into the current HalfPW/NNUE candidate.

For future self-play, consider its promotion discipline: gate against the
current incumbent plus frozen anchors and an external fixed sentinel, so
self-play drift cannot look like Elo gain. The present 200-game incumbent gate
remains mandatory.

### Claustrophobia-derived training repair now active (2026-07-16)

The previous fresh+5%-refresh training recipe is superseded. It did not
guarantee that trusted teacher rows appeared in optimizer batches and therefore
was not a real anti-forgetting scheme. Treat all prior Titanium training as
unproven until an externally anchored candidate passes strength gates.

The repaired HalfPW input boundary has five CATv5 planes, all canonicalized to
the side to move and bounded in `[0,1]`:

- mover and opponent precise path-rank maps: raw `0..4` divided by `4`;
- mover and opponent propagated CAT maps: raw `0..200` divided by `200`;
- combined propagated CAT map: raw `0..400` divided by `400`.

The raw rank maps preserve which of the four deterministic unique shortest
paths owns each cell; the paths may share only the already-defined first ply.
Separate binary planes were rejected for now because they would add eight
81-cell hot-eval dot products. The five-plane representation adds only the two
missing per-side propagated dot products. Search retains its existing integer
CAT resolution and thresholds. The NN boundary alone normalizes CAT.

P2 canonicalization remains genuine 180-degree rotation plus role swap.
Left/right reflection is a separate 50% training augmentation applied after
side-to-move canonicalization. Both transformations use lookup tables. A
second pre-rotated CAT-weight table was considered and deliberately omitted:
one existing `NET_MIRC` lookup per cell is negligible beside CAT construction
and five dot products, and no measured speedup justified extra machinery.

The accepted three-CAT-plane epoch-1 blob is converted losslessly at load:
old raw-witness weights are multiplied by `4`, old combined `/256` weights by
`400/256`, and the two new per-side propagated weights start at zero. The
start-position evaluation was exactly `56 cp` before and after conversion.

Training is now explicitly staged:

1. Bootstrap only from protected `teacher_dataset_good` rows. No Titanium
   self-play is allowed into this epoch.
2. After the bootstrap candidate passes both the incumbent and frozen-anchor
   gates, continuous epochs use 80% fresh, 10% recent replay, and 10% protected
   teacher anchor. Cohorts are interleaved through every full minibatch.
3. The trainer writes `cohort_manifest.json`, attaches cohort metadata to every
   optimizer batch, and fails closed on missing or drifting composition.
4. Candidate export uses EMA. Bootstrap uses `0.99` because it has only about
   223 steps; `0.999` would retain about 80% initialization over such a short
   run. Longer continuous training may use `0.999` after measuring its horizon.
5. No candidate may deploy from validation loss alone. Gate against accepted
   epoch 1 and a frozen absolute baseline using paired openings; retain the
   existing 200-game protocol for a promotion decision.

Focused verification completed:

- Python: normalized range/role-rotation, LR-reflection involution, and exact
  per-batch cohort mix tests: 3 passed;
- Rust CATv5 focused tests: 2 passed, including fixed normalization bounds;
- native `cargo check --bin titanium` passed;
- real engine identity/parity preflight passed;
- 1,024-row anchor-only optimizer smoke: 979 train / 45 validation, train loss
  `0.39459 -> 0.39132`, validation loss `0.38353`, 100% reported external
  teacher anchor, EMA weights exported.

Active bootstrap epoch (reproducible restart):

- run: `training/runs/catv5_normalized5_teacher_bootstrap_epoch1_20260716_r3`;
- launch PID: `21216` (PID is observational; inspect the log/process rather
  than assuming it survives a reboot);
- exact sample: 120,000 protected teacher positions; deterministic split
  114,011 optimizer rows / 5,989 validation rows; hard minimum 100,000;
- 50% LR augmentation, batch 512, LR `2e-4`, weight decay `1e-5`, EMA `0.99`;
- accepted seed: `training/runs/v16/accepted/epoch_0001.bin`, SHA-256
  `3d92ec1d1ed9aa935a7eb82ebe5d271373a7dc9500f0da4b5e9fa423b6bb0b86`;
- feature extraction uses `RAYON_NUM_THREADS=4`; the H=32 optimizer remains one
  PyTorch thread because measured 4-thread steps were slower (`177 ms` vs
  `129 ms`). This uses four cores where they accelerate CAT/BFF extraction.
- clean engine source worktree: `engine/.clean-build-426361a`, detached exactly
  at pushed commit `426361a`; unrelated dirty engine edits are excluded;
- engine: `engine/target-catv5-anchored-clean-426361a/release/titanium.exe`,
  SHA-256 `a4444f8036de69ba255b1e5f31181f8ecff4e5ec2e2631c4cf963a26b473e8b8`;
- stdout/stderr are in the run directory. At handoff the process was active and
  epoch 1 had entered its optimizer/featurization loop.

The first launch without a suffix was intentionally stopped after 25/223 steps
because its binary included an uncommitted, semantically neutral pre-rotated
CAT weight table. `_r2` was then stopped before optimizer progress because the
main engine worktree also contains unrelated tracked edits. No usage state was
consumed by either. `_r3` uses the clean detached commit build, passed 6/6
Rust/Python parity at exact 0 cp difference, and is the only valid bootstrap.

Do not deploy this run automatically. When it finishes: verify cohort and
weight diagnostics, run Rust/Python parity with the new blob, then gate first
against accepted epoch 1 and separately against the frozen absolute baseline.

Persistent rollback DSU for wall-cycle legality was implemented and rejected:
it preserved exact results but was about 1.1% slower aggregate versus the
maintenance-only control. Do not revisit without a new measured design.

## Time management — evidence and ordered backlog (2026-07-16)

### What the existing game database says

Read-only scan of `tools/binary_match/runs/**/results_*.jsonl`: 74 files,
4,032 clocked records, of which 1,801 are completed `termination=goal` games
with both 60-second clocks recorded. The other records are incomplete,
engine-dead/no-move, or historical rows without a termination field, so they
must not be used as a game-length prior. This corpus contains different
experiments and retries; use it as a scheduling prior, not as independent Elo
evidence.

Completed-game length: mean 58.86 plies; P50 59, P90 71, P95 77, P99 86,
maximum 99. Conditional remaining-game horizon, where the own-move value is
remaining plies / 2:

| Current ply | Samples reaching ply | Mean remaining own moves | P90 | P95 |
|---:|---:|---:|---:|---:|
| 0 | 1,801 | 29.43 | 35.5 | 38.5 |
| 20 | 1,801 | 19.43 | 25.5 | 28.5 |
| 40 | 1,769 | 9.63 | 16.0 | 18.5 |
| 50 | 1,489 | 5.90 | 11.5 | 14.0 |
| 60 | 817 | 3.65 | 8.5 | 11.0 |
| 70 | 229 | 3.38 | 8.0 | 9.0 |
| 80 | 62 | 2.27 | 5.0 | 5.0 |

At a fresh 60-second clock this implies 2.04 s/move from the mean horizon,
1.69 s/move from the P90 horizon, and 1.56 s/move from the P95 horizon. These
are schedule priors only; recalculate from the *current* remaining clock at
every move. Existing terminal clocks are not a deadline-buffer measurement:
their per-side P05 is 19.42 s, median 29.17 s, and minimum 2.16 s, but the
logs do not record allocated time, actual elapsed time, deadline overshoot, or
transport latency.

Current Titanium already has a coarse soft bound (85%, or 92% when losing), a
max-of-last-two-iterations predictive start check, stable/easy-move stop, and
partial-iteration recovery. It does **not** have an empirical game-length
schedule, an observed safety-buffer quantile, or a persistent per-game bank.

**Do not assume every position has 30 moves left.** The database horizon is a
fallback/regularizer only. At every root calculate two different quantities:

- A strict, optimistic lower bound on terminal ply from geometry: a pawn can
  gain at most two goalward rows in one legal move (a straight jump), so for a
  player with `rows_to_goal` remaining, `own_turns_lb = ceil(rows_to_goal/2)`.
  If that player is to move, its earliest terminal move is
  `2*own_turns_lb - 1`; otherwise it is `2*own_turns_lb`. The minimum over the
  two players is a physical lower bound even if all future walls cooperate.
  Current wall/BFS distance alone is not this bound because a future pawn jump
  can beat a cell-edge distance.
- A statistical reserve horizon: position-conditioned expected/P90/P95 moves
  left, falling back to the table above when no calibrated model exists. This
  higher horizon, not the lower bound, protects against time loss. There is no
  useful finite game-length upper bound from Quoridor geometry alone; games can
  wander, so the database supplies a quantile rather than a guarantee.

PV-leaf distance-to-win is neither bound by itself: it is useful as a tension
signal, but walls, a changed PV, or a jump can invalidate it. It can release
banked time only when corroborated by a stable root/PV and a proven or
calibrated race condition; it must never replace the statistical reserve.

### Ranked implementation plan — easiest / most likely first

1. **Timing telemetry only — do first.** No decision change. Per move log:
   ply, side, clock on entry, allocated soft/hard time, reserve, elapsed,
   return/overshoot milliseconds, completed depth, last two iteration costs,
   best-move changes, score deltas, aspiration retries, partial-iteration use,
   and root RFP/LMP/LMR/pruning counters. Include browser/worker transport
   timing separately where relevant. Compute P50/P95/P99/P99.9 overshoot and
   transport delay by platform. The safety buffer becomes measured as
   `max(P99.9 overrun + P99.9 transport, fixed scheduler margin)`; do not
   invent a millisecond constant from terminal clocks.
2. **Position-aware game-length scheduler — simple, likely win.**
   Feature-gated; fixed-depth unchanged. First calculate the geometric terminal
   lower bound above; then use a calibrated position-conditioned P90/P95
   reserve horizon (or the conditional database table as fallback) to set
   `spendable_clock / reserve_moves`, with a minimum budget and the
   telemetry-derived safety buffer reserved first. Use the lower bound/PV only
   to control when saved time may be released, never to reserve too little.
   Compare against the present 85/92% policy under actual 60-second matches;
   report time losses, remaining clock, completed depth, and Elo.
3. **Search-stability modifier — bounded.** On top of the schedule only:
   save time when the best move is unchanged across depths, score variance is
   low, no aspiration retry occurs, and the root margin is clear. Permit extra
   time only for repeated move changes, score swings, aspiration failures, or
   an unfinished critical iteration. Require offline calibration against the
   next completed depth; do not guess thresholds from a few games.
4. **Futility/pruning tension modifier — plausible but unproven.** Add the
   already-instrumented root RFP/LMP/LMR pressure, cutoff volatility, and
   pruning-versus-full-search disagreement as *candidate* instability signals.
   First test whether they predict a later PV/score reversal after controlling
   for depth and position phase. Only then allow a small, capped multiplier;
   its sign must be learned from that correlation, not assumed.
5. **Conservative per-game time bank — medium complexity.** Save surplus from
   low-tension moves and allow high-tension moves to spend only saved credit;
   never borrow future clock. Start with a 0.4xâ€“1.5x multiplier around the
   scheduled base and persist/reset the bank exactly once per game. This is
   the useful idea from downloaded `search.rs`; evaluate after steps 1â€“3 so
   the bank has calibrated inputs.
6. **Critical-position verification budget — hard/risky.** Reserve a bounded
   optional re-search or root verification only when the calibrated stability
   and futility signals agree. It competes directly with normal ID depth and
   needs separate ablations; do not combine it with the first time-bank test.
7. **Learned time policy — hardest / least justified now.** Train from the
   telemetry to predict value of one more iteration or probability of PV
   reversal. This needs a held-out corpus, calibration, and strict clock-loss
   constraints; it is not a near-term port from Claustrophobia.

For every behavior-changing timing candidate: run a small timed smoke first,
then the existing mirrored 200-game 60-second gate. Fixed-depth parity is not
applicable to the allocator itself, but its search semantics must remain
identical at a fixed supplied budget.

## Immediate next task: validate CATv5 precise

1. Inspect the uncommitted diff. Preserve protected dirty `dist.rs` and
   `net_weights.bin`; do not touch unrelated `search_bench.rs` instrumentation
   or generated targets/runs.
2. Confirm no CATv6 path-only source or training-data path is reachable. Do
   not delete old experimental artifacts merely to clean the tree.
3. Build isolated accepted-CATv5 and precise-CATv5 release binaries. Use the
   isolated epoch-1 candidate with `TITANIUM_NET_WEIGHTS_PATH`; never overwrite
   embedded production weights.
4. Run exact fixed-depth parity on startpos, c3h-midgame, and wall-maze, plus
   relevant movegen/perft differential tests. Then report pinned-core,
   alternating opening/midgame/endgame CAT calls/sec and end-to-end NPS/wall
   time.
5. If behavior differs and validation holds, run the required 200-game gate
   against accepted CATv5. Otherwise explicitly reject/park with evidence.
   Update this file in the same scoped commit only when a decision is made.

## Later task: disciplined low-hanging search optimization

1. Finish a fresh `bench-instrument` build of the **current** engine source
   under an isolated generated target directory, then profile production v17 on
   `startpos`, `c3h-midgame`, and `wall-maze` / `endgame-c5`.
2. Select one small, semantics-preserving hotspot. Likely candidates to inspect
   first are:
   - repeated route-feature accumulation / bit-count work in CAT evaluation;
   - repeated direction-mask reconstruction if it remains hot in the fresh
     profile;
   - another cache/layout refinement only if a properly pinned benchmark can
     establish a stable signal.
3. Do not add a heuristic or change search decision semantics just because it
   makes a profile faster.
4. Make a candidate only in the engine source, run:
   - formatting/checking;
   - fixed-depth parity on `startpos`, `c3h-midgame`, and `wall-maze`;
   - current perft/movegen differential tests relevant to modified code;
   - pinned, alternating baseline/candidate timing on opening, midgame, and
     endgame, reporting NPS and wall time.
5. Commit only if parity is exact and the timing result is repeatable. For a
   pure performance change, then run a strength gate only if it changes search
   behaviour; otherwise preserve the benchmark evidence and promote it as a
   performance-only commit.

Do not run an Oracle strength gate merely to profile. No strength gate is active
now. Use the existing 200-game protocol only after a behaviour-changing
candidate is ready.

## Strength-gate protocol

- 200 games, randomized mirrored 10-ply openings, 60 s per side.
- Local: 4 workers, shards 0–3 of 17.
- Oracle: 13 workers, shards 4–16 of 17. Upload the prebuilt engine binary only,
  not a source archive.
- Merge into one `status.json` at the run root. Invalid `engine_dead` and
  `no_move` games are excluded, sessions restarted, and the same game requeued
  up to three times.
- Record raw W-D-L, valid game count, score, approximate Elo, artifact path,
  and the explicit promote/reject decision in `OVERNIGHT_ENGINE_HANDOFF.md`
  and this file.

## Website status and remaining work

Already fixed and pushed in website repository commit `63e33a3`:

- undo restores spent time correctly;
- applying settings clears a time-only forfeit and permits play to resume;
- canonical clock-log trimming and time-result clearing have tests.

Still to do later:

- feed time allocation explicit engine telemetry (PV stability and root-move
  uncertainty);
- ensure the UI uses exact DTM only when engine metadata says it is exact;
- never infer “Win/Loss in N” from a score magnitude alone.

## Files and commands worth using

- Canonical running log: `OVERNIGHT_ENGINE_HANDOFF.md`
- Search benchmark: `engine\src\bin\search_bench.rs`
- Current benchmark positions: `startpos`, `c3h-midgame`, `wall-maze`,
  `low-wall`, `endgame-c5`, `dense-maze`.
- Production selector: `$env:TITANIUM_BENCH_ENGINE = 'titanium-v17'`
- Existing match artifacts: `tools\binary_match\runs\`
- Existing harness commands and examples are in `OVERNIGHT_ENGINE_HANDOFF.md`.

Important operational rule: do not launch overlapping Cargo builds. Check for
existing `cargo` / `rustc` first. Do not use `cargo fmt --all` at engine root
unless deliberately prepared for unrelated formatting changes.

## Last actions before handoff

- CATv6 path-only was explicitly discarded. Its old artifacts are retained
  only for audit; neither its labels nor weights are inputs to the new run.
- CATv5 precise witnesses were implemented without changing CATv5 wave
  propagation or LMR/move ordering. The uncommitted source passed the listed
  CAT and parity checks.
- A fresh isolated CATv5-precise epoch-1 completed from accepted epoch 0; its
  weights remain unaccepted and undeployed.
- No strength gate is active. The generic training coordinator is deliberately
  idle below its queue threshold and does not track this isolated run.
