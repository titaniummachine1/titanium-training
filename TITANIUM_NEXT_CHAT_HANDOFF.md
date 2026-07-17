# Titanium v17 — fresh-chat handoff

Last updated: 2026-07-17

This file is the **only canonical source of truth** for resuming engine work.
Read it before running an experiment or changing a search flag. Update this
file in the same commit as every accepted, rejected, parked, or superseded
engine decision; do not create a parallel status log or extra `HANDOFF_*.md`
files for task queues. It deliberately separates accepted production work,
rejected work, parked ideas, and uncommitted state.

## Repository state

- Workspace root: `C:\gitProjects\Quoridor best AI`
- Engine repository: `C:\gitProjects\Quoridor best AI\engine`
- Engine `HEAD`: `main` — **one-side-broke** (`e0d47d3`) + **jump-aware dual
  distance** (±1 tempo eval upgrade). Parent for jump A/B: `e0d47d3`.
- Root branch: `codex/diversity-prep-only` (meta commit pins engine submodule).
- Do not create a hidden engine branch. Work directly from the engine baseline,
  using a small isolated commit only after a feature passes its required tests.
- Do not touch Adaptive TT. It is a known-good design and is out of scope.
- Do not invent new remaining-wall monopoly-N solvers; prefer oracle/certify.

### Dirty state that belongs to the user / other work

In `engine`, leave unrelated dirty files alone if present. Do not clean, reset,
delete, stage, or commit unrelated work. The external reference files in
`C:\Users\Terminatort8000\Downloads\` are not to be copied into the engine or
committed.

## Cursor: start here

**Broke-side gate — KEEP (owner, 2026-07-17):** Adaptive stage 1 finished
100/100 games (`broke_side_adaptive_20260717_202318`). Combined **53–47**
(candidate A vs baseline B), score 0.53 (~+21 Elo). Wilson 95% CI inconclusive
(`lb≈0.43`, `ub≈0.62`) but owner keeps broke-side on `main`. Engine `main`:
`e0d47d3` + jump-aware mid-tier commit follows.

**Next immediate gate:** A/B **jump-aware dual distance** (precise BFF twin in
±1 tempo band) vs parent without it. Same harness (`run_broke_side_adaptive.ps1`),
100→150 adaptive, native build both SHAs.

```powershell
$env:RUSTFLAGS = '-C target-cpu=native'
Remove-Item Env:TITANIUM_ALLOW_SUBOPTIMAL -ErrorAction SilentlyContinue
cargo build --release -p titanium --bin titanium --manifest-path engine\Cargo.toml
```

Broke-side theorems (sound only, **active on main**):

- `wl[opp]==0` and STM wins pure race → `Lower(RACE_MATE - dtm)`
- `wl[stm]==0` and STM loses pure race → `Upper(-(RACE_MATE - dtm))`
- Not a cut: wallless side wins while opp still has walls; walled side loses pure race
- Proofs use `race_tbl` (jumps), never eval BFF

Certified race coverage now:

```text
both 0 walls     → Gate1 + exact race_tbl DTM
one side broke   → refuse-to-place Lower/Upper (e0d47d3)
1 wall           → sound subset
2 walls          → monopoly subset (often PV-only on v17)
both armed >2    → search / optional certify / wall_ignore (off)
```

### Ordered engine task queue (oracle-first)

Do **not** start P1+ until the **jump-aware** A/B gate above is recorded.

**Oracle organization and time management:** Length bounds are directly useful
for time management. A sound lower bound on remaining game length (minimum
plies to any terminal) prevents dumping the clock too early and helps pace
search. A known upper bound (maximum plies until a forced end) permits
aggressive spending or race-to-goal mode near a forced short game. Exact race
DTM, when available, is both a score bound and an exact length for time
management. Keep one search-facing facade (for example, a race/oracle API)
returning typed results. Internally organize producers by responsibility:
score-Lower, score-Upper, length-min, length-max, and exact-DTM. Do not split
the repository into independent Upper-only and Lower-only crates/modules that
duplicate shared topology/path logic. Existing
`RaceBound::{Lower,Upper,Exact,Unknown}` remains the alpha-beta cut type.
Add a future `LengthBound { min_plies, max_plies }` (or equivalent `Option`
fields), consumed by time management, horizon logic, and extensions; keep it
separate from score cuts so alpha-beta cannot confuse “I win” with “the game
ends in ≤N”. When budget allows, `certify()` should emit both a score
`RaceBound` and a `LengthBound`, aligned with C1.

**Jump-aware dual distance (mid-tier, landed):** When BFF dual distances put the
race inside the ±1-tempo band (`bff_tempo_margin_close`), eval upgrades to
`jump_aware_goal_distances` — same two-player distance shape as BFF, but
jump-correct via `gen_pawn_moves` BFS with frozen opponent. Cheaper than
`race_tbl`; used for soft eval / hands-empty heuristic only. Hard αβ
Lower/Upper still require audited Gate1 (`delta_eta > 1`) or `race_tbl`.
Stats: `jump_dist_calls`, `jump_dist_upgrades`, `jump_dist_cuts_avoided`.

1. **O1** After the A/B gate, warm/share `race_tbl` across broke / 0-wall /
   1w / 2w probes.
2. **O2** (or **O2b**) Expose exact/min/max length to time management and soft
   evaluation, not only the current ±1800 soft score.
3. **O3** Oracle regression pack (empty + walled + jump-heavy); **done** —
   tests landed and passed; still no Elo claim. Never ship race changes
   without it.
4. **C1** `certify()` → typed `RaceBound` + `LengthBound` + budget (kills
   future N-wall special cases).
5. **C2** Measure `wall_ignore` loss cert; enable only if stats + Elo say so.
6. **S1** Move ordering from race ETA / tempo margin.
7. **S2** LMR/extensions on race-critical positions.
8. **N1/N2** NNUE features from oracle (ETA, margin, corridor) — long cycle.
9. **F1/F2** Whole-game flamegraphs; optimize CAT heat / eval only with evidence
   (CAT was ~11–17% exclusive; race ≈ 0%; 1t NPS ~200k on current builds).

Non-goals: `three_wall_monopoly_bound`; BFF-only win/loss proofs; Elo claims
without the A/B gate.

Tools: `tools/profile_tt_per_think/` (collect / flamegraph / `measure_broke_side_stats.py`).

---

Accepted training/search baseline context below remains historical through
2026-07-16 unless superseded by the Cursor start-here block above.

## Current production baseline — active and accepted

The named production engine is `titanium-v17`.

| Change                                  | Commit(s)                               | Evidence / decision                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Correct race score semantics            | `815a2ca`                               | **Active.** A proof bound, approximate distance, and exact DTM are distinct. Approximate values must never be encoded in the exact `RACE_MATE - n` band or displayed as “Win in N.”                                                                                                                                                                                                                                                                                |
| CAT flood reuse                         | `164453b`                               | **Active.** Reuses CAT corridor flood results with identical CAT values; removes redundant work.                                                                                                                                                                                                                                                                                                                                                                   |
| One-wall race proof                     | `2792020`                               | **Active, all eligible nodes.** 200 valid 60 s mirrored games: 104–96 vs prior baseline, about +14 Elo.                                                                                                                                                                                                                                                                                                                                                            |
| Two-wall race proof                     | `325d2fa`, `078ef78`                    | **Active only on PV/full-window nodes.** Full tree beat race1 105–95 (+17 Elo); PV-only then beat full tree 109–91 (+31 Elo). `titanium-v17-race2w` remains the slower full-tree control.                                                                                                                                                                                                                                                                          |
| Normalized precise CATv5 five-plane net | `426361a`, `03856fe` + accepted epoch 2 | **Active, accepted by owner override, embedded and pushed.** Correct 180-degree side-to-move orientation; raw precise witnesses plus per-side and combined propagated CAT. The final 200-game gate was 96–104 (48%, no draws/errors), statistically consistent with equal strength. Blob SHA-256: `3e8a87965cce61b12c642db3cb74cb8a7613c618144c725b250869a2890df1e1`. Full checkpoint SHA-256: `bea6718721b211022e36aa801eafce6a4f8d490ea98e6bb9b9a370f82f303aaa`. |

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

| Candidate                                        | Result                                                                                                                                                                                                              | Decision                                                                                       |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Time-control-adaptive depth-4 RFP                | Corrected valid gate stopped at 153 games: local 18–16 and Oracle 45–74; combined 63–90, about −62 Elo. An earlier run was invalid because the candidate label accidentally disabled ACE RFP.                       | **Rejected.** Do not retry depth-4 RFP without a materially different mechanism.               |
| One-wall PV-only instead of all-node             | 99–101 over 200 valid games, about −3.5 Elo.                                                                                                                                                                        | **No promotion.** Production keeps all-node race1.                                             |
| Two-way static-eval cache with the same capacity | Fixed-depth parity held, but timing was unreliable: the direct-cache midgame control alone ranged from 190k to 279k NPS. Candidate was never committed.                                                             | **Discarded as inconclusive.** Revisit only with pinned-core alternating A/B runs.             |
| PV immutable-path projection                     | The full-wall minimax fixture did not confirm its claimed terminal depth. Candidate `df15485` was explicitly reverted by `cd494d7`.                                                                                 | **Parked; not deployed.** Need a standalone exhaustive verifier before any new implementation. |
| WALLQ-TC from `Downloads\search.rs`              | Exact one-wall leaf correction within ±150 cp of alpha/beta requires an exact per-side one-wall damage feature Titanium does not retain. Adding it at every leaf would reintroduce legal-wall/BFF cost CAT removed. | **Not low-hanging; do not port now.**                                                          |

The old “rfp-ace promoted +3 Elo” line in the historical section of
`OVERNIGHT_ENGINE_HANDOFF.md` is superseded by the later, corrected
time-control depth-4 RFP result above. Do not treat that historical line as an
active promotion.

## CATv5 normalized precise witnesses — accepted baseline

CATv6 path-only is **rejected / discarded**. The initial implementation
replaced CATv5's broad Lee-wave propagation with a sparse four-path map. Its
200-game 100–100 result is invalid strength evidence because it tested that
wrong semantic change; a later corrected match was stopped at 126 games,
candidate 60–66, and is also not evidence. Preserve the artifacts for audit,
but do not restore, train on, or deploy CATv6 path-only fields or weights.

The accepted CATv5-precise implementation keeps CATv5's existing symmetric
BFF/Lee-wave propagation, heat LUT, distance bias, CAT search guidance, and
all existing LMR/move-order behavior. It replaces only the loose route sources
with up to four deterministic shortest paths per player. Subsequent paths may
reuse the pawn/current square and first ply, but squares from the second ply
onward are blocked. The path ranks are raw witness values `4, 3, 2, 1`, with
the higher rank taking overlap. Each precise path seeds the existing CATv5
Lee-wave propagation: this is deliberately not a sparse path-only heatmap.

NN inputs are exactly five CAT planes: raw mover witness (0–4), raw opponent
witness (0–4), propagated mover heat (0–200), propagated opponent heat
(0–200), and combined propagated heat (0–400). Rust and Python normalize them
to 0–1 using `/4`, `/200`, and `/400`. Do not add a raw combined witness plane:
it is a linear sum of the two raw inputs and adds no information. Do not alter
CAT-dependent LMR/move ordering without a separately gated ablation.

The engine implementation and correct 180-degree canonicalization are committed
at `426361a`; accepted weights are embedded by `03856fe`. Both are pushed on
engine `main`. The failed CATv6 LMR/order experiment was reverted to production
behavior. Checks passed using isolated clean builds:

- `cargo check --bin titanium`;
- 18/18 `cat::build` tests, including deterministic witnesses and the
  assertion that witnesses seed propagation rather than form a sparse map;
- engine/Python parity against the isolated binary;
- release binary SHA-256
  `978059948daf16d57589fcaee13fbdc9686c1e6c3a736383896d6f254e51a6fe`.

`catv5_precise4_wave_comparison.png` is the corrected visualization. On its
legal 38-ply sample: baseline 57 active squares, precise CATv5 60, 34 changed,
L1 difference 2033. It is visualization only, not strength evidence.

**Input-orientation correction (2026-07-16):** Titanium's NNUE
side-to-move mapping was found to reflect only the row (`8-row, col`) instead
of rotating the board 180 degrees. `NET_MIRC` and `NET_MIRS` now reverse both
row and column in Rust and every active Python featurization/training path.
This changed feature coordinates. All earlier CATv5-precise blobs, including
`training/runs/catv5_precise_epoch1_20260715`, are stale and must never be
loaded, resumed, gated, or deployed. Accepted epoch 2 is the first baseline
trained after this correction.

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
   future time; constrain allocation to 0.4x–1.5x base;
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
data pipeline; first ablate planes 0–13 and then 14–15; cache threat labels;
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

Completed bootstrap epoch (reproducible foreground run, exit code 0):

- run: `training/runs/catv5_normalized5_teacher_bootstrap_epoch1_20260716_r4`;
- launch PID: `19776` (PID is observational; inspect the log/process rather
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
- stdout/stderr are in the run directory. The valid foreground process exited
  normally after validation, checkpointing, and EMA export.

The first launch without a suffix was intentionally stopped after 25/223 steps
because its binary included an uncommitted, semantically neutral pre-rotated
CAT weight table. `_r2` was then stopped before optimizer progress because the
main engine worktree also contains unrelated tracked edits. No usage state was
consumed by either. `_r3` was incorrectly believed dead, but its detached Python
process and Rust extractor were discovered still consuming CPU while `_r4` was
running. The obsolete `_r3` process tree was terminated; it produced no accepted
checkpoint and must never be resumed or used. `_r4` uses the same clean detached
commit build, passed 6/6 Rust/Python parity at exact 0 cp difference, runs in a
foreground persistent session, and is the only valid bootstrap.

Final result:

- 223 full optimizer batches, 113,964 positions actually optimized (47 rows in
  the incomplete final minibatch were intentionally dropped), 2,115.05 s train
  time, 53.88 positions/s;
- train loss `0.4209459 -> 0.3966549`; validation loss `0.3868207531` over the
  5,989-row validation split in 82.22 s;
- exact optimizer composition: 113,964/113,964 external teacher-anchor rows,
  unit sample weights, no fresh or recent self-play rows;
- `net_weights_best.bin` SHA-256
  `3e8a87965cce61b12c642db3cb74cb8a7613c618144c725b250869a2890df1e1`;
- resumable `best.pt` SHA-256
  `bea6718721b211022e36aa801eafce6a4f8d490ea98e6bb9b9a370f82f303aaa`;
  direct resume check restored step 223, epoch 1, best validation, all 16 model
  tensors, all 16 Adam state entries, and all 16 EMA tensors;
- candidate-vs-Python parity with the exact clean binary and explicit candidate
  blob: 6/6 positions at exactly 0 cp difference.

The initial post-training parity invocation exposed a validation-tool bug:
`parity_check.py` honored `TITANIUM_ENGINE_BIN` for Rust but loaded the embedded
production weight blob on Python even when `TITANIUM_NET_WEIGHTS_PATH` was set.
The checker now resolves its Python weights from the same environment variable.
The misleading cross-network failures were 0/6; after the fix the real same-net
result is the exact 6/6 above.

Acceptance and gate decision (2026-07-16):

- the audited gate used 200 mirrored games, seed 1337, 60-second sudden-death
  clocks, an audited 10-ply opening book, four local workers, and 13 Oracle
  workers;
- at the owner's decision snapshot, 170 games were complete and the result was
  85–85; the remote workers had already finished before they could be stopped,
  and the final result was candidate 96, accepted epoch 1 104, draws 0,
  errors 0 (48%, approximately -14 Elo);
- exact artifacts:
  `tools/binary_match/runs/catv5_normalized5_r4_vs_accepted_epoch1_clock60_200_17shard`;
- the final result is statistically consistent with equal strength. The owner
  explicitly accepted the corrected normalized CAT schema as the new baseline
  despite the numerical loss and without a separate frozen-anchor gate;
- it is accepted checkpoint-chain epoch 2 at
  `training/runs/v16/accepted/epoch_0002.bin`; its resumable state is
  `training/runs/v16/accepted/epoch_0002.pt`;
- opening-collapse sanity passed `e2 e8 e3 e7`. Do not repeat this incumbent
  gate or describe epoch 2 as an unaccepted candidate;
- clean engine commit `03856fe` built successfully at
  `engine/target-catv5-accepted-03856fe/release/titanium.exe`, SHA-256
  `dceb8f9de28215747c66491cb71b77ae0299e935b28fdf3be3763eb127ce0ea0`.

Persistent rollback DSU for wall-cycle legality was implemented and rejected:
it preserved exact results but was about 1.1% slower aggregate versus the
maintenance-only control. Do not revisit without a new measured design.

### Full Claustrophobia code audit (pinned 2026-07-16)

Reviewed current `Plaaasma/Claustrophobia` commit
`285e78d9e2023da2d4095ecdedc17bcf649948f6` across state/move generation,
encoding, MCTS, self-play, replay, training, orchestration, evaluation/gating,
targeted loss replay, and inference backends. This supersedes conclusions based
only on its README or an older snapshot.

Do not treat its README strength sentence as sufficient evidence. The committed
mixed historical `ka/ladder_results.jsonl` has, at the main 512-simulation vs
Ka-3000 setting, 169 wins, 146 losses, and 7 unfinished rows across multiple
network generations: 53.7% of decisive games, roughly +26 Elo, not a clean
final-champion match. `ka/external_elo.jsonl` is mostly noisy 10-game rungs and
its latest recorded rung is 4-6. These artifacts neither disprove nor establish
the release champion's claim against every public setting; they do establish why
Titanium must require its own paired 200-game and frozen-anchor evidence rather
than borrowing another project's constants or reputation.

Already covered or not a Titanium port:

- Its incremental rollback DSU is the design Titanium measured and rejected
  above. Its branchless wall-slot masks, lazy pseudo-legal wall validation,
  bitboard frontier floods, and wall-less race shortcuts all have existing
  Titanium equivalents; Titanium's topology/BFF path is already more developed.
- PUCT/FPU, virtual-loss batches, Gumbel sequential halving, forced playouts,
  MCTS-solver marks, and tree reuse are MCTS mechanisms, not low-risk alpha-beta
  additions. Do not transplant them by name.
- CUDA graphs, TensorRT, SM121, TF32, and multi-GPU DDP are NVIDIA-specific.
  They do not provide a usable path for the local AMD FirePro M5100. The current
  PyTorch is CPU-only and DirectML is absent.
- 180-degree side-to-move canonicalization, left/right augmentation, EMA,
  replay anchoring, frozen-reference gates, and rich path inputs have already
  informed the active CATv5 repair. Claustrophobia's pointwise worst-wall threat
  planes remain an offline future architecture experiment, not a HalfPW leaf
  feature.
- Its discounted game outcome and Q/Z mixture are appropriate for an AlphaZero
  value head. They must not replace Titanium's strong-teacher centipawn/WDL
  target. At most, game outcome or moves-left may be an auxiliary target in a
  separately gated ablation.

New bounded experiments and operating rules, ordered by expected value and ease:

1. **Resume full training state after this schema-changing bootstrap.** Starting
   `_r4` from the accepted engine-format weight blob was necessary because the
   normalized five-plane schema is new. Every later epoch on this same schema
   must use `_r4`'s `best.pt` (or a later accepted full checkpoint) with
   `--resume`/`--ckpt`, preserving model weights, Adam moments, EMA shadow,
   global step, epoch, and best-validation state. Do not create repeated
   fresh-Adam shocks by starting each epoch from `net_weights_best.bin`. The
   trainer checkpoint already serializes all required state; verify a resumed
   smoke reports the prior step and reproduces the saved model before allowing
   continuous training.
2. **Rebuild and use the versioned feature cache after CATv5 is frozen.**
   Claustrophobia stores encoded tensors once in shards. Titanium already has a
   stronger fingerprinted memmap cache, but the existing 1,411,376-row cache is
   stale: `FV_LEN=628`, old schema/orientation, while normalized five-plane
   CATv5 requires `FV_LEN=952`. After this epoch and parity validation, rebuild
   once with the exact clean-engine SHA, feature schema, split manifest, and
   position keys. Randomly compare cached rows bit-for-bit with fresh Rust
   featurization before allowing `--feature-cache`. This is a throughput change,
   not a strength claim, and should remove repeated CAT/BFS extraction from
   later epochs.
3. **Repair phase classification before mixed-corpus training.** In explicit
   cohort mode, `teacher:*` keys currently fall back to `midgame`, and the legacy
   phase-quota pass is bypassed. That did not change this 100%-anchor bootstrap's
   labels or sample weights, but its all-midgame phase diagnostic is not factual.
   Derive opening/midgame/endgame from each packed board state, then stratify
   _inside_ fresh/recent/anchor so the required 80/10/10 cohort ratio remains
   exact. Add tests proving both per-batch cohort composition and epoch-level
   phase coverage before enabling the later mixed stage.
4. **Exact early rejection in the 200-game gate.** After every completed paired
   chunk, calculate the candidate's best possible final score if it won every
   remaining game. If that is still below the promotion threshold, reject
   immediately. This cannot create a false promotion and preserves the full
   200-game requirement for every candidate still capable of passing. Record
   planned/played/saved games. A pentanomial paired-opening GSPRT may be studied
   later, but it must not enable early promotion until calibrated against the
   existing fixed 200-game protocol.
5. **Teacher-relabelled loss-position mining.** From valid games the candidate
   loses against the incumbent, frozen anchor, or external sentinel, sample the
   second half and the first large-eval-swing region. Re-run those positions
   through the strong teacher and add them as a separately identified hard-case
   cohort. Never imitate the losing move at one-hot confidence and never treat a
   raw final loss as a precise position evaluation. Track duplicate rate,
   teacher disagreement, phase/wall-reserve distribution, and held-out hard-case
   loss separately before assigning any fixed training percentage.
6. **Per-anchor floors, not aggregate only.** Keep the current incumbent gate
   and frozen absolute gate separate. If multiple frozen anchors are added, a
   candidate must clear a minimum score against every anchor as well as the
   aggregate, using color-swapped paired starts/common seeds. This prevents a
   gain against one recent family member from hiding catastrophic forgetting
   against an older style. Tune floors from confidence intervals; do not copy
   Claustrophobia's `0.45`/`0.50` constants blindly.
7. **Cheap trajectory generation, expensive teacher labels only on sampled
   positions.** The alpha-beta analogue of playout-cap randomization is to use a
   cheap accepted-engine search to reach diverse states, then spend the strong
   teacher budget on a stratified subset of plies. Preserve opening/midgame/
   endgame and wall-reserve coverage, and keep unlabeled cheap moves out of the
   optimizer. Measure trusted labels per CPU-hour and strength per trusted label
   against the current label-every-position pipeline.
8. **Rollback alarm only if data follows the candidate.** If a future continuous
   loop generates fresh positions with the trainer/candidate rather than the
   frozen accepted net, reset trainer, optimizer, EMA, and data source to the
   accepted checkpoint after repeated statistically meaningful frozen-anchor
   deficits. It is unnecessary during the current 100% teacher bootstrap.
9. **Training-only auxiliary heads (low priority).** Moves-to-end and final race
   differential can regularize a shared representation with no inference head
   required after export, but HalfPW capacity is tiny and much of the geometric
   signal is already explicit. Try only after the anchored CATv5 candidate is
   established, with target availability audited and a one-variable ablation.

Do not combine items 4-9 in one candidate. Epoch 2 is now accepted. The
immediate sequence is: rebuild and audit the 952-feature cache, repair phase
metadata, then verify a full-state resume from accepted `epoch_0002.pt` before
any subsequent epoch. Loss mining and data-generation changes remain later,
separate experiments. A future candidate must still be checked against both
accepted epoch 2 and a frozen absolute anchor; the owner override applies only
to epoch 2 and is not a permanent relaxation of the gate.

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

| Current ply | Samples reaching ply | Mean remaining own moves |  P90 |  P95 |
| ----------: | -------------------: | -----------------------: | ---: | ---: |
|           0 |                1,801 |                    29.43 | 35.5 | 38.5 |
|          20 |                1,801 |                    19.43 | 25.5 | 28.5 |
|          40 |                1,769 |                     9.63 | 16.0 | 18.5 |
|          50 |                1,489 |                     5.90 | 11.5 | 14.0 |
|          60 |                  817 |                     3.65 |  8.5 | 11.0 |
|          70 |                  229 |                     3.38 |  8.0 |  9.0 |
|          80 |                   62 |                     2.27 |  5.0 |  5.0 |

At a fresh 60-second clock this implies 2.04 s/move from the mean horizon,
1.69 s/move from the P90 horizon, and 1.56 s/move from the P95 horizon. These
are schedule priors only; recalculate from the _current_ remaining clock at
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
   pruning-versus-full-search disagreement as _candidate_ instability signals.
   First test whether they predict a later PV/score reversal after controlling
   for depth and position phase. Only then allow a small, capped multiplier;
   its sign must be learned from that correlation, not assumed.
5. **Conservative per-game time bank — medium complexity.** Save surplus from
   low-tension moves and allow high-tension moves to spend only saved credit;
   never borrow future clock. Start with a 0.4x–1.5x multiplier around the
   scheduled base and persist/reset the bank exactly once per game. This is
   the useful idea from downloaded `search.rs`; evaluate after steps 1–3 so
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

## Cursor takeover order — execute in this sequence

The CATv5 source, corrected orientation, anchored bootstrap, parity checks, and
incumbent gate are finished. Accepted epoch 2 is the baseline. Do not retrain
epoch 1, repeat its gate, change CAT semantics, or start a search experiment
before completing steps 1–3.

1. **Rebuild and audit the 952-feature cache.** Use accepted engine commit
   `03856fe` and feature schema
   `halfpw-sparse-route5-catv5-normalized5-ws20-v1`. Write a new versioned
   cache; never reinterpret the old 628-feature cache. Compare a deterministic
   random sample bit-for-bit against fresh Rust extraction, including P2
   positions that prove both row and column reverse under the 180-degree map.
   This is a throughput task and needs no strength gate.
2. **Repair phase metadata without changing cohort ratios.** Derive
   opening/midgame/endgame from packed board state for `teacher:*` rows. Add
   tests for correct phase classification, exact per-batch cohort composition,
   and epoch-level phase coverage. Do not start mixed training until they pass.
3. **Prove full-state resume from accepted epoch 2.** Resume from
   `training/runs/v16/accepted/epoch_0002.pt`, not the `.bin`. Run a bounded
   smoke and verify restored global step 223, epoch 1, all model tensors, all
   Adam state entries, EMA shadow, and best-validation state before optimizing
   further. A new `.bin` start would discard optimizer history and is forbidden.
4. **Run one controlled next epoch.** Only after steps 1–3, use the explicit
   80% fresh / 10% recent replay / 10% protected external-teacher-anchor mix.
   Report exact realized cohort and phase counts, held-out losses by cohort,
   parity, opening sanity, and checkpoint-resume identity. Do not silently use
   the generic 241/450 queue as if it satisfied this experiment's corpus plan.
5. **Gate the next candidate.** Compare it separately against accepted epoch 2
   and the frozen absolute anchor using the protocol below. The epoch-2 owner
   override does not lower future standards. Promote only with an explicit
   recorded decision.
6. **Only then choose one new experiment.** Prefer the time-management telemetry
   task below because it changes no decisions. Loss-position mining and search
   heuristics must remain separate candidates.

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
  and the explicit promote/reject decision in this file. Older handoff files
  are historical references only and must not override this one.

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

- Canonical handoff (this file only for task queues): `TITANIUM_NEXT_CHAT_HANDOFF.md`
- Historical detailed log: `OVERNIGHT_ENGINE_HANDOFF.md` (non-canonical)
- Search benchmark: `engine\src\bin\search_bench.rs`
- Broke-side / flamegraph helpers: `tools\profile_tt_per_think\`
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
  propagation or LMR/move ordering. The normalized five-plane source is pushed
  in engine commit `426361a` and the clean detached build passed the listed CAT,
  identity, and parity checks.
- The 120,000-position `_r4` teacher-only bootstrap completed from accepted
  epoch 1. Its exported weights are exact 6/6 Rust/Python parity and opening
  sanity passed `e2 e8 e3 e7`.
- Its 200-game incumbent gate finished 96–104 with no draws or errors. The
  owner explicitly accepted it as epoch 2 despite the numerical loss and
  skipped separate frozen-anchor evidence for this one promotion. The accepted
  `.bin` and full `.pt` hashes are recorded above and in the checkpoint chain;
  the blob is embedded and pushed in engine commit `03856fe`.
- Fixed `parity_check.py` so `TITANIUM_NET_WEIGHTS_PATH` selects the same blob
  for the Python and Rust sides; prior cross-network output is invalid.
- Completed the current Claustrophobia code audit and added only the bounded
  experiments above. Its rollback DSU remains rejected and its MCTS/NVIDIA
  mechanisms are not Titanium ports.
- 2026-07-16 Cursor follow-up: repaired packed-state phase classification +
  in-cohort phase stratification; proved full resume from `epoch_0002.pt`;
  started versioned 952-feature cache rebuild under
  `training/data/feature_cache_catv5_normalized5_952/` (Pass 2 featurizing).
  Do not start the mixed 80/10/10 epoch until that cache finishes and passes
  `training/audit_feature_cache_parity.py`. No strength gate is active. The
  generic training coordinator remains deliberately idle at ~241/450.
