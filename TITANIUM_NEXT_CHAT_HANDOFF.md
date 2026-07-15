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
