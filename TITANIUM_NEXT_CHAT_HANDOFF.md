# Titanium v17 — fresh-chat handoff

Last updated: 2026-07-15

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

## Next task: disciplined low-hanging search optimization

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

- The immutable projection was parked and reverted.
- The uncommitted eval-cache candidate was restored to the direct-cache
  baseline; only a documentation record was committed.
- A fresh isolated current-source `bench-instrument` build was started in
  `engine\target-current-instr`; it may need to be checked or rebuilt before
  profiling. It changes no production source.
