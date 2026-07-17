# Handoff for GPT — one-side-broke race bounds (in progress)

Paste this to GPT for cheap guidance. Do **not** ask it to write full patches unless needed; prefer fixture/logic verdicts.

## Goal

Simple Titanium strength experiment: when **exactly one** player has 0 walls in hand, use the existing exact no-future-walls race solver (`race_tbl` via `zero_wall_winner_for_current_topology`) for:

1. **Sound αβ cuts** (narrow refuse-to-place theorems only)
2. **Soft eval** when those theorems fire (`±1800`, not `RACE_WIN_FLOOR`)

Explicitly **not** building monopoly-3 / N-wall special cases. GPT-5.6 / prior discussion: prefer ETA for eval + careful αβ; strengthen `certify()` later for both-armed positions.

## Soundness contract (agreed)

### Arrival time (always safe as information)

If player A has 0 walls → exact race arrival for A on current topology is a **lower bound on A's arrival** (A cannot improve own path with walls).

### Game-result αβ cuts (only these two)

| Condition                                               | Pure-race winner | Bound                                                |
| ------------------------------------------------------- | ---------------- | ---------------------------------------------------- |
| `wl[opp]==0` and STM wins pure race                     | STM              | **Lower** — STM can refuse to place; opp cannot wall |
| `wl[stm]==0` and STM loses pure race                    | opp              | **Upper** — opp can refuse to place; stm cannot wall |
| both hands 0                                            | —                | existing exact DTM path                              |
| both hands > 0                                          | —                | Unknown (this feature)                               |
| wallless STM **wins** pure race but opp still has walls | STM              | **Unknown** — opp can spoil                          |
| walled STM **loses** pure race but opp has 0 walls      | opp              | **Unknown** — STM can spend walls to reverse         |

**Important:** projected “Black wins pure race” with `White=2, Black=0` is **not** an automatic Lower for Black.

## What was already implemented (this chat)

Files:

- `engine/src/titanium/search.rs` — `one_side_broke_race_bound()`, hooked in `ab()` before 1w/2w layers; soft eval `±1800` when decisive
- `engine/src/titanium/race.rs` — stats `broke_calls` / `broke_decisive` / `broke_unknown` in `RaceOutcomeStats` + JSON

Core function (conceptual):

```rust
fn one_side_broke_race_bound(&mut self) -> RaceBound {
    if !race_proof { return Unknown }
    if (wl[0]==0) == (wl[1]==0) { return Unknown } // both broke or both armed
    let pure_winner = zero_wall_winner_for_current_topology()?; // temporarily wl=[0,0], race_tbl
    let stm = turn; let opp = stm^1;
    if wl[opp]==0 && pure_winner==stm { return Lower(RACE_WIN_FLOOR) }
    if wl[stm]==0 && pure_winner!=stm { return Upper(-RACE_WIN_FLOOR) }
    Unknown
}
```

Uses existing helper:

```rust
fn zero_wall_winner_for_current_topology(&mut self) -> Option<usize>
// saves wl, sets [0,0], exact_hands_empty_score(false), restores wl → winner side
```

## Test status (FAILED — needs fixture/logic help)

Command:

```powershell
cd engine
$env:RUSTFLAGS='-C target-cpu=native'
cargo test -p titanium --lib broke_side_ -- --nocapture
```

| Test                                                                    | Expected         | Actual              | Notes                                                                                                                                                                                       |
| ----------------------------------------------------------------------- | ---------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `broke_side_lower_when_opp_out_of_walls_and_stm_wins_pure_race`         | Lower            | Lower ✅            | fixture `a8` vs `i2`, hands `[3,0]`, turn 0                                                                                                                                                 |
| `broke_side_declines_when_wallless_player_wins_but_opp_still_has_walls` | Unknown          | Unknown ✅          | `a8` vs `i2`, hands `[0,3]`, turn 0                                                                                                                                                         |
| `broke_side_declines_when_both_armed_or_both_broke`                     | Unknown, calls=0 | ✅                  |                                                                                                                                                                                             |
| `broke_side_upper_when_stm_out_of_walls_and_loses_pure_race`            | Upper            | **Unknown** ❌      | fixture `a2` vs `i8`, hands `[0,4]`, turn 0 — pure race did **not** make STM the loser (or tbl miss)                                                                                        |
| `broke_side_declines_when_walled_stm_loses_pure_race`                   | Unknown          | **Lower(30976)** ❌ | fixture `a2` vs `i8`, hands `[4,0]`, turn 0 — we thought STM loses pure race, but with opp broke STM actually **wins** → correctly emits Lower under the theorem; fixture premise was wrong |

Fixture helper (already in `search.rs` tests):

```rust
fn two_wall_fixture(p0: &str, p1: &str, hands: [i32; 2], turn: usize) -> GameState
```

Empty board topology (no walls placed). Need jump-aware pure-race fixtures that match intended winners.

## Ask GPT for

1. **Correct empty-board fixtures** (algebraic squares + hands + turn) for:
   - Upper case: STM has 0 walls, loses pure race vs armed opp
   - Decline-Upper case: STM has walls, opp has 0, STM **loses** pure race (so cut must stay Unknown — not Lower)
2. Confirm logic of `one_side_broke_race_bound` matches the soundness table (any bug?)
3. Whether soft eval `±1800` in `evaluate()` is OK short-term, or should wait until exact ETA feature exists
4. Minimal next step only — no monopoly-3, no full certify rewrite

## Deliberately out of scope

- `three_wall_monopoly_bound`
- Expanding 1w/2w special cases
- Exact ETA NNUE feature plane (follow-up)
- Strength match / Elo (after tests green)

## Related docs / prior art in repo

- `HANDOFF_REMAINING_WALL_RACE_LAYERS.md` — 0/1/2 wall layers
- `one_wall_race_bound` / `two_wall_monopoly_race_bound` — different, sum==1/2 monopoly subsets
- `wall_ignore_cert` — separate experimental path, default off

## Current live work note

Flamegraph race2w single-game measurement was started earlier then interrupted; not required for this broke-side feature.

## Status (2026-07-17) — DONE for v1

GPT guidance applied:

- Theorems confirmed sound
- Fixtures now oracle-scanned via `find_empty_board_pure_race_pair` (empty `race_tbl` once, scan pawn pairs)
- Bounds preserve exact DTM: `Lower/Upper(±(RACE_MATE - dtm))` not just floor
- Soft eval `±1800` kept
- Helper confirmed: `exact_hands_empty_score` only checks temporary `wl=[0,0]`

Tests: `cargo test -p titanium --lib broke_side_` → **5/5 pass**

Next optional: Elo smoke / exact ETA feature plane (not required for this cut).
