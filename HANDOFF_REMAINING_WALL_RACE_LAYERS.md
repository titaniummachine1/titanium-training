# Handoff: Remaining-wall race layers (0 / 1 / 2 walls)

**Goal of this note:** enough context to reason about extending the sound race-bound stack beyond 2 walls (e.g. monopoly-3, split hands, delayed placement).

**Date:** 2026-07-17  
**Primary code:** `engine/src/titanium/search.rs` (~4720–5095, ab hooks ~6264–6312), `engine/src/titanium/race.rs`, `engine/src/titanium/session.rs`, `engine/src/titanium/certify.rs`

---

## 1. Mental model — three different systems

| System | When | What it proves | Exact DTM? |
|--------|------|----------------|------------|
| **Service A (Gate1)** | Hands empty (`wl=[0,0]`) | Cheap αβ bound if path-sets disjoint | No |
| **Service B (`race_tbl`)** | Hands empty | Exact retrograde DTM on **fixed wall topology** | Yes (`RACE_MATE − k`) |
| **1-wall / 2-wall subsets** | `sum(wl)==1` or `==2` | Sound αβ **Lower/Upper** only | No |
| **`certify()` (v13)** | `sum(wl) ≤ 3` | Budget-capped campaign search | Incomplete cert |
| **`wall_ignore_race_bound`** | Any remaining walls | Separate “ignorance” loss cert (opt-in) | No |

**Critical invariant:** `race_tbl` / `solve_race_config` only index **pawn×pawn×turn** on a **frozen** edge set. Remaining walls in hand are **not** part of the table. Any N-wall layer must either:

1. temporarily discard hands and use the table (pure-race ignore-walls), or  
2. explicitly branch on wall placements and recurse into a smaller layer, or  
3. decline (`Unknown`).

There is **no** N-wall DTM table today.

---

## 2. Bound type used by all remaining-wall layers

```82:92:engine/src/titanium/race.rs
pub enum RaceBound {
    Lower(i32),   // STM forced win: true score ≥ bound
    Upper(i32),   // STM forced loss: true score ≤ bound
    Exact(i32),   // only Service B / exact path
    Unknown,
}
```

Floor used by 1w/2w (and Gate1):

```62:76:engine/src/titanium/race.rs
pub const RACE_MATE: i32 = 32_000;
pub const RACE_WIN_FLOOR: i32 = RACE_MATE - RACE_MAX_PLIES;
```

In `ab()`, a Lower ≥ β fail-highs; Upper ≤ α fail-lows; otherwise search continues. Bounds are **never** promoted to exact mate scores by the 1w/2w layers.

---

## 3. Where they hook into search

Order inside `ab()`:

```6263:6312:engine/src/titanium/search.rs
        let pv_node = beta > alpha.saturating_add(1);
        if self.g.wl[0] + self.g.wl[1] == 1 && (!self.one_wall_race_pv_only || pv_node) {
            match self.one_wall_race_bound() { ... cut on window ... }
        }
        if self.g.wl[0] + self.g.wl[1] == 2 && (!self.two_wall_race_pv_only || pv_node) {
            match self.two_wall_monopoly_race_bound() { ... }
        }
        if self.g.wl[0] == 0 && self.g.wl[1] == 0 {
            // Gate1 (Service A) then exact_hands_empty_score / race_tbl
            ...
        }
```

Notes:

- Exact match on **sum of hands**, not “≤ N”.
- PV-only gates skip cut nodes (cheap miss, less wall-enumeration blow-up).
- 0-wall path is the only place that can return an **Exact** race score into `ab()`.

---

## 4. Enable flags / session wiring

**Env (if session did not set resolved flags):**

- `TITANIUM_RACE_ONE_WALL=1|true`
- `TITANIUM_RACE_TWO_WALL=1|true`

**API:**

```2974:2986:engine/src/titanium/search.rs
    pub fn set_remaining_wall_race_layers(&mut self, one_wall: bool, two_wall: bool);
    pub fn set_two_wall_race_pv_only(&mut self, on: bool);
    pub fn set_one_wall_race_pv_only(&mut self, on: bool);
```

**Session defaults** (`session.rs`):

| Engine flag | 1w | 2w | PV-only |
|-------------|----|----|---------|
| `titanium-v17-race1w` | on | off | — |
| `titanium-v17-race2w` | on | on | — |
| `titanium-v17`, `…-race2pv`, `…-rfp-tc-d4`, `…-race1pv` | on | on | **2w PV-only** |
| `titanium-v17-race1pv` | on | on | **1w + 2w PV-only** |

Default-off until session/env sets them (deliberate strength/audit gate).

Stats on `RaceOutcomeStats`: `one_wall_*`, `two_wall_*` (calls / decisive / unknown).

---

## 5. Shared primitives

### 5.1 Pure-race winner after discarding hands

```4883:4897:engine/src/titanium/search.rs
    fn zero_wall_winner_for_current_topology(&mut self) -> Option<usize> {
        let saved_hands = self.g.wl;
        self.g.wl = [0, 0];
        let score = self.exact_hands_empty_score(false); // race_tbl, budget-gated
        self.g.wl = saved_hands;
        score.map(|score| if score > 0 { self.g.turn } else { self.g.turn ^ 1 })
    }
```

- Returns **winner side**, not DTM.
- Can return `None` if `race_proof` off or `race_tbl` build gated by budget/deadline.
- This is the **only** link from N-wall subsets into Service B.

### 5.2 Separation gate

```4899:4904:engine/src/titanium/search.rs
    fn pawns_are_race_separated(&self) -> bool {
        // Chebyshev distance > 2
        (p0/9).abs_diff(p1/9).max((p0%9).abs_diff(p1%9)) > 2
    }
```

Intent: avoid jump/interaction positions where “ignore walls / place now” reasoning is unsafe or incomplete. If not separated → layer declines.

### 5.3 Opponent goal distance

```4906:4911:engine/src/titanium/search.rs
    fn opponent_distance_to_goal(&self) -> u8 { ... compute_dist(opponent) ... }
```

Used as the **urgency** gate: forced wall placement is only considered when opponent is **exactly 1** from goal (must act *now* or they step in).

---

## 6. One-wall layer (full logic)

Entry: `one_wall_race_bound()` — sum hands == 1.

```
one_wall_race_bound
├─ STM holds the wall  → one_wall_holder_winner
└─ STM has 0 walls     → one_wall_nonholder_winner
```

### 6.1 Holder moves (`wl[stm]==1`)

```4913:4960:engine/src/titanium/search.rs
fn one_wall_holder_winner(&mut self) -> Option<usize>
```

1. Require sum==1, STM holds it, separated.
2. Pure-race (`zero_wall_winner…`):
   - If **holder wins** → `Some(holder)` (wall irrelevant).
3. Else if opp dist ≠ 1 → `None` (**do not invent a loss**; delayed wall may matter).
4. Else: try **every legal wall** → child with hands empty → `zero_wall_winner…`:
   - Any child where holder wins → `Some(holder)`.
   - All legal walls resolved, none help → `Some(holder^1)` (forced loss).
   - Any child `None` (tbl miss) → `None`.

### 6.2 Non-holder moves (`wl[stm]==0`)

```4963:4997:engine/src/titanium/search.rs
fn one_wall_nonholder_winner(&mut self) -> Option<usize>
```

1. Generate pawn moves only.
2. For each move: if already won, or `one_wall_holder_winner` on child:
   - Any reply where mover wins → `Some(mover)`.
3. If **every** reply resolved and none wins → `Some(mover^1)`.
4. If any reply `None` → `None`.

**Soundness pattern (AND/OR):**

- Win proof = ∃ move into a proven-win child.  
- Loss proof = ∀ legal moves into proven-loss children (and no unknown).  
- Unknown = refuse.

---

## 7. Two-wall layer (full logic)

Entry: `two_wall_monopoly_race_bound()` — sum hands == 2.

```5026:5094:engine/src/titanium/search.rs
fn two_wall_monopoly_race_bound(&mut self) -> RaceBound
```

### Gates (all required)

1. Feature enabled.  
2. `wl[0]+wl[1]==2`.  
3. **Monopoly:** `wl[stm]==2` (both walls with mover).  
   - Split `[1,1]` or opponent monopoly → `Unknown`.  
4. `pawns_are_race_separated()`.

### Algorithm

```
pure = zero_wall_winner_for_current_topology()

if pure == holder:
    → Lower   // walls don't matter

else if opp_dist == 1:
    for each legal wall placement:
        make wall   // now sum(wl)==1, opponent to move
        child = one_wall_nonholder_winner()
        unmake
    if ∃ child where holder wins → Lower
    if all legal walls resolved & none help → Upper
    else → Unknown

else:
    → Unknown   // delayed second wall / non-urgent: decline
```

**Reduction:** 2-wall forced placement does **not** invent a second free wall tree. It places **one** wall now and reuses the **1-wall nonholder** subset (opponent moves, then holder has the last wall).

```
2w monopoly + opp dist1
  └─ place wall → 1w nonholder (opp STM)
        └─ pawn move → 1w holder
              └─ place last wall → 0w race_tbl
```

### What 2w explicitly refuses

| Situation | Result |
|-----------|--------|
| Hands `[1,1]` | Unknown |
| Opp holds both, STM to move | Unknown |
| Pawns Chebyshev ≤ 2 | Unknown |
| Pure race loss but opp dist > 1 | Unknown |
| `race_tbl` budget miss on any needed child | Unknown (no false loss) |

---

## 8. Hands-empty stack (for extension context)

```4726:4812:engine/src/titanium/search.rs
fn race_tbl(&mut self, force: bool) -> Option<usize>
fn exact_hands_empty_score(&mut self, force: bool) -> Option<i32>
```

- Key = wall topology hash (pawns/turn XOR’d out).  
- Value = retrograde over ~`RACE_STATES` = 81×81×2.  
- Build is **budget/deadline gated** in-tree (`rc_count_cap`, `rc_solve_cap`, `deadline`).  
- Module docs: `race.rs` Service A vs B.

Gate1 (`race_outcome_with_dist`) only when hands empty and `cheap_cert`; never invents bounds on overlapping path sets.

---

## 9. Related but separate: `certify` (≤3 walls)

```396:418:engine/src/titanium/certify.rs
pub fn certify(game: &mut GameState, opts: &CertifyOpts) -> CertifyReport
```

- Caps at `sum(wl) > 3` → no cert.  
- Hands empty → `no_wall_race_winner`.  
- Otherwise budget-capped recursive campaign (not the monopoly race layer).  
- Different API, different soundness/budget story — do **not** confuse with 1w/2w `RaceBound` subsets when designing N-wall extensions.

Also optional: `wall_ignore_race_bound` / `wall_ignore_cert` — proves losses when walls cannot change a pure-race loss (separate opt-in).

---

## 10. Tests to keep green / extend

In `search.rs` (approx 1505–1654):

| Test | Claim |
|------|--------|
| `one_wall_subset_accepts_holder_pure_race_win_as_bound` | Pure race → Lower |
| `one_wall_subset_declines_when_delayed_wall_can_matter` | No invented loss |
| `two_wall_subset_is_default_off` | Env/session gate |
| `two_wall_subset_accepts_monopoly_pure_race_win_as_bound` | Monopoly pure win |
| `two_wall_forced_placement_fixture_matches_ordinary_search_winner` | Forced wall ≡ search sign |
| `two_wall_subset_rejects_split_nonholder_and_interacting_pawns` | Gates |
| `two_wall_subset_random_empty_topology_counter_oracle` | Random countercheck |
| `two_wall_subset_ab_measurement` | AB cut measurement |

Any N-wall extension should add the same pattern: pure-win fixture, forced-placement fixture, decline fixtures, random counter-oracle vs deep search.

---

## 11. How to extend to more walls (design sketch)

### 11.1 Safe incremental pattern (recommended)

Keep the **sound subset** discipline:

```
N-wall monopoly layer (wl[stm]==N, separated):
  1. pure-race win → Lower
  2. if urgent (opp dist==1):
       ∃ wall → (N-1)-wall layer proves holder win → Lower
       ∀ walls resolved → holder loss → Upper
  3. else Unknown
```

Then:

- **3-wall monopoly** = same shape as 2w, child = `two_wall_…` after one placement (careful: after place, hands may be `[2,0]` with **opponent** to move — need a **nonholder** variant of each layer, like 1w already has).  
- **Split hands** (`[2,1]`, `[1,1]`, …) are much harder: opponent can also place. Need AND/OR over **both** players’ wall moves, or refuse.

### 11.2 What must exist per layer N

| Piece | Why |
|-------|-----|
| `N_wall_holder_winner` | STM holds all remaining |
| `N_wall_nonholder_winner` | STM has 0 of the remaining (opp holds them) — needed when you place then pass turn |
| Separation + urgency gates | Keep branching finite / sound |
| Stats counters | Measure hit rate before enabling in production |
| PV-only flag | Cost control (wall loops × race_tbl) |
| Default-off + session flag | Strength gate |

2w today only implements **holder monopoly**; after placing one wall it calls `one_wall_nonholder_winner` (opp is STM). For 3w you likely need `two_wall_nonholder_…` or generalize.

### 11.3 Complexity warning

Worst-case branching (naive):

- Legal walls ≲ 128.  
- Each placement may recurse into another wall loop.  
- Each leaf may `race_tbl` build.  

Even 2w forced branch is expensive → **PV-only** on v17. For 3w+, require:

- early exit on first winning wall,  
- PV-only or depth/node budget,  
- maybe only opp-dist==1 (or ==2 with a *different* sound proof — do not casually widen urgency).

### 11.4 Unsound traps (do not do)

1. **Invent loss** when opp dist > 1 (“I lose pure race so I lose with walls”). Delayed walls can flip.  
2. Treat `race_tbl` with hands non-zero.  
3. Use Gate1 on positions with walls in hand.  
4. Assume split `[1,1]` behaves like monopoly.  
5. Drop separation without a jump-aware proof.  
6. Promote `RACE_WIN_FLOOR` bounds to exact DTM / mate scores.

### 11.5 Alternative directions (not just “3w monopoly”)

1. **Widen urgency carefully** — e.g. opp dist==2 with forced double-step proofs (much harder).  
2. **Wall-ignore certificates** — expand `wall_ignore_cert` for “no wall can save the loser” without enumerating placements.  
3. **Budgeted certify** — already ≤3 walls; strengthen/budget rather than new monopoly layer.  
4. **True multi-wall retrograde** — enormous state space (hands + topology); not a small extension.

---

## 12. Suggested first experiment for “more walls”

1. Implement `three_wall_monopoly_race_bound` mirroring 2w, child = new `two_wall_nonholder_winner` (or generalized).  
2. Default **off**; enable under `titanium-v17-race3w` + **PV-only**.  
3. Fixtures: pure monopoly win; forced wall vs dist-1; decline on `[2,1]` / close pawns / dist>1.  
4. Measure `three_wall_calls` / decisive / unknown in self-play; check NPS impact (flamegraphs already show race ≈ 0% today — so hit rate may be low but cuts matter for Elo).  
5. Only then consider `[2,1]` / `[1,1]` theories.

---

## 13. File index

| File | Role |
|------|------|
| `engine/src/titanium/search.rs` | 1w/2w subsets, ab hooks, `race_tbl` wrapper, tests |
| `engine/src/titanium/race.rs` | Service A/B, `RaceBound`, `RACE_*`, stats JSON |
| `engine/src/titanium/session.rs` | v17 flag wiring |
| `engine/src/titanium/certify.rs` | ≤3-wall budget cert (separate) |
| `engine/src/titanium/wall_ignore_cert.rs` | optional ignore-walls loss cert |
| `engine/src/titanium/oracle.rs` | reference solve (tests / offline) |

---

## 14. One-paragraph summary

Hands-empty race is a real oracle (Gate1 bound + `race_tbl` DTM). With walls left, Titanium only has **sound monopoly subsets** for 1 and 2 walls: prove win if pure race already wins, or if the opponent is one step from goal and every/any forced wall placement reduces into a smaller proven layer; otherwise **Unknown**. Extending to more walls means stacking the same reduction carefully (especially nonholder variants and cost gates)—not growing `race_tbl` to include hands.
