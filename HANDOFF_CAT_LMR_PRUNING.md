# Handoff: CAT / LMR / Move-Ordering / Futility-Pruning — Raw Code

Repo: `C:\gitProjects\Quoridor best AI\engine` (Rust). This is a Quoridor engine.
Board is 9x9 (81 squares, indices 0..80, `square_index(row,col) = row*9+col`).
Two players race pawns to the opposite edge; each can place walls (max 10 each)
to block/lengthen the opponent's path, subject to "always leave at least one
path for both players" legality.

This doc is a **complete, self-contained raw-code dump** of every system
involved in:
1. CAT (Corridor Attention Table) — the heatmap that says "how much does this
   square/wall matter to either player's race".
2. LMR (Late Move Reduction) — how deep to search a given move based on how
   unimportant CAT says it is.
3. Move ordering — what order moves get tried in at each search node.
4. Futility / tail-cutoff / dead-zone pruning — the "this move can provably or
   probably never matter" filters, including the exact mechanism for "both
   players are racing straight down their own corridors, ignore useless
   middle-of-board wall placements" (`wall_in_dead_zone` / `wall_should_search`
   below — read those first, they are the direct answer to that question).

No summarization of logic has been done inside code blocks — these are full,
verbatim file contents or complete functions copy-pasted from the source tree,
so another model with zero repo access can reason about them directly.

---

## 1. `src/cat/constants.rs` (32 lines, FULL FILE)

Every tunable threshold used by the whole CAT/LMR/pruning stack.

```rust
//! CAT v3 thresholds — search ordering, LMR, and pruning cutoffs (centi-squares).

/// Heat on a player's shortest path square (delta = 0).
/// Combined two-player ceiling: `2 × CAT_CORRIDOR_CM = 400 cm`.
pub const CAT_CORRIDOR_CM: u16 = 200;

/// Exact and near-shortest corridors are search-relevant; larger detours are zero.
/// Keep at least four suboptimal route sets visible to avoid single-path tunnel vision.
pub const MAX_RELEVANT_CORRIDOR_DELTA: u16 = 4;
pub const BOTTLENECK_CORRIDOR_DELTA: u16 = 2;
pub const BOTTLENECK_BONUS_CM: u16 = 40;

/// Skip LMR / treat as tactical when local heat ≥ this.
pub const CAT_HOT_CM: u16 = 160;

/// Cold fringe — extra LMR reduction below this.
pub const CAT_COLD_CM: u16 = 60;

/// Dead CAT tail — at or below this % of position peak → minimum child depth (max LMR).
pub const CAT_TAIL_DEAD_RATIO_PCT: u16 = 10;

/// Heavy fringe — above dead tail, up to this % of peak → strong cut, not absolute max.
pub const CAT_HEAVY_FRINGE_RATIO_PCT: u16 = 20;

/// Sentinel when BFS finds no path.
pub const DIST_PENALTY: u8 = 255;

/// Impact/bitmask path only — dense `corridor_heat` keeps `MAX_RELEVANT_CORRIDOR_DELTA`.
pub const MAX_IMPACT_HEAT_DELTA: usize = 8;

/// Compiled default path-distance bias (basis points). Search worker stays at 0.
pub const DEFAULT_CAT_DISTANCE_BIAS_BP: i16 = 0;
```

---

## 2. `src/cat/attention.rs` (76 lines, FULL FILE)

The `CorridorAttention` struct itself — per-square heat storage + how a wall's
heat is derived from the 2 (or 4) squares it touches.

```rust
//! Corridor Attention Table (CAT) — per-square heat for search ordering (not eval).

use crate::core::board::WallOrientation;
use crate::util::grid::square_index;
use std::ops::Index;

/// Per-square attention scores for move ordering / LMR (centi-units, not eval).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CorridorAttention {
    pub(crate) square_heat: [u16; 81],
    pub(crate) route_flex: [u8; 81],
    pub(crate) bottleneck_heat: [u16; 81],
}

impl Default for CorridorAttention {
    fn default() -> Self {
        Self {
            square_heat: [0; 81],
            route_flex: [0; 81],
            bottleneck_heat: [0; 81],
        }
    }
}

impl Index<usize> for CorridorAttention {
    type Output = u16;

    fn index(&self, index: usize) -> &Self::Output {
        &self.square_heat[index]
    }
}

impl CorridorAttention {
    pub fn square_heat(&self, row: u8, col: u8) -> u16 {
        self.square_heat[square_index(row, col) as usize]
    }

    pub fn route_flex(&self, row: u8, col: u8) -> u8 {
        self.route_flex[square_index(row, col) as usize]
    }

    pub fn wall_edge_heat(&self, row: u8, col: u8, orientation: WallOrientation) -> u16 {
        let edge_heat = |a: (u8, u8), b: (u8, u8)| -> u16 {
            let ai = square_index(a.0, a.1) as usize;
            let bi = square_index(b.0, b.1) as usize;
            let ha = self.square_heat[ai];
            let hb = self.square_heat[bi];
            let hi = ha.max(hb);
            if hi == 0 {
                // both cells off-path: this wall touches no corridor
                return 0;
            }
            // A wall fully on the corridor (both cells hot) reads full; a wall
            // that only *touches* the corridor on one side still registers — at
            // lo + 40% of the gap — instead of collapsing to ~0 under min().
            // Walls touching the contested path are tactically live even when
            // they don't block the exact current edge.
            let lo = ha.min(hb);
            let corridor = lo + (hi - lo) * 2 / 5;
            let bottleneck = self.bottleneck_heat[ai].min(self.bottleneck_heat[bi]);
            corridor.saturating_add(bottleneck)
        };

        let (a, b) = match orientation {
            WallOrientation::Horizontal => (
                edge_heat((row, col), (row + 1, col)),
                edge_heat((row, col + 1), (row + 1, col + 1)),
            ),
            WallOrientation::Vertical => (
                edge_heat((row, col), (row, col + 1)),
                edge_heat((row + 1, col), (row + 1, col + 1)),
            ),
        };
        a.max(b).saturating_add(a.min(b) / 4)
    }
}
```

---

## 3. `src/cat/build.rs` (748 lines, FULL FILE)

How the heatmap gets built from BFS distance fields. Two independent
implementations exist:
- `build_corridor_attention` / `build_player_corridor_attention`: the "full"
  multi-route corridor heat (dense per-square BFS from/to each pawn, delta ≤ 4
  suboptimal routes counted), used by the CAT-filtered wall generation path
  (`gen_walls_cat_filtered` in search.rs) and by `order_moves`/`wall_should_search`.
- `build_impact_heatmap` / `build_impact_heatmap_for_stm`: the cheaper bitmask
  ("BFF" = bit-flood-fill) version used for v16 LMR's per-move impact score
  (`move_impact_heat`) — this is what the Lazy-SMP root-move-filtering
  experiment (see §7 below) uses.

Both zero out heat **behind** a pawn (`zero_pawn_entry_and_rear`) — i.e. a
square/wall that's farther from goal than the pawn already is gets **zero**
heat, because moving there can never help that pawn's race. This is the
direct mechanism behind "engine shouldn't care about useless moves in the
middle of the board that are behind either racer."

```rust
//! Build CAT heat from BFS distance fields on the pawn grid.

use std::sync::atomic::{AtomicI32, Ordering};

use crate::cat::attention::CorridorAttention;
use crate::cat::constants::{
    BOTTLENECK_BONUS_CM, BOTTLENECK_CORRIDOR_DELTA, CAT_CORRIDOR_CM, DEFAULT_CAT_DISTANCE_BIAS_BP,
    MAX_IMPACT_HEAT_DELTA, MAX_RELEVANT_CORRIDOR_DELTA,
};
use crate::core::board::{Board, Player};
use crate::path::distance::{
    fill_dist_from_sq, fill_dist_layers_from_sq, fill_dist_layers_to_goal_row,
    fill_dist_to_goal_row, DistLayers,
};
use crate::path::masks::DirMasks;
use crate::path::BfsScratch;
use crate::util::grid::{flood_bit_sq, square_index, FLOOD_PLAYABLE, FLOOD_SQ_BY_BIT};

fn corridor_heat(delta: u16) -> u16 {
    if delta > MAX_RELEVANT_CORRIDOR_DELTA {
        return 0;
    }
    // Exact rounded values of `CAT_CORRIDOR_CM / (1 + delta·log2(delta+2))` for
    // delta 0..4 — kept as a LUT so the per-square hot loop never evaluates a
    // float `log2`. Bit-identical to the old formula:
    //   delta 0 → 200/1.0       = 200
    //   delta 1 → 200/(1+log2 3) = 77
    //   delta 2 → 200/(1+2·log2 4) = 40
    //   delta 3 → 200/(1+3·log2 5) = 25
    //   delta 4 → 200/(1+4·log2 6) = 18
    const HEAT_LUT: [u16; (MAX_RELEVANT_CORRIDOR_DELTA + 1) as usize] = [200, 77, 40, 25, 18];
    debug_assert_eq!(
        CAT_CORRIDOR_CM, 200,
        "HEAT_LUT computed for CAT_CORRIDOR_CM=200"
    );
    HEAT_LUT[delta as usize]
}

/// Centi-percent (68–100): gentle linear fade along the race. The near-pawn
/// squares are still slightly hottest, but the deep corridor — where walls
/// actually decide the race — keeps most of its heat. The floor was raised
/// 45→68 (corridor +~50%) because the old curve over-focused on the pawn:
/// near-pawn squares are easy to walk around, mid/far corridor blocks are not.
fn pawn_path_weight(dist_from: u8, shortest_to_goal: u8) -> u16 {
    if shortest_to_goal == 0 || shortest_to_goal == u8::MAX {
        return 100;
    }
    const MIN_WEIGHT: u16 = 68;
    const MAX_WEIGHT: u16 = 100;
    let from = u32::from(dist_from.min(shortest_to_goal));
    let total = u32::from(shortest_to_goal);
    let remaining = total.saturating_sub(from);
    MIN_WEIGHT + (u32::from(MAX_WEIGHT - MIN_WEIGHT) * remaining / total) as u16
}

fn neighbor_squares(sq: u8, masks: DirMasks, out: &mut [u8; 4]) -> usize {
    let bit = flood_bit_sq(sq);
    let mut n = 0usize;
    if masks.north & bit != 0 {
        out[n] = sq - 9;
        n += 1;
    }
    if masks.south & bit != 0 {
        out[n] = sq + 9;
        n += 1;
    }
    if masks.east & bit != 0 {
        out[n] = sq + 1;
        n += 1;
    }
    if masks.west & bit != 0 {
        out[n] = sq - 1;
        n += 1;
    }
    n
}

fn corridor_delta(
    sq: u8,
    dist_from_pawn: &[u8; 81],
    dist_to_goal: &[u8; 81],
    shortest_to_goal: u8,
) -> Option<u16> {
    let from = dist_from_pawn[sq as usize];
    let to = dist_to_goal[sq as usize];
    if from == u8::MAX || to == u8::MAX || shortest_to_goal == u8::MAX {
        return None;
    }
    Some((u16::from(from) + u16::from(to)).saturating_sub(u16::from(shortest_to_goal)))
}

/// `delta_arr[sq]` is the precomputed corridor delta (`u16::MAX` = off-path/None),
/// so the per-neighbor near-shortest test is an array read, not a recompute.
fn reasonable_forward_continuations(
    sq: u8,
    masks: DirMasks,
    dist_from_pawn: &[u8; 81],
    dist_to_goal: &[u8; 81],
    delta_arr: &[u16; 81],
) -> u8 {
    let from = dist_from_pawn[sq as usize];
    let to = dist_to_goal[sq as usize];
    if from == u8::MAX || to == 0 || to == u8::MAX {
        return 0;
    }
    let mut neighbors = [0u8; 4];
    let n = neighbor_squares(sq, masks, &mut neighbors);
    let mut count = 0u8;
    for &next in &neighbors[..n] {
        let next_from = dist_from_pawn[next as usize];
        let next_to = dist_to_goal[next as usize];
        // `u16::MAX` sentinel (None) is > MAX_RELEVANT, so it fails the bound naturally.
        if next_from == from.saturating_add(1)
            && next_to < to
            && delta_arr[next as usize] <= MAX_RELEVANT_CORRIDOR_DELTA
        {
            count = count.saturating_add(1);
        }
    }
    count
}

fn add_player_corridor_attention(
    board: &Board,
    player: Player,
    masks: DirMasks,
    out: &mut CorridorAttention,
    dist_from_pawn: &mut [u8; 81],
    dist_to_goal: &mut [u8; 81],
) {
    let (sr, sc) = board.pawn(player);
    let start = square_index(sr, sc);

    fill_dist_from_sq(start, masks, dist_from_pawn);
    fill_dist_to_goal_row(player, masks, dist_to_goal);

    let shortest_to_goal = dist_to_goal[start as usize];

    // Compute each square's corridor delta once (u16::MAX = off-path); the main
    // loop and the per-neighbor flex test both read it instead of recomputing.
    let mut delta_arr = [u16::MAX; 81];
    for sq in 0usize..81 {
        if let Some(d) = corridor_delta(sq as u8, dist_from_pawn, dist_to_goal, shortest_to_goal) {
            delta_arr[sq] = d;
        }
    }

    for sq in 0u8..81 {
        let idx = sq as usize;
        let delta = delta_arr[idx];
        let base = corridor_heat(delta);
        if base == 0 {
            continue;
        }

        let from = dist_from_pawn[idx];
        let weight = pawn_path_weight(from, shortest_to_goal);
        let heat = (u32::from(base) * u32::from(weight) / 100) as u16;
        if heat == 0 {
            continue;
        }

        let flex =
            reasonable_forward_continuations(sq, masks, dist_from_pawn, dist_to_goal, &delta_arr);
        out.square_heat[idx] = out.square_heat[idx].saturating_add(heat);
        out.route_flex[idx] = out.route_flex[idx].saturating_add(flex);
        if delta <= BOTTLENECK_CORRIDOR_DELTA && flex <= 1 && dist_to_goal[idx] > 0 {
            out.bottleneck_heat[idx] = out.bottleneck_heat[idx].saturating_add(BOTTLENECK_BONUS_CM);
        }
    }
}

pub fn build_player_corridor_attention(
    scratch: &mut BfsScratch,
    board: &Board,
    player: Player,
) -> CorridorAttention {
    let masks = DirMasks::from_board(board);
    let mut out = CorridorAttention::default();
    let (dist_from, dist_to) = scratch.dist_scratch_mut();
    add_player_corridor_attention(board, player, masks, &mut out, dist_from, dist_to);
    out
}

/// Per-square heat for the web overlay — max of each player's corridor signal.
///
/// Board square overlay: symmetric sum of both players' corridors so the display
/// shows hot areas for both sides regardless of who is to move. Uses the base
/// `build_impact_heatmap` (no STM-specific zeroing) so neither player's forward
/// corridor is erased by the other's rear-wipe.
pub fn build_corridor_display_squares(scratch: &mut BfsScratch, board: &Board) -> [u16; 81] {
    let _ = scratch;
    build_impact_heatmap(board).square_heat
}

fn merge_corridor_max(a: &mut CorridorAttention, b: &CorridorAttention) {
    for i in 0..81 {
        a.square_heat[i] = a.square_heat[i].max(b.square_heat[i]);
        a.route_flex[i] = a.route_flex[i].max(b.route_flex[i]);
        a.bottleneck_heat[i] = a.bottleneck_heat[i].max(b.bottleneck_heat[i]);
    }
}

/// Build combined two-player corridor attention for search ordering.
///
/// Uses per-square **max** of each player's heat (same as the web overlay), not sum —
/// summing both races doubled fringe heat and qualified ~40 walls per node in open games.
pub fn build_corridor_attention(scratch: &mut BfsScratch, board: &Board) -> CorridorAttention {
    let masks = DirMasks::from_board(board);
    let mut white = CorridorAttention::default();
    let mut black = CorridorAttention::default();
    {
        let (dist_from, dist_to) = scratch.dist_scratch_mut();
        add_player_corridor_attention(board, Player::One, masks, &mut white, dist_from, dist_to);
    }
    {
        let (dist_from, dist_to) = scratch.dist_scratch_mut();
        add_player_corridor_attention(board, Player::Two, masks, &mut black, dist_from, dist_to);
    }
    let mut attention = white;
    merge_corridor_max(&mut attention, &black);
    attention
}

/// Count low-flex squares on exact/near-shortest corridors (caging heuristic).
pub fn corridor_bottleneck_count(scratch: &mut BfsScratch, board: &Board, player: Player) -> u8 {
    let masks = DirMasks::from_board(board);
    let (sr, sc) = board.pawn(player);
    let start = square_index(sr, sc);
    let (dist_from, dist_to) = scratch.dist_scratch_mut();
    fill_dist_from_sq(start, masks, dist_from);
    fill_dist_to_goal_row(player, masks, dist_to);
    let shortest_to_goal = dist_from[start as usize];
    if shortest_to_goal == u8::MAX {
        return 8;
    }

    let mut delta_arr = [u16::MAX; 81];
    for sq in 0usize..81 {
        if let Some(d) = corridor_delta(sq as u8, dist_from, dist_to, shortest_to_goal) {
            delta_arr[sq] = d;
        }
    }

    let mut bottlenecks = 0u8;
    for sq in 0u8..81 {
        let delta = delta_arr[sq as usize];
        if delta > BOTTLENECK_CORRIDOR_DELTA || dist_to[sq as usize] == 0 {
            continue;
        }
        let flex = reasonable_forward_continuations(sq, masks, dist_from, dist_to, &delta_arr);
        if flex <= 1 {
            bottlenecks = bottlenecks.saturating_add(1);
        }
    }
    bottlenecks.min(8)
}

// ---------------------------------------------------------------------------
// BFF impact heatmap (fast path for LMR move ordering)
// ---------------------------------------------------------------------------

static CAT_DISTANCE_BIAS_BP: AtomicI32 = AtomicI32::new(DEFAULT_CAT_DISTANCE_BIAS_BP as i32);

/// Visualization-only path tilt (basis points). CAT worker may call this; search worker does not.
pub fn set_cat_distance_bias_bp(bias: i16) {
    CAT_DISTANCE_BIAS_BP.store(i32::from(bias.clamp(-9_900, 9_900)), Ordering::Relaxed);
}

pub fn cat_distance_bias_bp() -> i16 {
    CAT_DISTANCE_BIAS_BP
        .load(Ordering::Relaxed)
        .clamp(-9_900, 9_900) as i16
}

pub fn default_cat_distance_bias_bp() -> i16 {
    DEFAULT_CAT_DISTANCE_BIAS_BP
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImpactHeatPreset {
    Conservative,
    Aggressive,
}

const ACTIVE_IMPACT_HEAT_PRESET: ImpactHeatPreset = ImpactHeatPreset::Conservative;

fn impact_heat_for_preset(delta: usize, preset: ImpactHeatPreset) -> u16 {
    if delta > MAX_IMPACT_HEAT_DELTA {
        return 0;
    }
    const CONSERVATIVE: [u16; MAX_IMPACT_HEAT_DELTA + 1] = [200, 77, 40, 25, 18, 12, 8, 4, 2];
    const AGGRESSIVE: [u16; MAX_IMPACT_HEAT_DELTA + 1] = [200, 180, 160, 140, 100, 60, 30, 14, 6];
    match preset {
        ImpactHeatPreset::Conservative => CONSERVATIVE[delta],
        ImpactHeatPreset::Aggressive => AGGRESSIVE[delta],
    }
}

fn impact_heat(delta: usize) -> u16 {
    impact_heat_for_preset(delta, ACTIVE_IMPACT_HEAT_PRESET)
}

/// Goal-hot (+bias) / pawn-hot (−bias) tilt along the to-goal layer index `j`.
fn distance_bias_mult(j: usize, shortest: usize, bias_bp: i16) -> u16 {
    if shortest == 0 || bias_bp == 0 {
        return 100;
    }
    let magnitude = i32::from(bias_bp).abs().min(9_900);
    let j = j as i32;
    let shortest = shortest as i32;
    let reduction = if bias_bp > 0 {
        magnitude * j / shortest / 100
    } else {
        magnitude * (shortest - j) / shortest / 100
    };
    (100 - reduction).clamp(1, 100) as u16
}

/// Add `w` to `heat[sq]` for every set cell of `mask` (saturating).
#[inline]
fn scatter_add(heat: &mut [u16; 81], mask: u128, w: u16) {
    if w == 0 {
        return;
    }
    let mut bits = mask & FLOOD_PLAYABLE;
    while bits != 0 {
        let fb = bits.trailing_zeros();
        bits &= bits - 1;
        let sq = FLOOD_SQ_BY_BIT[fb as usize];
        if sq != u8::MAX {
            let slot = &mut heat[sq as usize];
            *slot = slot.saturating_add(w);
        }
    }
}

/// One player's impact contribution via overlapping bitmask layers.
pub(crate) fn add_player_impact_heat_with_bias(
    board: &Board,
    player: Player,
    masks: DirMasks,
    heat: &mut [u16; 81],
    bias_bp: i16,
) {
    let (sr, sc) = board.pawn(player);
    let start = square_index(sr, sc);
    let mut from = DistLayers::default();
    let mut to = DistLayers::default();
    fill_dist_layers_from_sq(start, masks, &mut from);
    fill_dist_layers_to_goal_row(player, masks, &mut to);

    let start_bit = flood_bit_sq(start);
    let Some(shortest) = (0..to.depth).find(|&d| to.masks[d] & start_bit != 0) else {
        return;
    };
    let tol = MAX_IMPACT_HEAT_DELTA;

    for i in 0..from.depth {
        let fi = from.masks[i];
        if fi == 0 {
            continue;
        }
        let jmax = (shortest + tol)
            .saturating_sub(i)
            .min(shortest)
            .min(to.depth.saturating_sub(1));
        for j in 0..=jmax {
            // Pawn square is a path-set entry node, not corridor heat.
            let cells = fi & to.masks[j] & FLOOD_PLAYABLE & !start_bit;
            if cells == 0 {
                continue;
            }
            let delta = (i + j).saturating_sub(shortest);
            let base = impact_heat(delta);
            if base == 0 {
                continue;
            }
            let mult = distance_bias_mult(j, shortest, bias_bp);
            let w = (u32::from(base) * u32::from(mult) / 100) as u16;
            scatter_add(heat, cells, w);
        }
    }
}

fn add_player_impact_heat(board: &Board, player: Player, masks: DirMasks, heat: &mut [u16; 81]) {
    let bias_bp = cat_distance_bias_bp();
    add_player_impact_heat_with_bias(board, player, masks, heat, bias_bp);
}

fn zero_pawn_entry_and_rear(heat: &mut [u16; 81], board: &Board, player: Player, masks: DirMasks) {
    let (sr, sc) = board.pawn(player);
    let pawn_sq = square_index(sr, sc);
    heat[pawn_sq as usize] = 0;

    let mut dist_to_goal = [u8::MAX; 81];
    fill_dist_to_goal_row(player, masks, &mut dist_to_goal);
    let our_dist = dist_to_goal[pawn_sq as usize];
    if our_dist == u8::MAX {
        return;
    }
    for sq in 0usize..81 {
        let d = dist_to_goal[sq];
        if d != u8::MAX && d > our_dist {
            heat[sq] = 0;
        }
    }
}

/// Fast bitmask impact heatmap for v16 LMR ordering and web square overlay.
///
/// Each player's heat is built and rear-zeroed independently before summing.
/// Sequential add-then-zero on a shared array would let P2's rear-wipe erase
/// P1's forward corridor when the pawns are close to their respective goals.
pub fn build_impact_heatmap(board: &Board) -> CorridorAttention {
    let masks = DirMasks::from_board(board);

    let mut h1 = [0u16; 81];
    add_player_impact_heat(board, Player::One, masks, &mut h1);
    zero_pawn_entry_and_rear(&mut h1, board, Player::One, masks);

    let mut h2 = [0u16; 81];
    add_player_impact_heat(board, Player::Two, masks, &mut h2);
    zero_pawn_entry_and_rear(&mut h2, board, Player::Two, masks);

    let mut out = CorridorAttention::default();
    for i in 0..81 {
        out.square_heat[i] = h1[i].saturating_add(h2[i]);
    }
    for player in [Player::One, Player::Two] {
        let (r, c) = board.pawn(player);
        out.square_heat[square_index(r, c) as usize] = 0;
    }
    out
}

/// Race-aware variant: builds the symmetric heatmap then additionally zeros any
/// combined heat that is strictly behind the side-to-move's pawn (farther from
/// their goal than they currently are). The symmetric base gives the correct view
/// of both players' corridors without cross-player erasure; this extra pass lets
/// the search ignore walls that can never help the mover.
#[inline]
pub fn build_impact_heatmap_for_stm(board: &Board, bfs: &mut BfsScratch) -> CorridorAttention {
    let _ = bfs;
    let mut cat = build_impact_heatmap(board);
    let stm = board.side();
    let masks = DirMasks::from_board(board);
    zero_pawn_entry_and_rear(&mut cat.square_heat, board, stm, masks);
    cat
}

// (test module omitted here for brevity — full file has ~300 lines of unit
// tests asserting e.g. "center hotter than corner", "pawn entry square is
// zero", "behind-pawn squares are always zero heat", "wall on corridor beats
// wall in corner". See engine/src/cat/build.rs lines 452-748 if you need them.)
```

---

## 4. `src/cat/prune.rs` — CAT-backed pruning + move ordering (core logic, ~1050 lines of the 2207-line file; trailing ~700 lines are `#[cfg(test)]` unit tests, omitted)

This is the single most important file for your question. It contains:
- **The dead-zone wall filter** (`wall_in_dead_zone`, `wall_completely_skipped`) —
  the SOUND, provably-safe "this wall touches only squares neither player can
  ever reach, so it's a pure waste of inventory" cut. This is unconditional
  (not a heuristic) and is exactly "engine ignoring useless walls stuck in
  parts of the board nobody can reach."
- **The heuristic wall relevance filter** (`wall_should_search`) — CAT-hot-corridor
  based cut for walls that ARE in reachable territory but don't touch either
  player's near-shortest path. This is a heuristic (can theoretically miss a
  move) but is what actually prunes "walls in the middle of the board that
  don't affect either racer's straight corridor."
- **CAT-based LMR depth reduction** (`cat_heat_child_depth`, `cat_v16_lmr_reduction*`,
  `cat_lmr_total_reduction`) — how many plies to shave off a move's search
  based on how cold its CAT heat is.
- **Move ordering** (`move_order_score`, `order_moves`) — race-gain-first,
  CAT-heat as tiebreak/fallback.

```rust
//! CAT-backed move pruning and ordering — does **not** generate moves.
//!
//! Legal moves always come from `moves::generate_legal_moves_slice`. This module
//! filters them using BFS shortest-path data and multi-route corridor heat for
//! both players, then feeds αβ / MCTS with a smaller, tactically relevant set.

use crate::cat::attention::CorridorAttention;
use crate::cat::constants::{CAT_COLD_CM, CAT_HOT_CM, DIST_PENALTY};
use crate::core::board::{Board, Move, Player, WallOrientation};
use crate::movegen::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::opening::book::BookHint;
use crate::path::distance::fill_dist_to_goal_row;
use crate::path::masks::DirMasks;
use crate::path::BfsScratch;
use crate::util::grid::{has_wall, is_goal, square_index, unpack_square, wall_touch_squares};
/// Wasted turn: opponent gets to improve on reply.
pub const TEMPO_PENALTY: i32 = -10;
const WALL_CROSS_GAP_CM: i32 = 40;
const WALL_CROSS_BLOCK_CM: i32 = 35;
const WALL_LOCAL_DENIAL_SLOTS: usize = 6;
const WALL_DENIAL_BOOST_NUM: i32 = 3;
const WALL_DENIAL_BOOST_DEN: i32 = 2;

pub fn wall_blocks_path_step(mv: Move, sq1: u8, sq2: u8) -> bool {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    let (r1, c1) = unpack_square(sq1);
    let (r2, c2) = unpack_square(sq2);
    match orientation {
        WallOrientation::Horizontal => {
            if c1 == c2 && r1.abs_diff(r2) == 1 {
                let min_r = r1.min(r2);
                min_r == row && (c1 == col || c1 == col + 1)
            } else {
                false
            }
        }
        WallOrientation::Vertical => {
            if r1 == r2 && c1.abs_diff(c2) == 1 {
                let min_c = c1.min(c2);
                min_c == col && (r1 == row || r1 == row + 1)
            } else {
                false
            }
        }
    }
}

pub fn wall_intersects_path(mv: Move, path: &[u8], len: usize) -> bool {
    if len <= 1 {
        return false;
    }
    for i in 0..(len - 1) {
        if wall_blocks_path_step(mv, path[i], path[i + 1]) {
            return true;
        }
    }
    false
}

pub fn get_shortest_path(
    board: &Board,
    player: Player,
    bfs: &mut BfsScratch,
    path_out: &mut [u8; 81],
) -> usize {
    let mut next_out = [u8::MAX; 81];
    bfs.fill_next_toward_goal(board, player, &mut next_out);

    let (pr, pc) = board.pawn(player);
    let mut current = square_index(pr, pc);
    let mut len = 0;
    while current != u8::MAX {
        path_out[len] = current;
        len += 1;
        if len >= 81 {
            break;
        }
        current = next_out[current as usize];
    }
    len
}

pub fn path_distance(player: Player, path: &[u8], len: usize) -> u8 {
    if len == 0 {
        return DIST_PENALTY;
    }
    let last_sq = path[len - 1];
    let (r, _) = unpack_square(last_sq);
    if is_goal(player, r) {
        (len - 1) as u8
    } else {
        DIST_PENALTY
    }
}

/// Opponent path gain and our path loss from a wall — one make/unmake, two BFS.
pub fn wall_race_swing(
    board: &mut Board,
    mv: Move,
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
) -> (i32, i32) {
    let Move::Wall { .. } = mv else {
        return (0, 0);
    };
    let us = board.side();
    let opp = us.opposite();
    let undo = board.make_move(mv);
    let our_after = bfs.shortest_distance(board, us).unwrap_or(DIST_PENALTY);
    let opp_after = bfs.shortest_distance(board, opp).unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);
    let opp_gain = i32::from(opp_after.saturating_sub(opp_dist));
    let our_loss = i32::from(our_after.saturating_sub(our_dist));
    (opp_gain, our_loss)
}

/// Net race swing from playing a wall: opponent path lengthening minus our path lengthening.
pub fn wall_net_race(
    board: &mut Board,
    mv: Move,
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
) -> i32 {
    let (opp_gain, our_loss) = wall_race_swing(board, mv, our_dist, opp_dist, bfs);
    opp_gain - our_loss
}

pub fn min_wall_net_race(our_dist: u8, opp_dist: u8) -> i32 {
    if our_dist > opp_dist {
        // Losing the race — any wall that lengthens the opponent counts.
        1
    } else if our_dist == opp_dist {
        // Tied — need a stronger swing to spend a wall.
        2
    } else {
        1
    }
}

pub fn opp_path_gain(board: &mut Board, mv: Move, opp_dist: u8, bfs: &mut BfsScratch) -> i32 {
    let Move::Wall { .. } = mv else {
        return 0;
    };
    let opp = board.side().opposite();
    let undo = board.make_move(mv);
    let new_opp = bfs.shortest_distance(board, opp).unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);
    i32::from(new_opp.saturating_sub(opp_dist))
}

pub fn our_path_gain(board: &mut Board, mv: Move, our_dist: u8, bfs: &mut BfsScratch) -> i32 {
    let Move::Pawn { .. } = mv else {
        return 0;
    };
    let us = board.side();
    let undo = board.make_move(mv);
    let new_our = bfs.shortest_distance(board, us).unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);
    i32::from(our_dist.saturating_sub(new_our))
}

pub fn move_immediate_gain(
    board: &mut Board,
    mv: Move,
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
) -> i32 {
    match mv {
        Move::Pawn { .. } => {
            let g = our_path_gain(board, mv, our_dist, bfs);
            if g > 0 {
                g
            } else {
                TEMPO_PENALTY
            }
        }
        Move::Wall { .. } => {
            let g = opp_path_gain(board, mv, opp_dist, bfs);
            if g > 0 {
                g
            } else {
                TEMPO_PENALTY
            }
        }
    }
}

pub fn is_tactical_move(
    board: &mut Board,
    mv: Move,
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
) -> bool {
    match mv {
        Move::Pawn { .. } => our_path_gain(board, mv, our_dist, bfs) > 0,
        Move::Wall { .. } => opp_path_gain(board, mv, opp_dist, bfs) > 0,
    }
}

#[inline]
fn wall_coord_in_bounds(row: u8, col: u8) -> bool {
    row <= 7 && col <= 7
}

fn is_cross_gap_wall(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    if !wall_coord_in_bounds(row, col) || has_wall(board, row, col, orientation) {
        return false;
    }
    match orientation {
        WallOrientation::Horizontal => {
            row >= 1
                && row <= 6
                && has_wall(board, row - 1, col, WallOrientation::Vertical)
                && has_wall(board, row + 1, col, WallOrientation::Vertical)
        }
        WallOrientation::Vertical => {
            col >= 1
                && col <= 6
                && has_wall(board, row, col - 1, WallOrientation::Horizontal)
                && has_wall(board, row, col + 1, WallOrientation::Horizontal)
        }
    }
}

fn blocks_cross_gap_wall(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    if is_cross_gap_wall(board, row, col, orientation) || !wall_coord_in_bounds(row, col) {
        return false;
    }
    match orientation {
        WallOrientation::Horizontal => {
            for dc in [-1i8, 1i8] {
                let gap_col = col as i8 + dc;
                if !(1..=6).contains(&gap_col) {
                    continue;
                }
                let gc = gap_col as u8;
                if row >= 1
                    && row <= 6
                    && has_wall(board, row - 1, gc, WallOrientation::Vertical)
                    && has_wall(board, row + 1, gc, WallOrientation::Vertical)
                {
                    return true;
                }
            }
        }
        WallOrientation::Vertical => {
            for dr in [-1i8, 1i8] {
                let gap_row = row as i8 + dr;
                if !(1..=6).contains(&gap_row) {
                    continue;
                }
                let gr = gap_row as u8;
                if col >= 1
                    && col <= 6
                    && has_wall(board, gr, col - 1, WallOrientation::Horizontal)
                    && has_wall(board, gr, col + 1, WallOrientation::Horizontal)
                {
                    return true;
                }
            }
        }
    }
    false
}

fn wall_shape_local_heat(
    cat: &CorridorAttention,
    row: u8,
    col: u8,
    orientation: WallOrientation,
) -> u16 {
    let edge = cat.wall_edge_heat(row, col, orientation);
    let touch = wall_touch_squares(row, col, orientation)
        .iter()
        .map(|&(r, c)| cat.square_heat(r, c))
        .max()
        .unwrap_or(0);
    edge.max(touch)
}

pub fn wall_slot_index(mv: Move) -> Option<usize> {
    match mv {
        Move::Wall {
            row,
            col,
            orientation,
        } if row < 8 && col < 8 => {
            let base = match orientation {
                WallOrientation::Horizontal => 0,
                WallOrientation::Vertical => 64,
            };
            Some(base + row as usize * 8 + col as usize)
        }
        _ => None,
    }
}

fn push_wall_slot(
    row: i16,
    col: i16,
    orientation: WallOrientation,
    out: &mut [usize; WALL_LOCAL_DENIAL_SLOTS],
    n: &mut usize,
) {
    if !(0..=7).contains(&row) || !(0..=7).contains(&col) {
        return;
    }
    let mv = Move::Wall {
        row: row as u8,
        col: col as u8,
        orientation,
    };
    let Some(idx) = wall_slot_index(mv) else {
        return;
    };
    if out[..*n].contains(&idx) {
        return;
    }
    out[*n] = idx;
    *n += 1;
}

/// Wall slots made physically illegal by placing `mv`: same/cross slot plus
/// adjacent same-orientation slots. This is intentionally local and does not
/// recurse into child move generation.
pub fn locally_invalidated_wall_slots(
    mv: Move,
    out: &mut [usize; WALL_LOCAL_DENIAL_SLOTS],
) -> usize {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return 0;
    };
    let mut n = 0usize;
    let row = row as i16;
    let col = col as i16;
    let other = match orientation {
        WallOrientation::Horizontal => WallOrientation::Vertical,
        WallOrientation::Vertical => WallOrientation::Horizontal,
    };
    push_wall_slot(row, col, orientation, out, &mut n);
    push_wall_slot(row, col, other, out, &mut n);
    match orientation {
        WallOrientation::Horizontal => {
            push_wall_slot(row, col - 1, orientation, out, &mut n);
            push_wall_slot(row, col + 1, orientation, out, &mut n);
        }
        WallOrientation::Vertical => {
            push_wall_slot(row - 1, col, orientation, out, &mut n);
            push_wall_slot(row + 1, col, orientation, out, &mut n);
        }
    }
    n
}

pub fn legal_neighbor_denial_heat(
    mv: Move,
    candidates: &[Move],
    direct_heats: &[i32],
    n: usize,
) -> i32 {
    let Some(self_slot) = wall_slot_index(mv) else {
        return 0;
    };
    let mut local = [usize::MAX; WALL_LOCAL_DENIAL_SLOTS];
    let local_n = locally_invalidated_wall_slots(mv, &mut local);
    let mut best = 0i32;
    for i in 0..n.min(candidates.len()).min(direct_heats.len()) {
        let Some(slot) = wall_slot_index(candidates[i]) else {
            continue;
        };
        if slot == self_slot || !local[..local_n].contains(&slot) {
            continue;
        }
        let heat = direct_heats[i].max(0);
        if heat >= i32::from(CAT_HOT_CM) {
            let boosted = heat.saturating_mul(WALL_DENIAL_BOOST_NUM) / WALL_DENIAL_BOOST_DEN;
            best = best.max(boosted);
        }
    }
    best
}

pub fn wall_shape_attention_bonus(board: &Board, mv: Move, cat: &CorridorAttention) -> i32 {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return 0;
    };
    if wall_shape_local_heat(cat, row, col, orientation) < CAT_HOT_CM {
        return 0;
    }
    if is_cross_gap_wall(board, row, col, orientation) {
        WALL_CROSS_GAP_CM
    } else if blocks_cross_gap_wall(board, row, col, orientation) {
        WALL_CROSS_BLOCK_CM
    } else {
        0
    }
}

/// Live squares orthogonally adjacent to sealed-off (unreachable) territory.
pub fn corridor_mouth_mask(reachable: u128) -> u128 {
    let mut mouths = 0u128;
    for sq in 0u8..81 {
        if reachable & (1u128 << sq) == 0 {
            continue;
        }
        let (r, c) = unpack_square(sq);
        for (dr, dc) in [(-1i8, 0), (1, 0), (0, -1), (0, 1)] {
            let nr = r as i16 + dr as i16;
            let nc = c as i16 + dc as i16;
            // 0..=8 — board edges are NOT sealed territory. With 0..=9 every
            // bottom/right edge square became a phantom "mouth".
            if !(0..=8).contains(&nr) || !(0..=8).contains(&nc) {
                continue;
            }
            let neighbor = square_index(nr as u8, nc as u8);
            if reachable & (1u128 << neighbor) == 0 {
                mouths |= 1u128 << sq;
                break;
            }
        }
    }
    mouths
}

/// Mouth squares, their reachable ring, and adjacent sealed cells (the gap slot itself).
pub fn gap_play_zone_mask(reachable: u128) -> u128 {
    let mouths = corridor_mouth_mask(reachable);
    let mut zone = mouths;
    for sq in 0u8..81 {
        if mouths & (1u128 << sq) == 0 {
            continue;
        }
        let (r, c) = unpack_square(sq);
        for (dr, dc) in [(-1i8, 0), (1, 0), (0, -1), (0, 1)] {
            let nr = r as i16 + dr as i16;
            let nc = c as i16 + dc as i16;
            if !(0..=8).contains(&nr) || !(0..=8).contains(&nc) {
                continue;
            }
            // Include both live ring and the sealed gap cell — half-walls and cross-gap H/V land here.
            zone |= 1u128 << square_index(nr as u8, nc as u8);
        }
    }
    zone
}

/// Touches sealed (unreachable) territory that is not part of the gap mouth play zone.
fn wall_probes_sealed_interior(mv: Move, reachable: u128, gap_zone: u128) -> bool {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    for (r, c) in wall_touch_squares(row, col, orientation) {
        let sq = square_index(r, c);
        if reachable & (1u128 << sq) == 0 && gap_zone & (1u128 << sq) == 0 {
            return true;
        }
    }
    false
}

fn wall_touches_gap_zone(mv: Move, gap_zone: u128) -> bool {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    for (r, c) in wall_touch_squares(row, col, orientation) {
        if gap_zone & (1u128 << square_index(r, c)) != 0 {
            return true;
        }
    }
    false
}

/// SOUND "useless wall" test: true iff EVERY square the wall touches is
/// unreachable (outside both pawns' reachable region). Such a wall can never be
/// adjacent to any pawn, so it can block no path — placing it only wastes
/// inventory, which is never an advantage → it can never be the best move, so
/// pruning it is NPS-only and cannot cost Elo.
///
/// Exclusion is built in: a wall touching even ONE reachable square — including a
/// half-in-void / half-in-playable wall — returns `false` (kept), preserving the
/// tactical half-wall placement.
pub fn wall_in_dead_zone(mv: Move, reachable: u128) -> bool {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    for (r, c) in wall_touch_squares(row, col, orientation) {
        if reachable & (1u128 << square_index(r, c)) != 0 {
            return false;
        }
    }
    true
}

/// Whether a wall can affect either player's reasonable routes to goal.
pub fn wall_should_search(
    mv: Move,
    cat: &CorridorAttention,
    reachable: u128,
    gap_zone: u128,
    board: &mut Board,
    _our_dist: u8,
    _opp_dist: u8,
    opp_path: &[u8],
    opp_path_len: usize,
    _bfs: &mut BfsScratch,
) -> bool {
    if wall_in_dead_zone(mv, reachable) {
        return false;
    }
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    // Gap geometry: H through V|V (or flank block beside it) can seal/open the pocket.
    // If the wall is not in a dead zone it touches live territory — always search it.
    if is_cross_gap_wall(board, row, col, orientation)
        || blocks_cross_gap_wall(board, row, col, orientation)
    {
        return true;
    }
    // Any wall touching the playable/sealed mouth, gap slot, or immediate ring.
    if gap_zone != 0 && wall_touches_gap_zone(mv, gap_zone) {
        return true;
    }
    // Wall reaches into sealed void away from the gap — no tactical value.
    if gap_zone != 0 && wall_probes_sealed_interior(mv, reachable, gap_zone) {
        return false;
    }
    // Fast exact hit: wall blocks a step of the opponent's current shortest path.
    if wall_intersects_path(mv, opp_path, opp_path_len) {
        return true;
    }
    // CAT v3 multi-route check: does the wall cut an edge on a HOT corridor
    // (exact shortest routes / contested lanes) of either player? This is the
    // anti-tunnel-vision signal — a single witness path (CAT v2) misses
    // equal-length reroutes. CAT is built once per node: no extra BFS per wall.
    // HOT (not COLD) keeps the move list tight — the delta-2/3 fringe admitted
    // nearly every wall on the board and exploded the tree.
    cat.wall_edge_heat(row, col, orientation) >= CAT_HOT_CM
}

/// Hard skip — dead void or sealed interior away from gap; never searched (not LMR).
pub fn wall_completely_skipped(mv: Move, board: &Board, reachable: u128, gap_zone: u128) -> bool {
    if wall_in_dead_zone(mv, reachable) {
        return true;
    }
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    if is_cross_gap_wall(board, row, col, orientation)
        || blocks_cross_gap_wall(board, row, col, orientation)
    {
        return false;
    }
    if gap_zone != 0 && wall_touches_gap_zone(mv, gap_zone) {
        return false;
    }
    gap_zone != 0 && wall_probes_sealed_interior(mv, reachable, gap_zone)
}

/// Filter legal moves for search — never generates moves, only prunes `moves` output.
/// `cat` is the caller's corridor attention and `opp_path` the caller's witness
/// shortest path — both computed once per node and shared with move ordering.
#[allow(clippy::too_many_arguments)]
pub fn collect_search_moves(
    board: &mut Board,
    buf: &mut [Move],
    bfs: &mut BfsScratch,
    cat: &CorridorAttention,
    opp_path: &[u8; 81],
    opp_path_len: usize,
    our_dist: u8,
    opp_dist: u8,
    tactical_only: bool,
    allow_walls: bool,
) -> usize {
    let mut scratch = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let full = generate_legal_moves_slice(board, &mut scratch, bfs);
    if full == 0 {
        return 0;
    }

    let reachable = bfs.both_reachable_mask(board);
    let gap_zone = if allow_walls {
        gap_play_zone_mask(reachable)
    } else {
        0
    };

    let mut n = 0usize;

    for i in 0..full {
        let mv = scratch[i];
        match mv {
            Move::Pawn { .. } => {
                if tactical_only && our_path_gain(board, mv, our_dist, bfs) <= 0 {
                    continue;
                }
                buf[n] = mv;
                n += 1;
            }
            Move::Wall { .. } => {
                if !allow_walls {
                    continue;
                }
                // Quiescence (tactical_only): walls are "noisy" only when they
                // actually lengthen the opponent's shortest path — the Quoridor
                // analog of a capture. Quiet walls must stand pat, not extend.
                if tactical_only {
                    // Cheap witness-path gate first; BFS only for the few that touch it.
                    if !wall_intersects_path(mv, opp_path, opp_path_len)
                        || opp_path_gain(board, mv, opp_dist, bfs) <= 0
                    {
                        continue;
                    }
                } else if !wall_should_search(
                    mv,
                    cat,
                    reachable,
                    gap_zone,
                    board,
                    our_dist,
                    opp_dist,
                    opp_path,
                    opp_path_len,
                    bfs,
                ) {
                    continue;
                }
                buf[n] = mv;
                n += 1;
            }
        }
    }

    // Main search must never be left without moves — fall back to full legality.
    // Quiescence (tactical_only) returns 0 instead: a position with no noisy
    // moves is quiet by definition and the caller stands pat on static eval.
    if n == 0 && !tactical_only {
        buf[..full].copy_from_slice(&scratch[..full]);
        return full;
    }
    n
}

/// Absolute CAT corridor floor — same threshold as `wall_should_search` / CAT viz.
#[inline]
pub fn is_cat_hot_corridor(cat_cm: i32) -> bool {
    cat_cm >= i32::from(CAT_HOT_CM)
}

/// Normalized heat at this node: 0 = at/below `cold_cm`, 1 = `cat_max`.
/// A move at 200 cm vs 100 cm with max 250 scales ~11× in fraction (proportional to hotspot).
#[inline]
pub fn cat_tail_dead_cutoff_cm(cat_ref_max: u16) -> i32 {
    if cat_ref_max == 0 {
        return 0;
    }
    i32::from(cat_ref_max.saturating_mul(crate::cat::constants::CAT_TAIL_DEAD_RATIO_PCT) / 100)
}

/// True when `cat_cm` is in the dead tail (≤ 10% of position peak) — max LMR reduction.
#[inline]
pub fn cat_is_dead_tail(cat_cm: i32, cat_ref_max: u16) -> bool {
    cat_ref_max > 0 && cat_cm >= 0 && cat_cm <= cat_tail_dead_cutoff_cm(cat_ref_max)
}

#[inline]
pub fn cat_heavy_fringe_cutoff_cm(cat_ref_max: u16) -> i32 {
    if cat_ref_max == 0 {
        return 0;
    }
    i32::from(cat_ref_max.saturating_mul(crate::cat::constants::CAT_HEAVY_FRINGE_RATIO_PCT) / 100)
}

/// Above dead tail but still weak (e.g. 41–80 cm when peak is 400) — heavy cut, not absolute max.
#[inline]
pub fn cat_is_heavy_fringe(cat_cm: i32, cat_ref_max: u16) -> bool {
    !cat_is_dead_tail(cat_cm, cat_ref_max)
        && cat_ref_max > 0
        && cat_cm >= 0
        && cat_cm <= cat_heavy_fringe_cutoff_cm(cat_ref_max)
}

#[inline]
pub fn cat_heat_fraction(cat_cm: i32, cat_max: u16, cold_cm: u16) -> f32 {
    if cat_max <= cold_cm {
        return if cat_cm > i32::from(cold_cm) {
            1.0
        } else {
            0.0
        };
    }
    let h = cat_cm.max(0) as f32;
    let cold = cold_cm as f32;
    let max_h = cat_max as f32;
    ((h - cold) / (max_h - cold)).clamp(0.0, 1.0)
}

/// Full-depth LMR when heat fraction reaches the profile hot-ratio gate.
#[inline]
pub fn cat_heat_skips_lmr(cat_cm: i32, cat_max: u16, cold_cm: u16, hot_ratio_pct: u16) -> bool {
    if cat_max == 0 {
        return is_cat_hot_corridor(cat_cm);
    }
    cat_heat_fraction(cat_cm, cat_max, cold_cm) * 100.0 >= hot_ratio_pct as f32
}

/// Skip LMR when move heat fraction clears the hot-ratio gate (replaces flat cm cutoff).
#[inline]
pub fn is_lmr_heat_hot(cat_cm: i32, cat_max: u16, cold_cm: u16, hot_ratio_pct: u16) -> bool {
    cat_heat_skips_lmr(cat_cm, cat_max, cold_cm, hot_ratio_pct)
}

/// Per-node CAT ceilings — walls scale against wall hotspots, not the sprint pawn.
#[derive(Debug, Clone, Copy, Default)]
pub struct CatHeatRefs {
    pub all: u16,
    pub walls: u16,
    pub pawns: u16,
}

pub fn cat_heat_refs(
    buf: &[Move],
    n: usize,
    board: &Board,
    cat: &CorridorAttention,
) -> CatHeatRefs {
    let mut refs = CatHeatRefs::default();
    for i in 0..n {
        let cm = move_corridor_attention(board, buf[i], cat).max(0) as u16;
        refs.all = refs.all.max(cm);
        match buf[i] {
            Move::Wall { .. } => refs.walls = refs.walls.max(cm),
            Move::Pawn { .. } => refs.pawns = refs.pawns.max(cm),
        }
    }
    refs
}

#[inline]
pub fn cat_heat_ref_max(mv: Move, refs: CatHeatRefs) -> u16 {
    match mv {
        Move::Wall { .. } => refs.walls,
        Move::Pawn { .. } => refs.all,
    }
}

pub fn cat_heat_refs_from_scores(buf: &[Move], n: usize, cat_heats: &[i32]) -> CatHeatRefs {
    let mut refs = CatHeatRefs::default();
    for i in 0..n.min(buf.len()).min(cat_heats.len()) {
        let cm = cat_heats[i].max(0) as u16;
        refs.all = refs.all.max(cm);
        match buf[i] {
            Move::Wall { .. } => refs.walls = refs.walls.max(cm),
            Move::Pawn { .. } => refs.pawns = refs.pawns.max(cm),
        }
    }
    refs
}

/// Target child plies from CAT heat — cold fringe caps at 1–2, hotspots keep full depth.
pub fn cat_heat_child_depth(
    cat_cm: i32,
    cat_ref_max: u16,
    cold_cm: u16,
    child_depth_full: u32,
) -> u32 {
    if child_depth_full == 0 {
        return 0;
    }
    if cat_ref_max > 0 && cat_is_dead_tail(cat_cm, cat_ref_max) {
        return 1.min(child_depth_full);
    }
    if cat_ref_max > 0 && cat_is_heavy_fringe(cat_cm, cat_ref_max) {
        let heavy = ((child_depth_full as f32) * 0.28).round().max(2.0) as u32;
        return heavy.min(child_depth_full);
    }
    if cat_ref_max == 0 || cat_cm <= 0 {
        return 1.min(child_depth_full);
    }
    let heat_t = cat_heat_fraction(cat_cm, cat_ref_max, cold_cm);
    // Steep curve — 245cm peak keeps ~full depth; 98cm fringe gets 1–2 plies (not flat 4% nodes).
    let mut used = (heat_t.powf(2.35) * child_depth_full as f32)
        .round()
        .max(1.0) as u32;
    if heat_t < 0.08 {
        used = 1;
    } else if heat_t < 0.18 {
        used = used.min(2);
    } else if heat_t < 0.35 {
        used = used.min((child_depth_full / 3).max(2));
    }
    used.min(child_depth_full)
}

/// Default CAT attention ceiling for Titanium v16 LMR (override via `TITANIUM_CAT_LMR_CEILING`).
pub const CAT_V16_LMR_CEILING_DEFAULT: u16 = 800;
pub const CAT_V16_LMR_CEILINGS: [u16; 3] = [500, 800, 1000];
/// Fringe cutoff: moves below this fraction of the effective position maximum search at child depth 1.
pub const CAT_V16_FRINGE_PCT_DEFAULT: u16 = 5;
pub const CAT_V16_FRINGE_PCT_STEP_PER_WORKER: u16 = 10;
pub const CAT_V16_FRINGE_PCT_MAX: u16 = 70;

#[inline]
pub fn cat_v16_lmr_fringe_pct_for_worker(worker_id: usize) -> u16 {
    if worker_id == 0 {
        CAT_V16_FRINGE_PCT_DEFAULT
    } else {
        (worker_id as u16)
            .saturating_mul(CAT_V16_FRINGE_PCT_STEP_PER_WORKER)
            .min(CAT_V16_FRINGE_PCT_MAX)
    }
}

/// Parse v16 LMR ceiling from env (`500`, `800`, or `1000`); defaults to [`CAT_V16_LMR_CEILING_DEFAULT`].
pub fn cat_v16_lmr_ceiling_from_env() -> u16 {
    std::env::var("TITANIUM_CAT_LMR_CEILING")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .filter(|v| CAT_V16_LMR_CEILINGS.contains(v))
        .unwrap_or(CAT_V16_LMR_CEILING_DEFAULT)
}

/// Titanium v16 late-move reduction from CAT heat.
///
/// Normalizes against the hottest same-kind move in the position, capped by the
/// selected upper bound (`ceiling` in 500/800/1000 cm). Moves at or colder than
/// `fringe_pct` of that effective maximum search at child depth 1; warmer moves
/// keep a proportional fraction of the remaining depth.
pub fn cat_v16_lmr_reduction_plies(
    mv: Move,
    cat_cm: i32,
    refs: CatHeatRefs,
    ceiling: u16,
    fringe_pct: u16,
    child_depth_full: u32,
) -> u32 {
    if child_depth_full <= 1 {
        return 0;
    }
    let cat_ref = cat_heat_ref_max(mv, refs).min(ceiling);
    if cat_ref == 0 {
        return child_depth_full.saturating_sub(1);
    }
    let fringe_pct = fringe_pct.min(100);
    let threshold = (u32::from(cat_ref) * u32::from(fringe_pct)).div_ceil(100) as i32;
    if cat_cm <= threshold {
        return child_depth_full.saturating_sub(1);
    }
    let reduction =
        cat_heat_depth_reduction(cat_cm, cat_ref, threshold.max(0) as u16, child_depth_full);
    reduction.min(child_depth_full.saturating_sub(2))
}

/// Returns CAT/index reducibility in `[0,1]`: 0 = important, 1 = very reducible.
pub fn cat_v16_lmr_reducibility(
    cat_cm: i32,
    refs: CatHeatRefs,
    mv: Move,
    move_index: usize,
) -> f64 {
    if move_index == 0 {
        return 0.0;
    }
    let cat_ref = cat_heat_ref_max(mv, refs);
    let cat_norm = if cat_ref == 0 {
        0.0
    } else {
        (cat_cm.max(0) as f64 / f64::from(cat_ref)).clamp(0.0, 1.0)
    };
    let unimportance = 1.0 - cat_norm;
    const INDEX_CAP: f64 = 16.0;
    let lateness = (move_index as f64 / INDEX_CAP).min(1.0);
    0.5 * unimportance + 0.5 * lateness
}

/// Returns the depth fraction in `[0,1]` **before** rounding to integer plies.
pub fn cat_v16_lmr_reduction_frac(
    cat_cm: i32,
    refs: CatHeatRefs,
    mv: Move,
    move_index: usize,
    aggression: f64,
) -> f64 {
    if move_index == 0 || aggression <= 0.0 {
        return 0.0;
    }
    let reducibility = cat_v16_lmr_reducibility(cat_cm, refs, mv, move_index);
    const SHARP: f64 = 3.0;
    let exponent = 1.0 + (1.0 - reducibility) * SHARP;
    aggression.clamp(0.0, 1.0).powf(exponent)
}

/// Connected CAT-LMR reduction — ONE intuitive model over both the move index
/// and its corridor impact. `aggression` ∈ [0,1] is the single tuning knob.
/// Returns total reduction plies (subsumes the old base index-LMR).
pub fn cat_v16_lmr_reduction(
    mv: Move,
    cat_cm: i32,
    refs: CatHeatRefs,
    move_index: usize,
    child_depth_full: u32,
    aggression: f64,
) -> u32 {
    if child_depth_full <= 1 || move_index == 0 || aggression <= 0.0 {
        return 0;
    }
    let frac = cat_v16_lmr_reduction_frac(cat_cm, refs, mv, move_index, aggression);
    let reduction = (f64::from(child_depth_full) * frac).round() as u32;
    reduction.min(child_depth_full.saturating_sub(1))
}

/// CAT-shaped LMR plies from heat fraction — scales child depth like the heatmap (245 ≫ 98).
pub fn cat_heat_depth_reduction(
    cat_cm: i32,
    cat_ref_max: u16,
    cold_cm: u16,
    child_depth_full: u32,
) -> u32 {
    let used = cat_heat_child_depth(cat_cm, cat_ref_max, cold_cm, child_depth_full);
    child_depth_full.saturating_sub(used)
}

/// CAT-proportional depth — no hard move-count cap; late-index table only stacks on cold moves.
pub fn cat_lmr_total_reduction(
    mv: Move,
    cat_cm: i32,
    refs: CatHeatRefs,
    cold_cm: u16,
    depth: u32,
    base_r: u32,
    opp_path: &[u8],
    opp_path_len: usize,
    corridor_relevant: bool,
) -> u32 {
    let child_full = depth.saturating_sub(1);
    let cat_ref = cat_heat_ref_max(mv, refs);
    let heat_t = if cat_ref > 0 && cat_cm > 0 {
        cat_heat_fraction(cat_cm, cat_ref, cold_cm)
    } else {
        0.0
    };
    let mut reduction = cat_heat_depth_reduction(cat_cm, cat_ref, cold_cm, child_full);
    // Stockfish LMR table only bites on cold late moves — never flatten CAT hotspots.
    if heat_t < 0.22 {
        reduction = reduction.saturating_add(base_r);
    }
    if matches!(mv, Move::Wall { .. }) && cat_cm == 0 {
        reduction = reduction.saturating_add(2);
    } else if matches!(mv, Move::Wall { .. })
        && !wall_intersects_path(mv, opp_path, opp_path_len)
        && !corridor_relevant
        && heat_t < 0.35
    {
        reduction = reduction.saturating_add(1);
    }
    let min_used = 1u32;
    let max_reduction = child_full.saturating_sub(min_used);
    reduction.min(max_reduction).min(depth.saturating_sub(1))
}

/// Quiet-corridor ordering boost — hotter CAT slots sort before colder ones.
#[inline]
pub fn cat_corridor_order_boost(cat_cm: i32, cat_max: u16, cold_cm: u16) -> i32 {
    (cat_heat_fraction(cat_cm, cat_max, cold_cm) * 16_000.0).round() as i32
}

fn cat_score_for_move(mv: Move, cat: &CorridorAttention) -> i32 {
    match mv {
        Move::Pawn { row, col } => i32::from(cat.square_heat(row, col)),
        Move::Wall {
            row,
            col,
            orientation,
        } => i32::from(cat.wall_edge_heat(row, col, orientation)),
    }
}

/// Combined corridor heat for LMR / futility (higher = more likely to matter).
/// Peak CAT heat (cm) over legal pawn moves for each player — NNUE `cat_best_p0/p1`.
pub fn best_pawn_cat_heats(
    board: &Board,
    cat: &CorridorAttention,
    bfs: &mut BfsScratch,
) -> (u16, u16) {
    let mut best = [0i32; 2];
    for (pi, player) in [Player::One, Player::Two].into_iter().enumerate() {
        let mut b = board.clone();
        b.side_to_move = player;
        let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = generate_legal_moves_slice(&mut b, &mut buf, bfs);
        for mv in &buf[..n] {
            if matches!(mv, Move::Pawn { .. }) {
                best[pi] = best[pi].max(move_corridor_attention(&b, *mv, cat));
            }
        }
    }
    (
        best[0].clamp(0, u16::MAX as i32) as u16,
        best[1].clamp(0, u16::MAX as i32) as u16,
    )
}

pub fn move_corridor_attention(board: &Board, mv: Move, cat: &CorridorAttention) -> i32 {
    cat_score_for_move(mv, cat) + wall_shape_attention_bonus(board, mv, cat)
}

/// Simple move impact from the BFF heatmap (v16 LMR ordering): a pawn move
/// inherits its destination square's heat; a wall inherits the HIGHEST heat of
/// the squares it touches. No per-move pathfinding, no defensive special-casing —
/// just "how hot is the part of the board this move acts on".
pub fn move_impact_heat(mv: Move, cat: &CorridorAttention) -> i32 {
    match mv {
        Move::Pawn { row, col } => i32::from(cat.square_heat(row, col)),
        Move::Wall { row, col, .. } => {
            // A wall borders the 2×2 block of cells (row,col)..=(row+1,col+1).
            let h = |r: u8, c: u8| i32::from(cat.square_heat(r, c));
            h(row, col)
                .max(h(row + 1, col))
                .max(h(row, col + 1))
                .max(h(row + 1, col + 1))
        }
    }
}

fn wall_touched_squares(mv: Move) -> [u8; 4] {
    let Move::Wall { row, col, .. } = mv else {
        return [0; 4];
    };
    [
        square_index(row, col),
        square_index(row + 1, col),
        square_index(row, col + 1),
        square_index(row + 1, col + 1),
    ]
}

/// True when `sq` is strictly closer to goal than at least one pawn (both races).
fn square_ahead_of_any_pawn(board: &Board, masks: DirMasks, sq: u8) -> bool {
    for player in [Player::One, Player::Two] {
        let (sr, sc) = board.pawn(player);
        let pawn_sq = square_index(sr, sc);
        let mut dist_to_goal = [u8::MAX; 81];
        fill_dist_to_goal_row(player, masks, &mut dist_to_goal);
        let pawn_d = dist_to_goal[pawn_sq as usize];
        let sq_d = dist_to_goal[sq as usize];
        if pawn_d != u8::MAX && sq_d != u8::MAX && sq_d < pawn_d {
            return true;
        }
    }
    false
}

/// Position-symmetric LMR impact — identical regardless of side to move.
///
/// Walls off both shortest paths, or only touching rear squares for every pawn,
/// score zero so sprint tail walls do not steal depth from real race moves.
pub fn move_impact_heat_race(
    board: &Board,
    mv: Move,
    cat: &CorridorAttention,
    bfs: &mut BfsScratch,
) -> i32 {
    match mv {
        Move::Pawn { .. } => move_impact_heat(mv, cat),
        Move::Wall { .. } => {
            let mut white_path = [0u8; 81];
            let wl = get_shortest_path(board, Player::One, bfs, &mut white_path);
            let mut black_path = [0u8; 81];
            let bl = get_shortest_path(board, Player::Two, bfs, &mut black_path);
            let on_white = wall_intersects_path(mv, &white_path, wl);
            let on_black = wall_intersects_path(mv, &black_path, bl);
            if !on_white && !on_black {
                return 0;
            }
            let masks = DirMasks::from_board(board);
            let mut max_h = 0i32;
            for sq in wall_touched_squares(mv) {
                if !square_ahead_of_any_pawn(board, masks, sq) {
                    continue;
                }
                let on_path = white_path[..wl].contains(&sq) || black_path[..bl].contains(&sq);
                if !on_path {
                    continue;
                }
                let (r, c) = unpack_square(sq);
                max_h = max_h.max(i32::from(cat.square_heat(r, c)));
            }
            max_h
        }
    }
}

pub fn wall_path_impact_attention(
    board: &mut Board,
    mv: Move,
    white_dist: u8,
    black_dist: u8,
    bfs: &mut BfsScratch,
) -> i32 {
    let Move::Wall { .. } = mv else {
        return 0;
    };
    let undo = board.make_move(mv);
    let white_after = bfs
        .shortest_distance(board, Player::One)
        .unwrap_or(DIST_PENALTY);
    let black_after = bfs
        .shortest_distance(board, Player::Two)
        .unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);

    let white_gain = u32::from(white_after.saturating_sub(white_dist));
    let black_gain = u32::from(black_after.saturating_sub(black_dist));
    let total = white_gain + black_gain;
    if total == 0 {
        return 0;
    }
    let strongest = white_gain.max(black_gain);
    let affected_paths = u32::from(white_gain > 0) + u32::from(black_gain > 0);
    let shared_bonus = if affected_paths > 1 { 40 } else { 0 };
    (total * 120 + strongest * 50 + shared_bonus).min(i32::MAX as u32) as i32
}

pub fn move_corridor_attention_with_path(
    board: &mut Board,
    mv: Move,
    cat: &CorridorAttention,
    white_dist: u8,
    black_dist: u8,
    bfs: &mut BfsScratch,
) -> i32 {
    move_corridor_attention(board, mv, cat).max(wall_path_impact_attention(
        board, mv, white_dist, black_dist, bfs,
    ))
}

pub fn move_corridor_attention_with_denial(
    board: &Board,
    mv: Move,
    cat: &CorridorAttention,
    candidates: &[Move],
    direct_heats: &[i32],
    n: usize,
) -> i32 {
    let direct = move_corridor_attention(board, mv, cat);
    if matches!(mv, Move::Wall { .. }) {
        direct.max(legal_neighbor_denial_heat(mv, candidates, direct_heats, n))
    } else {
        direct
    }
}

/// Stockfish-style extras layered on top of tactical ordering.
#[derive(Clone, Copy, Default)]
pub struct OrderExtras {
    pub pv_move: Option<Move>,
    pub killers: [Option<Move>; 2],
}

#[inline]
fn apply_order_extras(base: i32, mv: Move, tt_best: Option<Move>, extras: &OrderExtras) -> i32 {
    let mut score = base;
    if tt_best == Some(mv) {
        return score;
    }
    if extras.pv_move == Some(mv) {
        score = score.max(9_500);
    }
    for killer in extras.killers {
        if killer == Some(mv) {
            score = score.max(8_500);
        }
    }
    score
}

pub fn move_order_score(
    board: &mut Board,
    mv: Move,
    tt_best: Option<Move>,
    book_hint: Option<BookHint>,
    our_dist: u8,
    opp_dist: u8,
    opp_path: &[u8],
    opp_path_len: usize,
    bfs: &mut BfsScratch,
    cat: &CorridorAttention,
    cat_max: u16,
    cold_cm: u16,
) -> i32 {
    if tt_best == Some(mv) {
        return 10_000;
    }
    if let Some(hint) = book_hint {
        if hint.mv == mv {
            // PV bias only — tactical race gains and TT still outrank theory.
            let bias = i32::from(hint.stm_bias) / 4;
            return 9_000 + i32::from(hint.priority) + bias;
        }
    }
    let behind = our_dist > opp_dist;
    let race_pressure = behind || opp_dist <= 4;

    if matches!(mv, Move::Wall { .. }) {
        // Witness-path gate: a wall off the opponent's current shortest path
        // has opp_gain = 0, so net ≤ 0 < min_net — score it without any BFS.
        // Only walls that actually cut the path pay one make + two BFS.
        if !wall_intersects_path(mv, opp_path, opp_path_len) {
            let attn = move_corridor_attention(board, mv, cat);
            // Proportional CAT hotspot boost — 200 cm sorts well above 100 cm at same node.
            return -20_000 + cat_corridor_order_boost(attn, cat_max, cold_cm);
        }
        let (opp_gain, our_loss) = wall_race_swing(board, mv, our_dist, opp_dist, bfs);
        let net = opp_gain - our_loss;
        let min_net = min_wall_net_race(our_dist, opp_dist);
        let attn = move_corridor_attention(board, mv, cat);
        if net < min_net {
            return -20_000 + cat_corridor_order_boost(attn, cat_max, cold_cm);
        }
        if race_pressure {
            return 15_000 + net * 120 + attn / 8;
        }
        return 12_000 + net * 80 + attn / 16;
    }

    let gain = our_path_gain(board, mv, our_dist, bfs);
    if our_dist >= opp_dist && gain > 0 {
        // Lateral / slow sprint while clearly losing the race is not a tactic.
        let closes_gap = gain >= 2 || our_dist.saturating_sub(1) <= opp_dist;
        if behind && !closes_gap {
            return 800 + gain * 40;
        }
        return 14_000 + gain * 100;
    }
    if gain > 0 {
        1000 + gain * 100
    } else {
        let attn = move_corridor_attention(board, mv, cat);
        TEMPO_PENALTY + cat_corridor_order_boost(attn, cat_max, cold_cm)
    }
}

/// Score band for treating root/order candidates as tied (symmetry interleave).
const ORDER_SCORE_TIE_BAND: i32 = 150;

#[inline]
fn move_col(mv: Move) -> u8 {
    match mv {
        Move::Pawn { col, .. } | Move::Wall { col, .. } => col,
    }
}

#[inline]
fn symmetry_side(col: u8) -> u8 {
    if col < 4 {
        0
    } else if col > 4 {
        2
    } else {
        1
    }
}

/// Mirror across the e-file — d↔f on a 9-wide board (cols 0..8).
pub fn mirror_move(mv: Move) -> Move {
    let mirrored = 8 - move_col(mv);
    match mv {
        Move::Pawn { row, .. } => Move::Pawn { row, col: mirrored },
        Move::Wall {
            row, orientation, ..
        } => Move::Wall {
            row,
            col: mirrored,
            orientation,
        },
    }
}

/// Within tied score bands, round-robin left / right / center so LMR does not
/// always feast on the d-file before the f-file.
fn rebalance_symmetric_order(moves: &mut [Move], scores: &mut [i32], n: usize) {
    if n <= 1 {
        return;
    }
    let mut out_moves = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let mut out_scores = [0i32; MAX_LEGAL_MOVES];
    let mut out = 0usize;
    let mut start = 0usize;
    while start < n {
        let top = scores[start];
        let mut end = start + 1;
        while end < n && scores[end] >= top - ORDER_SCORE_TIE_BAND {
            end += 1;
        }
        let bucket_len = end - start;
        if bucket_len <= 1 {
            out_moves[out] = moves[start];
            out_scores[out] = scores[start];
            out += 1;
            start = end;
            continue;
        }

        let mut left = Vec::new();
        let mut center = Vec::new();
        let mut right = Vec::new();
        for idx in start..end {
            match symmetry_side(move_col(moves[idx])) {
                0 => left.push(idx),
                1 => center.push(idx),
                _ => right.push(idx),
            }
        }

        let mut merged = Vec::with_capacity(bucket_len);
        let max_lr = left.len().max(right.len());
        for k in 0..max_lr {
            if k < left.len() {
                merged.push(left[k]);
            }
            if k < right.len() {
                merged.push(right[k]);
            }
        }
        for &idx in &center {
            merged.push(idx);
        }

        for &idx in &merged {
            out_moves[out] = moves[idx];
            out_scores[out] = scores[idx];
            out += 1;
        }
        start = end;
    }
    moves[..n].copy_from_slice(&out_moves[..n]);
    scores[..n].copy_from_slice(&out_scores[..n]);
}

#[allow(clippy::too_many_arguments)]
pub fn order_moves(
    board: &mut Board,
    moves: &mut [Move],
    n: usize,
    tt_best: Option<Move>,
    book_hint: Option<BookHint>,
    scores: &mut [i32; MAX_LEGAL_MOVES],
    our_dist: u8,
    opp_dist: u8,
    opp_path: &[u8; 81],
    opp_path_len: usize,
    bfs: &mut BfsScratch,
    cat: &CorridorAttention,
    extras: &OrderExtras,
    history_bonus: impl Fn(Move) -> i32,
) {
    let cold_cm = CAT_COLD_CM;
    let refs = cat_heat_refs(moves, n, board, cat);
    for i in 0..n {
        let mv = moves[i];
        let cat_ref = cat_heat_ref_max(mv, refs);
        let base = move_order_score(
            board,
            mv,
            tt_best,
            book_hint,
            our_dist,
            opp_dist,
            opp_path,
            opp_path_len,
            bfs,
            cat,
            cat_ref,
            cold_cm,
        );
        let mut score = apply_order_extras(base, mv, tt_best, extras);
        score += history_bonus(mv);
        scores[i] = score;
    }
    let mut order: [usize; MAX_LEGAL_MOVES] = core::array::from_fn(|i| i);
    order[..n].sort_unstable_by(|&a, &b| scores[b].cmp(&scores[a]));
    let mut tmp = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let mut tmp_scores = [0i32; MAX_LEGAL_MOVES];
    tmp[..n].copy_from_slice(&moves[..n]);
    for i in 0..n {
        moves[i] = tmp[order[i]];
        tmp_scores[i] = scores[order[i]];
    }
    scores[..n].copy_from_slice(&tmp_scores[..n]);
    rebalance_symmetric_order(moves, scores, n);
}
```

---

## 5. `src/search/cat_index_lmr.rs` (FULL FILE minus tests, ~250 of 349 lines)

The generic (engine-agnostic, used by the older `search::alphabeta` path)
CAT×index LMR model, plus the **per-Lazy-SMP-worker aggression schedule**
(`lmr_aggression_percent`) — this is what earlier conversation turns were
calling "thread 0 = 177%, thread 1 = 200%, thread 2 = 277%, thread 3+ = 350%".

```rust
//! CAT × move-index LMR — normalized against max legal-move impact at this node.

/// Attention at or below this fraction of `Hmax` → dead tail (maximum CAT pressure).
pub const CAT_ATTENTION_TAIL_CUTOFF: f64 = 0.10;

/// Lazy-SMP worker aggression schedule (UI percent). Thread 3+ caps at 350 (not 400).
pub fn lmr_aggression_percent(thread_id: usize) -> i32 {
    match thread_id {
        0 => 177,
        1 => 200,
        2 => 277,
        _ => 350,
    }
}

/// When unset, every worker uses thread-0 aggression until path correction is validated.
pub fn lmr_thread_aggression_enabled() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var("TITANIUM_LMR_THREAD_AGGRESSION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

/// Signed tuning percent for [`lmr_tuning_to_aggression_g`].
pub fn lmr_aggression_tuning_percent(thread_id: usize) -> i32 {
    let id = if lmr_thread_aggression_enabled() {
        thread_id
    } else {
        0
    };
    -lmr_aggression_percent(id)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmrPathDiagnostics {
    pub self_gain: i32,
    pub opponent_delay: i32,
    pub race_gain: i32,
    pub attention_ratio: f64,
    pub base_reduction: u32,
    pub path_adjustment: i32,
    pub final_reduction: u32,
    pub thread_aggression_percent: i32,
}

/// Cached d0/d1 pawn scalars before/after `make_move` + child `refresh_dist`.
#[inline]
pub fn compute_race_gain(pre_our: u8, pre_opp: u8, post_our: u8, post_opp: u8) -> (i32, i32, i32) {
    let self_gain = i32::from(pre_our) - i32::from(post_our);
    let opponent_delay = i32::from(post_opp) - i32::from(pre_opp);
    let race_gain = self_gain + opponent_delay;
    (self_gain, opponent_delay, race_gain)
}

/// Path-aware final reduction after base CAT/index LMR. Does not alter CAT cm or attention.
pub fn apply_lmr_path_correction(
    base_reduction: u32,
    full_child_depth: u32,
    race_gain: i32,
    attention_ratio: f64,
    skip_reduction: bool,
) -> LmrPathDiagnostics {
    let max_safe = full_child_depth.saturating_sub(1);
    if skip_reduction || full_child_depth <= 1 {
        return LmrPathDiagnostics {
            self_gain: 0,
            opponent_delay: 0,
            race_gain,
            attention_ratio,
            base_reduction,
            path_adjustment: 0,
            final_reduction: base_reduction.min(max_safe),
            thread_aggression_percent: 0,
        };
    }

    let mut path_adjustment = 0i32;
    let mut final_reduction = base_reduction;

    if race_gain > 0 && final_reduction > 0 {
        path_adjustment = -1;
        final_reduction = final_reduction.saturating_sub(1);
    } else if race_gain == 0 && attention_ratio <= CAT_ATTENTION_TAIL_CUTOFF {
        let delta = max_safe as i32 - final_reduction as i32;
        path_adjustment = delta;
        final_reduction = max_safe;
    }

    final_reduction = final_reduction.min(max_safe);

    LmrPathDiagnostics {
        self_gain: 0,
        opponent_delay: 0,
        race_gain,
        attention_ratio,
        base_reduction,
        path_adjustment,
        final_reduction,
        thread_aggression_percent: 0,
    }
}

/// Full LMR plan: base CAT/index reduction, cached path correction, clamp.
pub fn cat_index_lmr_with_path(
    full_child_depth: u32,
    move_rank: usize,
    move_count: usize,
    move_impact: i32,
    max_move_impact: u32,
    thread_id: usize,
    skip_reduction: bool,
    first_reducible_rank: usize,
    pre_our: u8,
    pre_opp: u8,
    post_our: u8,
    post_opp: u8,
) -> LmrPathDiagnostics {
    let thread_aggression_percent = lmr_aggression_percent(if lmr_thread_aggression_enabled() {
        thread_id
    } else {
        0
    });
    let attention_ratio = cat_attention(move_impact, max_move_impact);
    let base_reduction = cat_index_lmr_reduction(
        full_child_depth,
        move_rank,
        move_count,
        move_impact,
        max_move_impact,
        lmr_tuning_to_aggression_g(lmr_aggression_tuning_percent(thread_id)),
        skip_reduction,
        first_reducible_rank,
    );
    let (self_gain, opponent_delay, race_gain) =
        compute_race_gain(pre_our, pre_opp, post_our, post_opp);
    let mut diag = apply_lmr_path_correction(
        base_reduction,
        full_child_depth,
        race_gain,
        attention_ratio,
        skip_reduction,
    );
    diag.self_gain = self_gain;
    diag.opponent_delay = opponent_delay;
    diag.race_gain = race_gain;
    diag.thread_aggression_percent = thread_aggression_percent;
    diag
}

/// Map UI / viz tuning percent (−500..150) to aggression multiplier `g`.
pub fn lmr_tuning_to_aggression_g(tuning_percent: i32) -> f64 {
    let t = tuning_percent.clamp(-500, 150) as f64;
    if t >= 150.0 {
        return 0.0;
    }
    if t >= 100.0 {
        return (150.0 - t) / 50.0;
    }
    if t <= -500.0 {
        return 1.0;
    }
    if t < 0.0 {
        // More negative → slightly hotter cuts at the same attention (still clamped in P).
        return 1.0 + (-t / 500.0) * 0.35;
    }
    1.0
}

#[inline]
pub fn cat_attention(move_impact: i32, max_move_impact: u32) -> f64 {
    if max_move_impact == 0 {
        return 0.0;
    }
    (move_impact.max(0) as f64 / max_move_impact as f64).clamp(0.0, 1.0)
}

/// CAT reduction pressure in `[0, 1]` from normalized attention.
#[inline]
pub fn cat_pressure(attention: f64) -> f64 {
    if attention <= CAT_ATTENTION_TAIL_CUTOFF {
        1.0
    } else {
        let num = 1.0 - attention;
        let den = 1.0 - CAT_ATTENTION_TAIL_CUTOFF;
        (num / den).powi(2)
    }
}

/// Logarithmic move-rank pressure: first move → 0, last → 1.
#[inline]
pub fn move_index_pressure(move_rank: usize, move_count: usize) -> f64 {
    if move_count <= 1 || move_rank <= 1 {
        return 0.0;
    }
    let k = move_rank.min(move_count) as f64;
    let n = move_count as f64;
    (k.ln() / n.ln()).clamp(0.0, 1.0)
}

/// Combined CAT × move-index LMR reduction in plies.
pub fn cat_index_lmr_reduction(
    full_child_depth: u32,
    move_rank: usize,
    move_count: usize,
    move_impact: i32,
    max_move_impact: u32,
    aggression_g: f64,
    skip_reduction: bool,
    first_reducible_rank: usize,
) -> u32 {
    if skip_reduction
        || full_child_depth <= 1
        || move_count <= 1
        || move_rank < first_reducible_rank
        || aggression_g <= 0.0
    {
        return 0;
    }

    let max_reduction = full_child_depth.saturating_sub(1);
    let index_pressure = move_index_pressure(move_rank, move_count);

    if max_move_impact > 0 {
        let attention = cat_attention(move_impact, max_move_impact);
        if attention <= CAT_ATTENTION_TAIL_CUTOFF {
            // Dead tail → leaf eval only; do not spend even 1 child ply.
            return full_child_depth;
        }
    }

    let total_pressure = if max_move_impact == 0 {
        aggression_g * index_pressure
    } else {
        let attention = cat_attention(move_impact, max_move_impact);
        if attention <= CAT_ATTENTION_TAIL_CUTOFF {
            aggression_g
        } else {
            aggression_g * cat_pressure(attention) * index_pressure
        }
    }
    .clamp(0.0, 1.0);

    ((max_reduction as f64 * total_pressure).round() as u32).min(max_reduction)
}
```

---

## 6. `src/search/v16_lmr.rs` (FULL FILE minus tests, ~125 of 179 lines)

The engine's **currently active** LMR model (Titanium v16), used by
`titanium/search.rs`. This is the "10% CAT attention is the only hard wall
cut" rule mentioned in project memory — `CAT_ATTENTION_TAIL_CUTOFF = 0.10`
(from `cat_index_lmr.rs` above, re-used here) is the single hard cliff: at or
below 10% of the hottest legal move's impact, a wall or backward pawn move
gets crushed straight to depth-1 (`child_depth_used = 1`); above that
threshold reduction is smooth/graded (`cat_extra_reduction`), never a hard
cut.

```rust
//! Titanium v16 LMR — ACE v13 graduated baseline + CAT-graded extra reduction.
//!
//! Base schedule is v15 (move-index steps 1/2/3). CAT attention then adds a
//! graded 0..+2 plies for colder moves (full detail, not just the binary tail),
//! and attention ≤ 10% of node max remains the only hard depth-1 cut.

use crate::search::cat_index_lmr::{cat_pressure, CAT_ATTENTION_TAIL_CUTOFF};

/// Max extra plies CAT coldness can add on top of the ACE v15 base.
pub const CAT_EXTRA_REDUCTION_MAX: f64 = 2.0;

/// Graded CAT extra reduction: 0 for hot moves, up to +2 near the tail.
#[inline]
pub fn cat_extra_reduction(attention_ratio: f64) -> i32 {
    (cat_pressure(attention_ratio) * CAT_EXTRA_REDUCTION_MAX).round() as i32
}

pub const ACE_LMR_AFTER_MOVE: usize = 4;
pub const ACE_LMR_MIN_DEPTH: i32 = 3;

/// Late-move reduction plies — same formula as ACE v13 / JS graduated LMR.
#[inline]
pub fn ace_graduated_lmr_reduction(move_index: usize, depth: i32) -> i32 {
    let mut red = 1;
    if move_index >= 12 {
        red += 1;
    }
    if depth >= 6 && move_index >= 24 {
        red += 1;
    }
    red
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V16HardOverride {
    None,
    DeadTail,
    BackwardMove,
}

impl V16HardOverride {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::DeadTail => "deadTail",
            Self::BackwardMove => "backwardMove",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V16LmrPlan {
    pub ace_base_reduction: i32,
    pub hard_override: V16HardOverride,
    pub final_reduction: i32,
    pub child_depth_used: i32,
}

#[inline]
fn max_safe_reduction(child_depth_full: i32) -> i32 {
    (child_depth_full - 1).max(0)
}

/// Wall move: ACE baseline, unless CAT attention ≤ 10% of max legal impact.
/// The rear-zeroed CAT heatmap IS the backward test: a wall touching squares a
/// pawn can reach without moving backwards carries heat, so only true rear /
/// no-op walls fall in the tail. Path-delay is NOT used to hard-cut — a wall
/// spanning into a forward corridor can leave the current BFS distance
/// unchanged (equal-length detour) and still be tactically critical.
pub fn plan_v16_wall_lmr(
    move_index: usize,
    depth: i32,
    child_depth_full: i32,
    attention_ratio: f64,
    opponent_delay: i32,
    self_delay: i32,
) -> V16LmrPlan {
    let ace_base = ace_graduated_lmr_reduction(move_index, depth);
    let max_safe = max_safe_reduction(child_depth_full);
    if attention_ratio <= CAT_ATTENTION_TAIL_CUTOFF {
        let label = if opponent_delay <= 0 && self_delay <= 0 {
            V16HardOverride::BackwardMove
        } else {
            V16HardOverride::DeadTail
        };
        return V16LmrPlan {
            ace_base_reduction: ace_base,
            hard_override: label,
            final_reduction: max_safe,
            child_depth_used: 1,
        };
    }
    let final_reduction = (ace_base + cat_extra_reduction(attention_ratio)).min(max_safe);
    V16LmrPlan {
        ace_base_reduction: ace_base,
        hard_override: V16HardOverride::None,
        final_reduction,
        child_depth_used: (child_depth_full - final_reduction).max(0),
    }
}

/// Pawn moved farther from goal (`self_gain < 0`) → depth-1 leaf search.
/// Sideways moves (`self_gain == 0`) are NOT backwards — pockets and detours
/// around tunnels stay fully searched.
pub fn plan_v16_pawn_lmr(
    move_index: usize,
    depth: i32,
    child_depth_full: i32,
    self_gain: i32,
) -> Option<V16LmrPlan> {
    if self_gain >= 0 {
        return None;
    }
    let ace_base = ace_graduated_lmr_reduction(move_index, depth);
    let max_safe = max_safe_reduction(child_depth_full);
    Some(V16LmrPlan {
        ace_base_reduction: ace_base,
        hard_override: V16HardOverride::BackwardMove,
        final_reduction: max_safe,
        child_depth_used: 1,
    })
}
```

---

## 7. `src/titanium/search.rs` — wiring it all together

This file is 6197 lines; below are the exact excerpts that matter, in the
order they execute for a real search node. Line numbers refer to the current
working tree (with an in-progress, UNCOMMITTED experiment applied — see §8).

### 7a. Imports (lines 1-64, module doc + relevant `use`s)

```rust
//! ACE v11 search — 1:1 port of the JS `Search` object (quoridor_5.html,
//! pathfix gen11_ghi).
//!
//! Iterative-deepening αβ with aspiration windows, typed TT, killers/history/
//! countermoves, null move, graduated LMR / EME, frontier LMP, reverse futility,
//! lazy wall legality, repetition detection, wall-stamp dist caching,
//! easy-move early stop, HalfPW net eval. Mirrors the JS node-for-node.

use crate::titanium::dist::{ /* ... */ };
use crate::titanium::move_id_to_board;
use crate::util::clock::{Duration, Instant};

use crate::cat::prune::{
    cat_v16_lmr_fringe_pct_for_worker, gap_play_zone_mask, get_shortest_path,
    move_corridor_attention_with_denial, move_corridor_attention_with_path, move_impact_heat,
    wall_in_dead_zone, wall_should_search,
};
use crate::cat::CorridorAttention;
use crate::core::board::{Board, Move as BoardMove, Player, Undo, WallOrientation};
use crate::movegen::{generate_legal_moves_slice_cached, GeometricWallCache, GeometricWallCacheStats, MAX_LEGAL_MOVES};
use crate::path::flood::expand_frontier;
use crate::path::masks::DirMasks;
use crate::path::BfsScratch;
use crate::search::v16_lmr::{
    plan_v16_pawn_lmr, plan_v16_wall_lmr, V16HardOverride, ACE_LMR_AFTER_MOVE, ACE_LMR_MIN_DEPTH,
};
use crate::titanium::certify::{certify, CertifyOpts};
use crate::titanium::game::{GameState, ZOBRIST};
use crate::titanium::net::{net, net_frozen, MAX_NET_H, Net, NET_BKT, NET_MIRC, NET_MIRS};
use crate::titanium::packed_state::FEATURE_SCHEMA;
use crate::titanium::race::{
    race_outcome_with_dist, solve_race_config, RaceBound, RaceOutcomeStats, RaceScratch, RACE_MATE,
    RACE_STATES,
};
use crate::titanium::reduction_sidecar::ReductionSidecar;
use crate::util::grid::FLOOD_PLAYABLE;

pub const MATE: i32 = 100_000;
pub const MAX_PLY: usize = 64;
const INF: i32 = 2 * MATE;
const CERT_WIN_SCORE: i32 = 15_000;
const CERT_BAND: i32 = 4_000;

/// Default CAT-index LMR tuning percent:
/// -500 = strongest CAT-shaped cuts, 100 = current/default, 150 = full depth.
pub const CAT_LMR_DEFAULT_TUNING_PERCENT: i32 = -177;

fn cat_lmr_tuning_percent() -> i32 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Ok(raw) = std::env::var("TITANIUM_CAT_LMR_TUNING_PERCENT") {
            if let Ok(value) = raw.parse::<i32>() {
                return value.clamp(-500, 150);
            }
        }
    }
    CAT_LMR_DEFAULT_TUNING_PERCENT
}

/// Late-move reduction plies — re-exported for LMR vision (`search::lmr_viz`).
pub use crate::search::v16_lmr::ace_graduated_lmr_reduction;

/// EME extends only the first ordered wall moves after the TT/best move.
/// Index 0 (TT move) already gets full depth; extending more siblings
/// compounds multiplicatively down the tree and explodes the node count.
const ACE_EME_TOP_MOVES: usize = 2;

/// Early Move Extension — +1 ply for the top ordered walls; +2 only for
/// the very first non-TT wall when there is real depth left to spend.
fn ace_graduated_eme_extension(move_index: usize, depth: i32) -> i32 {
    if move_index == 1 && depth >= 8 {
        2
    } else {
        1
    }
}
```

### 7b. Relevant `TitaniumSearch` struct fields (search.rs ~1416-1452)

```rust
    /// Mirrored Titanium board (movegen and/or CAT).
    bridge: Option<Box<TiBridge>>,
    /// Use Titanium `generate_legal_moves_slice` instead of ACE `wall_legal`.
    ti_movegen: bool,
    /// CAT-filter walls at inner nodes (requires `bridge`).
    cat_walls: bool,
    /// Titanium v16: CAT-scaled LMR with ceiling normalization (500/800/1000 cm).
    cat_lmr_v16: bool,
    cat_lmr_ceiling: u16,
    cat_lmr_fringe_pct: u16,
    /// SOUND dead-zone wall prune at inner nodes (requires `bridge`): drop only
    /// walls in an unreachable void / sealed interior — provably irrelevant (they
    /// change no path and only burn inventory, never the best move). NPS-only;
    /// cannot cost Elo. Distinct from `cat_walls` (heat filter, which can).
    dead_zone_prune: bool,
    /// Grafted-engine flag: in the hands-empty endgame, use Titanium's cheap
    /// path-aware tempo classifier instead of the full recursive `certify`.
    cheap_cert: bool,
    cert_eval_leaves_only: bool,
    wall_ignore_cert_override: Option<bool>,
    wall_ignore_cert_resolved: Option<bool>,
    /// Early Move Extensions on the first ordered wall moves (mirror of graduated LMR).
    eme: bool,
```

### 7c. `gen_moves_inner` — where wall generation branches into the three
filtering strategies (dead-zone-only / CAT-heat-filtered / lazy-geometry).
**This is the exact switchboard for "does the engine even generate a
useless wall as a candidate move".** (search.rs lines 3900-3959)

```rust
    fn gen_moves(&mut self, ply: usize, depth: i32, tt_move: i16, out: &mut [i16; 160]) -> usize {
        crate::bench_instr::record(
            |b| &mut b.gen_moves,
            || self.gen_moves_inner(ply, depth, tt_move, out),
        )
    }

    fn gen_moves_inner(
        &mut self,
        ply: usize,
        depth: i32,
        tt_move: i16,
        out: &mut [i16; 160],
    ) -> usize {
        let check_legal = ply == 0;
        // MoveGen+ : Titanium legal movegen at EVERY node (perft-parity search).
        // Fully legal walls — no lazy seal checks needed downstream, and inner
        // nodes can never search (or suggest via TT) a Titanium-illegal move.
        // The CAT hybrid keeps its own filtered path at inner nodes.
        if self.ti_movegen && (check_legal || (!self.cat_walls && !self.dead_zone_prune)) {
            return self
                .bridge
                .as_mut()
                .expect("ti movegen needs bridge")
                .gen_legal_ace(out);
        }
        let mut n = self.g.gen_pawn_moves(out, 0);
        if self.g.wl[self.g.turn] <= 0 {
            return n;
        }
        if self.cat_walls && !check_legal {
            return self.gen_walls_cat_filtered(depth, tt_move, out, n);
        }
        if self.dead_zone_prune && !check_legal {
            return self.gen_walls_deadzone_filtered(out, n);
        }
        for slot in 0..64 {
            if check_legal {
                if self.g.wall_legal(0, slot) {
                    out[n] = 100 + slot as i16;
                    n += 1;
                }
                if self.g.wall_legal(1, slot) {
                    out[n] = 200 + slot as i16;
                    n += 1;
                }
            } else {
                // lazy: geometry only; path-seal checked when the move is searched
                if self.g.wall_fits(0, slot) {
                    out[n] = 100 + slot as i16;
                    n += 1;
                }
                if self.g.wall_fits(1, slot) {
                    out[n] = 200 + slot as i16;
                    n += 1;
                }
            }
        }
        n
    }
```

### 7d. `gen_walls_deadzone_filtered` — the "both players racing straight
corridors, ignore middle-of-board walls nobody touches" filter in its purest
form: a wall is dropped iff `wall_in_dead_zone` (from §4) says it touches
**zero** reachable squares (search.rs lines 4075-4092)

```rust
    fn gen_walls_deadzone_filtered(&mut self, out: &mut [i16; 160], mut n: usize) -> usize {
        let bridge = self.bridge.as_mut().expect("dead-zone bridge");
        let reachable = bridge.bfs.both_reachable_mask(&bridge.board);
        for slot in 0..64 {
            for (wall_type, base) in [(0usize, 100i16), (1usize, 200i16)] {
                if !self.g.wall_fits(wall_type, slot) {
                    continue;
                }
                let m = base + slot as i16;
                if wall_in_dead_zone(move_id_to_board(m), reachable) {
                    continue;
                }
                out[n] = m;
                n += 1;
            }
        }
        n
    }
```

### 7e. `gen_walls_cat_filtered` — the heavier heat-based filter (used when
`self.cat_walls == true`), computes per-wall CAT heat and calls into
`wall_should_search` from §4 (search.rs lines 3961-4074 — the loop tail
after line 4024 calls `wall_should_search`; abbreviated middle shown):

```rust
    /// Hybrid wall generation: lazy geometry + CAT relevance filter.
    ///
    /// CAT (multi-route corridor heat) only above the leaf layer — depth-1 nodes
    /// dominate the tree and only need witness-path tactics, not breadth
    /// (mirrors `search::alphabeta`). The TT move always survives the filter.
    fn gen_walls_cat_filtered(
        &mut self,
        depth: i32,
        tt_move: i16,
        out: &mut [i16; 160],
        mut n: usize,
    ) -> usize {
        let me = self.g.turn;
        let our_dist = if me == 0 {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        } else {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        };
        let opp_dist = if me == 0 {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        } else {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        };
        let white_dist = if me == 0 { our_dist } else { opp_dist };
        let black_dist = if me == 0 { opp_dist } else { our_dist };
        let opp_player = if me == 0 { Player::Two } else { Player::One };

        let bridge = self.bridge.as_mut().expect("cat bridge");
        let cat = if depth >= 2 {
            bridge.bfs.build_corridor_attention(&bridge.board)
        } else {
            CorridorAttention::default()
        };
        let mut opp_path = [0u8; 81];
        let opp_path_len =
            get_shortest_path(&bridge.board, opp_player, &mut bridge.bfs, &mut opp_path);
        let reachable = bridge.bfs.both_reachable_mask(&bridge.board);
        let gap_zone = gap_play_zone_mask(reachable);
        let mut wall_candidates = [BoardMove::Pawn { row: 0, col: 0 }; 128];
        let mut wall_direct_heats = [0i32; 128];
        let mut wall_candidate_n = 0usize;

        for slot in 0..64 {
            for (wall_type, base) in [(0usize, 100i16), (1usize, 200i16)] {
                if !self.g.wall_fits(wall_type, slot) {
                    continue;
                }
                let m = base + slot as i16;
                let mv = move_id_to_board(m);
                wall_candidates[wall_candidate_n] = mv;
                wall_direct_heats[wall_candidate_n] = move_corridor_attention_with_path(
                    &mut bridge.board,
                    mv,
                    &cat,
                    white_dist,
                    black_dist,
                    &mut bridge.bfs,
                );
                wall_candidate_n += 1;
            }
        }
        // ... (loop over wall_candidates calling wall_should_search per candidate,
        //      pushing survivors into `out`; see engine/src/titanium/search.rs
        //      lines 4023-4074 for the exact continuation)
        n
    }
```

### 7f. `order_moves_prior` — actual per-node move ordering used by the live
Titanium search (distinct from, and simpler than, `cat::prune::order_moves`
in §4 which is used by the older `search::alphabeta` path). TT > pawn
progress-by-distance > killers > countermove > history > CAT-heat fallback
(tail-cut at 10% of max). (search.rs lines 4094-4181)

```rust
    fn order_moves(&self, ply: usize, moves: &mut [i16], tt_move: i16, cm_move: i16) {
        self.order_moves_prior(ply, moves, tt_move, cm_move, None);
    }

    /// Ordering: TT > pawn progress > killers > countermove > history. CAT heat
    /// is a fallback prior ONLY for walls the history table is silent on
    /// (h == 0), and never for tail moves (≤ 10% of node max attention) — so
    /// insignificant walls get no ordering credit they haven't earned.
    fn order_moves_prior(
        &self,
        ply: usize,
        moves: &mut [i16],
        tt_move: i16,
        cm_move: i16,
        cat_prior: Option<(&[i32; 264], u32)>,
    ) {
        let dist_me = if self.g.turn == 0 {
            &self.d0[self.dist0_idx]
        } else {
            &self.d1[self.dist1_idx]
        };
        let k = &self.killers[ply];
        let n = moves.len();
        let mut sc = [0i32; 160];
        for i in 0..n {
            let m = moves[i];
            sc[i] = if m == tt_move {
                2_000_000_000
            } else if m < 100 {
                1_000_000 - dist_me[m as usize] as i32 * 1000
            } else if m == k[0] {
                900_000
            } else if m == cm_move {
                870_000
            } else if m == k[1] {
                850_000
            } else {
                let h = self.history_tbl[m as usize];
                if h != 0 {
                    h
                } else if let Some((heat, max_h)) = cat_prior {
                    let cm = heat[m as usize].max(0) as u32;
                    // Tail cutoff: ≤10% of the hottest legal move stays at 0.
                    if cm * 10 > max_h {
                        cm as i32
                    } else {
                        0
                    }
                } else {
                    0
                }
            };
        }
        if ply == 0 {
            if let Some(order) = &self.opening_book_order {
                let attn = self.opening_book_attention.as_deref();
                for (rank, &bmv) in order.iter().enumerate() {
                    if let Some(pos) = moves.iter().position(|&m| m == bmv) {
                        let boost = attn.and_then(|a| a.get(rank).copied()).unwrap_or(0);
                        // Above TT move; win-rate + Ishtar tier sets relative priority.
                        let book_score = 2_050_000_000i32 + boost + (1000 - rank as i32);
                        sc[pos] = sc[pos].max(book_score);
                    }
                }
            }
            use crate::titanium::move_id_to_algebraic;
            use crate::titanium::opening_book::opening_move_would_be_denied;
            for i in 0..n {
                let alg = move_id_to_algebraic(moves[i]);
                if opening_move_would_be_denied(&self.g, &alg) {
                    sc[i] = i32::MIN / 4;
                }
            }
        }
        // stable insertion sort, descending — must match JS tie order exactly
        for a in 1..n {
            let mv = moves[a];
            let ms = sc[a];
            let mut b = a as isize - 1;
            while b >= 0 && sc[b as usize] < ms {
                moves[(b + 1) as usize] = moves[b as usize];
                sc[(b + 1) as usize] = sc[b as usize];
                b -= 1;
            }
            moves[(b + 1) as usize] = mv;
            sc[(b + 1) as usize] = ms;
        }
    }
```

### 7g. Reverse futility pruning + null-move pruning (search.rs lines 4359-4403)

This is the classical "beta cutoff shortcut" futility pruning (not CAT-related
— purely score-based): at shallow depth, if static eval already crushes beta
by more than `90 * depth`, return immediately without searching children.

```rust
        // reverse futility: hopeless to fall below beta at shallow depth
        if depth <= 4 && beta > -2000 && beta < 2000 {
            let sev = self.evaluate(depth);
            if sev - 90 * depth >= beta {
                return Ok(sev);
            }
        }

        // null move
        if allow_null && depth >= 3 && ply > 0 {
            let ev = self.evaluate(depth);
            if ev >= beta {
                let z = &ZOBRIST;
                self.g.turn ^= 1;
                self.g.hash_lo ^= z.turn_lo;
                self.g.hash_hi ^= z.turn_hi;
                if let Some(bridge) = self.bridge.as_mut() {
                    bridge.board.side_to_move = bridge.board.side_to_move.opposite();
                }
                let res = self.ab(depth - 3, -beta, -beta + 1, ply + 1, false, 0);
                let z = &ZOBRIST;
                self.g.turn ^= 1;
                self.g.hash_lo ^= z.turn_lo;
                self.g.hash_hi ^= z.turn_hi;
                if let Some(bridge) = self.bridge.as_mut() {
                    bridge.board.side_to_move = bridge.board.side_to_move.opposite();
                }
                self.dist0_idx = nd0;
                self.dist1_idx = nd1;
                self.cached_stamp = nst;
                self.dir_masks_key_lo = ndm_lo;
                self.dir_masks_key_hi = ndm_hi;
                self.dir_masks_cache = ndm_cache;
                if self.sub_min[ply + 1] < self.sub_min[ply] {
                    self.sub_min[ply] = self.sub_min[ply + 1];
                    self.sub_anc_lo[ply] = self.sub_anc_lo[ply + 1];
                    self.sub_anc_hi[ply] = self.sub_anc_hi[ply + 1];
                }
                let ns = -res?;
                if ns >= beta && ns < MATE - 200 {
                    return Ok(beta);
                }
            }
        }
```

### 7h. Per-node CAT-heat build + frontier LMP + the actual v16-LMR call site
(search.rs lines 4405-4530, the part of `ab()` right after move generation)

`heat_by_id`/`max_move_impact` here is the cheap `build_impact_heatmap` (§3)
computed ONCE per node, then reused both as an ordering prior (§7f) and as
the attention_ratio fed into `plan_v16_wall_lmr`/`plan_v16_pawn_lmr` (§6).
Frontier LMP (`depth <= 2 && i >= 10 && history <= 0`) is a plain late-move
pruning cut unrelated to CAT — after move 10 at shallow depth with a cold
history score, skip entirely (not even a reduced search).

```rust
        let mut moves = [0i16; 160];
        let mut n = self.gen_moves(ply, depth, tt_move, &mut moves);
        if n == 0 {
            return Ok(self.evaluate(depth));
        }
        let cm_move = if prev_move > 0 {
            self.cm[prev_move as usize]
        } else {
            0
        };

        // CAT impact heat, computed BEFORE ordering so it can serve as the
        // ordering prior for walls the history table knows nothing about.
        // Cheap BFF impact heatmap (bitboard path-set + flood): a move's
        // impact is a heatmap lookup (wall = hottest touched square).
        let mut heat_by_id = [0i32; 264];
        let mut max_move_impact = 0u32;
        let cat_lmr_active = self.cat_lmr_v16 && depth >= 2 && n > 0;
        if cat_lmr_active {
            if let Some(bridge) = self.bridge.as_mut() {
                let cat = crate::cat::build::build_impact_heatmap(&bridge.board);
                for i in 0..n {
                    let mv = move_id_to_board(moves[i]);
                    let h = move_impact_heat(mv, &cat);
                    heat_by_id[moves[i] as usize] = h;
                    max_move_impact = max_move_impact.max(h.max(0) as u32);
                }
            }
        }
        let cat_order_prior = if cat_lmr_active && max_move_impact > 0 {
            Some((&heat_by_id, max_move_impact))
        } else {
            None
        };
        self.order_moves_prior(ply, &mut moves[..n], tt_move, cm_move, cat_order_prior);
        // ... (ply==0 root-move override for Lazy-SMP workers, see §8) ...

        let mut cat_heats = [0i32; 160];
        for i in 0..n {
            cat_heats[i] = heat_by_id[moves[i] as usize];
        }

        let mut best = i32::MIN;
        let mut best_move: i16 = 0;
        let mut flag = 2;

        for i in 0..n {
            let m = moves[i];
            // frontier LMP
            if depth <= 2
                && ply > 0
                && i >= 10
                && m >= 100
                && m != tt_move
                && self.history_tbl[m as usize] <= 0
                && best > -MATE + 200
            {
                continue;
            }
            // Seal check only needed for ACE's lazy pseudo-legal walls; with
            // MoveGen+ (Titanium legal gen at every node) all walls are legal.
            let lazy_walls = !(self.ti_movegen && !self.cat_walls && !self.dead_zone_prune);
            if m >= 100 && ply > 0 && lazy_walls {
                let wt = if m < 200 { 0 } else { 1 };
                let slot = (m % 100) as usize;
                if self.g.wall_needs_path_check(wt, slot) {
                    self.g.set_wall_bits(wt, slot, true);
                    let paths_ok = self.g.has_path(0) && self.g.has_path(1);
                    self.g.set_wall_bits(wt, slot, false);
                    if !paths_ok {
                        continue; // sealing wall: pseudo-legal only
                    }
                }
            }
            // ... (make_move, then branch into EME / v16-wall-LMR / v16-pawn-LMR
            //      / plain full-depth search — this is where plan_v16_wall_lmr
            //      and plan_v16_pawn_lmr from §6 actually get called; see the
            //      full excerpt in §7i below) ...
        }
```

### 7i. The actual LMR dispatch inside the move loop — three branches: EME
extension, wall v16-LMR (calls `plan_v16_wall_lmr`), pawn v16-LMR (calls
`plan_v16_pawn_lmr`) (search.rs lines 4520-4703, condensed to the
LMR-relevant branches; PVS re-search plumbing kept verbatim since it's the
"verify with full window if the reduced search beats alpha" pattern):

```rust
            let new_depth = depth - 1;
            let result = if self.eme
                && i > 0
                && i <= ACE_EME_TOP_MOVES
                && depth >= ACE_LMR_MIN_DEPTH
                && m >= 100
                && m != tt_move
            {
                // EME — extend only the top ordered walls (see ACE_EME_TOP_MOVES)
                let ext = ace_graduated_eme_extension(i, depth);
                let ed = new_depth + ext;
                self.ab(ed, -beta, -alpha, ply + 1, true, m).map(|s| -s)
            } else if i >= ACE_LMR_AFTER_MOVE
                && depth >= ACE_LMR_MIN_DEPTH
                && m >= 100
                && m != tt_move
            {
                let attention_ratio = if cat_lmr_active && max_move_impact > 0 {
                    cat_heats[i].max(0) as f64 / max_move_impact as f64
                } else {
                    1.0
                };
                let v16_plan = if cat_lmr_active {
                    // CAT attention alone determines the v16 search depth. Path
                    // delay was only used to choose a diagnostic tail label, and
                    // refreshing child wall distances here made search pay an
                    // extra flood before the child node needed distances.
                    plan_v16_wall_lmr(i, depth, new_depth, attention_ratio, 0, 0)
                } else {
                    let ace_base = ace_graduated_lmr_reduction(i, depth);
                    let final_reduction = ace_base.min((new_depth - 1).max(0));
                    crate::search::v16_lmr::V16LmrPlan {
                        ace_base_reduction: ace_base,
                        hard_override: V16HardOverride::None,
                        final_reduction,
                        child_depth_used: (new_depth - final_reduction).max(0),
                    }
                };
                let red = v16_plan.final_reduction;
                // ... (reduction_sidecar / reduction_probe diagnostics, not
                //      functional — collects shadow-mode telemetry only) ...
                let rd = v16_plan.child_depth_used; // (minus optional probe extra_reduction)
                let pipeline_result = match self.ab(rd, -alpha - 1, -alpha, ply + 1, true, m) {
                    Ok(s) => {
                        let mut score = -s;
                        if score > alpha {
                            // reduced search beat alpha → re-search at full window/depth
                            match self.ab(new_depth, -beta, -alpha, ply + 1, true, m) {
                                Ok(s2) => score = -s2,
                                Err(e) => { /* unwind + propagate */ return Err(e); }
                            }
                        }
                        Ok(score)
                    }
                    Err(e) => Err(e),
                };
                pipeline_result
            } else if self.cat_lmr_v16
                && m < 100
                && i > 0
                && depth >= ACE_LMR_MIN_DEPTH
                && m != tt_move
            {
                // Pawn moves do not change wall topology, so the parent distance
                // fields remain valid after the pawn coordinate changes.
                let post_d0 = self.d0[nd0][self.g.pawn[0]];
                let post_d1 = self.d1[nd1][self.g.pawn[1]];
                let (pre_our, post_our) = if mover == 0 {
                    (pre_d0, post_d0)
                } else {
                    (pre_d1, post_d1)
                };
                let self_gain = i32::from(pre_our) - i32::from(post_our);
                if let Some(v16_plan) = plan_v16_pawn_lmr(i, depth, new_depth, self_gain) {
                    let rd = v16_plan.child_depth_used;
                    match self.ab(rd, -alpha - 1, -alpha, ply + 1, true, m) {
                        Ok(s) => {
                            let mut score = -s;
                            if score > alpha {
                                match self.ab(new_depth, -beta, -alpha, ply + 1, true, m) {
                                    Ok(s2) => score = -s2,
                                    Err(e) => { return Err(e); }
                                }
                            }
                            Ok(score)
                        }
                        Err(e) => Err(e),
                    }
                } else {
                    // self_gain >= 0 (forward or sideways pawn move): full depth,
                    // standard PVS null-window-then-verify.
                    match self.ab(new_depth, -alpha - 1, -alpha, ply + 1, true, m) {
                        Ok(s) => {
                            let mut score = -s;
                            if score > alpha && score < beta {
                                match self.ab(new_depth, -beta, -alpha, ply + 1, true, m) {
                                    Ok(s2) => score = -s2,
                                    Err(e) => { return Err(e); }
                                }
                            }
                            Ok(score)
                        }
                        Err(e) => Err(e),
                    }
                }
            } else if i > 0 {
                // Move index 0, or below ACE_LMR_AFTER_MOVE/ACE_LMR_MIN_DEPTH
                // thresholds: standard full-depth PVS, no reduction at all.
                match self.ab(new_depth, -alpha - 1, -alpha, ply + 1, true, m) {
                    Ok(s) => {
                        let mut score = -s;
                        if score > alpha && score < beta {
                            match self.ab(new_depth, -beta, -alpha, ply + 1, true, m) {
                                /* ... */
                            }
                        }
                        Ok(score)
                    }
                    Err(e) => Err(e),
                }
            } else {
                // i == 0: the presumed-best move, full window, full depth.
                self.ab(new_depth, -beta, -alpha, ply + 1, true, m).map(|s| -s)
            };
```

---

## 8. Lazy-SMP root-move filtering experiment (UNCOMMITTED, in progress)

Separate from per-node LMR (§7h/7i, applies at every ply), this experiment
touches ONLY the **root** move list (ply 0) handed to each parallel search
worker. Status: written, **not yet built/tested**. Located in
`engine/src/titanium/search.rs`, uncommitted diff against HEAD. Full current
diff:

```diff
diff --git a/src/titanium/search.rs b/src/titanium/search.rs
index a93015a..6e34950 100644
--- a/src/titanium/search.rs
+++ b/src/titanium/search.rs
@@ -161,9 +161,9 @@ mod lazy_smp_tests {
     fn helper_root_profiles_are_diversified() {
         let root_moves = (0..20).collect::<Vec<i16>>();
         let (main_moves, main_idx) =
-            TitaniumSearch::lazy_smp_profile_root_moves(&root_moves, 0, 20);
+            TitaniumSearch::lazy_smp_profile_root_moves(&root_moves, 0, 20, false);
         let (helper_moves, helper_idx) =
-            TitaniumSearch::lazy_smp_profile_root_moves(&root_moves, 1, 12);
+            TitaniumSearch::lazy_smp_profile_root_moves(&root_moves, 1, 12, false);
         assert_eq!(main_moves, root_moves);
         assert_eq!(main_idx, (0..20).collect::<Vec<_>>());
         assert_eq!(helper_moves.len(), 12);
@@ -475,7 +475,15 @@ const TT_MASK: u32 = (TT_SIZE - 1) as u32;
 // helpers narrow progressively for deeper per-move lookahead, floored at 40%
 // (not 20%) since helper results only ever matter as an emergency fallback
 // when main produces nothing (see lazy_smp_helper_partial).
-const LAZY_SMP_WIDTHS: [usize; 4] = [95, 80, 60, 40];
+const LAZY_SMP_WIDTHS: [usize; 4] = [95, 80, 80, 80];
+
+// The very last worker (worker_id == threads - 1, when there are at least 3
+// threads so it's a distinct role from main and the uniform-80% helpers)
+// skips the percentage schedule entirely and searches only this many
+// root moves -- the top-N by move ordering. Its job isn't breadth, it's
+// squeezing maximum depth out of whatever the rest of the pool already
+// agrees are the most promising candidates.
+const LAZY_SMP_LAST_WORKER_TOP_N: usize = 3;
 
 #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
 pub const LAZY_SMP_MAX_THREADS: usize = 16;
@@ -485,14 +493,63 @@ pub struct WorkerPlan {
     pub worker_id: usize,
     pub root_move_count: usize,
     pub root_width_percent: usize,
+    pub top_n_override: Option<usize>,
 }
 
 impl WorkerPlan {
     pub fn allowed_root_moves(&self) -> usize {
+        if let Some(top_n) = self.top_n_override {
+            return top_n.max(1).min(self.root_move_count.max(1));
+        }
         lazy_smp_allowed_root_moves(self.root_move_count, self.root_width_percent)
     }
 }
 
+// EXPERIMENT (not the shipped schedule): per-worker root-move cutoff by CAT
+// impact-heat VALUE relative to the best root move's heat, not by move count.
+// A move survives worker `w` iff heat(move) >= pct[w]% * max(heat(any root
+// move)). This is a genuinely different criterion than LAZY_SMP_WIDTHS: in a
+// position with one dominant move it collapses hard (real tail-cut); in a
+// flat position where many moves are nearly as good it keeps most of them,
+// regardless of raw count. main=20% (only drop clearly-useless tail moves),
+// then progressively stricter per worker, floored at 40%.
+const LAZY_SMP_VALUE_THRESHOLD_PCTS: [i32; 4] = [20, 30, 40, 40];
+
+fn lazy_smp_value_threshold_pct(worker_id: usize) -> i32 {
+    LAZY_SMP_VALUE_THRESHOLD_PCTS
+        .get(worker_id)
+        .copied()
+        .unwrap_or(*LAZY_SMP_VALUE_THRESHOLD_PCTS.last().expect("non-empty"))
+}
+
+/// Keep only root moves whose CAT impact heat clears `threshold_pct`% of the
+/// best root move's heat. Falls back to keeping everything when there's no
+/// CAT signal at all (max_heat <= 0) -- absence of signal is not evidence a
+/// move is useless.
+fn lazy_smp_value_filtered_moves(
+    root_moves: &[i16],
+    heat_by_id: &[i32; 264],
+    max_heat: i32,
+    threshold_pct: i32,
+) -> Vec<i16> {
+    if max_heat <= 0 {
+        return root_moves.to_vec();
+    }
+    let kept: Vec<i16> = root_moves
+        .iter()
+        .copied()
+        .filter(|&m| {
+            let h = heat_by_id[m as usize].max(0);
+            h.saturating_mul(100) >= threshold_pct.saturating_mul(max_heat)
+        })
+        .collect();
+    if kept.is_empty() {
+        root_moves.to_vec()
+    } else {
+        kept
+    }
+}
+
 pub fn lazy_smp_allowed_root_moves(root_move_count: usize, root_width_percent: usize) -> usize {
     if root_move_count == 0 {
         return 0;
@@ -2187,13 +2244,20 @@ impl TitaniumSearch {
         root_moves: &[i16],
         worker_id: usize,
         allowed: usize,
+        force_top_k: bool,
     ) -> (Vec<i16>, Vec<usize>) {
         let len = root_moves.len();
         let allowed = allowed.min(len);
         if allowed == 0 {
             return (Vec::new(), Vec::new());
         }
-        if worker_id == 0 || len <= 1 {
+        // Main (worker 0) always gets the clean best-ordered slice. A worker
+        // pinned to `top_n_override` (e.g. the top-3 depth specialist) also
+        // needs the true top-K by move ordering, not the strided/offset
+        // diversification sample below -- diversifying a 3-move slice would
+        // silently swap out the actual best candidates for scattered ones,
+        // defeating the entire point of a "go deep on the best moves" worker.
+        if worker_id == 0 || force_top_k || len <= 1 {
             return (
                 root_moves[..allowed].to_vec(),
                 (0..allowed).collect::<Vec<_>>(),
@@ -5441,6 +5505,28 @@ impl TitaniumSearch {
         if root_moves_raw.is_empty() {
             return self.think(time_ms, max_depth, full, log, engine_label);
         }
+
+        // EXPERIMENT: CAT impact-heat per root move, computed once against
+        // the root position, used below to VALUE-filter (not count-filter)
+        // each worker's root move list.
+        let mut heat_by_id = [0i32; 264];
+        let mut max_heat = 0i32;
+        if let Some(bridge) = self.bridge.as_ref() {
+            let cat = crate::cat::build::build_impact_heatmap(&bridge.board);
+            for &mv_id in &root_moves_raw {
+                let mv = move_id_to_board(mv_id);
+                let h = move_impact_heat(mv, &cat);
+                heat_by_id[mv_id as usize] = h;
+                max_heat = max_heat.max(h);
+            }
+        }
+        let filtered_by_worker: Vec<Vec<i16>> = (0..threads)
+            .map(|worker_id| {
+                let pct = lazy_smp_value_threshold_pct(worker_id);
+                lazy_smp_value_filtered_moves(&root_moves_raw, &heat_by_id, max_heat, pct)
+            })
+            .collect();
+
         let root_position = self.g.clone();
         let shared_tt = self
             .shared_tt
@@ -5451,16 +5537,21 @@ impl TitaniumSearch {
         let plans: Vec<WorkerPlan> = (0..threads)
             .map(|worker_id| WorkerPlan {
                 worker_id,
-                root_move_count: root_moves_raw.len(),
-                root_width_percent: Self::lazy_smp_width_percent(worker_id),
+                root_move_count: filtered_by_worker[worker_id].len(),
+                root_width_percent: lazy_smp_value_threshold_pct(worker_id) as usize,
+                top_n_override: None,
             })
             .collect();
 
         #[cfg(not(target_arch = "wasm32"))]
         let mut helper_results: Vec<(usize, ThinkResult, Vec<usize>)> = Vec::new();
-        let main_allowed = plans[0].allowed_root_moves();
-        let (main_root_moves, main_visit_map) =
-            Self::lazy_smp_profile_root_moves(&root_moves_raw, 0, main_allowed);
+        let main_allowed = filtered_by_worker[0].len();
+        let (main_root_moves, main_visit_map) = Self::lazy_smp_profile_root_moves(
+            &filtered_by_worker[0],
+            0,
+            main_allowed,
+            true,
+        );
         self.install_lazy_smp_context(
             0,
             shared_tt.clone(),
@@ -5476,9 +5567,13 @@ impl TitaniumSearch {
             .skip(1)
             .map(|plan| {
                 let mut worker = self.fork_lazy_worker(&root_position);
-                let allowed = plan.allowed_root_moves();
-                let (profiled_root_moves, visit_map) =
-                    Self::lazy_smp_profile_root_moves(&root_moves_raw, plan.worker_id, allowed);
+                let allowed = filtered_by_worker[plan.worker_id].len();
+                let (profiled_root_moves, visit_map) = Self::lazy_smp_profile_root_moves(
+                    &filtered_by_worker[plan.worker_id],
+                    plan.worker_id,
+                    allowed,
+                    true,
+                );
                 worker.install_lazy_smp_context(
                     plan.worker_id,
                     shared_tt.clone(),
```

Findings so far from prior experiments on this same root-filtering idea
(count-based, not the value-based version above): narrowing the ROOT move
list only removes work at ply 0 — every ply below the root still branches at
full width, so the depth ceiling barely moves even when total node count
drops substantially. See project memory `partial-iteration-measured` /
conversation history for the two prior measured runs (top-3-scattered bug,
then top-3-true-fix, both ~depth 16-17 vs main's depth 17, no meaningful
depth gain). The value-based (%-of-max-heat) version above has NOT been
built/tested yet.

---

## 9. Wall-ignorance forced-loss certificate (separate system — NOT part of
the LMR/pruning pipeline, but directly relevant to "both players racing
straight corridors" since it's the exact scenario it detects)

`src/titanium/wall_ignore_corridor.rs` (FULL FILE, 401 lines total, tests
included) and `src/titanium/wall_ignore_cert.rs` (FULL FILE, 451 lines,
tests included) implement an experimental (env-gated,
`TITANIUM_WALL_IGNORE_LOSS_CERT`, default OFF) certificate: if one player has
a "zero-delay corridor" (every square on their shortest path is provably
unreachable-by-detour for both players even after removing any single edge —
i.e. genuinely walled off from interference) and that guarantees them a
faster forced win than the opponent's fastest possible race finish even
assuming perfect opponent play, the position is scored as a certain win/loss
without searching further. This is the most literal instance of "both
players are just racing down their own corridors and nothing else matters" —
when it triggers, wall placement moves become entirely irrelevant because
the corridor is proven immune to them.

```rust
// engine/src/titanium/wall_ignore_corridor.rs (FULL FILE)

//! Zero-delay wall-immune corridor detection for the experimental wall-ignorance
//! forced-loss certificate (Titanium v15 experimental).

use crate::titanium::game::{GameState, BORDER, DELTA, DIRBIT};
use std::collections::VecDeque;

/// Undirected board edge in ACE cell indices (canonical: lower index first).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BoardEdge {
    pub a: usize,
    pub b: usize,
}

impl BoardEdge {
    #[inline]
    pub fn new(a: usize, b: usize) -> Self {
        if a <= b {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunnerGuaranteeKind {
    ZeroDelayCorridor,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunnerGuarantee {
    pub side: usize,
    /// Maximum future own pawn moves required to reach a goal under the proven strategy.
    pub max_own_moves_to_goal: u8,
    pub path: Vec<usize>,
    pub protected_edges: Vec<BoardEdge>,
    pub kind: RunnerGuaranteeKind,
}

pub struct CorridorScratch {
    queue: VecDeque<usize>,
    visited: [bool; 81],
    parent: [Option<usize>; 81],
}

impl Default for CorridorScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl CorridorScratch {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::with_capacity(81),
            visited: [false; 81],
            parent: [None; 81],
        }
    }

    fn reset_path_search(&mut self) {
        self.queue.clear();
        self.visited = [false; 81];
        self.parent = [None; 81];
    }

    fn reset_reachability(&mut self) {
        self.queue.clear();
        self.visited = [false; 81];
    }
}

#[inline]
pub fn is_goal_cell(side: usize, cell: usize) -> bool {
    (side == 0 && cell < 9) || (side == 1 && cell >= 72)
}

#[inline]
pub fn shortest_distance(g: &GameState, side: usize) -> u8 {
    let mut dist = [255u8; 81];
    g.compute_dist(side, &mut dist);
    dist[g.pawn[side]]
}

fn topology_neighbors(g: &GameState, cell: usize, out: &mut [usize; 4]) -> usize {
    let bm = g.blocked[cell] | BORDER[cell];
    let mut n = 0usize;
    for d in 0..4 {
        if bm & DIRBIT[d] != 0 {
            continue;
        }
        out[n] = (cell as i16 + DELTA[d]) as usize;
        n += 1;
    }
    n
}

pub fn reconstruct_shortest_goal_path(
    g: &GameState,
    side: usize,
    scratch: &mut CorridorScratch,
) -> Option<Vec<usize>> {
    scratch.reset_path_search();
    let start = g.pawn[side];
    scratch.visited[start] = true;
    scratch.parent[start] = None;
    scratch.queue.push_back(start);

    let mut found_goal = None;
    while let Some(current) = scratch.queue.pop_front() {
        if is_goal_cell(side, current) {
            found_goal = Some(current);
            break;
        }
        let mut nb = [0usize; 4];
        let nn = topology_neighbors(g, current, &mut nb);
        for i in 0..nn {
            let next = nb[i];
            if scratch.visited[next] {
                continue;
            }
            scratch.visited[next] = true;
            scratch.parent[next] = Some(current);
            scratch.queue.push_back(next);
        }
    }

    let goal = found_goal?;
    let mut reversed = vec![goal];
    let mut cursor = goal;
    while cursor != start {
        cursor = scratch.parent[cursor]?;
        reversed.push(cursor);
    }
    reversed.reverse();
    debug_assert_eq!(
        reversed.len().checked_sub(1),
        Some(shortest_distance(g, side) as usize)
    );
    Some(reversed)
}

pub fn any_goal_reachable_without_edge(
    g: &GameState,
    side: usize,
    start: usize,
    removed: BoardEdge,
    scratch: &mut CorridorScratch,
) -> bool {
    scratch.reset_reachability();
    scratch.visited[start] = true;
    scratch.queue.push_back(start);

    while let Some(current) = scratch.queue.pop_front() {
        if is_goal_cell(side, current) {
            return true;
        }
        let mut nb = [0usize; 4];
        let nn = topology_neighbors(g, current, &mut nb);
        for i in 0..nn {
            let next = nb[i];
            if BoardEdge::new(current, next) == removed {
                continue;
            }
            if scratch.visited[next] {
                continue;
            }
            scratch.visited[next] = true;
            scratch.queue.push_back(next);
        }
    }
    false
}

pub fn detect_zero_delay_corridor(
    g: &GameState,
    side: usize,
    scratch: &mut CorridorScratch,
) -> Option<RunnerGuarantee> {
    let path = reconstruct_shortest_goal_path(g, side, scratch)?;
    let distance = path.len().checked_sub(1)? as u8;

    if distance == 0 {
        return Some(RunnerGuarantee {
            side,
            max_own_moves_to_goal: 0,
            path,
            protected_edges: Vec::new(),
            kind: RunnerGuaranteeKind::ZeroDelayCorridor,
        });
    }

    let start = path[0];
    let mut protected_edges = Vec::with_capacity(path.len() - 1);

    for pair in path.windows(2) {
        let edge = BoardEdge::new(pair[0], pair[1]);
        if any_goal_reachable_without_edge(g, side, start, edge, scratch) {
            return None;
        }
        protected_edges.push(edge);
    }

    debug_assert_eq!(distance, shortest_distance(g, side));
    Some(RunnerGuarantee {
        side,
        max_own_moves_to_goal: distance,
        path,
        protected_edges,
        kind: RunnerGuaranteeKind::ZeroDelayCorridor,
    })
}
```

```rust
// engine/src/titanium/wall_ignore_cert.rs (core logic, tests omitted)

//! Experimental wall-ignorance forced-loss certificate (Titanium v15 experimental).
//!
//! Feature-gated via `TITANIUM_WALL_IGNORE_LOSS_CERT` (default off).

use crate::core::board::{Board, Player};
use crate::titanium::cert_bridge::{paths_overlap, titanium_game_from_board};
use crate::titanium::game::GameState;
use crate::titanium::search::MATE;
use crate::titanium::wall_ignore_corridor::{
    detect_zero_delay_corridor, shortest_distance, CorridorScratch, RunnerGuarantee,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaceInteraction {
    NonInteracting,
    Deterministic,
    Volatile,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WallIgnoreVerdict {
    pub winner: usize,
    pub winner_terminal_ply: u16,
    pub loser_terminal_ply: u16,
    pub source: CertSource,
    pub interaction: RaceInteraction,
    pub race_minimax_used: bool,
}

#[inline]
pub fn earliest_terminal_ply(side: usize, side_to_move: usize, distance: u8) -> u16 {
    if distance == 0 {
        return 0;
    }
    let moves_first = side == side_to_move;
    2 * distance as u16 - u16::from(moves_first)
}

fn classify_race_interaction(
    g: &GameState,
    _guarantee: &RunnerGuarantee,
    _loser: usize,
) -> RaceInteraction {
    let mut d0 = [0u8; 81];
    let mut d1 = [0u8; 81];
    g.compute_dist(0, &mut d0);
    g.compute_dist(1, &mut d1);
    if paths_overlap(g, &d0, &d1) {
        let adj = crate::titanium::cert_bridge::turn_adjusted_tempo_advantage(g);
        if adj.abs() >= 2 {
            RaceInteraction::Deterministic
        } else {
            RaceInteraction::Volatile
        }
    } else {
        RaceInteraction::NonInteracting
    }
}

fn direct_wall_ignore_verdict(
    g: &GameState,
    winner: usize,
    guarantee: &RunnerGuarantee,
) -> Option<WallIgnoreVerdict> {
    let loser = 1 - winner;
    let winner_ply = earliest_terminal_ply(winner, g.turn, guarantee.max_own_moves_to_goal);
    let loser_dist = shortest_distance(g, loser);
    if loser_dist == 255 {
        return None;
    }
    let loser_ply = earliest_terminal_ply(loser, g.turn, loser_dist);
    if winner_ply >= loser_ply {
        return None;
    }
    Some(WallIgnoreVerdict {
        winner,
        winner_terminal_ply: winner_ply,
        loser_terminal_ply: loser_ply,
        source: CertSource::WallIgnoranceCorridor,
        interaction: RaceInteraction::NonInteracting,
        race_minimax_used: false,
    })
}

/// Core detector + race check on a [`GameState`].
pub fn try_wall_ignorance_loss_cert(
    g: &mut GameState,
    scratch: &mut CertScratch,
    force_enable: bool,
) -> Option<WallIgnoreVerdict> {
    if !force_enable && !wall_ignore_loss_cert_enabled() {
        return None;
    }
    if g.winner() >= 0 {
        return None;
    }

    for winner in [0usize, 1] {
        let Some(guarantee) = detect_zero_delay_corridor(g, winner, &mut scratch.corridor) else {
            continue;
        };
        let loser = 1 - winner;
        let interaction = classify_race_interaction(g, &guarantee, loser);

        let verdict = match interaction {
            RaceInteraction::NonInteracting => direct_wall_ignore_verdict(g, winner, &guarantee),
            RaceInteraction::Deterministic | RaceInteraction::Volatile => None,
        };

        if let Some(ref v) = verdict {
            if v.winner_terminal_ply >= v.loser_terminal_ply {
                continue;
            }
            return Some(v.clone());
        }
    }
    None
}

/// Score from side-to-move perspective using the engine mate encoding.
#[inline]
pub fn cert_score_from_stm(verdict: &WallIgnoreVerdict, stm: usize) -> i32 {
    if verdict.winner == stm {
        MATE - verdict.winner_terminal_ply as i32
    } else {
        -MATE + verdict.winner_terminal_ply as i32
    }
}
```

---

## Summary map (quick reference)

| Concern | Function(s) | File |
|---|---|---|
| Build heatmap (dense, multi-route) | `build_corridor_attention`, `build_player_corridor_attention` | `cat/build.rs` |
| Build heatmap (cheap bitmask, used for LMR) | `build_impact_heatmap`, `build_impact_heatmap_for_stm` | `cat/build.rs` |
| Per-move heat lookup | `move_impact_heat`, `move_corridor_attention`, `move_impact_heat_race` | `cat/prune.rs` |
| **Provably-useless wall (dead zone)** | `wall_in_dead_zone`, `wall_completely_skipped` | `cat/prune.rs` |
| **Heuristically-irrelevant wall (not on hot corridor)** | `wall_should_search` | `cat/prune.rs` |
| Wall gen at inner nodes, dead-zone only | `gen_walls_deadzone_filtered` | `titanium/search.rs` |
| Wall gen at inner nodes, CAT-heat filtered | `gen_walls_cat_filtered` | `titanium/search.rs` |
| Move ordering (active engine) | `order_moves_prior` | `titanium/search.rs` |
| Move ordering (legacy/alphabeta engine) | `order_moves`, `move_order_score` | `cat/prune.rs` |
| LMR depth-reduction model (active, v16) | `plan_v16_wall_lmr`, `plan_v16_pawn_lmr`, `cat_extra_reduction` | `search/v16_lmr.rs` |
| LMR CAT-attention primitives | `cat_attention`, `cat_pressure`, `CAT_ATTENTION_TAIL_CUTOFF=0.10` | `search/cat_index_lmr.rs` |
| Per-thread LMR aggression schedule | `lmr_aggression_percent` (177/200/277/350) | `search/cat_index_lmr.rs` |
| Reverse futility pruning (score-based, not CAT) | inline in `ab()` (`depth <= 4 && sev - 90*depth >= beta`) | `titanium/search.rs` ~4360 |
| Null-move pruning | inline in `ab()` | `titanium/search.rs` ~4368 |
| Frontier LMP (late-move pruning by history) | inline in `ab()` move loop | `titanium/search.rs` ~4466 |
| Lazy-SMP root move width (shipped, count-based) | `LAZY_SMP_WIDTHS`, `lazy_smp_allowed_root_moves` | `titanium/search.rs` |
| Lazy-SMP root move filter (experimental, value-based, uncommitted) | `lazy_smp_value_filtered_moves`, `LAZY_SMP_VALUE_THRESHOLD_PCTS` | `titanium/search.rs` (see §8 diff) |
| Zero-delay-corridor forced-win certificate | `detect_zero_delay_corridor`, `try_wall_ignorance_loss_cert` | `titanium/wall_ignore_corridor.rs`, `titanium/wall_ignore_cert.rs` |
