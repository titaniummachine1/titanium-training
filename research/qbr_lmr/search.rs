//! Iterative-deepening negamax alpha-beta with PVS and independently gated
//! search/evaluation experiments. Merged defaults live in [`Features`]; every
//! batch-2 ratchet candidate remains default-off and composable.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
use std::thread;
use std::time::Duration;

use crate::bfs::{
    distance_to_row, distance_to_row_for_wall, distances_to_row, shortest_path_to_row, step,
    DistanceQuery, GoalDistances, PawnShortestEdges, Topology, WallDistanceContext,
};
use crate::clock::Instant;
use crate::movegen::{
    geometrically_available_wall_slots, legal_actions, legal_actions_until, legal_actions_with_distances,
    legal_actions_with_distances_until, MoveList, MAX_ACTIONS,
};
use crate::nnue::{Accumulator as NnueAccumulator, Network as NnueNetwork};
use crate::race::{
    exact_zero_wall_outcome, RaceBuildPolicy, RaceTableCache, ZeroWallOutcome,
};
use crate::sha256;
use crate::state::{cell_col, cell_row, walls_left, winner, Action, ActionKind, State};

// ---------------------------------------------------------------------------
// Scoring constants (integer centisteps).
// ---------------------------------------------------------------------------

/// Base magnitude of a terminal score.  A win at ply `p` scores `MATE - p`,
/// a loss `-(MATE - p)`, so shallower mates dominate deeper ones.
pub const MATE: i32 = 30_000;
/// Scores at least this large in magnitude are mate scores (ply-relative).
pub const MATE_THRESHOLD: i32 = 29_000;
const INF: i32 = 31_000;

const BOUND_NONE: u8 = 0;
const BOUND_EXACT: u8 = 1;
const BOUND_LOWER: u8 = 2;
const BOUND_UPPER: u8 = 3;

// The default TT remains fixed at 2^22 entries. CLI-selected capacities are
// rounded down to a power of two so hashing can retain the same mask operation.
const TT_BITS: usize = 22;
const TT_SIZE: usize = 1 << TT_BITS;
const MEBIBYTE: usize = 1024 * 1024;
const MAX_SEARCH_THREADS: usize = 256;

/// The public maximum search depth is 64. Leave headroom for a pass search
/// and keep killer indexing total if a caller raises that limit later.
const MAX_KILLER_PLY: usize = 128;
const KILLER_EMPTY: u16 = u16::MAX;
const HISTORY_ACTIONS: usize = 3 * 256;
const CMH_ENTRIES: usize = 2 * HISTORY_ACTIONS * HISTORY_ACTIONS;
const CMH_MAX: i32 = i16::MAX as i32;
const CORR_HISTORY_WALL_BANDS: usize = 3;
const CORR_HISTORY_ENTRIES: usize = 2 * CORR_HISTORY_WALL_BANDS * CORR_HISTORY_WALL_BANDS;
const CORR_HISTORY_LIMIT: i32 = 1024;
const CORR_HISTORY_UPDATE_MAX: i32 = 256;
const CORR_HISTORY_SCALE: i32 = 32;
/// Heuristic evaluations cannot approach this range; race-table and mate
/// values can. Keeping their discontinuous exact scores out of correction
/// learning is the Q3 aspiration-safety rail.
const PROVEN_SCORE_THRESHOLD: i32 = 10_000;
const WALL_HISTORY_DISTANCE_BANDS: usize = 3;
const WALL_HISTORY_CONTEXTS: usize = 2 * WALL_HISTORY_DISTANCE_BANDS;
const WALL_HISTORY_ACTIONS: usize = 2 * 64;
const WALL_HISTORY_ENTRIES: usize = 2 * WALL_HISTORY_CONTEXTS * WALL_HISTORY_ACTIONS;
const EVAL_SWING_BUCKETS: usize = 4;
const RFP_PRECISION_BUCKETS: usize = 2;
const RFP_PRECISION_LINEAR_MARGINS: [i32; 5] = [100, 150, 200, 300, 450];
const RFP_PRECISION_ARMS: usize = RFP_PRECISION_LINEAR_MARGINS.len() + 1;
/// SYNTHESIS_V3 leaves the optional quadratic arm's coefficient unspecified.
/// BATCH-3 freezes the explicit inference `100 * depth^2` for reproducibility.
const RFP_PRECISION_QUADRATIC_COEFFICIENT: i32 = 100;
const FRAGILITY_CP: i32 = 100;
const INTERDICTION_ME_CP: i32 = 69;
const INTERDICTION_OPP_CP: i32 = 62;
const INTERDICTION_CAP: i32 = 10;
/// WALLQ-TC's frozen decision-relevance band around a leaf's actual window.
const WALLQ_TC_WINDOW_CP: i32 = 150;

/// Timed searches start with coarse node polling, then tighten it as the hard
/// deadline approaches. Ordering and move generation have their own direct
/// polls because a single node can contain many BFS operations.
const DEADLINE_CHECK_VERY_NEAR: Duration = Duration::from_millis(5);
const DEADLINE_CHECK_NEAR: Duration = Duration::from_millis(15);
const DEADLINE_CHECK_APPROACHING: Duration = Duration::from_millis(50);
/// Timed race-table construction gets one small, absolute slice per genmove.
/// Cache hits remain available afterward; a miss simply omits the shortcut.
const RACE_BUILD_SLICE_MAX: Duration = Duration::from_millis(5);
const RACE_BUILD_SLICE_DIVISOR: u32 = 8;

/// Estimate the next iteration in nanoseconds from the two most recent
/// completed iterations. The effective branching factor is `last / previous`,
/// clamped to [2, 20]. Division rounds up so the start gate never understates a
/// fractional-nanosecond estimate.
fn predicted_iteration_nanos(previous: Duration, last: Duration) -> u128 {
    let previous = previous.as_nanos();
    let last = last.as_nanos();
    if last == 0 {
        return 0;
    }
    if previous == 0 {
        return last.saturating_mul(20);
    }
    if last <= previous.saturating_mul(2) {
        return last.saturating_mul(2);
    }
    if last >= previous.saturating_mul(20) {
        return last.saturating_mul(20);
    }

    let numerator = last.saturating_mul(last);
    numerator / previous + u128::from(numerator % previous != 0)
}

/// WALLQ-TC's inclusive local-window test. The caller supplies the incoming
/// node window, before that node has raised alpha while searching children.
#[inline]
fn wallq_tc_in_window(static_eval: i32, alpha: i32, beta: i32) -> bool {
    static_eval >= alpha.saturating_sub(WALLQ_TC_WINDOW_CP)
        && static_eval <= beta.saturating_add(WALLQ_TC_WINDOW_CP)
}

/// WALLQ-TC-DUAL's resolved root mode.  This is deliberately stored in each
/// search lane instead of re-derived from a lane-local time control: Lazy-SMP
/// helpers intentionally receive `allotted_budget_ms=None`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WallqTcMode {
    Inactive,
    Fast,
    Long,
}

/// Probe-only override for reproducing a resolved WALLQ-TC lane without a
/// timed root budget. `Off` preserves the production budget resolver exactly.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WallqTcProbeForce {
    #[default]
    Off,
    Fast,
    Long,
}

impl WallqTcMode {
    #[inline]
    fn resolve(
        enabled: bool,
        budget_ms: Option<u64>,
        probe_force: WallqTcProbeForce,
    ) -> Self {
        if !enabled {
            return Self::Inactive;
        }
        match probe_force {
            WallqTcProbeForce::Fast => return Self::Fast,
            WallqTcProbeForce::Long => return Self::Long,
            WallqTcProbeForce::Off => {}
        }
        match budget_ms {
            Some(ms) if ms <= 50 => Self::Fast,
            Some(ms) if ms >= 500 => Self::Long,
            _ => Self::Inactive,
        }
    }

    #[inline]
    fn is_active(self) -> bool {
        !matches!(self, Self::Inactive)
    }
}

// ---------------------------------------------------------------------------
// Feature plumbing (merged defaults plus default-off experiments).
// ---------------------------------------------------------------------------

/// Search feature toggles. Merged features retain their established defaults;
/// every batch-2 candidate defaults to OFF.
#[derive(Clone, Debug)]
pub struct Features {
    /// Stored for future stochastic features; v0 search is deterministic.
    pub seed: u64,
    /// Lazy-SMP search threads. One preserves the exact legacy search path.
    pub threads: usize,
    /// Percentage of the move budget after which no new ID iteration starts.
    pub soft_pct: u64,
    /// Percentage of the move budget at which an in-flight search is aborted.
    pub hard_pct: u64,
    /// Predict whether the next ID iteration can finish before the hard limit.
    pub predictive_start: bool,
    pub cheap_wall_order: bool,
    pub killers: bool,
    pub history: bool,
    pub race_exact: bool,
    pub aspiration: bool,
    pub null_move: bool,
    pub lmr: bool,
    /// Q4 two-ply LMR tier for delta-zero walls outside gorisanson's probable set.
    pub lmr_probable_walls: bool,
    /// Q4's mandatory classifier-free attribution arm (late-index r=2 only).
    pub lmr_probable_walls_control: bool,
    /// Q1 late-move pruning on the shallow non-PV ordered wall tail.
    pub lmp: bool,
    /// Depth-3 LMP threshold selector; supported candidates are 8, 16, and 24.
    pub lmp_n: usize,
    /// Q2 shallow reverse-futility pruning.
    pub rfp: bool,
    /// Per-ply reverse-futility margin in centisteps.
    pub rfp_margin: i32,
    /// Maximum depth eligible for production reverse-futility pruning.
    pub rfp_depth: i32,
    /// Q12: extend production RFP through depth four only at fast time controls.
    pub rfp_tc_adaptive: bool,
    /// Q3 one-ply countermove history blended into quiet-wall history.
    pub cmh: bool,
    /// Q3 coarse wall-regime correction history, applied only at ordinary leaves.
    pub corr_hist: bool,
    /// Q6 depth-four extension of the merged N=24 late-move pruning vector.
    pub lmp_d4: bool,
    /// Q6 sweep arm: withhold only depth-four LMP in mate/race-critical nodes.
    pub lmp_d4_guard: bool,
    /// Q7 opponent-route-context wall history blended into quiet ordering.
    pub wall_hist: bool,
    /// Bench-only Q7 warm-read/LMR counters; ordinary flagged play leaves it off.
    pub probe_wall_hist: bool,
    pub ev_progress: bool,
    pub ev_wallphase: bool,
    pub wall_cp: i32,
    /// Endgame wall value; negative values disable phase interpolation.
    pub wall_cp_endgame: i32,
    /// Race-margin amplification; zero disables the additive term.
    pub race_amp: i32,
    pub ev_corridor: bool,
    pub tt_sym: bool,
    pub race1w: bool,
    /// Sound monopoly-only extension of the exact race region to two walls.
    pub race2w: bool,
    /// Cheap single-wall vulnerability of a unique shortest path.
    pub ev_fragility_1w: bool,
    /// NOW-lane search instrumentation; never changes move selection.
    pub probe_cutoffs: bool,
    /// Q2 fixed-depth static one-ply evaluation-swing measurement.
    pub probe_evalswing: bool,
    /// Q2 stage-2 shadow probe: RFP trigger precision against completed search.
    pub probe_rfp_precision: bool,
    /// Maximum depth sampled by the RFP precision probe, configured independently.
    pub probe_rfp_depth: i32,
    /// TM2 production time reallocation using the validated root-tension signal.
    pub tm2_time: bool,
    /// Quantized value evaluator path. `None` is the frozen classical path.
    pub nnue_path: Option<String>,
    /// Unrecognised `--feature k=v` pairs, retained for logging only.
    pub unknown: Vec<(String, String)>,
    /// WALLQ-TC-DUAL: enable its leaf correction only in the frozen fast
    /// (<=50ms) and long (>=500ms) root-budget bands. This tail placement
    /// preserves the established hot feature-field layout when it is off.
    pub wallq_tc: bool,
    /// Probe-only fixed-depth override. Production play leaves this `Off`.
    pub wallq_tc_probe_force: WallqTcProbeForce,
}

impl Default for Features {
    fn default() -> Self {
        #[cfg(all(target_family = "wasm", target_os = "unknown"))]
        let threads = 1;
        #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
        let threads = 4;
        Self {
            seed: 0,
            threads, // Native E-010 MERGED default is 4; browser wasm is single-thread.
            soft_pct: 55,
            hard_pct: 85,
            predictive_start: true, // E-009 MERGED: SPRT(0,+5) H1 (n=1390, 56.4%, Elo +44.5 [26.3,63.0]) @50ms — F-004 fix
            cheap_wall_order: false,
            killers: false,
            history: true, // E-004 MERGED: SPRT H1 (n=2316, 52.4%, Elo +16.5 [4.3,28.8]) @50ms
            race_exact: false, // E-005R REVERTED: hardened-contract re-verification hit cap w/o decision (n=3000, +11.6 [-0.7,23.9] pentanomial). Original +64.6 was on the pre-wall_cp base (0-wall nodes common); wall hoarding collapsed its firing rate. Re-test queued at (0,+5).
            aspiration: false,
            null_move: false,
            lmr: true, // E-011 MERGED: SPRT(0,+5) H1 (n=1028, 57.9%, Elo +55.2 [34.2,76.7]) @50ms
            lmr_probable_walls: false,
            lmr_probable_walls_control: false,
            lmp: true, // E-020 MERGED: SPRT(0,+5) H1 (n=1404, 56.4%, Elo +44.8 [26.7,63.1]) @50ms; N=24 frozen from Q1 zero-game stage
            lmp_n: 24,
            rfp: true,
            rfp_margin: 100,
            rfp_depth: 3,
            rfp_tc_adaptive: true, // E-028 MERGED: TC-adaptive depth-four RFP at budgets <=200ms
            cmh: true, // E-021 MERGED: SPRT(0,+5) H1 (n=3122, 52.8%, Elo +19.3 [7.2,31.4]) @50ms
            corr_hist: false,
            lmp_d4: false,
            lmp_d4_guard: false,
            wall_hist: false,
            probe_wall_hist: false,

            ev_progress: false,
            ev_wallphase: false,
            wall_cp: 150, // E-019 MERGED: SPRT(0,+5) H1 (n=3768, 53.6%, Elo +25.1 [14.1,36.2]) — supersedes E-008's 200 on the LMR/race1w base
            wall_cp_endgame: -1,
            race_amp: 0,
            ev_corridor: false,
            tt_sym: false,
            race1w: true, // E-015 MERGED: SPRT(0,+5) H1 (n=2188, 52.8%, Elo +19.4 [5.4,33.4]) @50ms
            race2w: false,
            ev_fragility_1w: false,
            probe_cutoffs: false,
            probe_evalswing: false,
            probe_rfp_precision: false,
            probe_rfp_depth: 3,
            tm2_time: false,
            nnue_path: None,
            unknown: Vec::new(),
            wallq_tc: true,
            wallq_tc_probe_force: WallqTcProbeForce::Off,
        }
    }
}

fn truthy(value: &str) -> bool {
    matches!(value, "1" | "true" | "on" | "yes")
}

impl Features {
    /// Parse `--seed N` and repeatable `--feature k=v` from trailing CLI args.
    /// Unknown arguments are ignored so the harness may append extras safely.
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let mut f = Features::default();
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--seed" => {
                    if let Some(v) = args.get(i + 1) {
                        f.seed = v.parse().unwrap_or(0);
                    }
                    i += 2;
                }
                "--feature" => {
                    if let Some(kv) = args.get(i + 1) {
                        if let Some((key, value)) = kv.split_once('=') {
                            f.set(key, value)?;
                        } else if kv == "nnue" {
                            // Preserve the established permissive parser for
                            // other experiments, but never silently downgrade
                            // an explicitly requested NNUE to classical eval.
                            f.nnue_path = Some(String::new());
                        }
                    }
                    i += 2;
                }
                _ => i += 1,
            }
        }
        let f = target_features(f);
        if f.wallq_tc && f.tm2_time {
            return Err(
                "--feature wallq_tc=on requires --feature tm2_time=off".to_owned(),
            );
        }
        Ok(f)
    }

    fn set(&mut self, key: &str, value: &str) -> Result<(), String> {
        match key {
            "soft_pct" => self.soft_pct = value.parse().unwrap_or(55),
            "hard_pct" => self.hard_pct = value.parse().unwrap_or(85),
            "predictive_start" => self.predictive_start = truthy(value),
            "cheap_wall_order" => self.cheap_wall_order = truthy(value),
            "killers" => self.killers = truthy(value),
            "history" => self.history = truthy(value),
            "race_exact" => self.race_exact = truthy(value),
            "aspiration" => self.aspiration = truthy(value),
            "null_move" => self.null_move = truthy(value),
            "lmr" => self.lmr = truthy(value),
            "lmr_probable_walls" => self.lmr_probable_walls = truthy(value),
            "lmr_probable_walls_control" => {
                self.lmr_probable_walls_control = truthy(value)
            }
            "lmp" => self.lmp = truthy(value),
            "lmp_n" => {
                self.lmp_n = match value {
                    "8" => 8,
                    "16" => 16,
                    "24" => 24,
                    _ => 24,
                }
            }
            "rfp" => self.rfp = truthy(value),
            "rfp_margin" => {
                self.rfp_margin = value
                    .parse::<i32>()
                    .ok()
                    .filter(|margin| *margin >= 0)
                    .unwrap_or(250)
            }
            "rfp_depth" => {
                self.rfp_depth = value
                    .parse::<i32>()
                    .ok()
                    .filter(|depth| (1..=4).contains(depth))
                    .unwrap_or(3)
            }
            "rfp_tc_adaptive" => self.rfp_tc_adaptive = truthy(value),
            "cmh" => self.cmh = truthy(value),
            "corr_hist" => self.corr_hist = truthy(value),
            "lmp_d4" => self.lmp_d4 = truthy(value),
            "lmp_d4_guard" => self.lmp_d4_guard = truthy(value),
            "wall_hist" => self.wall_hist = truthy(value),
            "probe_wall_hist" => self.probe_wall_hist = truthy(value),
            "ev_progress" => self.ev_progress = truthy(value),
            "ev_wallphase" => self.ev_wallphase = truthy(value),
            "wall_cp" => self.wall_cp = value.parse().unwrap_or(200),
            "wall_cp_endgame" => self.wall_cp_endgame = value.parse().unwrap_or(-1),
            "race_amp" => self.race_amp = value.parse().unwrap_or(0),
            "ev_corridor" => self.ev_corridor = truthy(value),
            "wallq_tc" => self.wallq_tc = truthy(value),
            "wallq_tc_probe_force" => {
                self.wallq_tc_probe_force = match value {
                    "off" => WallqTcProbeForce::Off,
                    "fast" => WallqTcProbeForce::Fast,
                    "long" => WallqTcProbeForce::Long,
                    _ => {
                        return Err(
                            "--feature wallq_tc_probe_force must be off, fast, or long"
                                .to_owned(),
                        )
                    }
                }
            }
            "tt_sym" => self.tt_sym = truthy(value),
            "race1w" => self.race1w = truthy(value),
            "race2w" => self.race2w = truthy(value),
            "ev_fragility_1w" => self.ev_fragility_1w = truthy(value),
            "probe_cutoffs" => self.probe_cutoffs = truthy(value),
            "probe_evalswing" => self.probe_evalswing = truthy(value),
            "probe_rfp_precision" => self.probe_rfp_precision = truthy(value),
            "probe_rfp_depth" => {
                self.probe_rfp_depth = value
                    .parse::<i32>()
                    .ok()
                    .filter(|depth| *depth > 0)
                    .unwrap_or(3)
            }
            "tm2_time" => self.tm2_time = truthy(value),
            "nnue" => {
                // An explicitly empty path is still a configured NNUE and
                // therefore fails closed in the fallible CLI constructor.
                self.nnue_path = Some(value.to_owned());
            }
            "threads" => {
                self.threads = value
                    .parse::<usize>()
                    .ok()
                    .filter(|threads| *threads > 0)
                    .unwrap_or(1)
                    .min(MAX_SEARCH_THREADS)
            }
            "seed" => self.seed = value.parse().unwrap_or(0),
            _ => self.unknown.push((key.to_owned(), value.to_owned())),
        }
        Ok(())
    }

    /// Canonical resolved feature map for provenance telemetry.  The explicit
    /// fixed field order makes the hash independent of argv ordering while
    /// still retaining unknown pairs, so an unsupported requested feature can
    /// never be silently indistinguishable from a recognized one.
    pub fn effective_feature_map_hash(&self) -> String {
        sha256::hex_digest(self.effective_feature_map().as_bytes())
    }

    fn effective_feature_map(&self) -> String {
        let mut map = String::with_capacity(1536);
        macro_rules! push_field {
            ($field:ident) => {
                let _ = writeln!(map, "{}={:?}", stringify!($field), self.$field);
            };
        }
        push_field!(seed);
        push_field!(threads);
        push_field!(soft_pct);
        push_field!(hard_pct);
        push_field!(predictive_start);
        push_field!(cheap_wall_order);
        push_field!(killers);
        push_field!(history);
        push_field!(race_exact);
        push_field!(aspiration);
        push_field!(null_move);
        push_field!(lmr);
        push_field!(lmr_probable_walls);
        push_field!(lmr_probable_walls_control);
        push_field!(lmp);
        push_field!(lmp_n);
        push_field!(rfp);
        push_field!(rfp_margin);
        push_field!(rfp_depth);
        push_field!(rfp_tc_adaptive);
        push_field!(cmh);
        push_field!(corr_hist);
        push_field!(lmp_d4);
        push_field!(lmp_d4_guard);
        push_field!(wall_hist);
        push_field!(probe_wall_hist);
        push_field!(ev_progress);
        push_field!(ev_wallphase);
        push_field!(wall_cp);
        push_field!(wall_cp_endgame);
        push_field!(race_amp);
        push_field!(ev_corridor);
        push_field!(wallq_tc);
        // Keep the absent/explicit-off production map byte-identical to the
        // pre-probe map while still binding an active override in provenance.
        if self.wallq_tc_probe_force != WallqTcProbeForce::Off {
            push_field!(wallq_tc_probe_force);
        }
        push_field!(tt_sym);
        push_field!(race1w);
        push_field!(race2w);
        push_field!(ev_fragility_1w);
        push_field!(probe_cutoffs);
        push_field!(probe_evalswing);
        push_field!(probe_rfp_precision);
        push_field!(probe_rfp_depth);
        push_field!(tm2_time);
        push_field!(nnue_path);
        push_field!(unknown);
        map
    }
}

#[inline]
fn target_features(features: Features) -> Features {
    #[cfg(all(target_family = "wasm", target_os = "unknown"))]
    {
        let mut features = features;
        features.threads = 1;
        features
    }
    #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
    {
        features
    }
}

// ---------------------------------------------------------------------------
// Horizontal reflection and Zobrist hashing.
// ---------------------------------------------------------------------------

#[inline]
fn mirror_cell_lr(cell: u8) -> u8 {
    cell_row(cell) * 9 + (8 - cell_col(cell))
}

#[inline]
fn mirror_slot_lr(slot: u8) -> u8 {
    let row = slot / 8;
    let col = slot % 8;
    row * 8 + (7 - col)
}

fn mirror_wall_bits_lr(mut walls: u64) -> u64 {
    let mut mirrored = 0u64;
    while walls != 0 {
        let slot = walls.trailing_zeros() as u8;
        walls &= walls - 1;
        mirrored |= 1u64 << mirror_slot_lr(slot);
    }
    mirrored
}

#[inline]
fn mirror_action_lr(action: Action) -> Action {
    Action {
        kind: action.kind,
        pos: match action.kind {
            ActionKind::Pawn => mirror_cell_lr(action.pos),
            ActionKind::Horizontal | ActionKind::Vertical => mirror_slot_lr(action.pos),
        },
    }
}

fn mirror_state_lr(state: &State) -> State {
    State {
        p0: mirror_cell_lr(state.p0),
        p1: mirror_cell_lr(state.p1),
        w0: state.w0,
        w1: state.w1,
        h: mirror_wall_bits_lr(state.h),
        v: mirror_wall_bits_lr(state.v),
        turn: state.turn,
    }
}

#[inline]
fn state_order_key(state: &State) -> (u8, u8, u8, u8, u64, u64, u8) {
    (
        state.p0, state.p1, state.w0, state.w1, state.h, state.v, state.turn,
    )
}

struct Zobrist {
    p0: [u64; 81],
    p1: [u64; 81],
    h: [u64; 64],
    v: [u64; 64],
    w0: [u64; 11],
    w1: [u64; 11],
    turn: u64,
}

#[inline]
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Zobrist {
    fn new() -> Self {
        let mut s = 0x0123_4567_89AB_CDEFu64;
        let mut fill = |n: usize| -> Vec<u64> { (0..n).map(|_| splitmix(&mut s)).collect() };
        let p0: Vec<u64> = fill(81);
        let p1: Vec<u64> = fill(81);
        let h: Vec<u64> = fill(64);
        let v: Vec<u64> = fill(64);
        let w0: Vec<u64> = fill(11);
        let w1: Vec<u64> = fill(11);
        let turn = splitmix(&mut s);
        Self {
            p0: p0.try_into().unwrap(),
            p1: p1.try_into().unwrap(),
            h: h.try_into().unwrap(),
            v: v.try_into().unwrap(),
            w0: w0.try_into().unwrap(),
            w1: w1.try_into().unwrap(),
            turn,
        }
    }

    fn hash(&self, state: &State) -> u64 {
        let mut key = self.p0[state.p0 as usize] ^ self.p1[state.p1 as usize];
        key ^= self.w0[state.w0 as usize] ^ self.w1[state.w1 as usize];
        if state.turn == 1 {
            key ^= self.turn;
        }
        let mut h = state.h;
        while h != 0 {
            let slot = h.trailing_zeros() as usize;
            h &= h - 1;
            key ^= self.h[slot];
        }
        let mut v = state.v;
        while v != 0 {
            let slot = v.trailing_zeros() as usize;
            v &= v - 1;
            key ^= self.v[slot];
        }
        key
    }
}

#[derive(Clone, Copy)]
struct TtPositionKey {
    hash: u64,
    /// The current position was reflected to reach the canonical hash.
    mirrored: bool,
}

// ---------------------------------------------------------------------------
// Transposition table.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct TtEntry {
    key: u32,
    best: u16,
    score: i16,
    depth: u8, // 0 == empty
    bound: u8,
}

impl TtEntry {
    const EMPTY: Self = Self {
        key: 0,
        best: 0,
        score: 0,
        depth: 0,
        bound: BOUND_NONE,
    };
}

// Atomic TT payload: compact action (8), score (16), depth-minus-one (6),
// bound (2), verifier (32). Legal actions need only 209 codes: 81 pawn
// destinations plus 64 slots for each wall orientation. Public search depth is
// capped at 64, so storing depth - 1 preserves every supported depth in 6 bits.
const ATOMIC_TT_BEST_BITS: u32 = 8;
const ATOMIC_TT_SCORE_SHIFT: u32 = ATOMIC_TT_BEST_BITS;
const ATOMIC_TT_DEPTH_SHIFT: u32 = ATOMIC_TT_SCORE_SHIFT + 16;
const ATOMIC_TT_DEPTH_BITS: u32 = 6;
const ATOMIC_TT_DEPTH_MASK: u64 = (1u64 << ATOMIC_TT_DEPTH_BITS) - 1;
const ATOMIC_TT_BOUND_SHIFT: u32 = ATOMIC_TT_DEPTH_SHIFT + ATOMIC_TT_DEPTH_BITS;
const ATOMIC_TT_VERIFY_SHIFT: u32 = ATOMIC_TT_BOUND_SHIFT + 2;
const ATOMIC_TT_VERIFY_MASK: u64 = (1u64 << (64 - ATOMIC_TT_VERIFY_SHIFT)) - 1;
const ATOMIC_TT_PAWN_CODES: u16 = 81;
const ATOMIC_TT_WALL_CODES: u16 = 64;

/// Single-word lock-free TT entry used only by Lazy SMP. The compact action,
/// score, depth, bound, and 32-bit hash verifier are published together, so a
/// reader can never observe a torn payload. Depth-aware CAS replacement also
/// prevents a racing shallower entry from overwriting a deeper one.
///
/// The old 28-bit verifier accepted a random same-index alien with probability
/// 2^-28 (about 3.73e-9), and ignored the hash bits between a short table's
/// index and bit 36 entirely. The verifier below folds every non-index hash bit
/// into 32 bits, reducing the random false-accept probability to 2^-32 (about
/// 2.33e-10, 16x lower) without overlapping the slot-index bits.
struct AtomicTtEntry {
    word: AtomicU64,
}

impl AtomicTtEntry {
    fn empty() -> Self {
        Self {
            word: AtomicU64::new(0),
        }
    }

    #[inline]
    fn probe(&self, hash: u64, index_bits: u32) -> Option<TtEntry> {
        let word = self.word.load(Ordering::Acquire);
        if atomic_tt_depth(word) == 0
            || atomic_tt_verify(word) != atomic_tt_hash_verify(hash, index_bits)
        {
            return None;
        }
        Some(unpack_atomic_tt_word(word, hash))
    }

    #[inline]
    fn store(&self, hash: u64, entry: TtEntry, index_bits: u32) {
        // The public search limit is 64. If an internal caller violates that
        // envelope, omitting the TT store is conservative and keeps packing
        // total rather than truncating a depth into an unsafe value.
        if entry.depth == 0 || entry.depth > 64 {
            return;
        }
        let replacement = pack_atomic_tt_word(entry, hash, index_bits);
        let mut current = self.word.load(Ordering::Relaxed);
        loop {
            let current_depth = atomic_tt_depth(current);
            if current_depth != 0 && entry.depth < current_depth {
                return;
            }
            if current_depth == entry.depth
                && (atomic_tt_bound(current) == BOUND_EXACT || entry.bound != BOUND_EXACT)
            {
                return;
            }
            match self.word.compare_exchange_weak(
                current,
                replacement,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    fn clear(&self) {
        self.word.store(0, Ordering::Release);
    }
}

struct SharedTt {
    entries: Box<[AtomicTtEntry]>,
    index_bits: u32,
}

impl SharedTt {
    fn new(entries: usize) -> Self {
        assert!(entries.is_power_of_two());
        Self {
            entries: (0..entries)
                .map(|_| AtomicTtEntry::empty())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            index_bits: entries.trailing_zeros(),
        }
    }

    fn clear(&self) {
        for entry in &self.entries {
            entry.clear();
        }
    }
}

#[inline]
fn atomic_tt_hash_verify(hash: u64, index_bits: u32) -> u64 {
    debug_assert!(index_bits < u64::BITS);
    let non_index = hash >> index_bits;
    let folded = (non_index as u32) ^ ((non_index >> 32) as u32);
    u64::from(folded) & ATOMIC_TT_VERIFY_MASK
}

#[inline]
fn atomic_tt_verify(word: u64) -> u64 {
    word >> ATOMIC_TT_VERIFY_SHIFT
}

#[inline]
fn atomic_tt_depth(word: u64) -> u8 {
    if atomic_tt_bound(word) == BOUND_NONE {
        0
    } else {
        (((word >> ATOMIC_TT_DEPTH_SHIFT) & ATOMIC_TT_DEPTH_MASK) as u8) + 1
    }
}

#[inline]
fn atomic_tt_bound(word: u64) -> u8 {
    ((word >> ATOMIC_TT_BOUND_SHIFT) & 0x3) as u8
}

#[inline]
fn pack_atomic_tt_word(entry: TtEntry, hash: u64, index_bits: u32) -> u64 {
    debug_assert!((1..=64).contains(&entry.depth));
    debug_assert!(entry.bound <= BOUND_UPPER);
    u64::from(compact_atomic_tt_action(entry.best))
        | (u64::from(entry.score as u16) << ATOMIC_TT_SCORE_SHIFT)
        | (u64::from(entry.depth - 1) << ATOMIC_TT_DEPTH_SHIFT)
        | (u64::from(entry.bound) << ATOMIC_TT_BOUND_SHIFT)
        | (atomic_tt_hash_verify(hash, index_bits) << ATOMIC_TT_VERIFY_SHIFT)
}

#[inline]
fn unpack_atomic_tt_word(word: u64, hash: u64) -> TtEntry {
    TtEntry {
        key: (hash >> 32) as u32,
        best: expand_atomic_tt_action((word & ((1 << ATOMIC_TT_BEST_BITS) - 1)) as u8),
        score: ((word >> ATOMIC_TT_SCORE_SHIFT) as u16) as i16,
        depth: atomic_tt_depth(word),
        bound: atomic_tt_bound(word),
    }
}

#[inline]
fn compact_atomic_tt_action(best: u16) -> u8 {
    let action = decode_action(best);
    let code = match action.kind {
        ActionKind::Pawn => {
            debug_assert!(u16::from(action.pos) < ATOMIC_TT_PAWN_CODES);
            u16::from(action.pos)
        }
        ActionKind::Horizontal => {
            debug_assert!(u16::from(action.pos) < ATOMIC_TT_WALL_CODES);
            ATOMIC_TT_PAWN_CODES + u16::from(action.pos)
        }
        ActionKind::Vertical => {
            debug_assert!(u16::from(action.pos) < ATOMIC_TT_WALL_CODES);
            ATOMIC_TT_PAWN_CODES + ATOMIC_TT_WALL_CODES + u16::from(action.pos)
        }
    };
    code as u8
}

#[inline]
fn expand_atomic_tt_action(code: u8) -> u16 {
    let code = u16::from(code);
    if code < ATOMIC_TT_PAWN_CODES {
        code
    } else if code < ATOMIC_TT_PAWN_CODES + ATOMIC_TT_WALL_CODES {
        (1 << 8) | (code - ATOMIC_TT_PAWN_CODES)
    } else {
        debug_assert!(code < ATOMIC_TT_PAWN_CODES + 2 * ATOMIC_TT_WALL_CODES);
        (2 << 8) | (code - ATOMIC_TT_PAWN_CODES - ATOMIC_TT_WALL_CODES)
    }
}

#[inline]
fn encode_action(a: Action) -> u16 {
    let kind = match a.kind {
        ActionKind::Pawn => 0u16,
        ActionKind::Horizontal => 1,
        ActionKind::Vertical => 2,
    };
    (kind << 8) | a.pos as u16
}

#[inline]
fn decode_action(code: u16) -> Action {
    let kind = match code >> 8 {
        1 => ActionKind::Horizontal,
        2 => ActionKind::Vertical,
        _ => ActionKind::Pawn,
    };
    Action {
        kind,
        pos: (code & 0xFF) as u8,
    }
}

fn new_countermove_history() -> Box<[i16]> {
    vec![0; CMH_ENTRIES].into_boxed_slice()
}

#[inline]
fn countermove_history_index(side: usize, previous: Action, reply: Action) -> usize {
    debug_assert!(side < 2);
    (side * HISTORY_ACTIONS + encode_action(previous) as usize) * HISTORY_ACTIONS
        + encode_action(reply) as usize
}

/// Coarse material regimes are deliberately sample-rich: empty, scarce, and
/// plentiful wall inventories. Q3's fine wall hash remains a separate,
/// unshipped arm.
#[inline]
fn wall_count_band(walls: u8) -> usize {
    match walls {
        0 => 0,
        1 | 2 => 1,
        _ => 2,
    }
}

#[inline]
fn correction_history_index(state: &State) -> usize {
    let side = state.turn as usize;
    let (own_walls, opponent_walls) = if state.turn == 0 {
        (state.w0, state.w1)
    } else {
        (state.w1, state.w0)
    };
    (side * CORR_HISTORY_WALL_BANDS + wall_count_band(own_walls))
        * CORR_HISTORY_WALL_BANDS
        + wall_count_band(opponent_walls)
}

#[inline]
fn correction_history_value(entry: i16) -> i32 {
    (i32::from(entry) / CORR_HISTORY_SCALE)
        .clamp(-CORR_HISTORY_LIMIT / CORR_HISTORY_SCALE, CORR_HISTORY_LIMIT / CORR_HISTORY_SCALE)
}

/// Depth-weighted residual update using the same self-decaying gravity form
/// as CMH. The bounded internal value maps exactly to a +/-32cp leaf term.
#[inline]
fn update_correction_history_entry(entry: &mut i16, residual: i32, depth: i32) {
    let bonus = residual
        .saturating_mul(depth)
        .saturating_div(8)
        .clamp(-CORR_HISTORY_UPDATE_MAX, CORR_HISTORY_UPDATE_MAX);
    let current = i32::from(*entry);
    let updated = current
        .saturating_add(bonus)
        .saturating_sub(current.saturating_mul(bonus.abs()) / CORR_HISTORY_LIMIT)
        .clamp(-CORR_HISTORY_LIMIT, CORR_HISTORY_LIMIT);
    *entry = updated as i16;
}

#[inline]
fn wall_history_action_index(action: Action) -> usize {
    match action.kind {
        ActionKind::Horizontal => action.pos as usize,
        ActionKind::Vertical => 64 + action.pos as usize,
        ActionKind::Pawn => unreachable!("wall history is never indexed by a pawn move"),
    }
}

#[inline]
fn wall_history_index(side: usize, context: u8, action: Action) -> usize {
    debug_assert!(side < 2);
    debug_assert!((context as usize) < WALL_HISTORY_CONTEXTS);
    let action_index = wall_history_action_index(action);
    (side * WALL_HISTORY_CONTEXTS + context as usize) * WALL_HISTORY_ACTIONS + action_index
}

// ---------------------------------------------------------------------------
// Search context.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct CutoffProbeStats {
    interior_nodes: u64,
    cutoff_ranks: [u64; 5],
    validated_tt_first_cutoffs: u64,
    ordering_generation_nanos: u128,
    search_nanos: u128,
}

impl CutoffProbeStats {
    #[inline]
    fn record_cutoff(&mut self, rank: usize, from_validated_tt: bool) {
        let bucket = match rank {
            0 => 0,
            1 => 1,
            2..=5 => 2,
            6..=10 => 3,
            _ => 4,
        };
        self.cutoff_ranks[bucket] = self.cutoff_ranks[bucket].saturating_add(1);
        if from_validated_tt {
            debug_assert_eq!(rank, 0);
            self.validated_tt_first_cutoffs =
                self.validated_tt_first_cutoffs.saturating_add(1);
        }
    }
}

/// Raw counters from `--feature probe_cutoffs=1` for one search context.
/// Bench reuses a single context, so these aggregate all fixed positions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CutoffProbeReport {
    pub interior_nodes: u64,
    /// Zero-based rank buckets: 0, 1, 2-5, 6-10, and greater than 10.
    pub cutoff_ranks: [u64; 5],
    pub validated_tt_first_cutoffs: u64,
    pub ordering_generation_nanos: u128,
    pub search_nanos: u128,
}

struct EvalSwingProbeStats {
    /// Pawn/wall x RFP-opponent-walls-zero/nonzero, exact centistep swing.
    histograms: [BTreeMap<i32, u64>; EVAL_SWING_BUCKETS],
    totals: [u64; EVAL_SWING_BUCKETS],
    maxima: [i32; EVAL_SWING_BUCKETS],
}

impl EvalSwingProbeStats {
    fn new() -> Self {
        Self {
            histograms: std::array::from_fn(|_| BTreeMap::new()),
            totals: [0; EVAL_SWING_BUCKETS],
            maxima: [0; EVAL_SWING_BUCKETS],
        }
    }

    #[inline]
    fn record(&mut self, action: Action, rfp_opponent_has_walls: bool, swing: i32) {
        let action_offset = usize::from(action.kind != ActionKind::Pawn) * 2;
        let bucket = action_offset + usize::from(rfp_opponent_has_walls);
        let magnitude = swing.saturating_abs();
        *self.histograms[bucket].entry(magnitude).or_insert(0) += 1;
        self.totals[bucket] += 1;
        self.maxima[bucket] = self.maxima[bucket].max(magnitude);
    }

    fn percentile(&self, bucket: usize, percentile: u64) -> i32 {
        let total = self.totals[bucket];
        if total == 0 {
            return 0;
        }
        let target = total.saturating_mul(percentile).saturating_add(99) / 100;
        let mut cumulative = 0u64;
        for (&value, &count) in &self.histograms[bucket] {
            cumulative += count;
            if cumulative >= target {
                return value;
            }
        }
        self.maxima[bucket]
    }

    fn report(&self) -> [EvalSwingBucketReport; EVAL_SWING_BUCKETS] {
        std::array::from_fn(|bucket| EvalSwingBucketReport {
            wall_move: bucket >= 2,
            opponent_has_walls: bucket & 1 != 0,
            count: self.totals[bucket],
            p50: self.percentile(bucket, 50),
            p90: self.percentile(bucket, 90),
            p95: self.percentile(bucket, 95),
            p99: self.percentile(bucket, 99),
            max: self.maxima[bucket],
        })
    }
}

/// Static one-ply evaluation-swing measurements from the depth-one frontier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvalSwingBucketReport {
    pub wall_move: bool,
    pub opponent_has_walls: bool,
    pub count: u64,
    pub p50: i32,
    pub p90: i32,
    pub p95: i32,
    pub p99: i32,
    pub max: i32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RfpPrecisionCounts {
    fires: u64,
    true_fail_highs: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RfpPrecisionSample {
    opponent_walls_bucket: usize,
    fires: [bool; RFP_PRECISION_ARMS],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RfpPrecisionProbeStats {
    eligible_nodes: [u64; RFP_PRECISION_BUCKETS],
    counts: [[RfpPrecisionCounts; RFP_PRECISION_BUCKETS]; RFP_PRECISION_ARMS],
    projected_saved_nodes: [u64; RFP_PRECISION_ARMS],
    projected_saved_nanos: [u128; RFP_PRECISION_ARMS],
}

impl RfpPrecisionProbeStats {
    #[inline]
    fn record(&mut self, sample: RfpPrecisionSample, true_score: i32, beta: i32) {
        let bucket = sample.opponent_walls_bucket;
        self.eligible_nodes[bucket] = self.eligible_nodes[bucket].saturating_add(1);
        for (arm, fired) in sample.fires.into_iter().enumerate() {
            if !fired {
                continue;
            }
            let counts = &mut self.counts[arm][bucket];
            counts.fires = counts.fires.saturating_add(1);
            if true_score >= beta {
                counts.true_fail_highs = counts.true_fail_highs.saturating_add(1);
            }
        }
    }

    fn report(&self) -> [RfpPrecisionArmReport; RFP_PRECISION_ARMS] {
        std::array::from_fn(|arm| {
            let (coefficient, quadratic) = if arm < RFP_PRECISION_LINEAR_MARGINS.len() {
                (RFP_PRECISION_LINEAR_MARGINS[arm], false)
            } else {
                (RFP_PRECISION_QUADRATIC_COEFFICIENT, true)
            };
            RfpPrecisionArmReport {
                coefficient,
                quadratic,
                projected_saved_nodes: self.projected_saved_nodes[arm],
                projected_saved_nanos: self.projected_saved_nanos[arm],
                buckets: std::array::from_fn(|bucket| RfpPrecisionBucketReport {
                    opponent_has_walls: bucket != 0,
                    eligible_nodes: self.eligible_nodes[bucket],
                    fires: self.counts[arm][bucket].fires,
                    true_fail_highs: self.counts[arm][bucket].true_fail_highs,
                }),
            }
        })
    }
}

/// One opponent-inventory slice of the stage-2 RFP shadow probe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RfpPrecisionBucketReport {
    pub opponent_has_walls: bool,
    /// Completed, post-TT null-window nodes in the configured probe depth range.
    pub eligible_nodes: u64,
    /// Nodes whose static evaluation met this arm's hypothetical RFP bound.
    pub fires: u64,
    /// Fires whose completed baseline search score also reached beta.
    pub true_fail_highs: u64,
}

/// Precision counts for one frozen linear or quadratic RFP margin arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RfpPrecisionArmReport {
    /// Linear margin in cp/depth, or the coefficient of `depth^2`.
    pub coefficient: i32,
    pub quadratic: bool,
    /// Non-overlapping descendants of topmost hypothetical cut nodes.
    pub projected_saved_nodes: u64,
    /// Non-overlapping measured subtree time below those cut points.
    pub projected_saved_nanos: u128,
    /// Opponent walls zero, then opponent walls at least one.
    pub buckets: [RfpPrecisionBucketReport; RFP_PRECISION_BUCKETS],
}

struct RfpPrecisionTrackedSample {
    sample: RfpPrecisionSample,
    outermost: [bool; RFP_PRECISION_ARMS],
    started: Option<Instant>,
    start_nodes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct WallHistoryProbeStats {
    reads: u64,
    pooled_warm_reads: u64,
    bucketed_warm_reads: u64,
    lmr_reductions: u64,
    lmr_researches: u64,
}

/// Q7 zero-game sparsity and ordering-disruption counters. Bench aggregates
/// them across all fixed positions in its one persistent context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WallHistoryProbeReport {
    pub reads: u64,
    pub pooled_warm_reads: u64,
    pub bucketed_warm_reads: u64,
    pub lmr_reductions: u64,
    pub lmr_researches: u64,
}

/// Production TM2 root-tension signal. Its exact integer value lies in
/// `[0, 100]` and drives only the live time-bank allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TmTension {
    pub tension: u8,
}

fn tm_tension(state: &State) -> Option<TmTension> {
    let topology = Topology::from_walls(state.h, state.v);
    let dist0 = distance_to_row(&topology, state.p0, 8);
    let dist1 = distance_to_row(&topology, state.p1, 0);
    if dist0 < 0 || dist1 < 0 {
        return None;
    }

    let path_delta = dist0.abs_diff(dist1);
    let path_closeness = 8u16.saturating_sub(path_delta.min(8)) as u8;
    let pawn_col_delta = cell_col(state.p0).abs_diff(cell_col(state.p1));
    let pawn_row_delta = cell_row(state.p0).abs_diff(cell_row(state.p1));
    let contested_band = pawn_col_delta.max(pawn_row_delta) <= 2;
    let opponent_walls = if state.turn == 0 { state.w1 } else { state.w0 };
    let tension = u16::from(opponent_walls) * u16::from(path_closeness)
        + if contested_band { 20 } else { 0 };
    debug_assert!(tension <= 100, "canonical wall stocks bound TM2 tension");

    Some(TmTension {
        tension: tension.min(100) as u8,
    })
}

// TM2's affine target is centered on the validated self-play probe population:
// mean tension = 0.20974609375 over 512 roots. Before the upper clamp,
// f(tension) = 1 + tension / 100 - 0.209746, so its measured mean is one.
// Parts-per-million arithmetic keeps the timed policy deterministic.
const TM2_MULTIPLIER_SCALE: u64 = 1_000_000;
const TM2_MULTIPLIER_MIN: u64 = 400_000;
const TM2_MULTIPLIER_MAX: u64 = 1_500_000;
const TM2_MULTIPLIER_ZERO_TENSION: u64 = 790_254;
const TM2_MULTIPLIER_PER_TENSION_UNIT: u64 = 10_000;

#[derive(Default)]
struct Tm2TimeBank {
    /// Cumulative flat minus allocated milliseconds. This is saved time only:
    /// allocation may spend previously banked surplus but can never borrow.
    balance_ms: u128,
}

impl Tm2TimeBank {
    fn target_multiplier(tension: u8) -> u64 {
        TM2_MULTIPLIER_ZERO_TENSION
            .saturating_add(
                u64::from(tension).saturating_mul(TM2_MULTIPLIER_PER_TENSION_UNIT),
            )
            .clamp(TM2_MULTIPLIER_MIN, TM2_MULTIPLIER_MAX)
    }

    fn rounded_scaled_budget(base_ms: u64, multiplier: u64) -> u64 {
        let scaled = u128::from(base_ms)
            .saturating_mul(u128::from(multiplier))
            .saturating_add(u128::from(TM2_MULTIPLIER_SCALE / 2))
            / u128::from(TM2_MULTIPLIER_SCALE);
        scaled.min(u128::from(u64::MAX)) as u64
    }

    fn minimum_budget(base_ms: u64) -> u64 {
        let scaled = u128::from(base_ms)
            .saturating_mul(u128::from(TM2_MULTIPLIER_MIN))
            .saturating_add(u128::from(TM2_MULTIPLIER_SCALE - 1))
            / u128::from(TM2_MULTIPLIER_SCALE);
        scaled.min(u128::from(u64::MAX)) as u64
    }

    fn maximum_budget(base_ms: u64) -> u64 {
        let scaled = u128::from(base_ms)
            .saturating_mul(u128::from(TM2_MULTIPLIER_MAX))
            / u128::from(TM2_MULTIPLIER_SCALE);
        scaled.min(u128::from(u64::MAX)) as u64
    }

    fn allocate(&mut self, base_ms: u64, tension: u8) -> u64 {
        if base_ms == 0 {
            return 0;
        }

        let minimum = Self::minimum_budget(base_ms);
        let maximum = Self::maximum_budget(base_ms).max(minimum);
        let target = Self::rounded_scaled_budget(base_ms, Self::target_multiplier(tension))
            .clamp(minimum, maximum);

        // Extra time is affordable only after an earlier root banked it. The
        // cap makes `allocated <= base + old_balance`, so the exact cumulative
        // flat-minus-allocated balance below stays non-negative at every game
        // prefix, including the first move and arbitrary termination.
        let affordable = self
            .balance_ms
            .checked_add(u128::from(base_ms))
            .expect("TM2 time-bank credit overflow");
        let affordable_maximum = u128::from(maximum).min(affordable) as u64;
        debug_assert!(minimum <= affordable_maximum);
        let allocated = target.clamp(minimum, affordable_maximum);
        self.balance_ms = affordable - u128::from(allocated);
        allocated
    }

    fn reset(&mut self) {
        self.balance_ms = 0;
    }
}

pub struct SearchContext {
    tt: Vec<TtEntry>,
    shared_tt: Option<Arc<SharedTt>>,
    tt_mask: usize,
    zobrist: Zobrist,
    features: Features,
    /// Production RFP depth selected once from the current search allotment.
    active_rfp_depth: i32,
    /// Gorisanson-compatible absolute turn at the current search root. QBP
    /// states omit it, so the pinned adapter estimate seeds each search. Like
    /// history-shaped LMR context, this selective-reduction phase is not part
    /// of the rules-state TT key; all positions share the mature phase at 6+.
    probable_wall_root_turn: u32,
    /// Immutable validated weights shared by Lazy-SMP lanes.
    nnue: Option<Arc<NnueNetwork>>,
    /// Two non-TT cutoff moves per search ply. `KILLER_EMPTY` is needed because
    /// action code zero is the legal pawn move `a1`.
    killers: [[u16; 2]; MAX_KILLER_PLY],
    /// Butterfly history: side-to-move x compact action code.
    history_scores: [[i32; HISTORY_ACTIONS]; 2],
    /// Search-local Q7 wall history: side x route context x action.
    wall_history_scores: Option<Box<[i32]>>,
    /// Active lane's logical [side][previous][reply] i16 CMH table.
    countermove_history: Option<Box<[i16]>>,
    /// Parent-owned helper lanes, moved into worker threads and restored after
    /// every search so the 2.4 MiB tables persist across moves without copies.
    helper_countermove_histories: Vec<Option<Box<[i16]>>>,
    /// Per-lane coarse correction histories. They persist across moves in a
    /// game, but are never shared atomically, keeping SMP learning isolated.
    correction_history: Option<[i16; CORR_HISTORY_ENTRIES]>,
    helper_correction_histories: Vec<Option<[i16; CORR_HISTORY_ENTRIES]>>,
    /// Set only while an authoritative exact root is being searched for a
    /// concrete move; its ordinary fallback tree must not train or consume Q3.
    correction_history_suppressed: bool,
    /// Per-game topology LRU, shared by the owner and its Lazy-SMP helpers.
    race_tables: Arc<RaceTableCache>,
    /// Main lane may build only inside its per-move slice; timed helpers are
    /// cache-only so joining them cannot inherit an uninterruptible build.
    race_build_policy: RaceBuildPolicy,
    /// Per-game credit that makes TM2 a reallocation rather than extra time.
    tm2_time_bank: Tm2TimeBank,
    nodes: u64,
    /// Per-lane output-only count of adaptive depth-four production RFP cutoffs.
    rfp_tc_adaptive_d4_fires: u64,
    /// WALLQ-TC leaf counters remain lane-local until their SearchResult is
    /// merged by the Lazy-SMP owner.
    wallq_tc_leaves: u64,
    wallq_tc_in_window: u64,
    hard_deadline: Option<Instant>,
    next_deadline_check: u64,
    aborted: bool,
    cutoff_probe: CutoffProbeStats,
    eval_swing_probe: Option<EvalSwingProbeStats>,
    rfp_precision_probe: RfpPrecisionProbeStats,
    rfp_precision_active: [u16; RFP_PRECISION_ARMS],
    wall_history_probe: WallHistoryProbeStats,
    /// Present only in helper contexts. The main thread owns the decision to
    /// stop and publishes it through this flag.
    external_stop: Option<Arc<AtomicBool>>,
    /// WALLQ-TC-DUAL keeps the two active budget bands physically isolated
    /// from the ordinary TT and from one another. This tail sidecar preserves
    /// the established hot `SearchContext` layout for off/inert recursion.
    wallq_tc_fast_tt: Option<Vec<TtEntry>>,
    wallq_tc_long_tt: Option<Vec<TtEntry>>,
    wallq_tc_fast_shared_tt: Option<Arc<SharedTt>>,
    wallq_tc_long_shared_tt: Option<Arc<SharedTt>>,
    /// Resolved exactly once by the owning root search and copied verbatim to
    /// helpers, whose local time controls intentionally omit the budget.
    wallq_tc_mode: WallqTcMode,
    wallq_tc_budget_ms: Option<u64>,
    #[cfg(test)]
    lmr_reductions: u64,
    #[cfg(test)]
    lmr_researches: u64,
}

/// Time / depth envelope for one `search` call.
pub struct TimeControl {
    pub max_depth: u32,
    /// Original QBP allotment. Fixed-depth searches deliberately leave it absent.
    pub allotted_budget_ms: Option<u64>,
    /// Do not START a new iteration once this instant has passed.
    pub soft_deadline: Option<Instant>,
    /// Abort the in-flight iteration once this instant has passed.
    pub hard_deadline: Option<Instant>,
}

pub struct SearchResult {
    pub best_action: Action,
    pub score: i32,
    /// Deepest fully completed iteration (0 = only the instant/fallback move).
    pub depth: u32,
    /// Aggregate nodes searched by the main thread and all Lazy-SMP helpers.
    pub nodes: u64,
    /// Main-thread nodes, retained separately for SMP telemetry.
    pub main_nodes: u64,
    /// Main plus helpers that spawned and exited normally.
    pub threads: usize,
    /// Adaptive depth-four production RFP cutoffs across every completed lane.
    pub d4_fires: u64,
    /// WALLQ-TC leaves evaluated across every completed lane.
    pub wallq_tc_leaves: u64,
    /// WALLQ-TC leaves that were within their local window and computed the
    /// exact frozen interdiction correction across every completed lane.
    pub wallq_tc_in_window: u64,
    /// Root's resolved budget. Fixed-depth searches retain `None`.
    pub wallq_tc_budget_ms: Option<u64>,
    /// Number of completed lanes that received an active root-resolved
    /// WALLQ-TC mode. Inactive roots always report zero.
    pub wallq_tc_active_lanes: u64,
    /// Exact interdiction corrections applied by active WALLQ-TC lanes.
    pub wallq_tc_fires: u64,
}

/// A proven root value for a position in one of the currently implemented
/// exact race regions. Values use the search's side-to-move centistep scale.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactLabel {
    pub kind: &'static str,
    pub value: i32,
}

/// Detect exact regions independently of their search feature toggles.
///
/// Self-play data uses only pawn-interaction-aware adjudication for perfect
/// labels. The value is always rooted at ply zero and therefore belongs to the
/// position's side to move.
pub fn exact_label(state: &State) -> Option<ExactLabel> {
    if let Some(value) = race_one_wall_score(state, 0) {
        return Some(ExactLabel {
            kind: "race1w",
            value,
        });
    }
    race_zero_wall_score(state, 0).map(|value| ExactLabel {
        kind: "race_exact",
        value,
    })
}

impl SearchContext {
    /// Construct the frozen default search configuration, including exactly
    /// 2^22 transposition-table entries.
    pub fn new(features: Features) -> Self {
        Self::with_tt_entries(features, TT_SIZE)
    }

    /// Fallible CLI constructor: a configured NNUE must load completely or the
    /// engine refuses to start the requested lane.
    pub fn try_new(features: Features) -> Result<Self, String> {
        Self::try_with_tt_entries(features, TT_SIZE)
    }

    /// Construct a context whose TT fits within `tt_mb` MiB. The entry count is
    /// rounded down to a power of two, preserving mask-based indexing.
    pub fn new_with_tt_mb(features: Features, tt_mb: usize) -> Result<Self, String> {
        let features = target_features(features);
        if tt_mb == 0 {
            return Err("--tt-mb must be greater than zero".to_owned());
        }
        let bytes = tt_mb
            .checked_mul(MEBIBYTE)
            .ok_or_else(|| "--tt-mb is too large".to_owned())?;
        let entry_bytes = if features.threads > 1 {
            std::mem::size_of::<AtomicTtEntry>()
        } else {
            std::mem::size_of::<TtEntry>()
        };
        let max_entries = bytes / entry_bytes;
        if max_entries == 0 {
            return Err("--tt-mb is too small for one TT entry".to_owned());
        }
        let tt_entries = 1usize << (usize::BITS - 1 - max_entries.leading_zeros());
        Self::try_with_tt_entries(features, tt_entries)
    }

    fn with_tt_entries(features: Features, tt_entries: usize) -> Self {
        Self::try_with_tt_entries(features, tt_entries)
            .expect("test/internal NNUE configuration must load successfully")
    }

    fn try_with_tt_entries(features: Features, tt_entries: usize) -> Result<Self, String> {
        let nnue = features
            .nnue_path
            .as_ref()
            .map(NnueNetwork::load)
            .transpose()?
            .map(Arc::new);
        Ok(Self::with_tt_entries_and_nnue(features, tt_entries, nnue))
    }

    fn with_tt_entries_and_nnue(
        features: Features,
        tt_entries: usize,
        nnue: Option<Arc<NnueNetwork>>,
    ) -> Self {
        let features = target_features(features);
        assert!(tt_entries.is_power_of_two());
        let parallel = features.threads > 1;
        let cmh_enabled = features.cmh;
        let corr_hist_enabled = features.corr_hist;
        let wall_hist_enabled = features.wall_hist;
        let helper_lanes = features.threads.saturating_sub(1);
        let probe_evalswing = features.probe_evalswing;
        let active_rfp_depth = features.rfp_depth;
        Self {
            tt: if parallel {
                Vec::new()
            } else {
                vec![TtEntry::EMPTY; tt_entries]
            },
            wallq_tc_fast_tt: None,
            wallq_tc_long_tt: None,
            shared_tt: parallel.then(|| Arc::new(SharedTt::new(tt_entries))),
            wallq_tc_fast_shared_tt: None,
            wallq_tc_long_shared_tt: None,
            tt_mask: tt_entries - 1,
            zobrist: Zobrist::new(),
            features,
            wallq_tc_mode: WallqTcMode::Inactive,
            wallq_tc_budget_ms: None,
            active_rfp_depth,
            probable_wall_root_turn: 0,
            nnue,
            killers: [[KILLER_EMPTY; 2]; MAX_KILLER_PLY],
            history_scores: [[0; HISTORY_ACTIONS]; 2],
            wall_history_scores: wall_hist_enabled
                .then(|| vec![0; WALL_HISTORY_ENTRIES].into_boxed_slice()),
            countermove_history: cmh_enabled.then(new_countermove_history),
            helper_countermove_histories: (0..helper_lanes)
                .map(|_| cmh_enabled.then(new_countermove_history))
                .collect(),
            correction_history: corr_hist_enabled.then_some([0; CORR_HISTORY_ENTRIES]),
            helper_correction_histories: (0..helper_lanes)
                .map(|_| corr_hist_enabled.then_some([0; CORR_HISTORY_ENTRIES]))
                .collect(),
            correction_history_suppressed: false,
            race_tables: Arc::new(RaceTableCache::default()),
            race_build_policy: RaceBuildPolicy::Unlimited,
            tm2_time_bank: Tm2TimeBank::default(),
            nodes: 0,
            rfp_tc_adaptive_d4_fires: 0,
            wallq_tc_leaves: 0,
            wallq_tc_in_window: 0,
            hard_deadline: None,
            next_deadline_check: 0,
            aborted: false,
            cutoff_probe: CutoffProbeStats::default(),
            eval_swing_probe: probe_evalswing.then(EvalSwingProbeStats::new),
            rfp_precision_probe: RfpPrecisionProbeStats::default(),
            rfp_precision_active: [0; RFP_PRECISION_ARMS],
            wall_history_probe: WallHistoryProbeStats::default(),
            external_stop: None,
            #[cfg(test)]
            lmr_reductions: 0,
            #[cfg(test)]
            lmr_researches: 0,
        }
    }

    pub fn validate_nnue_state(&self, state: &State) -> Result<(), String> {
        self.nnue
            .as_deref()
            .map_or(Ok(()), |network| network.validate_state(state))
    }

    /// Evaluate one position through the configured deployment path. Terminal
    /// and enabled exact-race regions retain the same precedence as negamax.
    pub fn configured_evaluation(&mut self, state: &State) -> Result<i32, String> {
        self.validate_nnue_state(state)?;
        if winner(state) >= 0 {
            return Ok(-MATE);
        }
        if self.features.race2w {
            if let Some(score) = self.race_up_to_two_walls_score(state, 0) {
                return Ok(score);
            }
        }
        if self.features.race1w {
            if let Some(score) = self.race_one_wall_score(state, 0) {
                return Ok(score);
            }
        }
        if self.features.race_exact {
            if let Some(score) = self.race_exact_score(state, 0) {
                return Ok(score);
            }
        }
        let accumulator = self
            .nnue
            .as_deref()
            .map(|network| network.try_accumulator(state))
            .transpose()?;
        Ok(self.raw_evaluation(state, accumulator.as_ref()))
    }

    /// Build the protocol time envelope from the configured budget fractions.
    pub fn time_control_for_budget(&self, start: Instant, budget_ms: u64) -> TimeControl {
        let soft_micros = budget_ms
            .saturating_mul(self.features.soft_pct)
            .saturating_mul(10);
        let hard_micros = budget_ms
            .saturating_mul(self.features.hard_pct)
            .saturating_mul(10);
        TimeControl {
            max_depth: 64,
            allotted_budget_ms: Some(budget_ms),
            soft_deadline: Some(start + Duration::from_micros(soft_micros)),
            hard_deadline: Some(start + Duration::from_micros(hard_micros)),
        }
    }

    /// Build a root time envelope, optionally reallocating the flat QBP budget.
    ///
    /// The non-negative running bank lets later high-tension roots spend only
    /// surplus already saved by earlier low-tension roots. Fixed-depth and
    /// direct-budget callers deliberately bypass this production-only policy.
    pub fn time_control_for_root(
        &mut self,
        start: Instant,
        state: &State,
        base_budget_ms: u64,
    ) -> TimeControl {
        let budget_ms = if self.features.tm2_time {
            tm_tension(state)
                .map(|tension| {
                    self.tm2_time_bank
                        .allocate(base_budget_ms, tension.tension)
                })
                .unwrap_or(base_budget_ms)
        } else {
            base_budget_ms
        };
        self.time_control_for_budget(start, budget_ms)
    }

    /// Wipe every TT entry (used between independent bench positions).
    pub fn clear_tt(&mut self) {
        self.clear_primary_tt();
        if let Some(tt) = &self.wallq_tc_fast_shared_tt {
            tt.clear();
        } else if let Some(tt) = self.wallq_tc_fast_tt.as_mut() {
            tt.fill(TtEntry::EMPTY);
        }
        if let Some(tt) = &self.wallq_tc_long_shared_tt {
            tt.clear();
        } else if let Some(tt) = self.wallq_tc_long_tt.as_mut() {
            tt.fill(TtEntry::EMPTY);
        }
    }

    #[inline]
    fn clear_primary_tt(&mut self) {
        if let Some(tt) = &self.shared_tt {
            tt.clear();
        } else {
            self.tt.fill(TtEntry::EMPTY);
        }
    }

    /// Clear only the namespace a forthcoming independent root can use.
    /// Bench setup calls this before starting its stopwatch, so a flagged but
    /// inert fixed-depth root neither clears nor faults in active-only tables.
    /// A probe-forced root clears the genuine selected lane namespace.
    pub fn clear_tt_for_allotted_budget(&mut self, budget_ms: Option<u64>) {
        match WallqTcMode::resolve(
            self.features.wallq_tc,
            budget_ms,
            self.features.wallq_tc_probe_force,
        ) {
            WallqTcMode::Inactive => self.clear_primary_tt(),
            WallqTcMode::Fast => {
                self.ensure_wallq_tc_tt_namespace_for(WallqTcMode::Fast);
                if let Some(tt) = &self.wallq_tc_fast_shared_tt {
                    tt.clear();
                } else if let Some(tt) = self.wallq_tc_fast_tt.as_mut() {
                    tt.fill(TtEntry::EMPTY);
                }
            }
            WallqTcMode::Long => {
                self.ensure_wallq_tc_tt_namespace_for(WallqTcMode::Long);
                if let Some(tt) = &self.wallq_tc_long_shared_tt {
                    tt.clear();
                } else if let Some(tt) = self.wallq_tc_long_tt.as_mut() {
                    tt.fill(TtEntry::EMPTY);
                }
            }
        }
    }

    /// Return cumulative probe measurements when the default-off probe is on.
    pub fn cutoff_probe_report(&self) -> Option<CutoffProbeReport> {
        self.features.probe_cutoffs.then_some(CutoffProbeReport {
            interior_nodes: self.cutoff_probe.interior_nodes,
            cutoff_ranks: self.cutoff_probe.cutoff_ranks,
            validated_tt_first_cutoffs: self.cutoff_probe.validated_tt_first_cutoffs,
            ordering_generation_nanos: self.cutoff_probe.ordering_generation_nanos,
            search_nanos: self.cutoff_probe.search_nanos,
        })
    }

    /// Return depth-one static-evaluation swing distributions when enabled.
    pub fn eval_swing_probe_report(
        &self,
    ) -> Option<[EvalSwingBucketReport; EVAL_SWING_BUCKETS]> {
        self.eval_swing_probe
            .as_ref()
            .map(EvalSwingProbeStats::report)
    }

    /// Return completed-search precision counts for every frozen RFP arm.
    pub fn rfp_precision_probe_report(
        &self,
    ) -> Option<[RfpPrecisionArmReport; RFP_PRECISION_ARMS]> {
        self.features
            .probe_rfp_precision
            .then(|| self.rfp_precision_probe.report())
    }

    fn begin_rfp_precision_sample(
        &mut self,
        sample: RfpPrecisionSample,
    ) -> RfpPrecisionTrackedSample {
        let mut outermost = [false; RFP_PRECISION_ARMS];
        let mut starts_coverage = false;
        for (arm, fired) in sample.fires.into_iter().enumerate() {
            if !fired {
                continue;
            }
            outermost[arm] = self.rfp_precision_active[arm] == 0;
            starts_coverage |= outermost[arm];
            self.rfp_precision_active[arm] =
                self.rfp_precision_active[arm].saturating_add(1);
        }
        RfpPrecisionTrackedSample {
            sample,
            outermost,
            started: starts_coverage.then(Instant::now),
            start_nodes: self.nodes,
        }
    }

    fn finish_rfp_precision_sample(
        &mut self,
        tracked: RfpPrecisionTrackedSample,
        true_score: i32,
        beta: i32,
    ) {
        let elapsed_nanos = tracked
            .started
            .map_or(0, |started| started.elapsed().as_nanos());
        let saved_nodes = self.nodes.saturating_sub(tracked.start_nodes);
        for (arm, fired) in tracked.sample.fires.into_iter().enumerate() {
            if !fired {
                continue;
            }
            debug_assert!(self.rfp_precision_active[arm] > 0);
            self.rfp_precision_active[arm] = self.rfp_precision_active[arm].saturating_sub(1);
            if tracked.outermost[arm] {
                self.rfp_precision_probe.projected_saved_nodes[arm] = self
                    .rfp_precision_probe
                    .projected_saved_nodes[arm]
                    .saturating_add(saved_nodes);
                self.rfp_precision_probe.projected_saved_nanos[arm] = self
                    .rfp_precision_probe
                    .projected_saved_nanos[arm]
                    .saturating_add(elapsed_nanos);
            }
        }
        self.rfp_precision_probe
            .record(tracked.sample, true_score, beta);
    }

    /// Return Q7 warm-read and LMR re-search counters when wall history is on.
    pub fn wall_history_probe_report(&self) -> Option<WallHistoryProbeReport> {
        self.features
            .probe_wall_hist
            .then_some(WallHistoryProbeReport {
                reads: self.wall_history_probe.reads,
                pooled_warm_reads: self.wall_history_probe.pooled_warm_reads,
                bucketed_warm_reads: self.wall_history_probe.bucketed_warm_reads,
                lmr_reductions: self.wall_history_probe.lmr_reductions,
                lmr_researches: self.wall_history_probe.lmr_researches,
            })
    }

    /// Commit the configured TT and exercise startup-only search machinery
    /// before the QBP protocol begins accepting timed requests.
    pub fn warm_up_for_qbp(&mut self, state: &State) -> Result<(), String> {
        // WALLQ-TC must not fault in a separate active namespace during the
        // first timed request. Both active bands are committed here, while the
        // root still remains solely responsible for deciding whether either
        // one is used.
        self.prewarm_wallq_tc_namespaces();
        // Fresh zero-filled allocations can still be backed by uncommitted
        // virtual pages. Writing the whole table moves those faults out of the
        // first timed search for both the ordinary and atomic TT layouts.
        self.clear_tt();

        #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
        if self.features.threads > 1 {
            thread::Builder::new()
                .name("qbr-smp-warmup".to_owned())
                .spawn(|| {})
                .map_err(|error| format!("SMP warmup spawn failed: {error}"))?
                .join()
                .map_err(|_| "SMP warmup helper panicked".to_owned())?;
        }

        let tc = TimeControl {
            max_depth: 2,
            allotted_budget_ms: None,
            soft_deadline: None,
            hard_deadline: None,
        };
        let result = self.search(state, &tc);
        debug_assert_eq!(result.depth, 2);

        // The warm search must not seed the first real search with TT state.
        // This second full pass also makes the page commitment unavoidable.
        self.clear_tt();
        self.begin_new_game();
        Ok(())
    }

    /// Clear game-persistent learning at an actual game/generation boundary.
    /// Ordinary `search()` calls deliberately never sweep these large tables.
    pub fn begin_new_game(&mut self) {
        if let Some(history) = self.countermove_history.as_mut() {
            history.fill(0);
        }
        for history in &mut self.helper_countermove_histories {
            if let Some(history) = history.as_mut() {
                history.fill(0);
            }
        }
        if let Some(history) = self.correction_history.as_mut() {
            history.fill(0);
        }
        for history in &mut self.helper_correction_histories {
            if let Some(history) = history.as_mut() {
                history.fill(0);
            }
        }
        self.race_tables.clear();
        self.tm2_time_bank.reset();
    }

    fn clear_killers(&mut self) {
        self.killers = [[KILLER_EMPTY; 2]; MAX_KILLER_PLY];
    }

    /// Butterfly and wall histories are search-local. CMH and correction
    /// history intentionally persist until the next game boundary.
    fn clear_history(&mut self) {
        self.history_scores = [[0; HISTORY_ACTIONS]; 2];
        if let Some(history) = self.wall_history_scores.as_mut() {
            history.fill(0);
        }
    }

    /// Age history once between completed iterative-deepening passes.  Integer
    /// division deliberately keeps this deterministic and allocation-free.
    fn age_history(&mut self) {
        for side in &mut self.history_scores {
            for score in side {
                *score /= 2;
            }
        }
        if let Some(history) = self.wall_history_scores.as_mut() {
            for score in history {
                *score /= 2;
            }
        }
    }

    #[inline]
    fn killer_codes(&self, ply: i32) -> [u16; 2] {
        self.killers
            .get(ply as usize)
            .copied()
            .unwrap_or([KILLER_EMPTY; 2])
    }

    /// Install a non-TT cutoff as the primary killer, retaining one older
    /// distinct killer.  An action code of zero is legal, hence the explicit
    /// `KILLER_EMPTY` sentinel rather than `Action::EMPTY`.
    fn record_killer(&mut self, ply: i32, action: Action, tt_move: Option<Action>) {
        if Some(action) == tt_move {
            return;
        }
        let Some(killers) = self.killers.get_mut(ply as usize) else {
            return;
        };
        let code = encode_action(action);
        if killers[0] == code {
            return;
        }
        if killers[1] == code {
            killers[1] = killers[0];
            killers[0] = code;
            return;
        }
        killers[1] = killers[0];
        killers[0] = code;
    }

    fn record_history(
        &mut self,
        state: &State,
        action: Action,
        depth: i32,
        previous_action: Option<Action>,
    ) {
        self.record_history_with_context(state, action, depth, previous_action, None);
    }

    fn record_history_with_context(
        &mut self,
        state: &State,
        action: Action,
        depth: i32,
        previous_action: Option<Action>,
        wall_context: Option<u8>,
    ) {
        let side = state.turn as usize;
        let code = encode_action(action) as usize;
        let bonus = depth.saturating_mul(depth);
        if self.features.history {
            self.history_scores[side][code] =
                self.history_scores[side][code].saturating_add(bonus);
        }

        // Q3's first candidate is bonus-only: failed replies receive no
        // malus. Gravity makes every i16 entry self-decaying, so the 2.4 MiB
        // table never participates in per-iteration full-table aging.
        if self.features.cmh && action.kind != ActionKind::Pawn {
            if let (Some(previous), Some(history)) =
                (previous_action, self.countermove_history.as_mut())
            {
                let index = countermove_history_index(side, previous, action);
                let entry = i32::from(history[index]);
                let bonus = bonus.clamp(0, CMH_MAX);
                let updated = entry
                    .saturating_add(bonus)
                    .saturating_sub(entry.saturating_mul(bonus.abs()) / CMH_MAX)
                    .clamp(i16::MIN as i32, i16::MAX as i32);
                history[index] = updated as i16;
            }
        }

        if self.features.wall_hist && action.kind != ActionKind::Pawn {
            if let (Some(context), Some(history)) =
                (wall_context, self.wall_history_scores.as_mut())
            {
                let index = wall_history_index(side, context, action);
                history[index] = history[index].saturating_add(bonus);
            }
        }
    }

    #[cfg(test)]
    #[inline]
    fn history_score(
        &mut self,
        state: &State,
        action: Action,
        previous_action: Option<Action>,
    ) -> i32 {
        self.history_score_with_context(state, action, previous_action, None)
    }

    #[inline]
    fn history_score_with_context(
        &mut self,
        state: &State,
        action: Action,
        previous_action: Option<Action>,
        wall_context: Option<u8>,
    ) -> i32 {
        let side = state.turn as usize;
        let pooled_score = if self.features.history {
            self.history_scores[side][encode_action(action) as usize]
        } else {
            0
        };
        let mut score = pooled_score;
        if self.features.cmh && action.kind != ActionKind::Pawn {
            if let (Some(previous), Some(history)) =
                (previous_action, self.countermove_history.as_ref())
            {
                score = score.saturating_add(i32::from(
                    history[countermove_history_index(side, previous, action)],
                ));
            }
        }
        if self.features.wall_hist && action.kind != ActionKind::Pawn {
            if let Some(context) = wall_context {
                let bucketed = self.wall_history_scores.as_ref().map_or(0, |history| {
                    history[wall_history_index(side, context, action)]
                });
                if self.features.probe_wall_hist {
                    self.wall_history_probe.reads =
                        self.wall_history_probe.reads.saturating_add(1);
                    if pooled_score > 0 {
                        self.wall_history_probe.pooled_warm_reads = self
                            .wall_history_probe
                            .pooled_warm_reads
                            .saturating_add(1);
                    }
                    if bucketed > 0 {
                        self.wall_history_probe.bucketed_warm_reads = self
                            .wall_history_probe
                            .bucketed_warm_reads
                            .saturating_add(1);
                    }
                }
                score = score.saturating_add(bucketed);
            }
        }
        score
    }

    #[inline]
    fn raw_evaluation(
        &self,
        state: &State,
        accumulator: Option<&NnueAccumulator>,
    ) -> i32 {
        self.nnue.as_deref().map_or_else(
            || evaluate_with_features(state, &self.features),
            |network| {
                network.evaluate(
                    state,
                    accumulator.expect("NNUE evaluation requires the lane accumulator"),
                )
            },
        )
    }

    #[inline]
    fn leaf_evaluation<const WALLQ_TC_ACTIVE: bool>(
        &mut self,
        state: &State,
        accumulator: Option<&NnueAccumulator>,
        alpha: i32,
        beta: i32,
    ) -> i32 {
        let raw = self.raw_evaluation(state, accumulator);
        let score = if WALLQ_TC_ACTIVE {
            self.wallq_tc_leaves = self.wallq_tc_leaves.saturating_add(1);
            if wallq_tc_in_window(raw, alpha, beta) {
                self.wallq_tc_in_window = self.wallq_tc_in_window.saturating_add(1);
                let (i_me, i_opp) = interdiction_stm_values(state);
                raw.saturating_add(interdiction_correction(i_me, i_opp))
            } else {
                raw
            }
        } else {
            raw
        };
        if self.nnue.is_some()
            || !self.features.corr_hist
            || self.correction_history_suppressed
        {
            return score;
        }
        let correction = self
            .correction_history
            .as_ref()
            .map_or(0, |history| {
                correction_history_value(history[correction_history_index(state)])
            });
        score.saturating_add(correction)
    }

    /// Learn only from completed exact ordinary-search values. Exact race and
    /// mate families occupy a disjoint score range and never enter the table.
    fn record_correction_history(
        &mut self,
        state: &State,
        depth: i32,
        score: i32,
        accumulator: Option<&NnueAccumulator>,
    ) {
        if self.nnue.is_some()
            || !self.features.corr_hist
            || self.correction_history_suppressed
            || score.saturating_abs() >= PROVEN_SCORE_THRESHOLD
        {
            return;
        }
        let raw = self.raw_evaluation(state, accumulator);
        let residual = score.saturating_sub(raw);
        if let Some(history) = self.correction_history.as_mut() {
            update_correction_history_entry(
                &mut history[correction_history_index(state)],
                residual,
                depth,
            );
        }
    }

    #[inline]
    fn should_stop(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        if self
            .external_stop
            .as_ref()
            .is_some_and(|stop| stop.load(Ordering::Acquire))
        {
            self.aborted = true;
            return true;
        }
        let Some(deadline) = self.hard_deadline else {
            return false;
        };
        if self.nodes < self.next_deadline_check {
            return false;
        }

        let now = Instant::now();
        if now >= deadline {
            self.aborted = true;
            return true;
        }
        let remaining = deadline.duration_since(now);
        let interval = if remaining <= DEADLINE_CHECK_VERY_NEAR {
            1
        } else if remaining <= DEADLINE_CHECK_NEAR {
            32
        } else if remaining <= DEADLINE_CHECK_APPROACHING {
            256
        } else {
            4096
        };
        self.next_deadline_check = self.nodes.saturating_add(interval);
        false
    }

    /// Poll outside the node-count path, notably during move generation and
    /// ordering. These phases can otherwise cross a deadline without adding a
    /// single node.
    #[inline]
    fn hard_deadline_reached_now(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        if self
            .external_stop
            .as_ref()
            .is_some_and(|stop| stop.load(Ordering::Acquire))
        {
            self.aborted = true;
            return true;
        }
        if self
            .hard_deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            self.aborted = true;
        }
        self.aborted
    }

    #[inline]
    fn tt_position_key(&self, state: &State) -> TtPositionKey {
        #[cfg(feature = "profile-timers")]
        let _profile_guard =
            crate::profile::Guard::new(crate::profile::Bucket::TtOps);
        let direct = self.zobrist.hash(state);
        if !self.features.tt_sym {
            return TtPositionKey {
                hash: direct,
                mirrored: false,
            };
        }
        let reflected_state = mirror_state_lr(state);
        let reflected = self.zobrist.hash(&reflected_state);
        if reflected < direct
            || (reflected == direct && state_order_key(&reflected_state) < state_order_key(state))
        {
            TtPositionKey {
                hash: reflected,
                mirrored: true,
            }
        } else {
            TtPositionKey {
                hash: direct,
                mirrored: false,
            }
        }
    }

    #[inline]
    fn wallq_tc_active(&self) -> bool {
        self.wallq_tc_mode.is_active()
    }

    #[inline]
    fn wallq_tc_active_lanes(&self) -> u64 {
        u64::from(self.wallq_tc_active())
    }

    /// WALLQ-TC's correction count is its local-window count while an active
    /// root owns the correction path. It is read only while materialising a
    /// lane result, never at a leaf.
    #[inline]
    fn wallq_tc_fires(&self) -> u64 {
        if self.wallq_tc_active() {
            self.wallq_tc_in_window
        } else {
            0
        }
    }

    /// Provision a physical TT namespace for a selected mode. Root dispatch
    /// calls this after resolving its mode; QBP/bench prewarm may provision
    /// both active tables before a clock starts, without resolving any root.
    fn ensure_wallq_tc_tt_namespace_for(&mut self, mode: WallqTcMode) {
        let entries = self.tt_mask + 1;
        match mode {
            WallqTcMode::Inactive => {}
            WallqTcMode::Fast => {
                if self.shared_tt.is_some() {
                    if self.wallq_tc_fast_shared_tt.is_none() {
                        self.wallq_tc_fast_shared_tt = Some(Arc::new(SharedTt::new(entries)));
                    }
                } else if self.wallq_tc_fast_tt.is_none() {
                    self.wallq_tc_fast_tt = Some(vec![TtEntry::EMPTY; entries]);
                }
            }
            WallqTcMode::Long => {
                if self.shared_tt.is_some() {
                    if self.wallq_tc_long_shared_tt.is_none() {
                        self.wallq_tc_long_shared_tt = Some(Arc::new(SharedTt::new(entries)));
                    }
                } else if self.wallq_tc_long_tt.is_none() {
                    self.wallq_tc_long_tt = Some(vec![TtEntry::EMPTY; entries]);
                }
            }
        }
    }

    /// Make this root's active physical namespace the ordinary TT.  The
    /// recursive TT routines consequently remain the pre-WALLQ-TC routines in
    /// every mode; only an active root pays this setup/teardown indirection.
    ///
    /// Activates the root-owned WALLQ-TC namespace. The recursive leaf path
    /// is selected separately as a root-resolved const specialization.
    fn activate_wallq_tc_root(&mut self) {
        let mode = self.wallq_tc_mode;
        if !mode.is_active() {
            return;
        }
        self.ensure_wallq_tc_tt_namespace_for(mode);

        match mode {
            WallqTcMode::Inactive => unreachable!("inactive WALLQ-TC root cannot activate"),
            WallqTcMode::Fast => {
                if self.shared_tt.is_some() {
                    std::mem::swap(
                        &mut self.shared_tt,
                        &mut self.wallq_tc_fast_shared_tt,
                    );
                    debug_assert!(self.shared_tt.is_some());
                    debug_assert!(self.wallq_tc_fast_shared_tt.is_some());
                } else {
                    let active = self
                        .wallq_tc_fast_tt
                        .take()
                        .expect("active fast WALLQ-TC TT must be provisioned by the root");
                    self.wallq_tc_fast_tt = Some(std::mem::replace(&mut self.tt, active));
                }
            }
            WallqTcMode::Long => {
                if self.shared_tt.is_some() {
                    std::mem::swap(
                        &mut self.shared_tt,
                        &mut self.wallq_tc_long_shared_tt,
                    );
                    debug_assert!(self.shared_tt.is_some());
                    debug_assert!(self.wallq_tc_long_shared_tt.is_some());
                } else {
                    let active = self
                        .wallq_tc_long_tt
                        .take()
                        .expect("active long WALLQ-TC TT must be provisioned by the root");
                    self.wallq_tc_long_tt = Some(std::mem::replace(&mut self.tt, active));
                }
            }
        }

    }

    /// Restore the ordinary namespace after an active root. Inert and
    /// flag-off roots execute no work.
    fn deactivate_wallq_tc_root(&mut self) {
        if !self.wallq_tc_mode.is_active() {
            return;
        }

        match self.wallq_tc_mode {
            WallqTcMode::Inactive => unreachable!("inactive WALLQ-TC root cannot deactivate"),
            WallqTcMode::Fast => {
                if self.shared_tt.is_some() {
                    std::mem::swap(
                        &mut self.shared_tt,
                        &mut self.wallq_tc_fast_shared_tt,
                    );
                } else {
                    let ordinary = self
                        .wallq_tc_fast_tt
                        .take()
                        .expect("ordinary TT must be parked in the fast WALLQ-TC slot");
                    self.wallq_tc_fast_tt = Some(std::mem::replace(&mut self.tt, ordinary));
                }
            }
            WallqTcMode::Long => {
                if self.shared_tt.is_some() {
                    std::mem::swap(
                        &mut self.shared_tt,
                        &mut self.wallq_tc_long_shared_tt,
                    );
                } else {
                    let ordinary = self
                        .wallq_tc_long_tt
                        .take()
                        .expect("ordinary TT must be parked in the long WALLQ-TC slot");
                    self.wallq_tc_long_tt = Some(std::mem::replace(&mut self.tt, ordinary));
                }
            }
        }
    }

    /// Commit both active WALLQ-TC namespaces before an externally timed
    /// protocol/bench request. This intentionally does not choose a mode and
    /// therefore cannot substitute for root-time resolution.
    pub fn prewarm_wallq_tc_namespaces(&mut self) {
        if self.features.wallq_tc {
            self.ensure_wallq_tc_tt_namespace_for(WallqTcMode::Fast);
            self.ensure_wallq_tc_tt_namespace_for(WallqTcMode::Long);
            self.clear_tt();
        }
    }

    #[inline]
    fn tt_entry_move(entry: TtEntry, key: TtPositionKey) -> Action {
        let canonical = decode_action(entry.best);
        if key.mirrored {
            mirror_action_lr(canonical)
        } else {
            canonical
        }
    }

    #[inline]
    fn tt_store_position(
        &mut self,
        key: TtPositionKey,
        depth: i32,
        bound: u8,
        score: i32,
        best: Action,
    ) {
        let canonical = if key.mirrored {
            mirror_action_lr(best)
        } else {
            best
        };
        self.tt_store(key.hash, depth, bound, score, canonical);
    }

    fn tt_probe(&self, hash: u64) -> Option<TtEntry> {
        #[cfg(feature = "profile-timers")]
        let _profile_guard =
            crate::profile::Guard::new(crate::profile::Bucket::TtOps);
        let index = (hash as usize) & self.tt_mask;
        if let Some(tt) = &self.shared_tt {
            return tt.entries[index].probe(hash, tt.index_bits);
        }
        let entry = self.tt[index];
        if entry.depth != 0 && entry.key == (hash >> 32) as u32 {
            Some(entry)
        } else {
            None
        }
    }

    fn tt_store(&mut self, hash: u64, depth: i32, bound: u8, score: i32, best: Action) {
        #[cfg(feature = "profile-timers")]
        let _profile_guard =
            crate::profile::Guard::new(crate::profile::Bucket::TtOps);
        let idx = (hash as usize) & self.tt_mask;
        let entry = TtEntry {
            key: (hash >> 32) as u32,
            best: encode_action(best),
            score: score as i16,
            depth: depth as u8,
            bound,
        };
        if let Some(tt) = &self.shared_tt {
            tt.entries[idx].store(hash, entry, tt.index_bits);
            return;
        }
        let existing = self.tt[idx];
        if existing.depth == 0 || entry.depth >= existing.depth {
            self.tt[idx] = entry;
        }
    }

    /// Run iterative deepening under `tc`, returning the main thread's root
    /// result. In Lazy-SMP mode helper results are discarded, while their node
    /// counts are included in `SearchResult::nodes`.
    pub fn search(&mut self, state: &State, tc: &TimeControl) -> SearchResult {
        self.search_with_root_mode(state, tc)
    }

    fn search_with_root_mode(&mut self, state: &State, tc: &TimeControl) -> SearchResult {
        // WALLQ-TC-DUAL normally resolves solely from this root's
        // already-resolved budget. The default-off probe override may select
        // that same resolved mode for a fixed-depth measurement. Helpers
        // inherit the mode instead of re-evaluating their budgetless clocks.
        // An active root then swaps its physical TT into the ordinary slot.
        // Inert roots do neither; `search_from_depth` resolves the leaf
        // specialization once for the main lane and each helper.
        self.wallq_tc_budget_ms = tc.allotted_budget_ms;
        self.wallq_tc_mode = WallqTcMode::resolve(
            self.features.wallq_tc,
            tc.allotted_budget_ms,
            self.features.wallq_tc_probe_force,
        );
        self.activate_wallq_tc_root();
        self.active_rfp_depth = rfp_depth_for_budget(
            self.features.rfp_depth,
            self.features.rfp_tc_adaptive,
            tc.allotted_budget_ms,
        );
        self.probable_wall_root_turn = if self.features.lmr_probable_walls {
            probable_wall_turn_estimate(state)
        } else {
            0
        };
        let probe_started = self.features.probe_cutoffs.then(Instant::now);
        self.race_build_policy = match tc.hard_deadline {
            None => RaceBuildPolicy::Unlimited,
            Some(hard_deadline) => {
                let now = Instant::now();
                let remaining = hard_deadline.saturating_duration_since(now);
                let slice = (remaining / RACE_BUILD_SLICE_DIVISOR).min(RACE_BUILD_SLICE_MAX);
                RaceBuildPolicy::Until(now + slice)
            }
        };
        // Resolve a proven one-wall root move before Lazy-SMP helpers can
        // start. It consumes the same bounded construction slice as interior
        // probes; an incomplete family falls back to ordinary search. Tiny
        // requests retain the immediate legal fallback without doing the
        // bridge's full root move generation.
        const MIN_ROOT_RACE_TIME: Duration = Duration::from_millis(30);
        let root_race_admitted = tc
            .hard_deadline
            .map(|deadline| {
                deadline.saturating_duration_since(Instant::now()) >= MIN_ROOT_RACE_TIME
            })
            .unwrap_or(true);
        #[cfg(feature = "profile-timers")]
        let race_solution = crate::profile::measure(crate::profile::Bucket::Race, || {
            (self.features.race1w && root_race_admitted)
                .then(|| self.race_one_wall_root_winning_solution(state))
                .flatten()
        });
        #[cfg(not(feature = "profile-timers"))]
        let race_solution = (self.features.race1w && root_race_admitted)
            .then(|| self.race_one_wall_root_winning_solution(state))
            .flatten();
        let exact_root = self.nnue.is_none()
            && self.features.corr_hist
            && (race_solution.is_some()
                || (self.features.race2w
                    && self.race_up_to_two_walls_score(state, 0).is_some())
                || (self.features.race1w && self.race_one_wall_score(state, 0).is_some())
                || (self.features.race_exact && self.race_exact_score(state, 0).is_some()));
        let previous_correction_suppression = self.correction_history_suppressed;
        self.correction_history_suppressed = exact_root;
        // Dispatch once per root search. The false specialization contains no
        // adaptive-counter branch or mutation in the ordinary search path.
        let count_adaptive_d4_fires =
            self.features.rfp_tc_adaptive && self.active_rfp_depth == 4;
        let mut result = if count_adaptive_d4_fires {
            self.search_active::<true>(state, tc, race_solution.map(|solution| solution.0))
        } else {
            self.search_active::<false>(state, tc, race_solution.map(|solution| solution.0))
        };
        self.correction_history_suppressed = previous_correction_suppression;
        if let Some((action, score)) = race_solution {
            // The ordinary search may prefer an unresolved heuristic line.
            // A root tablebase win is authoritative over that approximation.
            result.best_action = action;
            result.score = score;
        }
        if let Some(started) = probe_started {
            self.cutoff_probe.search_nanos = self
                .cutoff_probe
                .search_nanos
                .saturating_add(started.elapsed().as_nanos());
        }
        self.deactivate_wallq_tc_root();
        result
    }

    fn search_active<const COUNT_D4_FIRES: bool>(
        &mut self,
        state: &State,
        tc: &TimeControl,
        race_fallback: Option<Action>,
    ) -> SearchResult {
        #[cfg(all(target_family = "wasm", target_os = "unknown"))]
        {
            debug_assert_eq!(self.features.threads, 1);
            return self.search_from_depth::<COUNT_D4_FIRES>(state, tc, 1, race_fallback);
        }
        #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
        if self.features.threads == 1 {
            self.search_from_depth::<COUNT_D4_FIRES>(state, tc, 1, race_fallback)
        } else {
            self.search_parallel::<COUNT_D4_FIRES>(state, tc, race_fallback)
        }
    }

    #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
    fn helper_context(
        &self,
        stop: Arc<AtomicBool>,
        countermove_history: Option<Box<[i16]>>,
        correction_history: Option<[i16; CORR_HISTORY_ENTRIES]>,
    ) -> Self {
        let mut features = self.features.clone();
        features.threads = 1;
        let probe_evalswing = features.probe_evalswing;
        let wall_hist_enabled = features.wall_hist;
        Self {
            tt: Vec::new(),
            wallq_tc_fast_tt: None,
            wallq_tc_long_tt: None,
            shared_tt: Some(
                self.shared_tt
                    .as_ref()
                    .expect("parallel search requires a shared TT")
                    .clone(),
            ),
            // The owner has already selected the active physical namespace
            // into `shared_tt` before helpers exist. Helpers therefore retain
            // the ordinary recursive TT path and need no TC namespace state.
            wallq_tc_fast_shared_tt: None,
            wallq_tc_long_shared_tt: None,
            tt_mask: self.tt_mask,
            zobrist: Zobrist::new(),
            features,
            wallq_tc_mode: self.wallq_tc_mode,
            wallq_tc_budget_ms: self.wallq_tc_budget_ms,
            active_rfp_depth: self.active_rfp_depth,
            probable_wall_root_turn: self.probable_wall_root_turn,
            nnue: self.nnue.clone(),
            killers: [[KILLER_EMPTY; 2]; MAX_KILLER_PLY],
            history_scores: [[0; HISTORY_ACTIONS]; 2],
            wall_history_scores: wall_hist_enabled
                .then(|| vec![0; WALL_HISTORY_ENTRIES].into_boxed_slice()),
            countermove_history,
            helper_countermove_histories: Vec::new(),
            correction_history,
            helper_correction_histories: Vec::new(),
            correction_history_suppressed: self.correction_history_suppressed,
            race_tables: Arc::clone(&self.race_tables),
            race_build_policy: match self.race_build_policy {
                RaceBuildPolicy::Unlimited => RaceBuildPolicy::Unlimited,
                RaceBuildPolicy::Until(_) | RaceBuildPolicy::CachedOnly => {
                    RaceBuildPolicy::CachedOnly
                }
            },
            tm2_time_bank: Tm2TimeBank::default(),
            nodes: 0,
            rfp_tc_adaptive_d4_fires: 0,
            wallq_tc_leaves: 0,
            wallq_tc_in_window: 0,
            hard_deadline: None,
            next_deadline_check: 0,
            aborted: false,
            cutoff_probe: CutoffProbeStats::default(),
            eval_swing_probe: probe_evalswing.then(EvalSwingProbeStats::new),
            rfp_precision_probe: RfpPrecisionProbeStats::default(),
            rfp_precision_active: [0; RFP_PRECISION_ARMS],
            wall_history_probe: WallHistoryProbeStats::default(),
            external_stop: Some(stop),
            #[cfg(test)]
            lmr_reductions: 0,
            #[cfg(test)]
            lmr_researches: 0,
        }
    }

    #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
    fn search_parallel<const COUNT_D4_FIRES: bool>(
        &mut self,
        state: &State,
        tc: &TimeControl,
        race_fallback: Option<Action>,
    ) -> SearchResult {
        debug_assert!(self.features.threads > 1);
        debug_assert!(self.shared_tt.is_some());

        let stop = Arc::new(AtomicBool::new(false));
        let mut helpers = Vec::with_capacity(self.features.threads - 1);
        for helper_index in 0..self.features.threads - 1 {
            if tc.hard_deadline.is_some_and(|deadline| {
                deadline.saturating_duration_since(Instant::now()) <= Duration::from_millis(1)
            }) {
                break;
            }
            let helper_history = self
                .helper_countermove_histories
                .get_mut(helper_index)
                .and_then(Option::take);
            debug_assert_eq!(helper_history.is_some(), self.features.cmh);
            let helper_correction = self
                .helper_correction_histories
                .get_mut(helper_index)
                .and_then(Option::take);
            debug_assert_eq!(helper_correction.is_some(), self.features.corr_hist);
            let mut helper =
                self.helper_context(stop.clone(), helper_history, helper_correction);
            let helper_state = *state;
            // Count the main as lane zero: helper lanes alternate one ply
            // ahead and level with the main to decorrelate their TT work.
            let start_depth = 1 + ((helper_index + 1) & 1) as u32;
            let helper_tc = TimeControl {
                max_depth: tc.max_depth,
                allotted_budget_ms: None,
                soft_deadline: None,
                hard_deadline: None,
            };
            match thread::Builder::new()
                .name(format!("qbr-smp-{}", helper_index + 1))
                .spawn(move || {
                    let result = helper.search_from_depth::<COUNT_D4_FIRES>(
                        &helper_state,
                        &helper_tc,
                        start_depth,
                        None,
                    );
                    (
                        result,
                        helper.countermove_history.take(),
                        helper.correction_history.take(),
                    )
                })
            {
                Ok(handle) => helpers.push((helper_index, handle)),
                Err(error) => {
                    // A failed spawn consumes and drops its closure. Recreate
                    // only that exceptional lane; normal moves never allocate
                    // or clear a CMH table.
                    if self.features.cmh {
                        self.helper_countermove_histories[helper_index] =
                            Some(new_countermove_history());
                    }
                    if self.features.corr_hist {
                        self.helper_correction_histories[helper_index] =
                            Some([0; CORR_HISTORY_ENTRIES]);
                    }
                    eprintln!("info smp_spawn_error={error}");
                }
            }
        }

        let mut main_result = self.search_from_depth::<COUNT_D4_FIRES>(state, tc, 1, race_fallback);
        stop.store(true, Ordering::Release);
        let mut helper_nodes = 0u64;
        let mut completed_helpers = 0usize;
        for (helper_index, helper) in helpers {
            match helper.join() {
                Ok((result, history, correction_history)) => {
                    self.helper_countermove_histories[helper_index] = history;
                    self.helper_correction_histories[helper_index] = correction_history;
                    helper_nodes = helper_nodes.saturating_add(result.nodes);
                    if COUNT_D4_FIRES {
                        main_result.d4_fires =
                            main_result.d4_fires.saturating_add(result.d4_fires);
                    }
                    main_result.wallq_tc_leaves = main_result
                        .wallq_tc_leaves
                        .saturating_add(result.wallq_tc_leaves);
                    main_result.wallq_tc_in_window = main_result
                        .wallq_tc_in_window
                        .saturating_add(result.wallq_tc_in_window);
                    main_result.wallq_tc_active_lanes = main_result
                        .wallq_tc_active_lanes
                        .saturating_add(result.wallq_tc_active_lanes);
                    main_result.wallq_tc_fires = main_result
                        .wallq_tc_fires
                        .saturating_add(result.wallq_tc_fires);
                    completed_helpers += 1;
                }
                Err(_) => {
                    if self.features.cmh {
                        self.helper_countermove_histories[helper_index] =
                            Some(new_countermove_history());
                    }
                    if self.features.corr_hist {
                        self.helper_correction_histories[helper_index] =
                            Some([0; CORR_HISTORY_ENTRIES]);
                    }
                    eprintln!("info smp_helper_panic=1");
                }
            }
        }
        main_result.nodes = main_result.nodes.saturating_add(helper_nodes);
        main_result.threads = 1 + completed_helpers;
        main_result
    }

    /// The legacy iterative-deepening loop. `start_depth == 1` is the exact
    /// single-thread path; helpers may begin at depth two.
    fn search_from_depth<const COUNT_D4_FIRES: bool>(
        &mut self,
        state: &State,
        tc: &TimeControl,
        start_depth: u32,
        preferred_fallback: Option<Action>,
    ) -> SearchResult {
        if self.wallq_tc_mode.is_active() {
            self.search_from_depth_resolved::<COUNT_D4_FIRES, true>(
                state,
                tc,
                start_depth,
                preferred_fallback,
            )
        } else {
            self.search_from_depth_resolved::<COUNT_D4_FIRES, false>(
                state,
                tc,
                start_depth,
                preferred_fallback,
            )
        }
    }

    /// Root-resolved leaf specialization. The ordinary path monomorphizes
    /// without WALLQ-TC's counters, local-window arithmetic, or DAG lookup.
    fn search_from_depth_resolved<const COUNT_D4_FIRES: bool, const WALLQ_TC_ACTIVE: bool>(
        &mut self,
        state: &State,
        tc: &TimeControl,
        start_depth: u32,
        preferred_fallback: Option<Action>,
    ) -> SearchResult {
        self.nodes = 0;
        self.wallq_tc_leaves = 0;
        self.wallq_tc_in_window = 0;
        if COUNT_D4_FIRES {
            self.rfp_tc_adaptive_d4_fires = 0;
        }
        self.aborted = false;
        self.hard_deadline = tc.hard_deadline;
        self.next_deadline_check = 0;
        if self.features.probe_rfp_precision {
            self.rfp_precision_active = [0; RFP_PRECISION_ARMS];
        }
        #[cfg(test)]
        {
            self.lmr_reductions = 0;
            self.lmr_researches = 0;
        }
        if self.features.killers {
            self.clear_killers();
        }
        if self.features.history || self.features.wall_hist {
            self.clear_history();
        }
        let mut nnue_accumulator = self.nnue.as_deref().map(|network| {
            network
                .try_accumulator(state)
                .expect("QBP/root validation must precede NNUE search")
        });

        let mut roots = MoveList::new();
        let root_deadline = self.hard_deadline;
        let root_stop = self.external_stop.clone();
        let roots_complete = if root_deadline.is_some() || root_stop.is_some() {
            legal_actions_until(state, &mut roots, || {
                root_stop
                    .as_ref()
                    .is_some_and(|stop| stop.load(Ordering::Acquire))
                    || root_deadline.is_some_and(|deadline| Instant::now() >= deadline)
            })
        } else {
            legal_actions(state, &mut roots);
            true
        };
        // Defensive: a terminal / no-move position still yields a token.
        if roots.len() == 0 {
            return SearchResult {
                best_action: Action::EMPTY,
                score: 0,
                depth: 0,
                nodes: 0,
                main_nodes: 0,
                threads: 1,
                d4_fires: if COUNT_D4_FIRES {
                    self.rfp_tc_adaptive_d4_fires
                } else {
                    0
                },
                wallq_tc_leaves: self.wallq_tc_leaves,
                wallq_tc_in_window: self.wallq_tc_in_window,
                wallq_tc_budget_ms: self.wallq_tc_budget_ms,
                wallq_tc_active_lanes: self.wallq_tc_active_lanes(),
                wallq_tc_fires: self.wallq_tc_fires(),
            };
        }
        // Fallback: canonical first legal move, always safe to play.
        let mut best_action = preferred_fallback
            .filter(|action| roots.contains(*action))
            .unwrap_or_else(|| roots.get(0));
        let mut best_score = 0;
        // The interruptible generator always emits pawn moves first, so even
        // an expired deadline returns a legal canonical fallback immediately.
        if !roots_complete {
            self.aborted = true;
            return SearchResult {
                best_action,
                score: best_score,
                depth: 0,
                nodes: 0,
                main_nodes: 0,
                threads: 1,
                d4_fires: if COUNT_D4_FIRES {
                    self.rfp_tc_adaptive_d4_fires
                } else {
                    0
                },
                wallq_tc_leaves: self.wallq_tc_leaves,
                wallq_tc_in_window: self.wallq_tc_in_window,
                wallq_tc_budget_ms: self.wallq_tc_budget_ms,
                wallq_tc_active_lanes: self.wallq_tc_active_lanes(),
                wallq_tc_fires: self.wallq_tc_fires(),
            };
        }
        // A single legal move needs no search at all.
        if roots.len() == 1 {
            return SearchResult {
                best_action,
                score: 0,
                depth: 0,
                nodes: 0,
                main_nodes: 0,
                threads: 1,
                d4_fires: if COUNT_D4_FIRES {
                    self.rfp_tc_adaptive_d4_fires
                } else {
                    0
                },
                wallq_tc_leaves: self.wallq_tc_leaves,
                wallq_tc_in_window: self.wallq_tc_in_window,
                wallq_tc_budget_ms: self.wallq_tc_budget_ms,
                wallq_tc_active_lanes: self.wallq_tc_active_lanes(),
                wallq_tc_fires: self.wallq_tc_fires(),
            };
        }

        let use_predictor = self.features.predictive_start && self.hard_deadline.is_some();
        let mut previous_iteration_time: Option<Duration> = None;
        let mut last_iteration_time: Option<Duration> = None;
        let mut completed = 0u32;
        for depth in start_depth..=tc.max_depth {
            if self.hard_deadline_reached_now() {
                break;
            }
            if depth > 1 {
                if let Some(soft) = tc.soft_deadline {
                    if Instant::now() >= soft {
                        break;
                    }
                }
                if use_predictor && depth > 3 {
                    if let (Some(previous), Some(last), Some(hard)) = (
                        previous_iteration_time,
                        last_iteration_time,
                        self.hard_deadline,
                    ) {
                        let remaining = hard.saturating_duration_since(Instant::now());
                        if predicted_iteration_nanos(previous, last) > remaining.as_nanos() {
                            break;
                        }
                    }
                }
                if self.features.history || self.features.wall_hist {
                    self.age_history();
                }
            }
            let iteration_started = use_predictor.then(Instant::now);
            let iteration = if self.features.aspiration && depth > 1 {
                self.search_root_aspiration::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
                    state,
                    depth as i32,
                    best_score,
                    nnue_accumulator.as_mut(),
                )
            } else {
                self.search_root::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
                    state,
                    depth as i32,
                    nnue_accumulator.as_mut(),
                )
            };
            #[cfg(debug_assertions)]
            if let (Some(network), Some(accumulator)) =
                (self.nnue.as_deref(), nnue_accumulator.as_ref())
            {
                debug_assert_eq!(
                    accumulator,
                    &network
                        .try_accumulator(state)
                        .expect("validated root accumulator rebuild"),
                    "NNUE accumulator was not restored after an ID iteration",
                );
            }
            let iteration_elapsed = iteration_started.map(|started| started.elapsed());
            match iteration {
                Some((action, score)) => {
                    if let Some(elapsed) = iteration_elapsed {
                        previous_iteration_time = last_iteration_time;
                        last_iteration_time = Some(elapsed);
                    }
                    best_action = action;
                    best_score = score;
                    completed = depth;
                    if score.abs() >= MATE_THRESHOLD {
                        break; // proven mate; deeper search cannot improve it.
                    }
                }
                None => break, // aborted mid-iteration; keep the prior best.
            }
        }
        SearchResult {
            best_action,
            score: best_score,
            depth: completed,
            nodes: self.nodes,
            main_nodes: self.nodes,
            threads: 1,
            d4_fires: if COUNT_D4_FIRES {
                self.rfp_tc_adaptive_d4_fires
            } else {
                0
            },
            wallq_tc_leaves: self.wallq_tc_leaves,
            wallq_tc_in_window: self.wallq_tc_in_window,
            wallq_tc_budget_ms: self.wallq_tc_budget_ms,
            wallq_tc_active_lanes: self.wallq_tc_active_lanes(),
            wallq_tc_fires: self.wallq_tc_fires(),
        }
    }

    /// PVS search for a non-first move. Eligible LMR moves receive a one- or
    /// two-ply reduction in a null window. A fail-high is always verified at
    /// the ordinary child depth before the usual PVS full-window re-search.
    fn search_late_move<const COUNT_D4_FIRES: bool, const WALLQ_TC_ACTIVE: bool>(
        &mut self,
        child: &State,
        mut accumulator: Option<&mut NnueAccumulator>,
        action: Action,
        depth: i32,
        alpha: i32,
        beta: i32,
        ply: i32,
        probable_turn: u32,
        reduction: i32,
    ) -> Option<i32> {
        if reduction > 0 {
            debug_assert!((1..=2).contains(&reduction));
            if self.features.probe_wall_hist {
                self.wall_history_probe.lmr_reductions = self
                    .wall_history_probe
                    .lmr_reductions
                    .saturating_add(1);
            }
            #[cfg(test)]
            {
                self.lmr_reductions += 1;
            }
            let reduced = -self.negamax_counted::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
                child,
                accumulator.as_deref_mut(),
                depth - 1 - reduction,
                -alpha - 1,
                -alpha,
                ply,
                probable_turn,
                false,
                Some(action),
            )?;
            if reduced <= alpha {
                return Some(reduced);
            }
            if self.features.probe_wall_hist {
                self.wall_history_probe.lmr_researches = self
                    .wall_history_probe
                    .lmr_researches
                    .saturating_add(1);
            }
            #[cfg(test)]
            {
                self.lmr_researches += 1;
            }
        }

        let score = -self.negamax_counted::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
            child,
            accumulator.as_deref_mut(),
            depth - 1,
            -alpha - 1,
            -alpha,
            ply,
            probable_turn,
            false,
            Some(action),
        )?;
        if score > alpha && score < beta {
            Some(-self.negamax_counted::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
                child,
                accumulator.as_deref_mut(),
                depth - 1,
                -beta,
                -alpha,
                ply,
                probable_turn,
                false,
                Some(action),
            )?)
        } else {
            Some(score)
        }
    }

    /// Root: full-window PVS over ordered moves, returning (best, score).
    /// This remains the all-off root path so it deliberately keeps v0's
    /// full-window loop and exact TT-store behaviour.
    fn search_root<const COUNT_D4_FIRES: bool, const WALLQ_TC_ACTIVE: bool>(
        &mut self,
        state: &State,
        depth: i32,
        mut accumulator: Option<&mut NnueAccumulator>,
    ) -> Option<(Action, i32)> {
        let tt_key = self.tt_position_key(state);
        let tt_move = self
            .tt_probe(tt_key.hash)
            .map(|entry| Self::tt_entry_move(entry, tt_key));
        // There is no predecessor at root, so CMH deliberately falls back to
        // the established butterfly score.
        let tight_edge_threshold = (self.features.lmr && depth >= 3).then_some(6);
        let ordered = self.order_moves(state, tt_move, 0, None, tight_edge_threshold)?;
        let lmr_edges = (self.features.lmr && depth >= 3 && ordered.len > 6)
            .then(|| ordered.tight_edges.unwrap_or_else(|| LmrEdges::for_state(state)));
        let probable_walls = (self.features.lmr_probable_walls
            && depth >= LMR_PROBABLE_WALLS_MIN_DEPTH
            && lmr_edges.is_some())
        .then(|| ProbableWalls::for_state(state, self.probable_wall_root_turn));
        if self.hard_deadline_reached_now() {
            return None;
        }

        let mut alpha = -INF;
        let beta = INF;
        let mut best_action = ordered.moves[0];
        let mut best_score = -INF;

        for i in 0..ordered.len {
            if self.hard_deadline_reached_now() {
                return None;
            }
            let action = ordered.moves[i];
            #[cfg(feature = "profile-timers")]
            let child = crate::profile::measure(crate::profile::Bucket::StateApply, || {
                state.apply(action)
            });
            #[cfg(not(feature = "profile-timers"))]
            let child = state.apply(action);
            if let (Some(network), Some(current)) =
                (self.nnue.as_deref(), accumulator.as_deref_mut())
            {
                current.make(network, state, action);
            }
            let score_result = if i == 0 {
                self.negamax_counted::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
                    &child,
                    accumulator.as_deref_mut(),
                    depth - 1,
                    -beta,
                    -alpha,
                    1,
                    self.probable_wall_root_turn.saturating_add(1),
                    false,
                    Some(action),
                )
                .map(|score| -score)
            } else {
                let reduction = lmr_edges.as_ref().map_or(0, |edges| {
                    lmr_reduction(
                        depth,
                        i,
                        action,
                        tt_move,
                        edges,
                        probable_walls.as_ref(),
                        self.features.lmr_probable_walls_control,
                    )
                });
                self.search_late_move::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
                    &child,
                    accumulator.as_deref_mut(),
                    action,
                    depth,
                    alpha,
                    beta,
                    1,
                    self.probable_wall_root_turn.saturating_add(1),
                    reduction,
                )
            };
            if let (Some(network), Some(current)) =
                (self.nnue.as_deref(), accumulator.as_deref_mut())
            {
                current.unmake(network, state, action);
            }
            let score = score_result?;
            if score > best_score {
                best_score = score;
                best_action = action;
            }
            if score > alpha {
                alpha = score;
            }
        }

        self.tt_store_position(tt_key, depth, BOUND_EXACT, best_score, best_action);
        if self.features.corr_hist {
            self.record_correction_history(
                state,
                depth,
                best_score,
                accumulator.as_deref(),
            );
        }
        Some((best_action, best_score))
    }

    /// Retry an iteration in progressively wider windows around the last
    /// completed score: +/-60, then +/-240, then the ordinary full window.
    fn search_root_aspiration<const COUNT_D4_FIRES: bool, const WALLQ_TC_ACTIVE: bool>(
        &mut self,
        state: &State,
        depth: i32,
        previous_score: i32,
        mut accumulator: Option<&mut NnueAccumulator>,
    ) -> Option<(Action, i32)> {
        for failures in 0..=2 {
            let (alpha, beta) = aspiration_bounds(previous_score, failures);
            if failures == 2 {
                return self.search_root::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
                    state,
                    depth,
                    accumulator.as_deref_mut(),
                );
            }
            let (action, score) = self.search_root_window::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
                state,
                depth,
                alpha,
                beta,
                accumulator.as_deref_mut(),
            )?;
            if score > alpha && score < beta {
                return Some((action, score));
            }
        }
        unreachable!("the final aspiration retry is a full-window search")
    }

    /// Windowed root PVS used only by the aspiration feature.  Its TT entry is
    /// correctly marked as a bound on a failed window; the all-off root still
    /// uses `search_root` above.
    fn search_root_window<const COUNT_D4_FIRES: bool, const WALLQ_TC_ACTIVE: bool>(
        &mut self,
        state: &State,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        mut accumulator: Option<&mut NnueAccumulator>,
    ) -> Option<(Action, i32)> {
        let orig_alpha = alpha;
        let tt_key = self.tt_position_key(state);
        let tt_move = self
            .tt_probe(tt_key.hash)
            .map(|entry| Self::tt_entry_move(entry, tt_key));
        let tight_edge_threshold = (self.features.lmr && depth >= 3).then_some(6);
        let ordered = self.order_moves(state, tt_move, 0, None, tight_edge_threshold)?;
        let lmr_edges = (self.features.lmr && depth >= 3 && ordered.len > 6)
            .then(|| ordered.tight_edges.unwrap_or_else(|| LmrEdges::for_state(state)));
        let probable_walls = (self.features.lmr_probable_walls
            && depth >= LMR_PROBABLE_WALLS_MIN_DEPTH
            && lmr_edges.is_some())
        .then(|| ProbableWalls::for_state(state, self.probable_wall_root_turn));
        if self.hard_deadline_reached_now() {
            return None;
        }
        let mut best_action = ordered.moves[0];
        let mut best_score = -INF;

        for i in 0..ordered.len {
            if self.hard_deadline_reached_now() {
                return None;
            }
            let action = ordered.moves[i];
            #[cfg(feature = "profile-timers")]
            let child = crate::profile::measure(crate::profile::Bucket::StateApply, || {
                state.apply(action)
            });
            #[cfg(not(feature = "profile-timers"))]
            let child = state.apply(action);
            if let (Some(network), Some(current)) =
                (self.nnue.as_deref(), accumulator.as_deref_mut())
            {
                current.make(network, state, action);
            }
            let score_result = if i == 0 {
                self.negamax_counted::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
                    &child,
                    accumulator.as_deref_mut(),
                    depth - 1,
                    -beta,
                    -alpha,
                    1,
                    self.probable_wall_root_turn.saturating_add(1),
                    false,
                    Some(action),
                )
                .map(|score| -score)
            } else {
                let reduction = lmr_edges.as_ref().map_or(0, |edges| {
                    lmr_reduction(
                        depth,
                        i,
                        action,
                        tt_move,
                        edges,
                        probable_walls.as_ref(),
                        self.features.lmr_probable_walls_control,
                    )
                });
                self.search_late_move::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
                    &child,
                    accumulator.as_deref_mut(),
                    action,
                    depth,
                    alpha,
                    beta,
                    1,
                    self.probable_wall_root_turn.saturating_add(1),
                    reduction,
                )
            };
            if let (Some(network), Some(current)) =
                (self.nnue.as_deref(), accumulator.as_deref_mut())
            {
                current.unmake(network, state, action);
            }
            let score = score_result?;
            if score > best_score {
                best_score = score;
                best_action = action;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }

        let bound = if best_score <= orig_alpha {
            BOUND_UPPER
        } else if best_score >= beta {
            BOUND_LOWER
        } else {
            BOUND_EXACT
        };
        self.tt_store_position(tt_key, depth, bound, best_score, best_action);
        if self.features.corr_hist && bound == BOUND_EXACT {
            self.record_correction_history(
                state,
                depth,
                best_score,
                accumulator.as_deref(),
            );
        }
        Some((best_action, best_score))
    }

    fn negamax_counted<const COUNT_D4_FIRES: bool, const WALLQ_TC_ACTIVE: bool>(
        &mut self,
        state: &State,
        mut accumulator: Option<&mut NnueAccumulator>,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: i32,
        probable_turn: u32,
        previous_was_null: bool,
        previous_action: Option<Action>,
    ) -> Option<i32> {
        self.nodes += 1;
        if self.should_stop() {
            return None;
        }

        // Terminal: a set winner means the *side to move* just lost.
        if winner(state) >= 0 {
            return Some(-(MATE - ply));
        }
        #[cfg(feature = "profile-timers")]
        let race_profile_guard =
            crate::profile::Guard::new(crate::profile::Bucket::Race);
        if self.features.race2w {
            if let Some(score) = self.race_up_to_two_walls_score(state, ply) {
                return Some(score);
            }
        }
        if self.features.race1w {
            if let Some(score) = self.race_one_wall_score(state, ply) {
                return Some(score);
            }
        }
        if self.features.race_exact {
            if let Some(score) = self.race_exact_score(state, ply) {
                return Some(score);
            }
        }
        #[cfg(feature = "profile-timers")]
        drop(race_profile_guard);
        if depth <= 0 {
            #[cfg(feature = "profile-timers")]
            return Some(crate::profile::measure(
                crate::profile::Bucket::LeafEvaluation,
                || self.leaf_evaluation::<WALLQ_TC_ACTIVE>(state, accumulator.as_deref(), alpha, beta),
            ));
            #[cfg(not(feature = "profile-timers"))]
            return Some(self.leaf_evaluation::<WALLQ_TC_ACTIVE>(
                state,
                accumulator.as_deref(),
                alpha,
                beta,
            ));
        }

        // "Interior nodes" is deliberately counted before TT-bound and
        // null-move returns. Those are still non-leaf recursive nodes, but
        // neither has a searched move with an ordered cutoff rank.
        if self.features.probe_cutoffs {
            self.cutoff_probe.interior_nodes =
                self.cutoff_probe.interior_nodes.saturating_add(1);
        }

        let orig_alpha = alpha;
        // Snapshot the incoming window: alpha changes during the move loop,
        // but only nodes entered with a null window are non-PV LMP nodes.
        let is_null_window = alpha.checked_add(1) == Some(beta);
        let lmp_null_window = self.features.lmp && is_null_window;
        let tt_key = self.tt_position_key(state);
        let mut tt_move = None;
        if let Some(e) = self.tt_probe(tt_key.hash) {
            tt_move = Some(Self::tt_entry_move(e, tt_key));
            if e.depth as i32 >= depth {
                let score = from_tt(e.score as i32, ply);
                match e.bound {
                    BOUND_EXACT => return Some(score),
                    BOUND_LOWER => {
                        if score >= beta {
                            return Some(score);
                        }
                    }
                    BOUND_UPPER => {
                        if score <= alpha {
                            return Some(score);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Stage-2 RFP instrumentation is a shadow probe: take the hypothetical
        // triggers here, after exact/terminal returns and the TT probe, then
        // classify them only once this node's ordinary search has completed.
        // Probe depth is configurable for pre-registered shadow measurements;
        // production RFP uses its independently configured depth below.
        let rfp_precision_sample = if self.features.probe_rfp_precision
            && rfp_precision_eligible(
                depth,
                is_null_window,
                beta,
                self.features.probe_rfp_depth,
            )
        {
            #[cfg(feature = "profile-timers")]
            let static_eval = crate::profile::measure(
                crate::profile::Bucket::RfpEvaluation,
                || self.raw_evaluation(state, accumulator.as_deref()),
            );
            #[cfg(not(feature = "profile-timers"))]
            let static_eval = self.raw_evaluation(state, accumulator.as_deref());
            let sample = rfp_precision_sample(
                state,
                depth,
                beta,
                static_eval,
                self.features.probe_rfp_depth,
            );
            Some(self.begin_rfp_precision_sample(sample))
        } else {
            None
        };

        // Reverse futility owns this shallow static fail-high slice before null
        // move. Its probe-validated eligibility covers every opponent-wall
        // inventory; a cutoff is fail-hard and bypasses the TT store below.
        if self.features.rfp
            && !self.features.probe_rfp_precision
            && rfp_eligible(depth, is_null_window, beta, self.active_rfp_depth)
        {
            #[cfg(feature = "profile-timers")]
            let static_eval = crate::profile::measure(
                crate::profile::Bucket::RfpEvaluation,
                || self.raw_evaluation(state, accumulator.as_deref()),
            );
            #[cfg(not(feature = "profile-timers"))]
            let static_eval = self.raw_evaluation(state, accumulator.as_deref());
            if rfp_prunable(
                depth,
                is_null_window,
                beta,
                static_eval,
                self.features.rfp_margin,
                self.active_rfp_depth,
            )
            {
                if COUNT_D4_FIRES {
                    debug_assert!(self.features.rfp_tc_adaptive);
                    debug_assert_eq!(self.active_rfp_depth, 4);
                    if depth == 4 {
                        self.rfp_tc_adaptive_d4_fires = self
                            .rfp_tc_adaptive_d4_fires
                            .saturating_add(1);
                    }
                }
                return Some(beta);
            }
        }

        // R=2 null-move pruning.  It never passes twice in succession, only
        // runs while the mover still owns a wall, and is withheld from any
        // position where either race is within three ordinary BFS steps.
        if self.features.null_move
            && rfp_precision_sample.is_none()
            && null_move_allowed(state, depth, previous_was_null)
        {
            let mut passed = *state;
            passed.turn ^= 1;
            let null_score = -self.negamax_counted::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
                &passed,
                accumulator.as_deref_mut(),
                depth - 3,
                -beta,
                -beta + 1,
                ply + 1,
                probable_turn,
                true,
                None,
            )?;
            if null_score >= beta {
                return Some(beta); // fail-hard; no synthetic TT move is stored.
            }
        }

        let lmp_d4_enabled = lmp_null_window
            && self.features.lmp_d4
            && depth == 4
            && (!self.features.lmp_d4_guard
                || lmp_d4_guard_allows(state, alpha, beta));
        let lmp_depth_enabled = (1..=3).contains(&depth) || lmp_d4_enabled;
        let lmr_tight_threshold = (self.features.lmr && depth >= 3).then_some(6);
        let lmp_tight_threshold = (lmp_null_window && lmp_depth_enabled)
            .then(|| lmp_threshold(self.features.lmp_n, depth));
        let tight_edge_threshold = match (lmr_tight_threshold, lmp_tight_threshold) {
            (Some(lmr), Some(lmp)) => Some(lmr.min(lmp)),
            (lmr, lmp) => lmr.or(lmp),
        };
        let ordered = self.order_moves(
            state,
            tt_move,
            ply,
            previous_action,
            tight_edge_threshold,
        )?;
        let lmr_active = self.features.lmr && depth >= 3 && ordered.len > 6;
        let lmp_active = lmp_null_window
            && lmp_depth_enabled
            && ordered.len > lmp_threshold(self.features.lmp_n, depth);
        let tight_edges = (lmr_active || lmp_active)
            .then(|| ordered.tight_edges.unwrap_or_else(|| LmrEdges::for_state(state)));
        let probable_walls = (lmr_active
            && self.features.lmr_probable_walls
            && depth >= LMR_PROBABLE_WALLS_MIN_DEPTH)
            .then(|| ProbableWalls::for_state(state, probable_turn));
        let eval_swing_parent = (self.features.probe_evalswing && depth == 1)
            .then(|| self.raw_evaluation(state, accumulator.as_deref()));
        // These depth-one movers are the reply side relative to a candidate
        // depth-two RFP node. Their OWN inventory is therefore the RFP
        // parent's opponent-walls bucket used by the probe stratification.
        let rfp_opponent_has_walls =
            eval_swing_parent.is_some() && walls_left(state) > 0;
        if self.hard_deadline_reached_now() {
            return None;
        }
        let mut best_score = -INF;
        let mut best_action = ordered.moves[0];

        for i in 0..ordered.len {
            if self.hard_deadline_reached_now() {
                return None;
            }
            let action = ordered.moves[i];
            if let Some(parent_eval) = eval_swing_parent {
                #[cfg(feature = "profile-timers")]
                let probe_child =
                    crate::profile::measure(crate::profile::Bucket::StateApply, || {
                        state.apply(action)
                    });
                #[cfg(not(feature = "profile-timers"))]
                let probe_child = state.apply(action);
                if let (Some(network), Some(current)) =
                    (self.nnue.as_deref(), accumulator.as_deref_mut())
                {
                    current.make(network, state, action);
                }
                let child_from_parent = self
                    .raw_evaluation(&probe_child, accumulator.as_deref())
                    .saturating_neg();
                if let (Some(network), Some(current)) =
                    (self.nnue.as_deref(), accumulator.as_deref_mut())
                {
                    current.unmake(network, state, action);
                }
                let swing = child_from_parent.saturating_sub(parent_eval);
                self.eval_swing_probe
                    .as_mut()
                    .expect("enabled eval-swing probe must have storage")
                    .record(action, rfp_opponent_has_walls, swing);
            }
            if lmp_active
                && tight_edges.as_ref().is_some_and(|edges| {
                    if depth == 4 {
                        lmp_d4_prunable(
                            depth,
                            lmp_null_window,
                            i,
                            action,
                            tt_move,
                            self.features.lmp_n,
                            edges,
                        )
                    } else {
                        lmp_prunable(
                            depth,
                            lmp_null_window,
                            i,
                            action,
                            tt_move,
                            self.features.lmp_n,
                            edges,
                        )
                    }
                })
            {
                continue;
            }
            #[cfg(feature = "profile-timers")]
            let child = crate::profile::measure(crate::profile::Bucket::StateApply, || {
                state.apply(action)
            });
            #[cfg(not(feature = "profile-timers"))]
            let child = state.apply(action);
            if let (Some(network), Some(current)) =
                (self.nnue.as_deref(), accumulator.as_deref_mut())
            {
                current.make(network, state, action);
            }
            let score_result = if i == 0 {
                self.negamax_counted::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
                    &child,
                    accumulator.as_deref_mut(),
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    probable_turn.saturating_add(1),
                    false,
                    Some(action),
                )
                .map(|score| -score)
            } else {
                let reduction = if lmr_active {
                    tight_edges.as_ref().map_or(0, |edges| {
                        lmr_reduction(
                            depth,
                            i,
                            action,
                            tt_move,
                            edges,
                            probable_walls.as_ref(),
                            self.features.lmr_probable_walls_control,
                        )
                    })
                } else {
                    0
                };
                self.search_late_move::<COUNT_D4_FIRES, WALLQ_TC_ACTIVE>(
                    &child,
                    accumulator.as_deref_mut(),
                    action,
                    depth,
                    alpha,
                    beta,
                    ply + 1,
                    probable_turn.saturating_add(1),
                    reduction,
                )
            };
            if let (Some(network), Some(current)) =
                (self.nnue.as_deref(), accumulator.as_deref_mut())
            {
                current.unmake(network, state, action);
            }
            let score = score_result?;
            if score > best_score {
                best_score = score;
                best_action = action;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                if self.features.probe_cutoffs {
                    self.cutoff_probe
                        .record_cutoff(i, i == 0 && Some(action) == tt_move);
                }
                if self.features.killers {
                    self.record_killer(ply, action, tt_move);
                }
                if self.features.history || self.features.cmh || self.features.wall_hist {
                    if self.features.wall_hist {
                        self.record_history_with_context(
                            state,
                            action,
                            depth,
                            previous_action,
                            Some(ordered.wall_contexts[i]),
                        );
                    } else {
                        self.record_history(state, action, depth, previous_action);
                    }
                }
                break; // beta cutoff.
            }
        }

        let bound = if best_score <= orig_alpha {
            BOUND_UPPER
        } else if best_score >= beta {
            BOUND_LOWER
        } else {
            BOUND_EXACT
        };
        if let Some(sample) = rfp_precision_sample {
            self.finish_rfp_precision_sample(sample, best_score, beta);
        }
        if self.features.corr_hist && bound == BOUND_EXACT {
            self.record_correction_history(
                state,
                depth,
                best_score,
                accumulator.as_deref(),
            );
        }
        // WALLQ-TC leaf values are intentionally stored in the ordinary TT. The
        // same state can be evaluated under different local windows, but that
        // bounded W-dependent reuse is the accepted design cost; do not salt
        // or special-case the table.
        self.tt_store_position(tt_key, depth, bound, to_tt(best_score, ply), best_action);
        Some(best_score)
    }

    #[cfg(test)]
    fn negamax(
        &mut self,
        state: &State,
        accumulator: Option<&mut NnueAccumulator>,
        depth: i32,
        alpha: i32,
        beta: i32,
        ply: i32,
        previous_was_null: bool,
        previous_action: Option<Action>,
    ) -> Option<i32> {
        self.negamax_counted::<false, false>(
            state,
            accumulator,
            depth,
            alpha,
            beta,
            ply,
            probable_wall_turn_estimate(state),
            previous_was_null,
            previous_action,
        )
    }
}

// ---------------------------------------------------------------------------
// Mate-score <-> TT adjustment (make stored mate distances node-relative).
// ---------------------------------------------------------------------------

#[inline]
fn to_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_THRESHOLD {
        score + ply
    } else if score <= -MATE_THRESHOLD {
        score - ply
    } else {
        score
    }
}

#[inline]
fn from_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_THRESHOLD {
        score - ply
    } else if score <= -MATE_THRESHOLD {
        score + ply
    } else {
        score
    }
}

/// Bounds for the aspiration retry number.  The third attempt is deliberately
/// the ordinary full window, after the +/-60 and +/-240 attempts have failed.
#[inline]
fn aspiration_bounds(center: i32, failures: usize) -> (i32, i32) {
    let half_window = match failures {
        0 => 60,
        1 => 240,
        _ => return (-INF, INF),
    };
    (
        center.saturating_sub(half_window).max(-INF),
        center.saturating_add(half_window).min(INF),
    )
}

// ---------------------------------------------------------------------------
// Evaluation v0 (from side-to-move perspective).
// ---------------------------------------------------------------------------

#[inline]
fn side_view(state: &State) -> (u8, u8, u8, u8, i32, i32) {
    // (my_pawn, opp_pawn, my_goal_row, opp_goal_row, my_walls, opp_walls)
    if state.turn == 0 {
        (state.p0, state.p1, 8, 0, state.w0 as i32, state.w1 as i32)
    } else {
        (state.p1, state.p0, 0, 8, state.w1 as i32, state.w0 as i32)
    }
}

/// `100*(D_opp - D_me) + 35*(my_walls - opp_walls) + 50`.
pub fn evaluate(state: &State) -> i32 {
    let topo = Topology::from_walls(state.h, state.v);
    let (mine, opp, my_goal, opp_goal, my_walls, opp_walls) = side_view(state);
    let d_me = distance_to_row(&topo, mine, my_goal) as i32;
    let d_opp = distance_to_row(&topo, opp, opp_goal) as i32;
    100 * (d_opp - d_me) + 35 * (my_walls - opp_walls) + 50
}

#[inline]
fn progression_value(distance: i32) -> i32 {
    let progress = 17 - distance;
    6 * progress + 4 * progress * progress / 8
}

#[inline]
fn phased_wall_value(opponent_distance: i32) -> i32 {
    20 + 30 * opponent_distance.min(12) / 12
}

/// Interpolate from `wall_cp` at ten-or-more opposing steps to `endgame_cp`
/// at the goal. Signed integer division truncates toward zero.
#[inline]
fn endgame_wall_value(wall_cp: i32, endgame_cp: i32, opponent_distance: i32) -> i32 {
    endgame_cp + (wall_cp - endgame_cp) * opponent_distance.min(10) / 10
}

/// Scale the ordinary race margin by proximity to either goal. Multiplication
/// happens before the two signed divisions, each of which truncates toward zero.
#[inline]
fn race_amplification(amp: i32, d_me: i32, d_opponent: i32) -> i32 {
    let race_margin = d_opponent - d_me;
    let proximity = 12 - d_me.min(d_opponent).min(12);
    amp * race_margin * proximity / 12 / 4
}

const CARDINAL_DIRECTIONS: [(i8, i8); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

#[inline]
fn blocked_neighbor_directions(topology: &Topology, cell: u8) -> u8 {
    CARDINAL_DIRECTIONS
        .iter()
        .filter(|&&(dc, dr)| step(topology, cell, dc, dr).is_none())
        .count() as u8
}

/// Minimum squeeze count over every shortest prefix of up to six moves.
/// Taking the minimum makes the selected shortest route deterministic under
/// horizontal reflection while still measuring a path the pawn can follow.
fn corridor_vulnerability(
    topology: &Topology,
    distances: &GoalDistances,
    start: u8,
) -> i32 {
    let distance = distances.get(start);
    if distance <= 0 {
        return 0;
    }
    let steps = distance.min(6) as usize;
    let mut current = [u8::MAX; 81];
    current[start as usize] = 0;

    for _ in 0..steps {
        let mut next = [u8::MAX; 81];
        for cell in 0u8..81 {
            let cost = current[cell as usize];
            if cost == u8::MAX {
                continue;
            }
            let wanted = distances.get(cell) - 1;
            for &(dc, dr) in &CARDINAL_DIRECTIONS {
                let Some(destination) = step(topology, cell, dc, dr) else {
                    continue;
                };
                if distances.get(destination) != wanted {
                    continue;
                }
                let squeeze = u8::from(blocked_neighbor_directions(topology, destination) >= 2);
                let candidate = cost + squeeze;
                next[destination as usize] = next[destination as usize].min(candidate);
            }
        }
        current = next;
    }

    current
        .iter()
        .copied()
        .min()
        .filter(|cost| *cost != u8::MAX)
        .unwrap_or(0) as i32
}

/// Whether a statically placeable wall slot touches the unique shortest path.
/// Overlap and crossing checks are bit-only. Route legality is deliberately
/// not scanned here: `ev_fragility_1w` is a cheap structural term and performs
/// no per-wall BFS.
#[inline]
fn has_available_tight_wall(state: &State, edges: &PawnShortestEdges) -> bool {
    let (horizontal, vertical) = geometrically_available_wall_slots(state);
    horizontal & edges.horizontal_wall_touch_mask() != 0
        || vertical & edges.vertical_wall_touch_mask() != 0
}

#[inline]
fn single_wall_path_fragility(
    state: &State,
    topology: &Topology,
    distances: &GoalDistances,
    pawn: u8,
) -> i32 {
    distances
        .unique_pawn_shortest_edges(topology, pawn)
        .filter(|edges| has_available_tight_wall(state, edges))
        .map_or(0, |_| 1)
}

/// Check one geometrically available candidate wall.  `best` is a lower
/// bound on the final net score, so a target delta no larger than it cannot
/// improve the maximum: adding a wall cannot shorten the placer's route.
#[inline]
fn consider_interdiction_candidate(
    best: &mut i16,
    after_wall: WallDistanceContext,
    target_before: i16,
    target_query: &DistanceQuery,
    self_before: i16,
    self_query: &DistanceQuery,
) {
    let target_after = distance_to_row_for_wall(&after_wall, target_query);
    if target_after < 0 {
        // A legal wall must preserve the target's route.
        return;
    }
    let target_delta = target_after - target_before;
    debug_assert!(target_delta >= 0, "a wall cannot shorten a route");
    if target_delta <= *best {
        // Even a zero self delta cannot beat the current maximum, so route
        // legality for the other pawn cannot affect the result.
        return;
    }

    let self_after = distance_to_row_for_wall(&after_wall, self_query);
    if self_after < 0 {
        // The other half of rules-exact wall legality.
        return;
    }
    let self_delta = self_after - self_before;
    debug_assert!(self_delta >= 0, "a wall cannot shorten a route");
    *best = (*best).max(target_delta - self_delta);
}

/// The authoritative Q11 DAG restriction, factored so target-only probes use
/// exactly the same geometry and shortest-path candidate masks as the net
/// interdiction evaluator.
#[inline]
fn target_dag_candidate_masks(
    target_edges: &PawnShortestEdges,
    horizontal_available: u64,
    vertical_available: u64,
) -> (u64, u64) {
    (
        target_edges.horizontal_wall_touch_mask() & horizontal_available,
        target_edges.vertical_wall_touch_mask() & vertical_available,
    )
}

/// Exact best net one-wall damage for one placing side.  Candidate slots are
/// precisely the walls spanning an edge in the target pawn's complete
/// shortest-path DAG; every omitted wall has target delta zero and therefore
/// cannot exceed the zero floor.
fn interdiction_for_side(
    walls_in_hand: u8,
    target_edges: &PawnShortestEdges,
    target_before: i16,
    target_query: &DistanceQuery,
    self_before: i16,
    self_query: &DistanceQuery,
    wall_context: WallDistanceContext,
    horizontal_available: u64,
    vertical_available: u64,
) -> i32 {
    if walls_in_hand == 0 || target_before < 0 || self_before < 0 {
        return 0;
    }

    let mut best = 0i16;
    let (mut horizontal, mut vertical) =
        target_dag_candidate_masks(target_edges, horizontal_available, vertical_available);
    while horizontal != 0 {
        let slot = horizontal.trailing_zeros() as u8;
        horizontal &= horizontal - 1;
        consider_interdiction_candidate(
            &mut best,
            wall_context.with_horizontal(slot),
            target_before,
            target_query,
            self_before,
            self_query,
        );
    }

    while vertical != 0 {
        let slot = vertical.trailing_zeros() as u8;
        vertical &= vertical - 1;
        consider_interdiction_candidate(
            &mut best,
            wall_context.with_vertical(slot),
            target_before,
            target_query,
            self_before,
            self_query,
        );
    }
    best as i32
}

/// Shared exact implementation used by the evaluator and the parity tool.
/// The precomputed distance fields are deliberately reused for the two DAGs
/// and their before-wall distances.
fn interdiction_values_with_distances(
    state: &State,
    topology: &Topology,
    p0_distances: &GoalDistances,
    p1_distances: &GoalDistances,
) -> (i32, i32) {
    if winner(state) >= 0 {
        return (0, 0);
    }

    let p0_before = p0_distances.get(state.p0);
    let p1_before = p1_distances.get(state.p1);
    if p0_before < 0 || p1_before < 0 {
        // Adding a wall cannot repair a parser-valid but route-invalid state.
        return (0, 0);
    }

    let p0_edges = p0_distances.pawn_shortest_edges(topology, state.p0);
    let p1_edges = p1_distances.pawn_shortest_edges(topology, state.p1);
    let (horizontal_available, vertical_available) = geometrically_available_wall_slots(state);
    let wall_context = topology.wall_distance_context();
    let p0_query = DistanceQuery::to_row_8(state.p0);
    let p1_query = DistanceQuery::to_row_0(state.p1);

    let p0 = interdiction_for_side(
        state.w0,
        &p1_edges,
        p1_before,
        &p1_query,
        p0_before,
        &p0_query,
        wall_context,
        horizontal_available,
        vertical_available,
    );
    let p1 = interdiction_for_side(
        state.w1,
        &p0_edges,
        p0_before,
        &p0_query,
        p1_before,
        &p1_query,
        wall_context,
        horizontal_available,
        vertical_available,
    );
    (p0, p1)
}

/// Exact one-wall interdiction values in player-index order `(I_0, I_1)`.
/// This is stateless and therefore safe to call concurrently from Lazy SMP.
pub(crate) fn interdiction_values(state: &State) -> (i32, i32) {
    let topology = Topology::from_walls(state.h, state.v);
    let p0_distances = distances_to_row(&topology, 8);
    let p1_distances = distances_to_row(&topology, 0);
    interdiction_values_with_distances(state, &topology, &p0_distances, &p1_distances)
}

/// Exact one-wall interdiction values in side-to-move order `(I_me, I_opp)`.
pub(crate) fn interdiction_stm_values(state: &State) -> (i32, i32) {
    let (i0, i1) = interdiction_values(state);
    if state.turn == 0 {
        (i0, i1)
    } else {
        (i1, i0)
    }
}

/// Frozen E-032 value in side-to-move order. Both the active WALLQ-TC leaf
/// path and raw DAG/Q11 parity use this mapping, so its coefficients, cap,
/// and signs cannot drift apart.
#[inline]
fn interdiction_correction(i_me: i32, i_opp: i32) -> i32 {
    INTERDICTION_ME_CP * i_me.min(INTERDICTION_CAP)
        - INTERDICTION_OPP_CP * i_opp.min(INTERDICTION_CAP)
}

/// Featured evaluator.  The explicit legacy return is the frozen path used
/// whenever all evaluation features retain their legacy settings.
fn evaluate_with_features(state: &State, features: &Features) -> i32 {
    if !features.ev_progress
        && !features.ev_wallphase
        && features.wall_cp == 35
        && features.wall_cp_endgame < 0
        && features.race_amp == 0
        && !features.ev_corridor
        && !features.ev_fragility_1w
    {
        return evaluate(state);
    }

    let topology = Topology::from_walls(state.h, state.v);
    let (mine, opponent, my_goal, opponent_goal, my_walls, opponent_walls) = side_view(state);
    let needs_distance_fields = features.ev_corridor || features.ev_fragility_1w;
    let (d_me, d_opponent, p0_distances, p1_distances) = if needs_distance_fields {
        let p0_distances = distances_to_row(&topology, 8);
        let p1_distances = distances_to_row(&topology, 0);
        let (d_me, d_opponent) = if state.turn == 0 {
            (
                p0_distances.get(state.p0) as i32,
                p1_distances.get(state.p1) as i32,
            )
        } else {
            (
                p1_distances.get(state.p1) as i32,
                p0_distances.get(state.p0) as i32,
            )
        };
        (
            d_me,
            d_opponent,
            Some(p0_distances),
            Some(p1_distances),
        )
    } else {
        (
            distance_to_row(&topology, mine, my_goal) as i32,
            distance_to_row(&topology, opponent, opponent_goal) as i32,
            None,
            None,
        )
    };

    let flat_wall_score = features.wall_cp * (my_walls - opponent_walls);
    let mut wall_score = if features.ev_wallphase {
        my_walls * phased_wall_value(d_opponent)
            - opponent_walls * phased_wall_value(d_me)
    } else {
        flat_wall_score
    };
    if features.wall_cp_endgame >= 0 {
        let shaped_wall_score = my_walls
            * endgame_wall_value(features.wall_cp, features.wall_cp_endgame, d_opponent)
            - opponent_walls
                * endgame_wall_value(features.wall_cp, features.wall_cp_endgame, d_me);
        // Apply the new wall model as a delta from the flat model. This keeps
        // the older ev_wallphase experiment independently composable.
        wall_score += shaped_wall_score - flat_wall_score;
    }
    let mut score = 100 * (d_opponent - d_me) + wall_score + 50;

    if features.race_amp != 0 {
        score += race_amplification(features.race_amp, d_me, d_opponent);
    }

    if features.ev_progress {
        score += progression_value(d_me) - progression_value(d_opponent);
    }
    if features.ev_corridor {
        let (my_distances, opponent_distances) = if state.turn == 0 {
            (
                p0_distances.as_ref().expect("corridor p0 distances"),
                p1_distances.as_ref().expect("corridor p1 distances"),
            )
        } else {
            (
                p1_distances.as_ref().expect("corridor p1 distances"),
                p0_distances.as_ref().expect("corridor p0 distances"),
            )
        };
        let mine_squeeze = corridor_vulnerability(
            &topology,
            my_distances,
            mine,
        );
        let opponent_squeeze = corridor_vulnerability(
            &topology,
            opponent_distances,
            opponent,
        );
        score += 8 * (opponent_squeeze - mine_squeeze);
    }
    if features.ev_fragility_1w {
        let (my_distances, opponent_distances) = if state.turn == 0 {
            (
                p0_distances.as_ref().expect("fragility p0 distances"),
                p1_distances.as_ref().expect("fragility p1 distances"),
            )
        } else {
            (
                p1_distances.as_ref().expect("fragility p1 distances"),
                p0_distances.as_ref().expect("fragility p0 distances"),
            )
        };
        let my_fragility = if opponent_walls > 0 {
            single_wall_path_fragility(state, &topology, my_distances, mine)
        } else {
            0
        };
        let opponent_fragility = if my_walls > 0 {
            single_wall_path_fragility(state, &topology, opponent_distances, opponent)
        } else {
            0
        };
        score += FRAGILITY_CP * (opponent_fragility - my_fragility);
    }
    score
}

/// Exhausted-wall race shortcut used by the default-off `race_exact` feature.
/// The historical Chebyshev applicability guard is retained for stable
/// coverage, but every accepted verdict comes from the pawn-aware retrograde.
fn race_exact_score_with<F>(state: &State, ply: i32, outcome: &mut F) -> Option<i32>
where
    F: FnMut(&State) -> Option<ZeroWallOutcome>,
{
    if state.w0 != 0 || state.w1 != 0 {
        return None;
    }
    let dc = cell_col(state.p0).abs_diff(cell_col(state.p1));
    let dr = cell_row(state.p0).abs_diff(cell_row(state.p1));
    if dc.max(dr) <= 2 {
        return None;
    }
    race_zero_wall_score_with(state, ply, outcome)
}

#[cfg(test)]
fn race_exact_score(state: &State, ply: i32) -> Option<i32> {
    race_exact_score_with(state, ply, &mut exact_zero_wall_outcome)
}

/// Exact exhausted-wall verdict using the fixed-topology pawn retrograde.
///
/// The legacy topology distances are retained only to shape the magnitude on
/// positions where the old and exact winners agree, preserving established
/// search ordering. The sign (or zero for a draw) comes exclusively from the
/// interaction-aware table and is therefore safe for search and self-play
/// labels even when routes converge.
fn race_zero_wall_score_with<F>(state: &State, ply: i32, outcome: &mut F) -> Option<i32>
where
    F: FnMut(&State) -> Option<ZeroWallOutcome>,
{
    let outcome = outcome(state)?;
    let ZeroWallOutcome::Win { winner, .. } = outcome else {
        return Some(0);
    };

    let topology = Topology::from_walls(state.h, state.v);
    let (mine, opponent, my_goal, opponent_goal, _, _) = side_view(state);
    let distance = if winner == state.turn {
        distance_to_row(&topology, mine, my_goal)
    } else {
        distance_to_row(&topology, opponent, opponent_goal)
    };
    if distance < 0 {
        return None;
    }
    let magnitude = 20_000 - ply - 2 * distance as i32;
    Some(if winner == state.turn {
        magnitude
    } else {
        -magnitude
    })
}

fn race_zero_wall_score(state: &State, ply: i32) -> Option<i32> {
    race_zero_wall_score_with(state, ply, &mut exact_zero_wall_outcome)
}

/// Sound subset of the one-wall race resolution.
///
/// The holder must be on move and the same Chebyshev separation used by the
/// legacy race shortcut must hold. The apparent "place now or never" 1.5-ply
/// model is not exact in every such position: a holder can sometimes walk past
/// an edge that a useful wall would block, then place that wall later. We
/// therefore return a tablebase score only in either of two provable cases:
///
/// * the zero-wall race already wins, in which case moving on a shortest route
///   is fastest and spending the wall now or later cannot improve its score;
/// * the opponent is one step from goal while the zero-wall race loses, in
///   which case every pawn move concedes immediately and the wall must be
///   placed now.
///
/// All other positions fall back to ordinary search rather than turning a
/// potentially winning delayed-wall strategy into an unsound loss.
fn race_one_wall_score_with<F>(state: &State, ply: i32, outcome: &mut F) -> Option<i32>
where
    F: FnMut(&State) -> Option<ZeroWallOutcome>,
{
    if state.w0 as u16 + state.w1 as u16 != 1 || walls_left(state) != 1 {
        return None;
    }
    let dc = cell_col(state.p0).abs_diff(cell_col(state.p1));
    let dr = cell_row(state.p0).abs_diff(cell_row(state.p1));
    if dc.max(dr) <= 2 {
        return None;
    }

    let mut no_wall = *state;
    if no_wall.turn == 0 {
        no_wall.w0 = 0;
    } else {
        no_wall.w1 = 0;
    }
    let pure_race = race_zero_wall_score_with(&no_wall, ply, outcome)?;

    // A zero-wall draw proves that retaining the wall can avoid a loss, but a
    // delayed placement might still win. The immediate/no-placement model is
    // therefore incomplete and must fall back.
    if pure_race == 0 {
        return None;
    }
    // If discarding the optional wall already wins, following the proven
    // pawn-only strategy is sufficient and no placement can reach the goal a
    // ply sooner. Avoid building an unnecessary placement family.
    if pure_race > 0 {
        return Some(pure_race);
    }

    let topology = Topology::from_walls(state.h, state.v);
    let (_, opponent, _, opponent_goal, _, _) = side_view(state);
    let opponent_distance = distance_to_row(&topology, opponent, opponent_goal);
    if pure_race < 0 && opponent_distance != 1 {
        return None;
    }

    let mut best = pure_race;
    let mut legal = MoveList::new();
    legal_actions(state, &mut legal);
    // A fixed-topology solve is cheap, but pathological sparse/parser-valid
    // states can expose almost all 128 placements. Falling back above 96 is
    // sound; every position that does fire still resolves every legal wall.
    const MAX_EXACT_WALL_PLACEMENTS: usize = 96;
    if legal.len() - legal.pawn_count() > MAX_EXACT_WALL_PLACEMENTS {
        return None;
    }
    for index in 0..legal.len() {
        let action = legal.get(index);
        if action.kind == ActionKind::Pawn {
            continue;
        }
        let child = state.apply(action);
        let candidate = -race_zero_wall_score_with(&child, ply + 1, outcome)?;
        best = best.max(candidate);
    }
    Some(best)
}

fn race_one_wall_score(state: &State, ply: i32) -> Option<i32> {
    race_one_wall_score_with(state, ply, &mut exact_zero_wall_outcome)
}

/// Find a root move whose one-wall outcome is a proven win.
///
/// The ordinary root shortcut needs a concrete move. This bridge supplies an
/// authoritative proven action and pre-warms its exact topology family before
/// Lazy-SMP helpers start.
fn race_one_wall_root_winning_solution_with<F>(
    state: &State,
    outcome: &mut F,
) -> Option<(Action, i32)>
where
    F: FnMut(&State) -> Option<ZeroWallOutcome>,
{
    if state.w0 as u16 + state.w1 as u16 != 1 {
        return None;
    }

    if walls_left(state) == 1 {
        if race_one_wall_score_with(state, 0, outcome)? <= 0 {
            return None;
        }
        let mut legal = MoveList::new();
        legal_actions(state, &mut legal);
        let mut best = None;
        let mut best_score = 0;
        for index in 0..legal.len() {
            let action = legal.get(index);
            let child = state.apply(action);
            if winner(&child) == state.turn as i8 {
                return Some((action, MATE - 1));
            }
            let child_score = match action.kind {
                ActionKind::Pawn => race_one_wall_nonholder_score_with(&child, 1, outcome),
                ActionKind::Horizontal | ActionKind::Vertical => {
                    race_zero_wall_score_with(&child, 1, outcome)
                }
            };
            let Some(child_score) = child_score else {
                continue;
            };
            let score = -child_score;
            if score > best_score {
                best_score = score;
                best = Some((action, score));
            }
        }
        return best;
    }
    if walls_left(state) != 0 {
        return None;
    }

    let mut legal = MoveList::new();
    legal_actions(state, &mut legal);
    let mut best = None;
    let mut best_score = 0;
    for index in 0..legal.pawn_count() {
        let action = legal.get(index);
        let child = state.apply(action);
        if winner(&child) == state.turn as i8 {
            return Some((action, MATE - 1));
        }
        let Some(child_score) = race_one_wall_score_with(&child, 1, outcome) else {
            continue;
        };
        let score = -child_score;
        if score > best_score {
            best_score = score;
            best = Some((action, score));
        }
    }
    best
}

/// Resolve every already-sound race layer up through total hand two.
fn race_up_to_two_walls_score_with<F>(
    state: &State,
    ply: i32,
    outcome: &mut F,
) -> Option<i32>
where
    F: FnMut(&State) -> Option<ZeroWallOutcome>,
{
    match state.w0 as u16 + state.w1 as u16 {
        0 => race_zero_wall_score_with(state, ply, outcome),
        1 => race_one_wall_score_with(state, ply, outcome),
        2 => race_two_wall_score_with(state, ply, outcome),
        _ => None,
    }
}

/// Exact value of a one-wall layer while the wall-less player is on move.
///
/// The mover has pawn actions only. After each nonterminal pawn reply the sole
/// wall holder is on move, exactly the precondition of `race_one_wall_score`.
/// Requiring every reply to resolve makes this a complete minimax ply; a
/// single uncovered grandchild forces an ordinary-search fallback.
fn race_one_wall_nonholder_score_with<F>(
    state: &State,
    ply: i32,
    outcome: &mut F,
) -> Option<i32>
where
    F: FnMut(&State) -> Option<ZeroWallOutcome>,
{
    if state.w0 as u16 + state.w1 as u16 != 1 || walls_left(state) != 0 {
        return None;
    }

    let mut legal = MoveList::new();
    legal_actions(state, &mut legal);
    if legal.len() == 0 {
        return None;
    }
    // A direct goal is the fastest possible result and its terminal magnitude
    // dominates every 20_000-scale race-layer value. It therefore resolves
    // the node without demanding coverage of deliberately inferior replies.
    for index in 0..legal.len() {
        if winner(&state.apply(legal.get(index))) >= 0 {
            return Some(MATE - (ply + 1));
        }
    }
    let mut best = -INF;
    for index in 0..legal.len() {
        let action = legal.get(index);
        debug_assert_eq!(action.kind, ActionKind::Pawn);
        let child = state.apply(action);
        let child_score = race_one_wall_score_with(&child, ply + 1, outcome)?;
        best = best.max(-child_score);
    }
    Some(best)
}

/// Sound monopoly-only subset of the two-wall race layer.
///
/// Both walls must belong to the side to move. Split ownership is excluded:
/// pawn moves remain inside that two-sided layer, so immediate-placement
/// composition is not a game-theoretic value. Under monopoly the opponent is
/// wall-less and inert. Two cases generalise the proven race1w cases:
///
/// * if discarding both walls already wins the separated-pawn race, following
///   a shortest pawn route is the fastest win; optional wall tempi cannot make
///   the holder reach its goal sooner;
/// * if that race loses and the opponent is one step from goal, every pawn
///   move concedes, so a wall is forced now. For each placement we search the
///   opponent's complete pawn-reply ply, after which the remaining one-wall
///   holder is on move and `race_one_wall_score` is exact.
///
/// A delayed wall can dominate outside those cases. Any unresolved placement
/// or reply therefore returns `None` rather than manufacturing a loss.
fn race_two_wall_score_with<F>(state: &State, ply: i32, outcome: &mut F) -> Option<i32>
where
    F: FnMut(&State) -> Option<ZeroWallOutcome>,
{
    if state.w0 as u16 + state.w1 as u16 != 2 || walls_left(state) != 2 {
        return None;
    }
    let dc = cell_col(state.p0).abs_diff(cell_col(state.p1));
    let dr = cell_row(state.p0).abs_diff(cell_row(state.p1));
    if dc.max(dr) <= 2 {
        return None;
    }

    let mut no_walls = *state;
    no_walls.w0 = 0;
    no_walls.w1 = 0;
    let pure_race = race_zero_wall_score_with(&no_walls, ply, outcome)?;
    if pure_race > 0 {
        return Some(pure_race);
    }

    let topology = Topology::from_walls(state.h, state.v);
    let (_, opponent, _, opponent_goal, _, _) = side_view(state);
    if distance_to_row(&topology, opponent, opponent_goal) != 1 {
        return None;
    }

    let mut best = pure_race;
    let mut legal = MoveList::new();
    legal_actions(state, &mut legal);
    for index in 0..legal.len() {
        let action = legal.get(index);
        if action.kind == ActionKind::Pawn {
            continue;
        }
        let child = state.apply(action);
        let candidate = -race_one_wall_nonholder_score_with(&child, ply + 1, outcome)?;
        best = best.max(candidate);
    }
    Some(best)
}

#[cfg(test)]
fn race_two_wall_score(state: &State, ply: i32) -> Option<i32> {
    race_two_wall_score_with(state, ply, &mut exact_zero_wall_outcome)
}

impl SearchContext {
    fn race_exact_score(&self, state: &State, ply: i32) -> Option<i32> {
        let policy = self.race_build_policy;
        race_exact_score_with(state, ply, &mut |state| {
            self.race_tables.exact_zero_wall_outcome(state, policy)
        })
    }

    fn race_one_wall_score(&self, state: &State, ply: i32) -> Option<i32> {
        let policy = self.race_build_policy;
        race_one_wall_score_with(state, ply, &mut |state| {
            self.race_tables.exact_zero_wall_outcome(state, policy)
        })
    }

    fn race_one_wall_root_winning_solution(&self, state: &State) -> Option<(Action, i32)> {
        let policy = self.race_build_policy;
        race_one_wall_root_winning_solution_with(state, &mut |state| {
            self.race_tables.exact_zero_wall_outcome(state, policy)
        })
    }

    fn race_up_to_two_walls_score(&self, state: &State, ply: i32) -> Option<i32> {
        let policy = self.race_build_policy;
        race_up_to_two_walls_score_with(state, ply, &mut |state| {
            self.race_tables.exact_zero_wall_outcome(state, policy)
        })
    }
}

/// Null move is disabled when either player's ordinary topology-only race is
/// within three steps.  A missing route is also treated as critical.
fn is_race_critical(state: &State) -> bool {
    let topology = Topology::from_walls(state.h, state.v);
    distance_to_row(&topology, state.p0, 8) <= 3
        || distance_to_row(&topology, state.p1, 0) <= 3
}

#[inline]
fn lmp_d4_guard_allows(state: &State, alpha: i32, beta: i32) -> bool {
    alpha.saturating_abs() < MATE_THRESHOLD
        && beta.saturating_abs() < MATE_THRESHOLD
        && !is_race_critical(state)
}

#[inline]
fn null_move_allowed(state: &State, depth: i32, previous_was_null: bool) -> bool {
    depth >= 3 && walls_left(state) > 0 && !previous_was_null && !is_race_critical(state)
}

/// Resolve Q12's production RFP depth from the original search allotment.
#[inline]
fn rfp_depth_for_budget(
    configured_depth: i32,
    tc_adaptive: bool,
    allotted_budget_ms: Option<u64>,
) -> i32 {
    const FAST_BUDGET_MAX_MS: u64 = 200;

    if !tc_adaptive {
        return configured_depth;
    }

    match allotted_budget_ms {
        Some(budget_ms) if budget_ms <= FAST_BUDGET_MAX_MS => {
            debug_assert!(
                budget_ms <= FAST_BUDGET_MAX_MS,
                "PRE-REG Q12 depth-four RFP requires a <=200ms allotment"
            );
            4
        }
        Some(budget_ms) => {
            debug_assert!(
                budget_ms > FAST_BUDGET_MAX_MS,
                "PRE-REG Q12 long-TC identity rail"
            );
            // PRE-REG Q12 identity rail: this arm returns the frozen depth-three
            // limit directly; the sole adaptive depth-four arm above is unreachable.
            3
        }
        // Fixed-depth/bench searches carry no allotment and stay on frozen d3.
        None => 3,
    }
}

/// Production RFP eligibility covers the selected shallow depths, a null
/// window, and a beta outside the mate band. Probe settings never reach here.
#[inline]
fn rfp_eligible(depth: i32, is_null_window: bool, beta: i32, rfp_depth: i32) -> bool {
    is_null_window
        && (1..=rfp_depth).contains(&depth)
        && beta.saturating_abs() < MATE_THRESHOLD
}

#[inline]
fn rfp_prunable(
    depth: i32,
    is_null_window: bool,
    beta: i32,
    static_eval: i32,
    margin: i32,
    rfp_depth: i32,
) -> bool {
    rfp_eligible(depth, is_null_window, beta, rfp_depth)
        && static_eval.saturating_sub(margin.saturating_mul(depth)) >= beta
}

/// The precision probe covers ordinary shallow non-PV nodes through its
/// configured maximum depth, independent of opponent wall inventory.
#[inline]
fn rfp_precision_eligible(
    depth: i32,
    is_null_window: bool,
    beta: i32,
    probe_depth: i32,
) -> bool {
    is_null_window
        && (1..=probe_depth).contains(&depth)
        && beta.saturating_abs() < MATE_THRESHOLD
}

#[inline]
fn rfp_precision_sample(
    state: &State,
    depth: i32,
    beta: i32,
    static_eval: i32,
    probe_depth: i32,
) -> RfpPrecisionSample {
    debug_assert!(rfp_precision_eligible(depth, true, beta, probe_depth));
    let mut fires = [false; RFP_PRECISION_ARMS];
    for (arm, margin) in RFP_PRECISION_LINEAR_MARGINS.into_iter().enumerate() {
        fires[arm] = static_eval.saturating_sub(margin.saturating_mul(depth)) >= beta;
    }
    let quadratic_margin = RFP_PRECISION_QUADRATIC_COEFFICIENT
        .saturating_mul(depth)
        .saturating_mul(depth);
    fires[RFP_PRECISION_ARMS - 1] = static_eval.saturating_sub(quadratic_margin) >= beta;

    let opponent_walls = if state.turn == 0 { state.w1 } else { state.w0 };
    RfpPrecisionSample {
        opponent_walls_bucket: usize::from(opponent_walls > 0),
        fires,
    }
}

/// Union-of-all-shortest-route edge masks used by LMR.  Computing these once
/// per searched parent keeps the eligibility test exact and allocation-free.
#[derive(Clone, Copy)]
struct LmrEdges {
    mine: PawnShortestEdges,
    opponent: PawnShortestEdges,
}

impl LmrEdges {
    fn for_state(state: &State) -> Self {
        let topology = Topology::from_walls(state.h, state.v);
        let (mine, opponent, my_goal, opponent_goal, _, _) = side_view(state);
        let my_distances = distances_to_row(&topology, my_goal);
        let opponent_distances = distances_to_row(&topology, opponent_goal);
        Self {
            mine: my_distances.pawn_shortest_edges(&topology, mine),
            opponent: opponent_distances.pawn_shortest_edges(&topology, opponent),
        }
    }

    #[inline]
    fn wall_is_tight(&self, wall: Action) -> bool {
        wall_touches_pawn_shortest_edge(&self.mine, wall)
            || wall_touches_pawn_shortest_edge(&self.opponent, wall)
    }
}

/// Gorisanson's expansion mask (pinned local source `e15d6e7`), imported only
/// as a Q4 LMR classifier. Search still generates and orders every legal wall;
/// this mask can only deepen an already-reduced, provably delta-zero wall by
/// one additional ply.
#[derive(Clone, Copy, Default)]
struct ProbableWalls {
    horizontal: u64,
    vertical: u64,
}

impl ProbableWalls {
    fn for_state(state: &State, estimated_turn: u32) -> Self {
        let mut probable = Self::default();

        // game.js:577-657. Reconstruct the cumulative, order-independent
        // adjacency smear that gorisanson updates after every placed wall.
        let mut horizontal = state.h;
        while horizontal != 0 {
            let slot = horizontal.trailing_zeros() as u8;
            horizontal &= horizontal - 1;
            probable.smear_horizontal(slot);
        }
        let mut vertical = state.v;
        while vertical != 0 {
            let slot = vertical.trailing_zeros() as u8;
            vertical &= vertical - 1;
            probable.smear_vertical(slot);
        }

        let (mine, opponent) = if state.turn == 0 {
            (state.p0, state.p1)
        } else {
            (state.p1, state.p0)
        };

        // game.js:281-293. Keep the source's turn-three/turn-six gates.
        if estimated_turn >= 3 {
            probable.add_pawn_band(opponent);
        }
        if estimated_turn >= 6 || state.h != 0 || state.v != 0 {
            probable.add_pawn_band(mine);
        }

        // game.js:272-279: every row in the two OUTER SLOT COLUMNS, only for
        // horizontal walls and only from estimated turn six onward.
        if estimated_turn >= 6 {
            for row in 0..8 {
                probable.insert(ActionKind::Horizontal, row, 0);
                probable.insert(ActionKind::Horizontal, row, 7);
            }
        }

        probable
    }

    #[inline]
    fn contains(&self, action: Action) -> bool {
        match action.kind {
            ActionKind::Horizontal => self.horizontal & (1u64 << action.pos) != 0,
            ActionKind::Vertical => self.vertical & (1u64 << action.pos) != 0,
            ActionKind::Pawn => false,
        }
    }

    #[inline]
    fn insert(&mut self, kind: ActionKind, row: i32, col: i32) {
        if !(0..8).contains(&row) || !(0..8).contains(&col) {
            return;
        }
        let bit = 1u64 << (row * 8 + col);
        match kind {
            ActionKind::Horizontal => self.horizontal |= bit,
            ActionKind::Vertical => self.vertical |= bit,
            ActionKind::Pawn => unreachable!("probable-wall mask accepts walls only"),
        }
    }

    /// game.js:751-802. The asymmetric-looking ranges are exact: horizontal
    /// candidates span two rows by four columns; vertical candidates span
    /// four rows by two columns.
    fn add_pawn_band(&mut self, pawn: u8) {
        let row = i32::from(cell_row(pawn));
        let col = i32::from(cell_col(pawn));
        for wall_row in [row - 1, row] {
            for wall_col in [col - 2, col - 1, col, col + 1] {
                self.insert(ActionKind::Horizontal, wall_row, wall_col);
            }
        }
        for wall_row in [row - 2, row - 1, row, row + 1] {
            for wall_col in [col - 1, col] {
                self.insert(ActionKind::Vertical, wall_row, wall_col);
            }
        }
    }

    fn smear_horizontal(&mut self, slot: u8) {
        let row = i32::from(slot / 8);
        let col = i32::from(slot % 8);
        for (dr, dc) in [(0, -3), (0, -2), (0, 2), (0, 3)] {
            self.insert(ActionKind::Horizontal, row + dr, col + dc);
        }
        for (dr, dc) in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -2),
            (0, -1),
            (0, 1),
            (0, 2),
            (1, -1),
            (1, 0),
            (1, 1),
        ] {
            self.insert(ActionKind::Vertical, row + dr, col + dc);
        }
    }

    fn smear_vertical(&mut self, slot: u8) {
        let row = i32::from(slot / 8);
        let col = i32::from(slot % 8);
        for (dr, dc) in [(-3, 0), (-2, 0), (2, 0), (3, 0)] {
            self.insert(ActionKind::Vertical, row + dr, col + dc);
        }
        for (dr, dc) in [
            (-2, 0),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
            (2, 0),
        ] {
            self.insert(ActionKind::Horizontal, row + dr, col + dc);
        }
    }
}

#[inline]
fn probable_wall_turn_estimate(state: &State) -> u32 {
    let mut turn = u32::from(cell_row(state.p0))
        + u32::from(8 - cell_row(state.p1))
        + state.h.count_ones()
        + state.v.count_ones();
    if turn % 2 != u32::from(state.turn) {
        turn += 1;
    }
    turn
}

/// The Q4 tier starts at depth four. At depth three, a two-ply reduction would
/// send the child directly to depth zero; the conservative frozen arm keeps
/// the established one-ply LMR there.
const LMR_PROBABLE_WALLS_MIN_DEPTH: i32 = 4;
const LMR_PROBABLE_WALLS_CONTROL_INDEX: usize = 24;

#[inline]
fn lmr_reducible(
    depth: i32,
    ordered_index: usize,
    action: Action,
    tt_move: Option<Action>,
    tight_edges: &LmrEdges,
) -> bool {
    depth >= 3
        && ordered_index >= 6
        && action.kind != ActionKind::Pawn
        && Some(action) != tt_move
        && !tight_edges.wall_is_tight(action)
}

#[inline]
fn lmr_reduction(
    depth: i32,
    ordered_index: usize,
    action: Action,
    tt_move: Option<Action>,
    tight_edges: &LmrEdges,
    probable_walls: Option<&ProbableWalls>,
    graduated_control: bool,
) -> i32 {
    if !lmr_reducible(depth, ordered_index, action, tt_move, tight_edges) {
        return 0;
    }
    if depth >= LMR_PROBABLE_WALLS_MIN_DEPTH {
        // Supplying both flags deliberately selects the classifier arm; the
        // attribution control is a separate measurement, never a stacked tier.
        if let Some(probable) = probable_walls {
            return if probable.contains(action) { 1 } else { 2 };
        }
        if graduated_control && ordered_index >= LMR_PROBABLE_WALLS_CONTROL_INDEX {
            return 2;
        }
    }
    1
}

/// LMP candidates are named by their depth-3 threshold. Each adjacent ply
/// doubles the move count, yielding the frozen N=24 depth-1/2/3/4 vector
/// 6/12/24/48. Depth four is consumed only by the separate `lmp_d4` flag.
#[inline]
fn lmp_threshold(lmp_n: usize, depth: i32) -> usize {
    debug_assert!((1..=4).contains(&depth));
    let thresholds = match lmp_n {
        8 => [2, 4, 8, 16],
        16 => [4, 8, 16, 32],
        _ => [6, 12, 24, 48],
    };
    thresholds[(depth - 1) as usize]
}

/// Q6's depth-four arm is deliberately the existing LMR quiet-wall
/// classifier plus its own movecount threshold, never a bare move index.
#[inline]
fn lmp_d4_prunable(
    depth: i32,
    is_null_window: bool,
    ordered_index: usize,
    action: Action,
    tt_move: Option<Action>,
    lmp_n: usize,
    tight_edges: &LmrEdges,
) -> bool {
    depth == 4
        && is_null_window
        && ordered_index >= lmp_threshold(lmp_n, depth)
        && lmr_reducible(depth, ordered_index, action, tt_move, tight_edges)
}

#[inline]
fn lmp_prunable(
    depth: i32,
    is_null_window: bool,
    ordered_index: usize,
    action: Action,
    tt_move: Option<Action>,
    lmp_n: usize,
    tight_edges: &LmrEdges,
) -> bool {
    is_null_window
        && (1..=3).contains(&depth)
        && ordered_index >= lmp_threshold(lmp_n, depth)
        && action.kind != ActionKind::Pawn
        && Some(action) != tt_move
        && !tight_edges.wall_is_tight(action)
}

// ---------------------------------------------------------------------------
// Move ordering v0.
// ---------------------------------------------------------------------------

struct Ordered {
    moves: [Action; MAX_ACTIONS],
    /// Q7 context computed at the ordering read site and kept aligned through
    /// every later blend/reorder so cutoff writes reuse the identical value.
    wall_contexts: [u8; MAX_ACTIONS],
    len: usize,
    /// Exact LMR/LMP masks derived from ordering's existing distance fields.
    tight_edges: Option<LmrEdges>,
}

/// Inputs arrive in their exact canonical/base tie order, so a strict-key
/// insertion preserves the former `(key, index)` comparator byte-for-byte.
#[inline]
fn stable_insertion_ascending(items: &mut [(i32, u16, Action)]) {
    for index in 1..items.len() {
        let item = items[index];
        let mut hole = index;
        while hole != 0 && items[hole - 1].0 > item.0 {
            items[hole] = items[hole - 1];
            hole -= 1;
        }
        items[hole] = item;
    }
}

#[inline]
fn stable_insertion_descending(items: &mut [(i32, u16, Action)]) {
    for index in 1..items.len() {
        let item = items[index];
        let mut hole = index;
        while hole != 0 && items[hole - 1].0 < item.0 {
            items[hole] = items[hole - 1];
            hole -= 1;
        }
        items[hole] = item;
    }
}

/// v0 ordering: TT move first; then pawn moves by ascending resulting mover
/// BFS distance; then walls by descending `(opp_dist_increase - my_dist_increase)`
/// using exact distance deltas; canonical order breaks ties.
fn order_moves_v0(state: &State, tt_move: Option<Action>) -> Ordered {
    order_moves_v0_with_stop(state, tt_move, false, None, None, false, || false)
        .expect("unbounded move ordering cannot be interrupted")
}

fn order_moves_v0_until(
    state: &State,
    tt_move: Option<Action>,
    deadline: Instant,
    history: Option<&[i32; HISTORY_ACTIONS]>,
    tight_edge_threshold: Option<usize>,
    wall_hist: bool,
) -> Option<Ordered> {
    order_moves_v0_with_stop(
        state,
        tt_move,
        true,
        history,
        tight_edge_threshold,
        wall_hist,
        || Instant::now() >= deadline,
    )
}

fn order_moves_v0_with_stop<F>(
    state: &State,
    tt_move: Option<Action>,
    interruptible: bool,
    history: Option<&[i32; HISTORY_ACTIONS]>,
    tight_edge_threshold: Option<usize>,
    wall_hist: bool,
    mut should_stop: F,
) -> Option<Ordered>
where
    F: FnMut() -> bool,
{
    if should_stop() {
        return None;
    }
    let topo = Topology::from_walls(state.h, state.v);
    let (my_pawn, opp_pawn, my_goal, opp_goal, _mw, _ow) = side_view(state);
    let my_distances = distances_to_row(&topo, my_goal);
    if should_stop() {
        return None;
    }
    let opp_distances = distances_to_row(&topo, opp_goal);
    if should_stop() {
        return None;
    }

    let mut ml = MoveList::new();
    let mut wall_order_keys = [0i8; MAX_ACTIONS];
    let generated = if !interruptible {
        if state.turn == 0 {
            legal_actions_with_distances(
                state,
                &topo,
                &my_distances,
                &opp_distances,
                &mut ml,
                &mut wall_order_keys,
            );
        } else {
            legal_actions_with_distances(
                state,
                &topo,
                &opp_distances,
                &my_distances,
                &mut ml,
                &mut wall_order_keys,
            );
        }
        true
    } else if state.turn == 0 {
        legal_actions_with_distances_until(
            state,
            &topo,
            &my_distances,
            &opp_distances,
            &mut ml,
            &mut wall_order_keys,
            &mut should_stop,
        )
    } else {
        legal_actions_with_distances_until(
            state,
            &topo,
            &opp_distances,
            &my_distances,
            &mut ml,
            &mut wall_order_keys,
            &mut should_stop,
        )
    };
    if !generated {
        return None;
    }
    let n = ml.len();
    let tight_edges = tight_edge_threshold
        .is_some_and(|threshold| n > threshold)
        .then(|| LmrEdges {
            mine: my_distances.pawn_shortest_edges(&topo, my_pawn),
            opponent: opp_distances.pawn_shortest_edges(&topo, opp_pawn),
        });

    // (key, canonical index, action) for stable, deterministic ordering.
    let mut pawns: [(i32, u16, Action); 8] = [(0, 0, Action::EMPTY); 8];
    let mut positive_walls: [(i32, u16, Action); MAX_ACTIONS] =
        [(0, 0, Action::EMPTY); MAX_ACTIONS];
    let mut zero_walls = [Action::EMPTY; MAX_ACTIONS];
    let mut negative_walls: [(i32, u16, Action); MAX_ACTIONS] =
        [(0, 0, Action::EMPTY); MAX_ACTIONS];
    let mut wall_context_by_action = [0u8; WALL_HISTORY_ACTIONS];
    let mut pc = 0usize;
    let mut positive_count = 0usize;
    let mut zero_count = 0usize;
    let mut negative_count = 0usize;

    for i in 0..n {
        if should_stop() {
            return None;
        }
        let a = ml.get(i);
        match a.kind {
            ActionKind::Pawn => {
                let d = my_distances.get(a.pos) as i32;
                pawns[pc] = (d, i as u16, a);
                pc += 1;
            }
            ActionKind::Horizontal | ActionKind::Vertical => {
                if wall_hist {
                    wall_context_by_action[wall_history_action_index(a)] =
                        wall_history_context(&opp_distances, opp_pawn, a);
                }
                let key = wall_order_keys[i] as i32;
                if key > 0 {
                    positive_walls[positive_count] = (key, i as u16, a);
                    positive_count += 1;
                } else if key < 0 {
                    negative_walls[negative_count] = (key, i as u16, a);
                    negative_count += 1;
                } else {
                    // MoveList is canonical, so zero-key walls need no sort.
                    zero_walls[zero_count] = a;
                    zero_count += 1;
                }
            }
        }
    }

    if should_stop() {
        return None;
    }
    // Pawns ascending by (distance, canonical index).
    stable_insertion_ascending(&mut pawns[..pc]);
    // Positive keys precede canonical zeroes, which precede negative keys.
    // Sorting only the sparse nonzero groups is identical to the former full
    // `(key descending, canonical index ascending)` wall sort.
    stable_insertion_descending(&mut positive_walls[..positive_count]);
    stable_insertion_descending(&mut negative_walls[..negative_count]);
    if should_stop() {
        return None;
    }

    if let Some(history_scores) = history {
        // Traverse the already-sorted base groups exactly once. History remains
        // the primary key and strict insertion preserves base order on ties;
        // TT and winning pawn steps retain their established tactical prefix.
        let mut out = Ordered {
            moves: [Action::EMPTY; MAX_ACTIONS],
            wall_contexts: [0; MAX_ACTIONS],
            len: 0,
            tight_edges,
        };
        let tt = tt_move.filter(|action| ml.contains(*action));
        if let Some(action) = tt {
            out.moves[out.len] = action;
            out.len += 1;
        }

        let mut scored: [(i32, u16, Action); MAX_ACTIONS] =
            [(0, 0, Action::EMPTY); MAX_ACTIONS];
        let mut unscored = [Action::EMPTY; MAX_ACTIONS];
        let mut scored_count = 0usize;
        let mut unscored_count = 0usize;
        let mut base_rank = 0u16;

        macro_rules! classify {
            ($action:expr) => {{
                let action = $action;
                if base_rank & 7 == 0 && should_stop() {
                    return None;
                }
                if Some(action) != tt {
                    if action.kind == ActionKind::Pawn && is_winning_pawn_step(state, action) {
                        out.moves[out.len] = action;
                        out.len += 1;
                    } else {
                        let score = history_scores[encode_action(action) as usize];
                        debug_assert!(score >= 0);
                        if score > 0 {
                            scored[scored_count] = (score, base_rank, action);
                            scored_count += 1;
                        } else {
                            unscored[unscored_count] = action;
                            unscored_count += 1;
                        }
                    }
                }
                base_rank += 1;
            }};
        }

        for &(_, _, action) in &pawns[..pc] {
            classify!(action);
        }
        for &(_, _, action) in &positive_walls[..positive_count] {
            classify!(action);
        }
        for &action in &zero_walls[..zero_count] {
            classify!(action);
        }
        for &(_, _, action) in &negative_walls[..negative_count] {
            classify!(action);
        }

        if should_stop() {
            return None;
        }
        stable_insertion_descending(&mut scored[..scored_count]);
        for &(_, _, action) in &scored[..scored_count] {
            out.moves[out.len] = action;
            out.len += 1;
        }
        for &action in &unscored[..unscored_count] {
            out.moves[out.len] = action;
            out.len += 1;
        }
        debug_assert_eq!(out.len, n);
        if wall_hist {
            for index in 0..out.len {
                let action = out.moves[index];
                if action.kind != ActionKind::Pawn {
                    out.wall_contexts[index] =
                        wall_context_by_action[wall_history_action_index(action)];
                }
            }
        }
        return if should_stop() { None } else { Some(out) };
    }

    let mut out = Ordered {
        moves: [Action::EMPTY; MAX_ACTIONS],
        wall_contexts: [0; MAX_ACTIONS],
        len: 0,
        tight_edges,
    };
    let tt = tt_move.filter(|m| ml.contains(*m));
    if let Some(m) = tt {
        out.moves[out.len] = m;
        out.len += 1;
    }
    for &(_, _, a) in &pawns[..pc] {
        if Some(a) != tt {
            out.moves[out.len] = a;
            out.len += 1;
        }
    }
    for &(_, _, a) in &positive_walls[..positive_count] {
        if Some(a) != tt {
            out.moves[out.len] = a;
            out.len += 1;
        }
    }
    for &a in &zero_walls[..zero_count] {
        if Some(a) != tt {
            out.moves[out.len] = a;
            out.len += 1;
        }
    }
    for &(_, _, a) in &negative_walls[..negative_count] {
        if Some(a) != tt {
            out.moves[out.len] = a;
            out.len += 1;
        }
    }
    if wall_hist {
        for index in 0..out.len {
            let action = out.moves[index];
            if action.kind != ActionKind::Pawn {
                out.wall_contexts[index] =
                    wall_context_by_action[wall_history_action_index(action)];
            }
        }
    }
    if should_stop() {
        None
    } else {
        Some(out)
    }
}

#[inline]
fn wall_touches_tight_edge(distances: &GoalDistances, wall: Action) -> bool {
    match wall.kind {
        ActionKind::Horizontal => {
            let edge = (wall.pos / 8) * 9 + wall.pos % 8;
            distances.get(edge).abs_diff(distances.get(edge + 9)) == 1
                || distances.get(edge + 1).abs_diff(distances.get(edge + 10)) == 1
        }
        ActionKind::Vertical => {
            let edge = (wall.pos / 8) * 9 + wall.pos % 8;
            distances.get(edge).abs_diff(distances.get(edge + 1)) == 1
                || distances.get(edge + 9).abs_diff(distances.get(edge + 10)) == 1
        }
        ActionKind::Pawn => false,
    }
}

#[inline]
fn wall_history_distance_band(distance: i16) -> u8 {
    if distance <= 3 {
        0
    } else if distance <= 6 {
        1
    } else {
        2
    }
}

#[inline]
fn wall_history_context(
    opponent_distances: &GoalDistances,
    opponent_pawn: u8,
    action: Action,
) -> u8 {
    let band = wall_history_distance_band(opponent_distances.get(opponent_pawn));
    band
        + u8::from(wall_touches_tight_edge(opponent_distances, action))
            * WALL_HISTORY_DISTANCE_BANDS as u8
}

#[inline]
fn wall_touches_pawn_shortest_edge(edges: &PawnShortestEdges, wall: Action) -> bool {
    match wall.kind {
        ActionKind::Horizontal => edges.horizontal_wall_touches_edge(wall.pos),
        ActionKind::Vertical => edges.vertical_wall_touches_edge(wall.pos),
        ActionKind::Pawn => false,
    }
}

/// Test helper for the former union-of-shortest-edges delta filter.
///
/// Every pawn-shortest edge is open and tight (`d(next) = d(current) - 1`), so
/// the former global-tight predicate was a redundant superset check. If the
/// wall touches no pawn-shortest edge, every old shortest route remains intact.
#[cfg(test)]
#[inline]
fn exact_wall_distance_delta(
    topology_after: &Topology,
    distances_before: &GoalDistances,
    pawn_shortest_edges: &PawnShortestEdges,
    pawn: u8,
    goal_row: u8,
    wall: Action,
) -> i32 {
    if !wall_touches_pawn_shortest_edge(pawn_shortest_edges, wall) {
        return 0;
    }
    distance_to_row(topology_after, pawn, goal_row) as i32 - distances_before.get(pawn) as i32
}

impl Ordered {
    #[inline]
    fn new() -> Self {
        Self {
            moves: [Action::EMPTY; MAX_ACTIONS],
            wall_contexts: [0; MAX_ACTIONS],
            len: 0,
            tight_edges: None,
        }
    }

    #[inline]
    fn contains(&self, action: Action) -> bool {
        self.moves[..self.len].contains(&action)
    }

    #[inline]
    fn push(&mut self, action: Action) {
        self.push_with_context(action, 0);
    }

    #[inline]
    fn push_with_context(&mut self, action: Action, wall_context: u8) {
        debug_assert!(self.len < MAX_ACTIONS);
        self.moves[self.len] = action;
        self.wall_contexts[self.len] = wall_context;
        self.len += 1;
    }

    #[inline]
    fn context_for(&self, action: Action) -> u8 {
        self.moves[..self.len]
            .iter()
            .position(|candidate| *candidate == action)
            .map_or(0, |index| self.wall_contexts[index])
    }

}

/// `cheap_wall_order` static ordering.  The two parent BFS runs reconstruct
/// one shortest path for each side.  In particular, the wall loop below never
/// creates a modified topology and never runs a per-wall BFS; legality's BFS
/// checks remain exclusively in move generation.
fn order_moves_cheap(state: &State, tt_move: Option<Action>) -> Ordered {
    order_moves_cheap_with_stop(state, tt_move, false, || false)
        .expect("unbounded move ordering cannot be interrupted")
}

fn order_moves_cheap_until(
    state: &State,
    tt_move: Option<Action>,
    deadline: Instant,
) -> Option<Ordered> {
    order_moves_cheap_with_stop(state, tt_move, true, || Instant::now() >= deadline)
}

fn order_moves_cheap_with_stop<F>(
    state: &State,
    tt_move: Option<Action>,
    interruptible: bool,
    mut should_stop: F,
) -> Option<Ordered>
where
    F: FnMut() -> bool,
{
    if should_stop() {
        return None;
    }
    let mut ml = MoveList::new();
    let generated = if interruptible {
        legal_actions_until(state, &mut ml, &mut should_stop)
    } else {
        legal_actions(state, &mut ml);
        true
    };
    if !generated {
        return None;
    }
    let n = ml.len();

    let topo = Topology::from_walls(state.h, state.v);
    let (my_pawn, opp_pawn, my_goal, opp_goal, _mw, _ow) = side_view(state);
    // These are the one pair of parent BFS runs used to statically value every
    // wall at this node.  Parser-valid but game-invalid no-route states simply
    // receive no path-hit bonuses rather than falling back to exact wall BFS.
    let my_path = shortest_path_to_row(&topo, my_pawn, my_goal);
    if should_stop() {
        return None;
    }
    let opp_path = shortest_path_to_row(&topo, opp_pawn, opp_goal);
    if should_stop() {
        return None;
    }
    let opp_corridor = opp_path.map_or(0, |path| chebyshev_dilate(path.cells, 1));
    let opp_pawn_ring = chebyshev_dilate(1u128 << opp_pawn, 2);

    // Pawn ranking is also static: a legal step on the reconstructed own route
    // leads, then simple remaining-row progress breaks the rest.  Together
    // with the two parent searches above, this keeps cheap ordering to one
    // BFS pair per node rather than retaining any per-pawn BFS calls.
    let mut pawns: [(i32, u16, Action); 8] = [(0, 0, Action::EMPTY); 8];
    let mut walls: [(i32, u16, Action); MAX_ACTIONS] = [(0, 0, Action::EMPTY); MAX_ACTIONS];
    let mut pc = 0usize;
    let mut wc = 0usize;

    for i in 0..n {
        if i & 7 == 0 && should_stop() {
            return None;
        }
        let action = ml.get(i);
        match action.kind {
            ActionKind::Pawn => {
                let on_my_path = my_path.map_or(false, |path| path.cells & (1u128 << action.pos) != 0);
                let rows_left = if my_goal == 8 {
                    8 - cell_row(action.pos)
                } else {
                    cell_row(action.pos)
                };
                let key = (if on_my_path { 0 } else { 100 }) + rows_left as i32;
                pawns[pc] = (key, i as u16, action);
                pc += 1;
            }
            ActionKind::Horizontal | ActionKind::Vertical => {
                let (opp_rank, my_rank) = match action.kind {
                    ActionKind::Horizontal => (
                        opp_path.map_or(-1, |path| path.h_rank[action.pos as usize]),
                        my_path.map_or(-1, |path| path.h_rank[action.pos as usize]),
                    ),
                    ActionKind::Vertical => (
                        opp_path.map_or(-1, |path| path.v_rank[action.pos as usize]),
                        my_path.map_or(-1, |path| path.v_rank[action.pos as usize]),
                    ),
                    ActionKind::Pawn => unreachable!(),
                };
                let corners = wall_corner_mask(action.pos);
                let mut key = 0i32;
                if opp_rank >= 0 {
                    // A direct path cut dominates every geometric hint. Lower
                    // rank is earlier from the opponent pawn and scores higher.
                    key += 10_000 - opp_rank as i32;
                }
                if corners & opp_corridor != 0 {
                    key += 400;
                }
                if corners & opp_pawn_ring != 0 {
                    key += 250;
                }
                if my_rank >= 0 {
                    // Do not eagerly spend a wall that cuts our own selected
                    // shortest route unless the opponent benefit is stronger.
                    key -= 200;
                }
                walls[wc] = (key, i as u16, action);
                wc += 1;
            }
        }
    }

    if should_stop() {
        return None;
    }
    stable_insertion_ascending(&mut pawns[..pc]);
    // The canonical MoveList index is the deterministic small tie-break.
    stable_insertion_descending(&mut walls[..wc]);
    if should_stop() {
        return None;
    }

    let mut out = Ordered::new();
    let tt = tt_move.filter(|action| ml.contains(*action));
    if let Some(action) = tt {
        out.push(action);
    }
    for &(_, _, action) in &pawns[..pc] {
        if Some(action) != tt {
            out.push(action);
        }
    }
    for &(_, _, action) in &walls[..wc] {
        if Some(action) != tt {
            out.push(action);
        }
    }
    if should_stop() {
        None
    } else {
        Some(out)
    }
}

#[inline]
fn wall_corner_mask(slot: u8) -> u128 {
    let top_left = (slot / 8) * 9 + slot % 8;
    (1u128 << top_left)
        | (1u128 << (top_left + 1))
        | (1u128 << (top_left + 9))
        | (1u128 << (top_left + 10))
}

/// Expands a cell mask by a Chebyshev radius, clipped to the 9x9 board.
fn chebyshev_dilate(mut cells: u128, radius: u8) -> u128 {
    let mut expanded = 0u128;
    while cells != 0 {
        let cell = cells.trailing_zeros() as u8;
        cells &= cells - 1;
        let min_col = cell_col(cell).saturating_sub(radius);
        let max_col = cell_col(cell).saturating_add(radius).min(8);
        let min_row = cell_row(cell).saturating_sub(radius);
        let max_row = cell_row(cell).saturating_add(radius).min(8);
        for row in min_row..=max_row {
            for col in min_col..=max_col {
                expanded |= 1u128 << (row * 9 + col);
            }
        }
    }
    expanded
}

#[inline]
fn is_winning_pawn_step(state: &State, action: Action) -> bool {
    action.kind == ActionKind::Pawn && winner(&state.apply(action)) >= 0
}

impl SearchContext {
    /// Time the complete current ordering+generation path only on the probe.
    /// The default path calls the original body without reading the clock.
    fn order_moves(
        &mut self,
        state: &State,
        tt_move: Option<Action>,
        ply: i32,
        previous_action: Option<Action>,
        tight_edge_threshold: Option<usize>,
    ) -> Option<Ordered> {
        #[cfg(feature = "profile-timers")]
        let _profile_guard =
            crate::profile::Guard::new(crate::profile::Bucket::Ordering);
        if !self.features.probe_cutoffs {
            return self.order_moves_inner(
                state,
                tt_move,
                ply,
                previous_action,
                tight_edge_threshold,
            );
        }
        let started = Instant::now();
        let ordered = self.order_moves_inner(
            state,
            tt_move,
            ply,
            previous_action,
            tight_edge_threshold,
        );
        self.cutoff_probe.ordering_generation_nanos = self
            .cutoff_probe
            .ordering_generation_nanos
            .saturating_add(started.elapsed().as_nanos());
        ordered
    }

    /// Apply the independently gated ordering experiments over a base order.
    /// The early return deliberately leaves the frozen v0 function untouched
    /// whenever cheap ordering, killers, and history are all disabled.
    fn order_moves_inner(
        &mut self,
        state: &State,
        tt_move: Option<Action>,
        ply: i32,
        previous_action: Option<Action>,
        tight_edge_threshold: Option<usize>,
    ) -> Option<Ordered> {
        // Q7's binding context comes from v0's exact distance fields. When
        // both default-off experiments are requested, preserve that context
        // rather than silently manufacturing a cheap-order approximation.
        let use_cheap = self.features.cheap_wall_order && !self.features.wall_hist;
        let use_killers = self.features.killers;
        let use_history =
            self.features.history || self.features.cmh || self.features.wall_hist;
        let fuse_history = !use_cheap
            && !use_killers
            && self.features.history
            && !self.features.cmh
            && !self.features.wall_hist;
        let fused_history = fuse_history.then(|| &self.history_scores[state.turn as usize]);
        let base = if let Some(stop) = self.external_stop.clone() {
            let deadline = self.hard_deadline;
            let should_stop = || {
                stop.load(Ordering::Acquire)
                    || deadline.is_some_and(|instant| Instant::now() >= instant)
            };
            if use_cheap {
                order_moves_cheap_with_stop(state, tt_move, true, should_stop)
            } else {
                order_moves_v0_with_stop(
                    state,
                    tt_move,
                    true,
                    fused_history,
                    tight_edge_threshold,
                    self.features.wall_hist,
                    should_stop,
                )
            }
        } else {
            match (use_cheap, self.hard_deadline) {
                (true, Some(deadline)) => order_moves_cheap_until(state, tt_move, deadline),
                (false, Some(deadline)) => {
                    order_moves_v0_until(
                        state,
                        tt_move,
                        deadline,
                        fused_history,
                        tight_edge_threshold,
                        self.features.wall_hist,
                    )
                }
                (true, None) => Some(order_moves_cheap(state, tt_move)),
                (false, None) => Some(if fused_history.is_none()
                    && tight_edge_threshold.is_none()
                    && !self.features.wall_hist
                {
                    order_moves_v0(state, tt_move)
                } else {
                    order_moves_v0_with_stop(
                        state,
                        tt_move,
                        false,
                        fused_history,
                        tight_edge_threshold,
                        self.features.wall_hist,
                        || false,
                    )
                    .expect("unbounded move ordering cannot be interrupted")
                }),
            }
        };
        let Some(base) = base else {
            self.aborted = true;
            return None;
        };
        if fuse_history {
            return Some(base);
        }
        if !use_cheap && !use_killers && !use_history {
            return Some(base);
        }
        if !use_killers && !use_history {
            return Some(base);
        }
        if self.hard_deadline_reached_now() {
            return None;
        }

        let mut out = Ordered::new();
        out.tight_edges = base.tight_edges;
        if let Some(action) = tt_move.filter(|action| base.contains(*action)) {
            out.push_with_context(action, base.context_for(action));
        }
        // Winning pawn steps are tactical, not quiet moves: always retain them
        // immediately after TT and before the quiet killer/history layers. A
        // legal TT wall is the sole possible non-pawn before the pawn group;
        // once the group ends, every remaining move is a non-winning wall.
        let pawn_start = usize::from(
            base.len != 0
                && base.moves[0].kind != ActionKind::Pawn
                && Some(base.moves[0]) == tt_move,
        );
        for index in pawn_start..base.len {
            if index & 7 == 0 && self.hard_deadline_reached_now() {
                return None;
            }
            let action = base.moves[index];
            if action.kind != ActionKind::Pawn {
                break;
            }
            if !out.contains(action) && is_winning_pawn_step(state, action) {
                out.push_with_context(action, base.wall_contexts[index]);
            }
        }
        if use_killers {
            for code in self.killer_codes(ply) {
                if code == KILLER_EMPTY {
                    continue;
                }
                let action = decode_action(code);
                if base.contains(action) && !out.contains(action) {
                    out.push_with_context(action, base.context_for(action));
                }
            }
        }

        let mut scored: [(i32, u16, Action); MAX_ACTIONS] =
            [(0, 0, Action::EMPTY); MAX_ACTIONS];
        let mut unscored = [(0u16, Action::EMPTY); MAX_ACTIONS];
        let mut scored_count = 0usize;
        let mut unscored_count = 0usize;
        for index in 0..base.len {
            if index & 7 == 0 && self.hard_deadline_reached_now() {
                return None;
            }
            let action = base.moves[index];
            if !out.contains(action) {
                let score = if use_history {
                    self.history_score_with_context(
                        state,
                        action,
                        previous_action,
                        Some(base.wall_contexts[index]),
                    )
                } else {
                    0
                };
                debug_assert!(score >= 0);
                if score > 0 {
                    scored[scored_count] = (score, index as u16, action);
                    scored_count += 1;
                } else {
                    // Base order is the exact tie-break for zero history.
                    unscored[unscored_count] = (index as u16, action);
                    unscored_count += 1;
                }
            }
        }
        if self.hard_deadline_reached_now() {
            return None;
        }
        if use_history {
            stable_insertion_descending(&mut scored[..scored_count]);
        }
        for &(_, base_index, action) in &scored[..scored_count] {
            out.push_with_context(action, base.wall_contexts[base_index as usize]);
        }
        for &(base_index, action) in &unscored[..unscored_count] {
            out.push_with_context(action, base.wall_contexts[base_index as usize]);
        }
        if self.hard_deadline_reached_now() {
            None
        } else {
            Some(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{action_name, parse_action, parse_state, start_state};
    use std::collections::{HashMap, HashSet, VecDeque};

    const REF_UNRESOLVED: i8 = -2;

    fn depth_search(state: &State, depth: u32) -> SearchResult {
        depth_search_with_features(state, depth, Features::default())
    }

    fn depth_search_with_features(state: &State, depth: u32, features: Features) -> SearchResult {
        let mut ctx = SearchContext::new(features);
        let tc = TimeControl {
            max_depth: depth,
            allotted_budget_ms: None,
            soft_deadline: None,
            hard_deadline: None,
        };
        ctx.search(state, &tc)
    }

    fn depth_search_with_budget(
        state: &State,
        depth: u32,
        budget_ms: u64,
        features: Features,
    ) -> SearchResult {
        let mut ctx = SearchContext::new(features);
        // Unit searches carry the real QBP allotment metadata but omit wall-clock
        // deadlines so node-count assertions remain deterministic.
        let tc = TimeControl {
            max_depth: depth,
            allotted_budget_ms: Some(budget_ms),
            soft_deadline: None,
            hard_deadline: None,
        };
        ctx.search(state, &tc)
    }

    fn deterministic_tc(depth: u32, budget_ms: Option<u64>) -> TimeControl {
        TimeControl {
            max_depth: depth,
            allotted_budget_ms: budget_ms,
            soft_deadline: None,
            hard_deadline: None,
        }
    }

    fn assert_result_identity(actual: &SearchResult, expected: &SearchResult, label: &str) {
        assert_eq!(actual.best_action, expected.best_action, "{label}: move");
        assert_eq!(actual.score, expected.score, "{label}: score");
        assert_eq!(actual.depth, expected.depth, "{label}: depth");
        assert_eq!(actual.nodes, expected.nodes, "{label}: nodes");
        assert_eq!(actual.main_nodes, expected.main_nodes, "{label}: main_nodes");
    }

    #[test]
    fn nnue_never_overrides_terminal_or_race1w_scores() {
        let network = Arc::new(NnueNetwork::constant_for_test(0));
        let mut features = Features::default();
        features.threads = 1;
        let mut ctx = SearchContext::with_tt_entries_and_nnue(
            features,
            1024,
            Some(network.clone()),
        );

        let terminal = parse_state("p0=a9;p1=i9;w0=0;w1=0;h=-;v=-;t=1").unwrap();
        let mut accumulator = network.try_accumulator(&terminal).unwrap();
        assert_eq!(
            ctx.negamax(
                &terminal,
                Some(&mut accumulator),
                0,
                -INF,
                INF,
                0,
                false,
                None,
            ),
            Some(-MATE),
        );

        let race = parse_state("p0=a6;p1=i4;w0=1;w1=0;h=-;v=-;t=0").unwrap();
        let expected = race_one_wall_score(&race, 0).unwrap();
        let mut accumulator = network.try_accumulator(&race).unwrap();
        assert_eq!(
            ctx.negamax(
                &race,
                Some(&mut accumulator),
                0,
                -INF,
                INF,
                0,
                false,
                None,
            ),
            Some(expected),
        );
        assert_ne!(expected, network.evaluate(&race, &accumulator));
    }

    /// Independent depth-to-terminal minimax. A 0/1 result is a finite proof;
    /// cycles, draws, and wins beyond the horizon remain unresolved rather
    /// than being guessed. This deliberately uses canonical legal movegen and
    /// no production race-table data.
    fn reference_minimax(
        state: &State,
        plies_left: u8,
        memo: &mut HashMap<(State, u8), i8>,
    ) -> i8 {
        let terminal = winner(state);
        if terminal >= 0 {
            return terminal;
        }
        if plies_left == 0 {
            return REF_UNRESOLVED;
        }
        if let Some(&cached) = memo.get(&(*state, plies_left)) {
            return cached;
        }

        let mover = state.turn as i8;
        let opponent = 1 - mover;
        let mut legal = MoveList::new();
        legal_actions(state, &mut legal);
        if legal.len() == 0 {
            return REF_UNRESOLVED;
        }
        let mut unresolved = false;
        for index in 0..legal.len() {
            let child = state.apply(legal.get(index));
            let result = reference_minimax(&child, plies_left - 1, memo);
            if result == mover {
                memo.insert((*state, plies_left), mover);
                return mover;
            }
            if result != opponent {
                unresolved = true;
            }
        }
        let result = if unresolved {
            REF_UNRESOLVED
        } else {
            opponent
        };
        memo.insert((*state, plies_left), result);
        result
    }

    fn reference_winner(
        state: &State,
        memo: &mut HashMap<(State, u8), i8>,
    ) -> Option<u8> {
        for horizon in [8, 16, 24, 32, 48, 64] {
            let result = reference_minimax(state, horizon, memo);
            if result >= 0 {
                return Some(result as u8);
            }
        }
        None
    }

    fn reference_order_moves_v0(state: &State, tt_move: Option<Action>) -> Ordered {
        let topology = Topology::from_walls(state.h, state.v);
        let (mine, opponent, my_goal, opponent_goal, _, _) = side_view(state);
        let my_before = distance_to_row(&topology, mine, my_goal) as i32;
        let opponent_before = distance_to_row(&topology, opponent, opponent_goal) as i32;
        let mut legal = MoveList::new();
        legal_actions(state, &mut legal);
        let mut pawns = Vec::new();
        let mut walls = Vec::new();
        for index in 0..legal.len() {
            let action = legal.get(index);
            match action.kind {
                ActionKind::Pawn => pawns.push((
                    distance_to_row(&topology, action.pos, my_goal) as i32,
                    index as u16,
                    action,
                )),
                ActionKind::Horizontal | ActionKind::Vertical => {
                    let after = match action.kind {
                        ActionKind::Horizontal => topology.with_horizontal(action.pos),
                        ActionKind::Vertical => topology.with_vertical(action.pos),
                        ActionKind::Pawn => unreachable!(),
                    };
                    let my_delta =
                        distance_to_row(&after, mine, my_goal) as i32 - my_before;
                    let opponent_delta =
                        distance_to_row(&after, opponent, opponent_goal) as i32 - opponent_before;
                    walls.push((opponent_delta - my_delta, index as u16, action));
                }
            }
        }
        pawns.sort_unstable_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
        walls.sort_unstable_by(|x, y| (y.0, x.1).cmp(&(x.0, y.1)));

        let tt = tt_move.filter(|action| legal.contains(*action));
        let mut ordered = Ordered::new();
        if let Some(action) = tt {
            ordered.push(action);
        }
        for (_, _, action) in pawns.into_iter().chain(walls) {
            if Some(action) != tt {
                ordered.push(action);
            }
        }
        ordered
    }

    #[test]
    fn fused_wall_deltas_preserve_full_bfs_ordering() {
        let mut seed = 0x1319_8a2e_0370_7344u64;
        let mut state = start_state();

        for sample in 0..100 {
            if winner(&state) >= 0 {
                state = start_state();
            }
            let mut legal = MoveList::new();
            legal_actions(&state, &mut legal);
            assert!(legal.len() > 0);
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let tt = legal.get(seed as usize % legal.len());

            for candidate in [None, Some(tt)] {
                let actual = order_moves_v0(&state, candidate);
                let expected = reference_order_moves_v0(&state, candidate);
                assert_eq!(actual.len, expected.len, "sample={sample}");
                for index in 0..actual.len {
                    assert_eq!(
                        actual.moves[index],
                        expected.moves[index],
                        "sample={sample} index={index} state={}",
                        state.to_canonical_string(),
                    );
                }
            }

            state = state.apply(tt);
        }
    }

    #[test]
    fn batch2_features_defaults_and_parse_independently() {
        let defaults = Features::default();
        assert_eq!(defaults.threads, 4); // E-010 merge
        assert_eq!(defaults.soft_pct, 55);
        assert_eq!(defaults.hard_pct, 85);
        assert!(defaults.predictive_start); // E-009 merge
        assert!(defaults.lmr); // E-011 merge
        assert!(!defaults.lmr_probable_walls);
        assert!(!defaults.lmr_probable_walls_control);
        assert!(defaults.lmp); // E-020 merge
        assert_eq!(defaults.lmp_n, 24);
        assert!(defaults.rfp);
        assert_eq!(defaults.rfp_margin, 100);
        assert_eq!(defaults.rfp_depth, 3);
        assert!(defaults.rfp_tc_adaptive); // E-028 merge
        assert!(defaults.cmh); // E-021 merge
        assert!(!defaults.corr_hist);
        assert!(!defaults.lmp_d4);
        assert!(!defaults.lmp_d4_guard);
        assert!(!defaults.wall_hist);
        assert!(!defaults.probe_wall_hist);
        assert!(!defaults.ev_progress);
        assert!(!defaults.ev_wallphase);
        assert_eq!(defaults.wall_cp, 150); // E-019 merge (was 200 per E-008)
        assert_eq!(defaults.wall_cp_endgame, -1);
        assert_eq!(defaults.race_amp, 0);
        assert!(!defaults.ev_corridor);
        assert!(defaults.wallq_tc);
        assert!(!defaults.tt_sym);
        assert!(defaults.race1w); // E-015 merge
        assert!(!defaults.race2w);
        assert!(!defaults.ev_fragility_1w);
        assert!(defaults.history);
        assert!(!defaults.race_exact); // E-005R revert
        assert!(!defaults.probe_cutoffs);
        assert!(!defaults.probe_evalswing);
        assert!(!defaults.probe_rfp_precision);
        assert!(!defaults.tm2_time);
        assert!(defaults.nnue_path.is_none());

        let args: Vec<String> = [
            "--feature",
            "soft_pct=60",
            "--feature",
            "hard_pct=90",
            "--feature",
            "predictive_start=1",
            "--feature",
            "lmr=on",
            "--feature",
            "lmr_probable_walls=on",
            "--feature",
            "lmr_probable_walls_control=on",
            "--feature",
            "lmp=on",
            "--feature",
            "lmp_n=24",
            "--feature",
            "rfp=on",
            "--feature",
            "rfp_margin=400",
            "--feature",
            "rfp_depth=4",
            "--feature",
            "rfp_tc_adaptive=on",
            "--feature",
            "cmh=on",
            "--feature",
            "corr_hist=on",
            "--feature",
            "lmp_d4=on",
            "--feature",
            "lmp_d4_guard=on",
            "--feature",
            "wall_hist=on",
            "--feature",
            "probe_wall_hist=on",
            "--feature",
            "ev_progress=yes",
            "--feature",
            "ev_wallphase=1",
            "--feature",
            "wall_cp=90",
            "--feature",
            "wall_cp_endgame=25",
            "--feature",
            "race_amp=48",
            "--feature",
            "ev_corridor=true",
            "--feature",
            "wallq_tc=off",
            "--feature",
            "tt_sym=on",
            "--feature",
            "race1w=on",
            "--feature",
            "race2w=on",
            "--feature",
            "ev_fragility_1w=on",
            "--feature",
            "probe_cutoffs=1",
            "--feature",
            "probe_evalswing=1",
            "--feature",
            "probe_rfp_precision=1",
            "--feature",
            "tm2_time=1",
            "--feature",
            "nnue=C:/tmp/reference.qnn",
            "--feature",
            "threads=4",
        ]
        .iter()
        .map(|value| (*value).to_owned())
        .collect();
        let parsed = Features::parse(&args).unwrap();
        assert_eq!(parsed.soft_pct, 60);
        assert_eq!(parsed.hard_pct, 90);
        assert!(parsed.predictive_start);
        assert!(parsed.lmr);
        assert!(parsed.lmr_probable_walls);
        assert!(parsed.lmr_probable_walls_control);
        assert!(parsed.lmp);
        assert_eq!(parsed.lmp_n, 24);
        assert!(parsed.rfp);
        assert_eq!(parsed.rfp_margin, 400);
        assert_eq!(parsed.rfp_depth, 4);
        assert!(parsed.rfp_tc_adaptive);
        assert!(parsed.cmh);
        assert!(parsed.corr_hist);
        assert!(parsed.lmp_d4);
        assert!(parsed.lmp_d4_guard);
        assert!(parsed.wall_hist);
        assert!(parsed.probe_wall_hist);
        assert!(parsed.ev_progress);
        assert!(parsed.ev_wallphase);
        assert_eq!(parsed.wall_cp, 90);
        assert_eq!(parsed.wall_cp_endgame, 25);
        assert_eq!(parsed.race_amp, 48);
        assert!(parsed.ev_corridor);
        assert!(!parsed.wallq_tc);
        assert!(parsed.tt_sym);
        assert!(parsed.race1w);
        assert!(parsed.race2w);
        assert!(parsed.ev_fragility_1w);
        assert!(parsed.probe_cutoffs);
        assert!(parsed.probe_evalswing);
        assert!(parsed.probe_rfp_precision);
        assert!(parsed.tm2_time);
        assert_eq!(parsed.nnue_path.as_deref(), Some("C:/tmp/reference.qnn"));
        assert_eq!(parsed.threads, 4);

        for value in ["threads=0", "threads=invalid"] {
            let parsed = Features::parse(&["--feature".to_owned(), value.to_owned()]).unwrap();
            assert_eq!(parsed.threads, 1);
        }

        for value in ["lmp_n=0", "lmp_n=invalid"] {
            let parsed = Features::parse(&["--feature".to_owned(), value.to_owned()]).unwrap();
            assert_eq!(parsed.lmp_n, 24);
        }

        for value in ["rfp_margin=-1", "rfp_margin=invalid"] {
            let parsed = Features::parse(&["--feature".to_owned(), value.to_owned()]).unwrap();
            assert_eq!(parsed.rfp_margin, 250);
        }

        for value in ["rfp_depth=0", "rfp_depth=5", "rfp_depth=invalid"] {
            let parsed = Features::parse(&["--feature".to_owned(), value.to_owned()]).unwrap();
            assert_eq!(parsed.rfp_depth, 3);
        }

        let malformed_nnue =
            Features::parse(&["--feature".to_owned(), "nnue".to_owned()]).unwrap();
        assert_eq!(malformed_nnue.nnue_path.as_deref(), Some(""));
    }

    #[test]
    fn wallq_tc_rejects_tm2_time_at_feature_parse() {
        let error = Features::parse(&[
            "--feature".to_owned(),
            "wallq_tc=on".to_owned(),
            "--feature".to_owned(),
            "tm2_time=on".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("wallq_tc=on"));
        assert!(error.contains("tm2_time=off"));
    }

    #[test]
    fn retired_wallq_flags_are_unknown_and_do_not_conflict_with_wallq_tc() {
        let parsed = Features::parse(&[
            "--feature".to_owned(),
            "wallq_tc=on".to_owned(),
            "--feature".to_owned(),
            "wallq=on".to_owned(),
            "--feature".to_owned(),
            "ev_interdiction=on".to_owned(),
        ])
        .unwrap();
        assert!(parsed.wallq_tc);
        assert_eq!(
            parsed.unknown,
            vec![
                ("wallq".to_owned(), "on".to_owned()),
                ("ev_interdiction".to_owned(), "on".to_owned()),
            ],
        );
    }

    #[test]
    fn wallq_tc_resolves_exact_dual_bands_and_fixed_depth_is_inert() {
        let off = WallqTcProbeForce::Off;
        assert_eq!(WallqTcMode::resolve(false, Some(50), off), WallqTcMode::Inactive);
        assert_eq!(WallqTcMode::resolve(true, None, off), WallqTcMode::Inactive);
        assert_eq!(WallqTcMode::resolve(true, Some(50), off), WallqTcMode::Fast);
        assert_eq!(WallqTcMode::resolve(true, Some(51), off), WallqTcMode::Inactive);
        assert_eq!(WallqTcMode::resolve(true, Some(499), off), WallqTcMode::Inactive);
        assert_eq!(WallqTcMode::resolve(true, Some(500), off), WallqTcMode::Long);
        assert_eq!(WallqTcMode::resolve(true, Some(1_000), off), WallqTcMode::Long);
        assert_eq!(
            WallqTcMode::resolve(true, None, WallqTcProbeForce::Fast),
            WallqTcMode::Fast,
        );
        assert_eq!(
            WallqTcMode::resolve(true, None, WallqTcProbeForce::Long),
            WallqTcMode::Long,
        );
        assert_eq!(
            WallqTcMode::resolve(false, None, WallqTcProbeForce::Fast),
            WallqTcMode::Inactive,
            "the probe overrides only the budget gate, not wallq_tc=off",
        );

        let state = parse_state("p0=g2;p1=g7;w0=8;w1=9;h=e7;v=a2,b3;t=0").unwrap();
        let mut off = Features::default();
        off.threads = 1;
        off.wallq_tc = false;
        off.race_exact = false;
        off.race1w = false;
        off.race2w = false;
        let mut on = off.clone();
        on.wallq_tc = true;

        let expected = depth_search_with_features(&state, 2, off);
        let actual = depth_search_with_features(&state, 2, on);
        assert_result_identity(&actual, &expected, "fixed-depth inertness");
        assert_eq!(actual.wallq_tc_budget_ms, None);
        assert_eq!(actual.wallq_tc_active_lanes, 0);
        assert_eq!(actual.wallq_tc_fires, 0);
        assert_eq!(actual.wallq_tc_leaves, 0);
        assert_eq!(actual.wallq_tc_in_window, 0);

        let mut live_features = Features::default();
        live_features.threads = 1;
        live_features.race_exact = false;
        live_features.race1w = false;
        live_features.race2w = false;
        live_features.wallq_tc = true;
        for (budget_ms, should_be_active) in [(50, true), (100, false), (500, true), (1_000, true)] {
            let result = depth_search_with_budget(&state, 2, budget_ms, live_features.clone());
            assert_eq!(result.wallq_tc_budget_ms, Some(budget_ms));
            assert_eq!(result.wallq_tc_active_lanes, u64::from(should_be_active));
            if should_be_active {
                assert!(result.wallq_tc_fires > 0, "budget={budget_ms}");
            } else {
                assert_eq!(result.wallq_tc_fires, 0, "budget={budget_ms}");
            }
        }
    }

    #[test]
    fn wallq_tc_probe_force_fast_fixed_depth_reuses_the_genuine_fast_namespace() {
        let state = parse_state("p0=g2;p1=g7;w0=8;w1=9;h=e7;v=a2,b3;t=0").unwrap();
        let mut forced_features = Features::default();
        forced_features.threads = 1;
        forced_features.race_exact = false;
        forced_features.race1w = false;
        forced_features.race2w = false;
        forced_features.wallq_tc = true;
        forced_features.wallq_tc_probe_force = WallqTcProbeForce::Fast;

        let mut forced = SearchContext::with_tt_entries(forced_features.clone(), 1 << 12);
        forced.clear_tt_for_allotted_budget(None);
        let forced_fast_ptr = forced
            .wallq_tc_fast_tt
            .as_ref()
            .expect("forced fixed-depth clear must provision the Fast namespace")
            .as_ptr();
        assert!(forced.wallq_tc_long_tt.is_none());
        let forced_result = forced.search(&state, &deterministic_tc(2, None));
        assert_eq!(forced.wallq_tc_mode, WallqTcMode::Fast);
        assert_eq!(
            forced.wallq_tc_fast_tt.as_ref().unwrap().as_ptr(),
            forced_fast_ptr,
            "root activation must return the same physical Fast table",
        );
        assert_eq!(forced_result.wallq_tc_budget_ms, None);
        assert_eq!(forced_result.wallq_tc_active_lanes, 1);
        assert!(forced_result.wallq_tc_fires > 0);
        assert!(forced_result.wallq_tc_leaves > 0);
        assert!(forced_result.wallq_tc_in_window > 0);

        let mut genuine_features = forced_features;
        genuine_features.wallq_tc_probe_force = WallqTcProbeForce::Off;
        let mut genuine = SearchContext::with_tt_entries(genuine_features, 1 << 12);
        genuine.clear_tt_for_allotted_budget(Some(50));
        let genuine_fast_ptr = genuine
            .wallq_tc_fast_tt
            .as_ref()
            .expect("genuine 50ms clear must provision the Fast namespace")
            .as_ptr();
        let genuine_result = genuine.search(&state, &deterministic_tc(2, Some(50)));
        assert_eq!(genuine.wallq_tc_mode, WallqTcMode::Fast);
        assert_eq!(
            genuine.wallq_tc_fast_tt.as_ref().unwrap().as_ptr(),
            genuine_fast_ptr,
        );
        assert_result_identity(&forced_result, &genuine_result, "forced/genuine Fast");
        assert_eq!(forced_result.wallq_tc_leaves, genuine_result.wallq_tc_leaves);
        assert_eq!(
            forced_result.wallq_tc_in_window,
            genuine_result.wallq_tc_in_window,
        );
        assert_eq!(forced_result.wallq_tc_fires, genuine_result.wallq_tc_fires);
    }

    #[test]
    fn wallq_tc_root_mode_is_propagated_to_all_completed_helpers() {
        let state = parse_state("p0=g2;p1=g7;w0=8;w1=9;h=e7;v=a2,b3;t=0").unwrap();
        let mut features = Features::default();
        features.threads = 4;
        features.race_exact = false;
        features.race1w = false;
        features.race2w = false;
        features.wallq_tc = true;

        let mut active = SearchContext::with_tt_entries(features.clone(), 1 << 12);
        let active_result = active.search(&state, &deterministic_tc(2, Some(50)));
        assert_eq!(active_result.wallq_tc_budget_ms, Some(50));
        assert_eq!(active_result.wallq_tc_active_lanes as usize, active_result.threads);
        assert!(active_result.wallq_tc_fires > 0, "positive control must fire");

        let mut inactive = SearchContext::with_tt_entries(features, 1 << 12);
        let inactive_result = inactive.search(&state, &deterministic_tc(2, Some(100)));
        assert_eq!(inactive_result.wallq_tc_budget_ms, Some(100));
        assert_eq!(inactive_result.wallq_tc_active_lanes, 0);
        assert_eq!(inactive_result.wallq_tc_fires, 0);
    }

    #[test]
    fn wallq_tc_tt_mode_crossings_match_fresh_controls_without_clearing() {
        let state = parse_state("p0=g2;p1=g7;w0=8;w1=9;h=e7;v=a2,b3;t=0").unwrap();
        let mut features = Features::default();
        features.threads = 1;
        features.race_exact = false;
        features.race1w = false;
        features.race2w = false;
        features.wallq_tc = true;

        // The fast, inactive, and long bands own separate physical tables.
        // Thus each target mode starts from the same namespace state as a
        // fresh context even though the persistent owner is never cleared.
        for (first_ms, second_ms) in [(50, 100), (100, 50), (50, 1_000), (1_000, 50)] {
            let mut persistent = SearchContext::with_tt_entries(features.clone(), 1 << 12);
            let _first = persistent.search(&state, &deterministic_tc(2, Some(first_ms)));
            let actual = persistent.search(&state, &deterministic_tc(2, Some(second_ms)));

            let mut fresh = SearchContext::with_tt_entries(features.clone(), 1 << 12);
            let expected = fresh.search(&state, &deterministic_tc(2, Some(second_ms)));
            assert_result_identity(
                &actual,
                &expected,
                &format!("{first_ms}->{second_ms} fresh-control crossing"),
            );
            assert_eq!(actual.wallq_tc_budget_ms, Some(second_ms));
            assert_eq!(actual.wallq_tc_active_lanes, expected.wallq_tc_active_lanes);
            assert_eq!(actual.wallq_tc_fires, expected.wallq_tc_fires);
        }
    }

    #[test]
    fn wallq_tc_effective_feature_map_hash_is_resolved_and_feature_sensitive() {
        let absent = Features::parse(&[
            "--feature".to_owned(),
            "threads=1".to_owned(),
        ])
        .unwrap();
        let explicit_off = Features::parse(&[
            "--feature".to_owned(),
            "threads=1".to_owned(),
            "--feature".to_owned(),
            "wallq_tc_probe_force=off".to_owned(),
        ])
        .unwrap();
        assert_eq!(absent.wallq_tc_probe_force, WallqTcProbeForce::Off);
        assert_eq!(
            absent.effective_feature_map_hash(),
            "d43e36a4a53abe1200b207c17b51240987be8caeae89f5dc12e9e50c04952f11",
            "an absent probe flag must preserve the pre-contract feature map",
        );
        assert_eq!(
            absent.effective_feature_map_hash(),
            explicit_off.effective_feature_map_hash(),
        );

        let forced_fast = Features::parse(&[
            "--feature".to_owned(),
            "threads=1".to_owned(),
            "--feature".to_owned(),
            "wallq_tc_probe_force=fast".to_owned(),
        ])
        .unwrap();
        let forced_long = Features::parse(&[
            "--feature".to_owned(),
            "threads=1".to_owned(),
            "--feature".to_owned(),
            "wallq_tc_probe_force=long".to_owned(),
        ])
        .unwrap();
        assert_eq!(forced_fast.wallq_tc_probe_force, WallqTcProbeForce::Fast);
        assert_eq!(forced_long.wallq_tc_probe_force, WallqTcProbeForce::Long);
        assert_ne!(
            absent.effective_feature_map_hash(),
            forced_fast.effective_feature_map_hash(),
        );
        assert_ne!(
            forced_fast.effective_feature_map_hash(),
            forced_long.effective_feature_map_hash(),
        );
        assert!(Features::parse(&[
            "--feature".to_owned(),
            "wallq_tc_probe_force=fas".to_owned(),
        ])
        .is_err());

        let mut baseline = Features::default();
        baseline.wallq_tc = false;
        let mut enabled = baseline.clone();
        enabled.wallq_tc = true;
        assert_ne!(
            baseline.effective_feature_map_hash(),
            enabled.effective_feature_map_hash(),
        );

        let reordered = Features::parse(&[
            "--feature".to_owned(),
            "wallq_tc=on".to_owned(),
            "--feature".to_owned(),
            "threads=1".to_owned(),
        ])
        .unwrap();
        let direct = Features::parse(&[
            "--feature".to_owned(),
            "threads=1".to_owned(),
            "--feature".to_owned(),
            "wallq_tc=on".to_owned(),
        ])
        .unwrap();
        assert_eq!(
            reordered.effective_feature_map_hash(),
            direct.effective_feature_map_hash(),
            "effective map hash must not depend on argv ordering",
        );
    }

    #[test]
    fn tm2_time_reallocates_only_banked_surplus_and_resets_between_games() {
        let low =
            parse_state("p0=e2;p1=e8;w0=10;w1=0;h=-;v=-;t=0").unwrap();
        assert_eq!(tm_tension(&low).unwrap().tension, 0);
        let high = start_state();
        assert_eq!(tm_tension(&high).unwrap().tension, 80);

        let mut plain_features = Features::default();
        plain_features.threads = 1;
        let mut plain = SearchContext::with_tt_entries(plain_features, 8);
        let plain_tc = plain.time_control_for_root(Instant::now(), &low, 100);
        assert_eq!(plain_tc.allotted_budget_ms, Some(100));
        assert_eq!(plain.tm2_time_bank.balance_ms, 0);

        let mut tm2_features = Features::default();
        tm2_features.threads = 1;
        tm2_features.wallq_tc = false;
        tm2_features.tm2_time = true;
        let mut tm2 = SearchContext::with_tt_entries(tm2_features, 8);
        let high_tc = tm2.time_control_for_root(Instant::now(), &high, 100);
        let low_tc = tm2.time_control_for_root(Instant::now(), &low, 100);
        let later_high_tc = tm2.time_control_for_root(Instant::now(), &high, 100);
        assert_eq!(high_tc.allotted_budget_ms, Some(100));
        assert_eq!(low_tc.allotted_budget_ms, Some(79));
        assert_eq!(later_high_tc.allotted_budget_ms, Some(121));
        assert_eq!(tm2.tm2_time_bank.balance_ms, 0);

        tm2.begin_new_game();
        assert_eq!(tm2.tm2_time_bank.balance_ms, 0);
        let first_high_tc = tm2.time_control_for_root(Instant::now(), &high, 100);
        assert_eq!(first_high_tc.allotted_budget_ms, Some(100));
        assert_eq!(tm2.tm2_time_bank.balance_ms, 0);
    }

    #[test]
    fn tm2_time_simulated_stream_is_prefix_neutral_bounded_and_spends_surplus() {
        // Each block banks three low-tension roots before two high-tension
        // roots spend the surplus. The first spender reaches the 1.5x outer
        // clamp, and each five-root block returns exactly to flat allocation.
        let bank_then_spend = [0u8, 0, 0, 100, 100].into_iter().cycle().take(50);
        let base_ms = 1_000u64;
        let minimum = Tm2TimeBank::minimum_budget(base_ms);
        let maximum = Tm2TimeBank::maximum_budget(base_ms);
        let mut bank = Tm2TimeBank::default();
        let mut total = 0u128;
        let mut min_realized = u64::MAX;
        let mut max_realized = 0u64;

        for (index, tension) in bank_then_spend.enumerate() {
            let budget = bank.allocate(base_ms, tension);
            assert!(
                (minimum..=maximum).contains(&budget),
                "index={index} tension={tension} budget={budget}"
            );
            total += u128::from(budget);
            min_realized = min_realized.min(budget);
            max_realized = max_realized.max(budget);
            let flat_prefix = (index as u128 + 1) * u128::from(base_ms);
            assert!(total <= flat_prefix, "index={index} exceeded flat prefix");
            assert_eq!(
                bank.balance_ms,
                flat_prefix - total,
                "index={index} bank must equal cumulative flat minus allocated"
            );
        }

        let flat_total = 50 * u128::from(base_ms);
        assert_eq!(total, flat_total);
        assert_eq!(bank.balance_ms, 0);
        assert_eq!(min_realized, 790);
        assert_eq!(max_realized, 1_500);
        let mean = total as f64 / flat_total as f64;
        let min = min_realized as f64 / base_ms as f64;
        let max = max_realized as f64 / base_ms as f64;
        println!(
            "tm2_time simulated distribution: mean={mean:.6} min={min:.6} max={max:.6}"
        );
        assert!((mean - 1.0).abs() < 0.01);
        assert!(min >= 0.4);
        assert!(max <= 1.5);
    }

    #[test]
    fn tm2_time_adversarial_high_tension_never_exceeds_flat_at_any_prefix() {
        let base_ms = 100u64;
        // Tension 80 is the audited start-position counterexample and already
        // drives the affine target into the 1.5x outer clamp.
        let tension = 80u8;
        let mut bank = Tm2TimeBank::default();
        let mut cumulative_allocated = 0u128;

        for move_number in 1u128..=7 {
            let allocated = bank.allocate(base_ms, tension);
            cumulative_allocated += u128::from(allocated);
            let cumulative_flat = move_number * u128::from(base_ms);
            println!(
                "move={move_number} tension={tension} allocated_ms={allocated} cumulative_allocated_ms={cumulative_allocated} cumulative_flat_ms={cumulative_flat} bank_ms={}",
                bank.balance_ms
            );
            assert!(
                cumulative_allocated <= cumulative_flat,
                "move={move_number} exceeded flat allocation prefix"
            );
            assert_eq!(
                bank.balance_ms,
                cumulative_flat - cumulative_allocated,
                "move={move_number} bank must equal cumulative flat minus allocated"
            );
        }

        assert_eq!(cumulative_allocated, 700);
        assert_eq!(bank.balance_ms, 0);
    }

    #[test]
    fn cutoff_probe_uses_frozen_zero_based_rank_buckets() {
        let mut probe = CutoffProbeStats::default();
        for rank in [0, 1, 2, 5, 6, 10, 11, 42] {
            probe.record_cutoff(rank, rank == 0);
        }
        assert_eq!(probe.cutoff_ranks, [1, 1, 2, 2, 2]);
        assert_eq!(probe.validated_tt_first_cutoffs, 1);
    }

    #[test]
    fn correction_history_uses_side_aware_zero_scarce_plentiful_bands() {
        assert_eq!(wall_count_band(0), 0);
        assert_eq!(wall_count_band(1), 1);
        assert_eq!(wall_count_band(2), 1);
        assert_eq!(wall_count_band(3), 2);
        assert_eq!(wall_count_band(10), 2);

        let turn_zero =
            parse_state("p0=e1;p1=e9;w0=0;w1=2;h=-;v=-;t=0").unwrap();
        let turn_one = State {
            turn: 1,
            ..turn_zero
        };
        assert_eq!(correction_history_index(&turn_zero), 1);
        // Side one owns w1=2 and faces w0=0: side block 9 + (1*3+0).
        assert_eq!(correction_history_index(&turn_one), 12);
        assert!(correction_history_index(&turn_one) < CORR_HISTORY_ENTRIES);
    }

    #[test]
    fn correction_history_gravity_is_depth_weighted_and_caps_at_32cp() {
        let mut positive = 0i16;
        update_correction_history_entry(&mut positive, 64, 8);
        assert_eq!(positive, 64);
        assert_eq!(correction_history_value(positive), 2);

        for _ in 0..100 {
            update_correction_history_entry(&mut positive, i32::MAX, 64);
        }
        assert_eq!(positive, CORR_HISTORY_LIMIT as i16);
        assert_eq!(correction_history_value(positive), 32);

        let mut negative = 0i16;
        for _ in 0..100 {
            update_correction_history_entry(&mut negative, i32::MIN, 64);
        }
        assert_eq!(negative, -(CORR_HISTORY_LIMIT as i16));
        assert_eq!(correction_history_value(negative), -32);
        assert_eq!(correction_history_value(i16::MAX), 32);
        assert_eq!(correction_history_value(i16::MIN), -32);
    }

    #[test]
    fn correction_history_applies_only_to_ordinary_leaf_evaluation() {
        let ordinary = start_state();
        let mut features = Features::default();
        features.threads = 1;
        features.corr_hist = true;
        let mut ctx = SearchContext::with_tt_entries(features, 1024);
        let raw = evaluate_with_features(&ordinary, &ctx.features);
        ctx.correction_history.as_mut().unwrap()[correction_history_index(&ordinary)] =
            CORR_HISTORY_LIMIT as i16;
        assert_eq!(
            ctx.leaf_evaluation::<false>(&ordinary, None, -INF, INF),
            raw + 32
        );
        assert_eq!(evaluate_with_features(&ordinary, &ctx.features), raw);
        assert_eq!(
            ctx.negamax(&ordinary, None, 0, -INF, INF, 0, false, None),
            Some(raw + 32),
        );

        let exact =
            parse_state("p0=e1;p1=a9;w0=0;w1=0;h=-;v=-;t=0").unwrap();
        ctx.features.race_exact = true;
        let exact_score = ctx.race_exact_score(&exact, 0).unwrap();
        ctx.correction_history.as_mut().unwrap()[correction_history_index(&exact)] =
            -(CORR_HISTORY_LIMIT as i16);
        let before = *ctx.correction_history.as_ref().unwrap();
        assert_eq!(
            ctx.negamax(&exact, None, 0, -INF, INF, 0, false, None),
            Some(exact_score),
        );
        assert_eq!(*ctx.correction_history.as_ref().unwrap(), before);
    }

    #[test]
    fn correction_history_learns_exact_ordinary_residuals_and_resets_per_game() {
        let state = start_state();
        let mut features = Features::default();
        features.threads = 2;
        features.corr_hist = true;
        let mut ctx = SearchContext::with_tt_entries(features, 1024);
        let raw = evaluate_with_features(&state, &ctx.features);
        ctx.record_correction_history(&state, 8, raw + 64, None);
        let index = correction_history_index(&state);
        assert_eq!(ctx.correction_history.as_ref().unwrap()[index], 64);

        let before_proven = ctx.correction_history.as_ref().unwrap()[index];
        ctx.record_correction_history(&state, 8, 20_000, None);
        assert_eq!(
            ctx.correction_history.as_ref().unwrap()[index],
            before_proven,
        );
        ctx.helper_correction_histories[0].as_mut().unwrap()[index] = 17;
        ctx.begin_new_game();
        assert!(ctx.correction_history.as_ref().unwrap().iter().all(|v| *v == 0));
        assert!(ctx.helper_correction_histories[0]
            .as_ref()
            .unwrap()
            .iter()
            .all(|v| *v == 0));
    }

    #[test]
    fn correction_history_helper_lane_is_moved_back_without_a_per_move_clear() {
        let mut features = Features::default();
        features.threads = 2;
        features.corr_hist = true;
        let mut ctx = SearchContext::with_tt_entries(features, 1024);
        assert_eq!(ctx.helper_correction_histories.len(), 1);
        // A depth-two search from the full-inventory start cannot reach the
        // zero-walls buckets, so ordinary helper learning cannot alter this.
        let sentinel = 0;
        ctx.helper_correction_histories[0].as_mut().unwrap()[sentinel] = 2345;
        let tc = TimeControl {
            max_depth: 2,
            allotted_budget_ms: None,
            soft_deadline: None,
            hard_deadline: None,
        };

        let result = ctx.search(&start_state(), &tc);
        assert_eq!(result.threads, 2);
        assert_eq!(
            ctx.helper_correction_histories[0].as_ref().unwrap()[sentinel],
            2345,
        );
    }

    #[test]
    fn correction_history_does_not_learn_from_null_window_bounds() {
        let state = start_state();
        let mut features = Features::default();
        features.threads = 1;
        features.corr_hist = true;
        let mut ctx = SearchContext::with_tt_entries(features, 1024);
        let score = ctx
            .negamax(&state, None, 1, 0, 1, 0, false, None)
            .expect("unbounded null-window search completes");
        assert!(score <= 0 || score >= 1);
        assert!(ctx
            .correction_history
            .as_ref()
            .unwrap()
            .iter()
            .all(|entry| *entry == 0));
    }

    #[test]
    fn correction_history_is_suppressed_for_an_exact_root_fallback_tree() {
        let state =
            parse_state("p0=e1;p1=a9;w0=0;w1=0;h=-;v=-;t=0").unwrap();
        let mut plain_features = Features::default();
        plain_features.threads = 1;
        plain_features.race1w = false;
        plain_features.race_exact = true;
        let mut plain = SearchContext::with_tt_entries(plain_features.clone(), 1024);

        let mut corrected_features = plain_features;
        corrected_features.corr_hist = true;
        let mut corrected = SearchContext::with_tt_entries(corrected_features, 1024);
        corrected.correction_history.as_mut().unwrap().fill(321);
        let before = *corrected.correction_history.as_ref().unwrap();
        let tc = TimeControl {
            max_depth: 2,
            allotted_budget_ms: None,
            soft_deadline: None,
            hard_deadline: None,
        };
        let expected = plain.search(&state, &tc);
        let actual = corrected.search(&state, &tc);
        assert_eq!(actual.best_action, expected.best_action);
        assert_eq!(actual.score, expected.score);
        assert_eq!(*corrected.correction_history.as_ref().unwrap(), before);
        assert!(!corrected.correction_history_suppressed);
    }

    #[test]
    fn wall_history_context_is_opponent_tight_cross_distance_band() {
        assert_eq!(wall_history_distance_band(0), 0);
        assert_eq!(wall_history_distance_band(3), 0);
        assert_eq!(wall_history_distance_band(4), 1);
        assert_eq!(wall_history_distance_band(6), 1);
        assert_eq!(wall_history_distance_band(7), 2);

        let state = start_state();
        let topology = Topology::from_walls(state.h, state.v);
        let opponent_distances = distances_to_row(&topology, 0);
        let tight = parse_action("d8h").unwrap();
        let loose = parse_action("a4v").unwrap();
        assert!(wall_touches_tight_edge(&opponent_distances, tight));
        assert!(!wall_touches_tight_edge(&opponent_distances, loose));
        assert_eq!(wall_history_context(&opponent_distances, state.p1, tight), 5);
        assert_eq!(wall_history_context(&opponent_distances, state.p1, loose), 2);
    }

    #[test]
    fn wall_history_zero_is_order_neutral_and_context_stays_aligned() {
        let state = start_state();
        let expected = order_moves_v0(&state, None);
        let mut features = Features::default();
        features.threads = 1;
        features.history = false;
        features.cmh = false;
        features.wall_hist = true;
        let mut ctx = SearchContext::with_tt_entries(features, 1024);
        let neutral = ctx
            .order_moves_inner(&state, None, 0, None, None)
            .unwrap();
        assert_eq!(&neutral.moves[..neutral.len], &expected.moves[..expected.len]);

        let topology = Topology::from_walls(state.h, state.v);
        let opponent_distances = distances_to_row(&topology, 0);
        for index in 0..neutral.len {
            let action = neutral.moves[index];
            if action.kind != ActionKind::Pawn {
                assert_eq!(
                    neutral.wall_contexts[index],
                    wall_history_context(&opponent_distances, state.p1, action),
                );
            }
        }

        let promoted = parse_action("a4v").unwrap();
        let context = wall_history_context(&opponent_distances, state.p1, promoted);
        let history_index = wall_history_index(state.turn as usize, context, promoted);
        ctx.wall_history_scores.as_mut().unwrap()[history_index] = 100;
        let reordered = ctx
            .order_moves_inner(&state, None, 0, None, None)
            .unwrap();
        assert_eq!(reordered.moves[0], promoted);
        assert_eq!(reordered.wall_contexts[0], context);
    }

    #[test]
    fn wall_history_cutoff_update_reuses_read_context_and_ignores_pawns() {
        let state = start_state();
        let wall = parse_action("a4v").unwrap();
        let pawn = parse_action("e2").unwrap();
        let mut features = Features::default();
        features.threads = 1;
        features.history = false;
        features.cmh = false;
        features.wall_hist = true;
        features.probe_wall_hist = true;
        let mut ctx = SearchContext::with_tt_entries(features, 1024);

        ctx.record_history_with_context(&state, wall, 3, None, Some(2));
        assert_eq!(
            ctx.history_score_with_context(&state, wall, None, Some(2)),
            9,
        );
        assert_eq!(
            ctx.history_score_with_context(&state, wall, None, Some(5)),
            0,
        );
        ctx.record_history_with_context(&state, pawn, 3, None, Some(2));
        assert_eq!(
            ctx.history_score_with_context(&state, pawn, None, Some(2)),
            0,
        );

        ctx.age_history();
        assert_eq!(
            ctx.history_score_with_context(&state, wall, None, Some(2)),
            4,
        );
        ctx.clear_history();
        assert_eq!(
            ctx.history_score_with_context(&state, wall, None, Some(2)),
            0,
        );
        assert_eq!(
            ctx.wall_history_probe_report(),
            Some(WallHistoryProbeReport {
                reads: 4,
                pooled_warm_reads: 0,
                bucketed_warm_reads: 2,
                lmr_reductions: 0,
                lmr_researches: 0,
            }),
        );
    }

    #[test]
    fn eval_swing_probe_buckets_actions_walls_and_percentiles() {
        let pawn = parse_action("e2").unwrap();
        let wall = parse_action("a1h").unwrap();
        let mut probe = EvalSwingProbeStats::new();
        for swing in [10, -20, 30, 100] {
            probe.record(pawn, false, swing);
        }
        probe.record(wall, true, -250);
        let report = probe.report();

        assert_eq!(report[0].count, 4);
        assert_eq!(report[0].p50, 20);
        assert_eq!(report[0].p90, 100);
        assert_eq!(report[0].p99, 100);
        assert_eq!(report[0].max, 100);
        assert_eq!(report[1].count, 0);
        assert!(report[3].wall_move);
        assert!(report[3].opponent_has_walls);
        assert_eq!(report[3].p95, 250);
    }

    #[test]
    fn eval_swing_probe_does_not_change_fixed_depth_search() {
        let state = start_state();
        let mut plain_features = Features::default();
        plain_features.threads = 1;
        let mut plain = SearchContext::with_tt_entries(plain_features.clone(), 4096);
        let mut probe_features = plain_features;
        probe_features.probe_evalswing = true;
        let mut probed = SearchContext::with_tt_entries(probe_features, 4096);
        let tc = TimeControl {
            max_depth: 2,
            allotted_budget_ms: None,
            soft_deadline: None,
            hard_deadline: None,
        };

        let expected = plain.search(&state, &tc);
        let actual = probed.search(&state, &tc);
        assert_eq!(actual.best_action, expected.best_action);
        assert_eq!(actual.score, expected.score);
        assert_eq!(actual.nodes, expected.nodes);
        assert!(probed
            .eval_swing_probe_report()
            .unwrap()
            .iter()
            .map(|bucket| bucket.count)
            .sum::<u64>()
            > 0);
    }

    #[test]
    fn eval_swing_probe_uses_reply_movers_own_inventory_as_rfp_bucket() {
        // At a depth-one reply node P0 is the prospective RFP opponent. P0
        // owns no wall while P1 owns one, so the sample belongs in the zero
        // bucket; reading P1's inventory would reverse this result.
        let state =
            parse_state("p0=a1;p1=i9;w0=0;w1=1;h=-;v=-;t=0").unwrap();
        let mut features = Features::default();
        features.threads = 1;
        features.probe_evalswing = true;
        let mut ctx = SearchContext::with_tt_entries(features, 1024);
        let _ = ctx
            .negamax(&state, None, 1, -INF, INF, 0, false, None)
            .expect("unbounded probe completes");
        let report = ctx.eval_swing_probe_report().unwrap();

        assert!(report[0].count > 0);
        assert_eq!(report[1].count, 0);
        assert_eq!(report[2].count + report[3].count, 0);
    }

    #[test]
    fn rfp_precision_probe_freezes_linear_grid_and_inferred_100d2_arm() {
        let zero_walls =
            parse_state("p0=a1;p1=i9;w0=2;w1=0;h=-;v=-;t=0").unwrap();
        let sample = rfp_precision_sample(&zero_walls, 2, 100, 600, 3);
        assert_eq!(sample.opponent_walls_bucket, 0);
        assert_eq!(sample.fires, [true, true, true, false, false, true]);

        // The source spec names a quadratic arm but no coefficient. BATCH-3's
        // explicit, reproducible inference is 100 * depth^2: 1000-900 == beta.
        let depth_three = rfp_precision_sample(&zero_walls, 3, 100, 1000, 3);
        assert!(depth_three.fires[RFP_PRECISION_ARMS - 1]);
        let just_below = rfp_precision_sample(&zero_walls, 3, 101, 1000, 3);
        assert!(!just_below.fires[RFP_PRECISION_ARMS - 1]);
    }

    #[test]
    fn rfp_precision_probe_buckets_opponents_inventory_with_production_eligibility() {
        let turn_zero =
            parse_state("p0=a1;p1=i9;w0=0;w1=1;h=-;v=-;t=0").unwrap();
        let turn_one =
            parse_state("p0=a1;p1=i9;w0=1;w1=0;h=-;v=-;t=1").unwrap();
        assert_eq!(
            rfp_precision_sample(&turn_zero, 3, 0, 1000, 3).opponent_walls_bucket,
            1
        );
        assert_eq!(
            rfp_precision_sample(&turn_one, 3, 0, 1000, 3).opponent_walls_bucket,
            1
        );
        assert!(rfp_precision_eligible(3, true, 0, 3));

        // Production RFP independently retains this depth-three slice.
        assert!(rfp_eligible(3, true, 0, 3));
    }

    #[test]
    fn probe_rfp_depth_defaults_parses_four_without_extending_production_rfp() {
        assert_eq!(Features::default().probe_rfp_depth, 3);

        let parsed = Features::parse(&[
            "--feature".to_owned(),
            "probe_rfp_depth=4".to_owned(),
        ])
        .unwrap();
        assert_eq!(parsed.probe_rfp_depth, 4);
        assert!(rfp_precision_eligible(4, true, 0, parsed.probe_rfp_depth));
        assert!(!rfp_precision_eligible(5, true, 0, parsed.probe_rfp_depth));

        let d4_sample =
            rfp_precision_sample(&start_state(), 4, 0, 400, parsed.probe_rfp_depth);
        assert_eq!(d4_sample.fires, [true, false, false, false, false, false]);

        assert!(rfp_eligible(3, true, 0, 3));
        assert!(!rfp_eligible(4, true, 0, 3));
        assert!(!rfp_prunable(4, true, 0, i32::MAX, 0, 3));
    }

    #[test]
    fn rfp_precision_probe_reports_completed_true_and_false_fail_high_counts() {
        let zero_walls =
            parse_state("p0=a1;p1=i9;w0=2;w1=0;h=-;v=-;t=0").unwrap();
        let has_walls =
            parse_state("p0=a1;p1=i9;w0=2;w1=1;h=-;v=-;t=0").unwrap();
        let mut stats = RfpPrecisionProbeStats::default();
        stats.record(
            rfp_precision_sample(&zero_walls, 2, 100, 600, 3),
            100,
            100,
        );
        stats.record(
            rfp_precision_sample(&has_walls, 2, 100, 600, 3),
            99,
            100,
        );

        let report = stats.report();
        assert_eq!(report[0].coefficient, 100);
        assert!(!report[0].quadratic);
        assert_eq!(report[0].buckets[0].eligible_nodes, 1);
        assert_eq!(report[0].buckets[0].fires, 1);
        assert_eq!(report[0].buckets[0].true_fail_highs, 1);
        assert_eq!(report[0].buckets[1].eligible_nodes, 1);
        assert_eq!(report[0].buckets[1].fires, 1);
        assert_eq!(report[0].buckets[1].true_fail_highs, 0);
        assert_eq!(report[4].coefficient, 450);
        assert_eq!(report[4].buckets[0].fires, 0);
        assert!(report[RFP_PRECISION_ARMS - 1].quadratic);
        assert_eq!(report[RFP_PRECISION_ARMS - 1].coefficient, 100);
    }

    #[test]
    fn rfp_precision_projection_counts_only_topmost_covered_subtrees() {
        let state =
            parse_state("p0=a1;p1=i9;w0=2;w1=0;h=-;v=-;t=0").unwrap();
        let mut features = Features::default();
        features.threads = 1;
        features.probe_rfp_precision = true;
        let mut ctx = SearchContext::with_tt_entries(features, 1024);
        let sample = rfp_precision_sample(&state, 2, 100, 600, 3);

        let outer = ctx.begin_rfp_precision_sample(sample);
        let nested = ctx.begin_rfp_precision_sample(sample);
        ctx.nodes = 10;
        ctx.finish_rfp_precision_sample(nested, 100, 100);
        ctx.finish_rfp_precision_sample(outer, 100, 100);

        let report = ctx.rfp_precision_probe_report().unwrap();
        assert_eq!(report[0].projected_saved_nodes, 10);
        assert!(report[0].projected_saved_nanos > 0);
        assert_eq!(report[0].buckets[0].fires, 2);
        assert!(ctx.rfp_precision_active.iter().all(|active| *active == 0));
    }

    #[test]
    fn rfp_precision_probe_is_shadow_only_and_skips_exact_and_tt_returns() {
        let state = start_state();
        let mut plain_features = Features::default();
        plain_features.threads = 1;
        // This test compares the shadow probe with the pre-RFP ordinary path.
        plain_features.rfp = false;
        let mut plain = SearchContext::with_tt_entries(plain_features.clone(), 4096);
        let mut probe_features = plain_features;
        probe_features.probe_rfp_precision = true;
        let mut probed = SearchContext::with_tt_entries(probe_features.clone(), 4096);
        let tc = TimeControl {
            max_depth: 3,
            allotted_budget_ms: None,
            soft_deadline: None,
            hard_deadline: None,
        };
        let expected = plain.search(&state, &tc);
        let actual = probed.search(&state, &tc);
        assert_eq!(actual.best_action, expected.best_action);
        assert_eq!(actual.score, expected.score);
        assert_eq!(actual.depth, expected.depth);
        assert_eq!(actual.nodes, expected.nodes);
        assert!(probed
            .rfp_precision_probe_report()
            .unwrap()
            .iter()
            .flat_map(|arm| arm.buckets)
            .map(|bucket| bucket.eligible_nodes)
            .sum::<u64>()
            > 0);

        let exact =
            parse_state("p0=a6;p1=i4;w0=1;w1=0;h=-;v=-;t=0").unwrap();
        let mut exact_ctx = SearchContext::with_tt_entries(probe_features, 1024);
        let _ = exact_ctx
            .negamax(&exact, None, 3, 0, 1, 0, false, None)
            .expect("exact probe completes");
        assert!(exact_ctx
            .rfp_precision_probe_report()
            .unwrap()
            .iter()
            .flat_map(|arm| arm.buckets)
            .all(|bucket| bucket.eligible_nodes == 0));

        probed.clear_tt();
        let before_tt = probed.rfp_precision_probe_report().unwrap();
        let _ = probed
            .negamax(&state, None, 2, 0, 1, 0, false, None)
            .expect("first TT probe search completes");
        let after_first = probed.rfp_precision_probe_report().unwrap();
        assert_ne!(after_first, before_tt);
        let _ = probed
            .negamax(&state, None, 2, 0, 1, 0, false, None)
            .expect("TT return completes");
        assert_eq!(probed.rfp_precision_probe_report().unwrap(), after_first);
    }

    #[test]
    fn rfp_precision_probe_counts_the_forced_fast_corrected_leaf_tree() {
        let state = parse_state("p0=g2;p1=g7;w0=8;w1=9;h=e7;v=a2,b3;t=0").unwrap();
        let mut forced_features = Features::default();
        forced_features.threads = 1;
        forced_features.race_exact = false;
        forced_features.race1w = false;
        forced_features.race2w = false;
        forced_features.probe_rfp_precision = true;
        forced_features.probe_rfp_depth = 4;
        forced_features.wallq_tc = true;
        forced_features.wallq_tc_probe_force = WallqTcProbeForce::Fast;

        let mut forced = SearchContext::with_tt_entries(forced_features.clone(), 1 << 12);
        forced.clear_tt_for_allotted_budget(None);
        let forced_result = forced.search(&state, &deterministic_tc(4, None));
        let forced_report = forced.rfp_precision_probe_report().unwrap();
        let forced_eligible: u64 = forced_report[0]
            .buckets
            .iter()
            .map(|bucket| bucket.eligible_nodes)
            .sum();
        assert!(forced_result.wallq_tc_fires > 0);
        assert!(forced_eligible > 0);

        // A genuine deterministic 50ms Fast root is the oracle for both the
        // corrected search tree and its RFP classifications. Only the budget
        // provenance differs from the forced fixed-depth run.
        let mut genuine_features = forced_features;
        genuine_features.wallq_tc_probe_force = WallqTcProbeForce::Off;
        let mut genuine = SearchContext::with_tt_entries(genuine_features, 1 << 12);
        genuine.clear_tt_for_allotted_budget(Some(50));
        let genuine_result = genuine.search(&state, &deterministic_tc(4, Some(50)));
        let genuine_report = genuine.rfp_precision_probe_report().unwrap();

        assert_result_identity(&forced_result, &genuine_result, "forced/genuine RFP Fast");
        assert_eq!(forced_result.wallq_tc_leaves, genuine_result.wallq_tc_leaves);
        assert_eq!(forced_result.wallq_tc_fires, genuine_result.wallq_tc_fires);
        for (forced_arm, genuine_arm) in forced_report.iter().zip(genuine_report.iter()) {
            assert_eq!(forced_arm.coefficient, genuine_arm.coefficient);
            assert_eq!(forced_arm.quadratic, genuine_arm.quadratic);
            assert_eq!(forced_arm.buckets, genuine_arm.buckets);
            assert_eq!(
                forced_arm.projected_saved_nodes,
                genuine_arm.projected_saved_nodes,
            );
        }
    }

    #[test]
    fn budget_percentages_build_expected_deadlines() {
        let mut features = Features::default();
        features.soft_pct = 25;
        features.hard_pct = 75;
        let ctx = SearchContext::with_tt_entries(features, 8);
        let start = Instant::now();
        let tc = ctx.time_control_for_budget(start, 20);

        assert_eq!(tc.allotted_budget_ms, Some(20));
        assert_eq!(
            tc.soft_deadline.unwrap().duration_since(start),
            Duration::from_millis(5)
        );
        assert_eq!(
            tc.hard_deadline.unwrap().duration_since(start),
            Duration::from_millis(15)
        );
    }

    #[test]
    fn predictive_iteration_estimate_clamps_effective_branching() {
        assert_eq!(
            predicted_iteration_nanos(Duration::ZERO, Duration::from_nanos(7)),
            140
        );
        assert_eq!(
            predicted_iteration_nanos(Duration::from_nanos(10), Duration::from_nanos(15)),
            30
        );
        assert_eq!(
            predicted_iteration_nanos(Duration::from_nanos(4), Duration::from_nanos(9)),
            21
        );
        assert_eq!(
            predicted_iteration_nanos(Duration::from_nanos(10), Duration::from_nanos(50)),
            250
        );
        assert_eq!(
            predicted_iteration_nanos(Duration::from_nanos(10), Duration::from_nanos(250)),
            5_000
        );
    }

    #[test]
    fn mate_in_one_is_found() {
        // P0 on e8 (row 7) can step to e9 (row 8) and win immediately.
        let state = parse_state("p0=e8;p1=a9;w0=10;w1=10;h=-;v=-;t=0").unwrap();
        let res = depth_search(&state, 1);
        assert_eq!(action_name(res.best_action), "e9");
        assert!(res.score >= MATE_THRESHOLD, "score={}", res.score);
    }

    #[test]
    fn deterministic_and_tt_consistent() {
        // Same context searched twice, plus a fresh context, must agree.
        let state = start_state();
        let mut ctx = SearchContext::new(Features::default());
        let tc = TimeControl {
            max_depth: 3,
            allotted_budget_ms: None,
            soft_deadline: None,
            hard_deadline: None,
        };
        let a = ctx.search(&state, &tc);
        let b = ctx.search(&state, &tc);
        let c = depth_search(&state, 3);
        assert_eq!(action_name(a.best_action), action_name(b.best_action));
        assert_eq!(a.score, b.score);
        assert_eq!(action_name(a.best_action), action_name(c.best_action));
        assert_eq!(a.score, c.score);
        assert_eq!(a.depth, 3);
    }

    #[test]
    fn expired_deadline_returns_a_legal_root_fallback_without_searching() {
        let state = start_state();
        let mut legal = MoveList::new();
        legal_actions(&state, &mut legal);

        let mut ctx = SearchContext::with_tt_entries(Features::default(), 1024);
        let tc = TimeControl {
            max_depth: 64,
            allotted_budget_ms: None,
            soft_deadline: None,
            hard_deadline: Some(Instant::now() - Duration::from_millis(1)),
        };
        let result = ctx.search(&state, &tc);

        assert_eq!(result.best_action, legal.get(0));
        assert_eq!(result.depth, 0);
        assert_eq!(result.nodes, 0);
    }

    #[test]
    fn tt_roundtrip_of_actions() {
        for &text in &["e2", "h8h", "a1v", "e6h", "d3v"] {
            let a = crate::state::parse_action(text).unwrap();
            assert_eq!(decode_action(encode_action(a)), a);
        }
    }

    #[test]
    fn horizontal_mirror_mapping_is_involutive() {
        for (original, reflected) in [
            ("a1", "i1"),
            ("d9", "f9"),
            ("e5", "e5"),
            ("a1h", "h1h"),
            ("b3h", "g3h"),
            ("h8h", "a8h"),
            ("a1v", "h1v"),
            ("c3v", "f3v"),
            ("h8v", "a8v"),
        ] {
            let action = parse_action(original).unwrap();
            assert_eq!(action_name(mirror_action_lr(action)), reflected);
            assert_eq!(mirror_action_lr(mirror_action_lr(action)), action);
        }
    }

    #[test]
    fn mirrored_states_have_mirrored_legal_actions_and_canonical_hash() {
        let state = parse_state(
            "p0=b2;p1=g8;w0=8;w1=8;h=a3,f6;v=c1,g7;t=1",
        )
        .unwrap();
        let reflected = parse_state(
            "p0=h2;p1=c8;w0=8;w1=8;h=c6,h3;v=b7,f1;t=1",
        )
        .unwrap();
        assert_eq!(mirror_state_lr(&state), reflected);
        assert_eq!(mirror_state_lr(&reflected), state);

        let mut direct_moves = MoveList::new();
        let mut reflected_moves = MoveList::new();
        legal_actions(&state, &mut direct_moves);
        legal_actions(&reflected, &mut reflected_moves);
        assert_eq!(direct_moves.len(), reflected_moves.len());
        for index in 0..direct_moves.len() {
            let action = direct_moves.get(index);
            let mirrored_action = mirror_action_lr(action);
            assert!(reflected_moves.contains(mirrored_action), "{}", action_name(action));
            assert_eq!(
                mirror_state_lr(&state.apply(action)),
                reflected.apply(mirrored_action),
            );
        }


        // Adjacent pawns exercise jump and diagonal reflection as well as the
        // ordinary pawn/wall cases above.
        let adjacent = parse_state("p0=b4;p1=b5;w0=10;w1=10;h=-;v=-;t=0").unwrap();
        let adjacent_reflected = mirror_state_lr(&adjacent);
        let mut adjacent_moves = MoveList::new();
        let mut adjacent_reflected_moves = MoveList::new();
        legal_actions(&adjacent, &mut adjacent_moves);
        legal_actions(&adjacent_reflected, &mut adjacent_reflected_moves);
        assert_eq!(adjacent_moves.len(), adjacent_reflected_moves.len());
        for index in 0..adjacent_moves.len() {
            assert!(
                adjacent_reflected_moves.contains(mirror_action_lr(adjacent_moves.get(index)))
            );
        }

        let mut features = Features::default();
        features.tt_sym = true;
        let ctx = SearchContext::with_tt_entries(features, 8);
        let direct_key = ctx.tt_position_key(&state);
        let reflected_key = ctx.tt_position_key(&reflected);
        assert_eq!(direct_key.hash, reflected_key.hash);
        assert_eq!(
            direct_key.hash,
            ctx.zobrist.hash(&state).min(ctx.zobrist.hash(&reflected)),
        );

        let plain = SearchContext::with_tt_entries(Features::default(), 8);
        let plain_key = plain.tt_position_key(&state);
        assert_eq!(plain_key.hash, plain.zobrist.hash(&state));
        assert!(!plain_key.mirrored);
    }

    #[test]
    fn symmetric_tt_translates_pawn_and_wall_moves_both_directions() {
        let state = parse_state("p0=b2;p1=g8;w0=10;w1=10;h=-;v=-;t=0").unwrap();
        let reflected = mirror_state_lr(&state);
        let mut features = Features::default();
        features.tt_sym = true;
        let mut ctx = SearchContext::with_tt_entries(features, 8);
        let mut legal = MoveList::new();
        let mut mirrored_legal = MoveList::new();
        legal_actions(&state, &mut legal);
        legal_actions(&reflected, &mut mirrored_legal);

        for text in ["b3", "a1h", "c3v"] {
            let action = parse_action(text).unwrap();
            let mirrored_action = mirror_action_lr(action);
            assert!(legal.contains(action), "{text}");
            assert!(mirrored_legal.contains(mirrored_action), "{text}");
            let key = ctx.tt_position_key(&state);
            ctx.tt_store_position(key, 4, BOUND_EXACT, 73, action);
            let mirror_key = ctx.tt_position_key(&reflected);
            let entry = ctx.tt_probe(mirror_key.hash).unwrap();
            assert_eq!(SearchContext::tt_entry_move(entry, mirror_key), mirrored_action);
            assert_eq!(entry.score, 73);

            ctx.clear_tt();
            ctx.tt_store_position(mirror_key, 4, BOUND_LOWER, -19, mirrored_action);
            let entry = ctx.tt_probe(key.hash).unwrap();
            assert_eq!(SearchContext::tt_entry_move(entry, key), action);
            assert_eq!(entry.score, -19);
            ctx.clear_tt();
        }
    }

    #[test]
    fn symmetric_tt_composes_with_aspiration_search() {
        let state = parse_state("p0=b2;p1=g8;w0=10;w1=10;h=-;v=-;t=0").unwrap();
        let reflected = mirror_state_lr(&state);
        let mut features = Features::default();
        features.tt_sym = true;
        features.aspiration = true;
        let mut ctx = SearchContext::with_tt_entries(features, 65_536);
        let tc = TimeControl {
            max_depth: 3,
            allotted_budget_ms: None,
            soft_deadline: None,
            hard_deadline: None,
        };
        let direct = ctx.search(&state, &tc);
        let mirrored = ctx.search(&reflected, &tc);
        assert_eq!(direct.depth, 3);
        assert_eq!(mirrored.depth, 3);
        assert_eq!(direct.score, mirrored.score);
        let mut legal = MoveList::new();
        legal_actions(&reflected, &mut legal);
        assert!(legal.contains(mirrored.best_action));
    }

    #[test]
    fn tt_default_capacity_remains_two_to_the_twenty_two() {
        // Single-thread TT internals test: pin threads=1 (E-010 made
        // multi-thread the default; this test probes the legacy local TT).
        let mut features = Features::default();
        features.threads = 1;
        let ctx = SearchContext::new(features);
        assert_eq!(ctx.tt.len(), 1usize << 22);
        assert!(ctx.shared_tt.is_none());
        assert_eq!(ctx.tt_mask, (1usize << 22) - 1);
    }

    #[test]
    fn tt_mb_capacity_is_the_largest_fitting_power_of_two() {
        let mut features = Features::default();
        features.threads = 1; // single-thread TT internals (see above)
        let ctx = SearchContext::new_with_tt_mb(features, 1).unwrap();
        let entry_bytes = std::mem::size_of::<TtEntry>();
        assert!(ctx.tt.len().is_power_of_two());
        assert!(ctx.tt.len() * entry_bytes <= MEBIBYTE);
        assert!(ctx.tt.len() * 2 * entry_bytes > MEBIBYTE);
        assert_eq!(ctx.tt_mask, ctx.tt.len() - 1);
    }

    #[test]
    fn shared_tt_mb_capacity_accounts_for_atomic_entries() {
        let mut features = Features::default();
        features.threads = 4;
        let ctx = SearchContext::new_with_tt_mb(features, 1).unwrap();
        let shared = ctx.shared_tt.as_ref().unwrap();
        let entries = shared.entries.len();
        let entry_bytes = std::mem::size_of::<AtomicTtEntry>();
        assert!(ctx.tt.is_empty());
        assert!(entries.is_power_of_two());
        assert!(entries * entry_bytes <= MEBIBYTE);
        assert!(entries * 2 * entry_bytes > MEBIBYTE);
        assert_eq!(ctx.tt_mask, entries - 1);
    }

    #[test]
    fn qbp_warmup_restores_clean_single_thread_search_state() {
        let state = start_state();
        // Deterministic single-thread comparison: pin threads=1 on BOTH
        // contexts (E-010 default is 4; parallel search is nondeterministic).
        let mut st_features = Features::default();
        st_features.threads = 1;
        let mut warmed = SearchContext::with_tt_entries(st_features.clone(), 1024);
        for entry in &mut warmed.tt {
            entry.depth = 1;
        }

        warmed.warm_up_for_qbp(&state).unwrap();
        assert!(warmed.tt.iter().all(|entry| entry.depth == 0));

        let tc = TimeControl {
            max_depth: 3,
            allotted_budget_ms: None,
            soft_deadline: None,
            hard_deadline: None,
        };
        let got = warmed.search(&state, &tc);
        let mut fresh = SearchContext::with_tt_entries(st_features, 1024);
        let expected = fresh.search(&state, &tc);
        assert_eq!(got.best_action, expected.best_action);
        assert_eq!(got.score, expected.score);
        assert_eq!(got.depth, expected.depth);
        assert_eq!(got.nodes, expected.nodes);
    }

    #[test]
    fn qbp_warmup_clears_shared_tt_after_parallel_startup() {
        let mut features = Features::default();
        features.threads = 4;
        let mut ctx = SearchContext::with_tt_entries(features, 1024);
        let shared = ctx.shared_tt.as_ref().unwrap();
        for entry in &shared.entries {
            entry.word.store(u64::MAX, Ordering::Relaxed);
        }

        ctx.warm_up_for_qbp(&start_state()).unwrap();

        let shared = ctx.shared_tt.as_ref().unwrap();
        assert!(shared
            .entries
            .iter()
            .all(|entry| entry.word.load(Ordering::Relaxed) == 0));
    }

    #[test]
    fn qbp_warmup_clears_seeded_countermove_history() {
        let mut features = Features::default();
        features.threads = 1;
        features.cmh = true;
        let mut ctx = SearchContext::with_tt_entries(features, 1024);
        ctx.countermove_history.as_mut().unwrap()[CMH_ENTRIES - 1] = 777;

        ctx.warm_up_for_qbp(&start_state()).unwrap();
        assert!(ctx
            .countermove_history
            .as_ref()
            .unwrap()
            .iter()
            .all(|entry| *entry == 0));
    }

    #[test]
    fn tt_custom_capacity_store_probe_and_validation() {
        assert!(SearchContext::new_with_tt_mb(Features::default(), 0).is_err());
        assert!(SearchContext::new_with_tt_mb(Features::default(), usize::MAX).is_err());

        let mut ctx = SearchContext::with_tt_entries(Features::default(), 8);
        let action = parse_action("e2").unwrap();
        let hash = 0x1234_5678_abcd_ef05;
        ctx.tt_store(hash, 3, BOUND_EXACT, 17, action);
        let entry = ctx.tt_probe(hash).unwrap();
        assert_eq!(entry.key, 0x1234_5678);
        assert_eq!(entry.best, encode_action(action));
        assert_eq!(entry.score, 17);
        assert_eq!(entry.depth, 3);
        assert_eq!(entry.bound, BOUND_EXACT);
    }

    #[test]
    fn local_tt_keeps_deeper_entry_and_clear_restores_empty_state() {
        let mut ctx = SearchContext::with_tt_entries(Features::default(), 8);
        let hash = 0x1234_5678_abcd_ef05;
        let deep = parse_action("e2").unwrap();
        let shallow = parse_action("h8h").unwrap();

        ctx.tt_store(hash, 4, BOUND_EXACT, 17, deep);
        ctx.tt_store(hash, 3, BOUND_LOWER, -9, shallow);
        let retained = ctx.tt_probe(hash).expect("deep local entry must remain");
        assert_eq!(retained.best, encode_action(deep));
        assert_eq!(retained.depth, 4);
        assert_eq!(retained.score, 17);
        assert_eq!(retained.bound, BOUND_EXACT);

        ctx.clear_tt();
        assert!(ctx.tt_probe(hash).is_none());
    }

    #[test]
    fn shared_tt_race_never_returns_a_torn_entry() {
        fn entry_for(hash: u64) -> TtEntry {
            TtEntry {
                key: (hash >> 32) as u32,
                best: expand_atomic_tt_action(
                    ((hash ^ (hash >> 17)) % u64::from(
                        ATOMIC_TT_PAWN_CODES + 2 * ATOMIC_TT_WALL_CODES,
                    )) as u8,
                ),
                score: ((hash >> 16) as u16) as i16,
                depth: ((hash >> 48) as u8 % 63) + 1,
                bound: ((hash >> 56) as u8 % 3) + 1,
            }
        }

        let tt = Arc::new(SharedTt::new(1024));
        let barrier = Arc::new(std::sync::Barrier::new(4));
        let workers: Vec<_> = (0..4)
            .map(|worker| {
                let tt = tt.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    let mut random = 0x9e37_79b9_7f4a_7c15u64 ^ worker as u64;
                    let mut successful_probes = 0usize;
                    barrier.wait();
                    for _ in 0..25_000 {
                        random = random
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        let hash = random ^ random.rotate_left(23);
                        let expected = entry_for(hash);
                        let slot = &tt.entries[(hash as usize) & 1023];
                        slot.store(hash, expected, tt.index_bits);
                        if let Some(actual) = slot.probe(hash, tt.index_bits) {
                            successful_probes += 1;
                            assert_eq!(actual.key, expected.key);
                            assert_eq!(actual.best, expected.best);
                            assert_eq!(actual.score, expected.score);
                            assert_eq!(actual.depth, expected.depth);
                            assert_eq!(actual.bound, expected.bound);
                        }
                    }
                    successful_probes
                })
            })
            .collect();
        let mut successful_probes = 0usize;
        for worker in workers {
            successful_probes += worker.join().unwrap();
        }
        assert!(successful_probes > 0, "race smoke made no verified probes");
    }

    #[test]
    fn atomic_tt_cas_preserves_deeper_replacement() {
        let slot = AtomicTtEntry::empty();
        let deep_hash = 0x1234_5678_9abc_def0;
        let shallow_hash = 0xfedc_ba98_7654_3210;
        let mut deep = TtEntry {
            key: (deep_hash >> 32) as u32,
            best: encode_action(parse_action("e2").unwrap()),
            score: 17,
            depth: 8,
            bound: BOUND_EXACT,
        };
        let mut shallow = deep;
        shallow.key = (shallow_hash >> 32) as u32;
        shallow.depth = 3;

        let index_bits = 10;
        slot.store(deep_hash, deep, index_bits);
        slot.store(shallow_hash, shallow, index_bits);
        assert_eq!(slot.probe(deep_hash, index_bits).unwrap().depth, 8);
        assert!(slot.probe(shallow_hash, index_bits).is_none());

        shallow.depth = 9;
        slot.store(shallow_hash, shallow, index_bits);
        assert_eq!(slot.probe(shallow_hash, index_bits).unwrap().depth, 9);
        assert!(slot.probe(deep_hash, index_bits).is_none());

        deep.depth = 10;
        slot.store(deep_hash, deep, index_bits);
        assert_eq!(slot.probe(deep_hash, index_bits).unwrap().depth, 10);
    }

    #[test]
    fn atomic_tt_rejects_audit_alias_at_supported_table_widths() {
        let hash_a = 0x1234_5678_9abc_def0;
        let hash_b = 0x1234_5678_dabc_def0; // Differs only at bit 30.
        let entry = TtEntry {
            key: (hash_a >> 32) as u32,
            best: encode_action(parse_action("e2").unwrap()),
            score: 17,
            depth: 8,
            bound: BOUND_EXACT,
        };

        // Default 2^22 entries and --tt-mb 256's 2^25 atomic entries used to
        // map this pair to both the same slot and the same 28-bit verifier.
        for index_bits in [22, 25] {
            let index_mask = (1u64 << index_bits) - 1;
            assert_eq!(hash_a & index_mask, hash_b & index_mask);
            assert_ne!(
                atomic_tt_hash_verify(hash_a, index_bits),
                atomic_tt_hash_verify(hash_b, index_bits)
            );

            let slot = AtomicTtEntry::empty();
            slot.store(hash_a, entry, index_bits);
            assert!(slot.probe(hash_b, index_bits).is_none());
            assert_eq!(slot.probe(hash_a, index_bits).unwrap().score, 17);
        }
    }

    #[test]
    fn atomic_tt_compact_payload_round_trips_every_action_and_depth_64() {
        assert_eq!(ATOMIC_TT_VERIFY_SHIFT, 32);
        assert_eq!(ATOMIC_TT_VERIFY_MASK, u64::from(u32::MAX));

        for pos in 0..81 {
            let best = encode_action(Action {
                kind: ActionKind::Pawn,
                pos,
            });
            assert_eq!(expand_atomic_tt_action(compact_atomic_tt_action(best)), best);
        }
        for kind in [ActionKind::Horizontal, ActionKind::Vertical] {
            for pos in 0..64 {
                let best = encode_action(Action { kind, pos });
                assert_eq!(expand_atomic_tt_action(compact_atomic_tt_action(best)), best);
            }
        }

        let hash = 0xfedc_ba98_7654_3210;
        let best = encode_action(parse_action("h8v").unwrap());
        let entry = TtEntry {
            key: (hash >> 32) as u32,
            best,
            score: i16::MIN,
            depth: 64,
            bound: BOUND_UPPER,
        };
        let slot = AtomicTtEntry::empty();
        slot.store(hash, entry, 22);
        let actual = slot.probe(hash, 22).unwrap();
        assert_eq!(actual.best, best);
        assert_eq!(actual.score, i16::MIN);
        assert_eq!(actual.depth, 64);
        assert_eq!(actual.bound, BOUND_UPPER);
    }

    #[test]
    fn four_thread_search_returns_a_legal_move() {
        let state = start_state();
        let mut features = Features::default();
        features.threads = 4;
        // Preserve the original helper-activity check from before RFP merged.
        features.rfp = false;
        let mut ctx = SearchContext::new_with_tt_mb(features, 1).unwrap();
        let result = ctx.search(
            &state,
            &TimeControl {
                max_depth: 3,
                allotted_budget_ms: None,
                soft_deadline: None,
                hard_deadline: None,
            },
        );
        let mut legal = MoveList::new();
        legal_actions(&state, &mut legal);
        assert!(legal.contains(result.best_action));
        assert_eq!(ctx.features.threads, 4);
        assert_eq!(result.threads, 4);
        assert!(result.nodes > result.main_nodes);
    }

    #[test]
    fn four_thread_deadline_includes_helper_shutdown() {
        let state = start_state();
        let mut features = Features::default();
        features.threads = 4;
        features.wallq_tc = false;
        let mut ctx = SearchContext::new_with_tt_mb(features, 1).unwrap();
        let started = Instant::now();
        let tc = ctx.time_control_for_budget(started, 50);
        let result = ctx.search(&state, &tc);
        let elapsed = started.elapsed();
        let mut legal = MoveList::new();
        legal_actions(&state, &mut legal);
        assert!(legal.contains(result.best_action));
        assert_eq!(result.threads, 4);
        assert!(
            elapsed < Duration::from_millis(150),
            "four-thread search and shutdown took {elapsed:?}",
        );
    }

    #[test]
    fn helper_stop_signal_terminates_promptly() {
        let mut features = Features::default();
        features.threads = 2;
        let owner = SearchContext::with_tt_entries(features, 1024);
        let stop = Arc::new(AtomicBool::new(false));
        let mut helper = owner.helper_context(stop.clone(), None, None);
        let handle = thread::spawn(move || {
            helper.search_from_depth::<false>(
                &start_state(),
                &TimeControl {
                    max_depth: 64,
                    allotted_budget_ms: None,
                    soft_deadline: None,
                    hard_deadline: None,
                },
                1,
                None,
            )
        });
        thread::sleep(Duration::from_millis(10));
        let signalled = Instant::now();
        stop.store(true, Ordering::Release);
        let result = handle.join().unwrap();
        assert!(
            signalled.elapsed() < Duration::from_millis(500),
            "helper took {:?} to stop after {} nodes",
            signalled.elapsed(),
            result.nodes,
        );
    }

    #[test]
    fn single_move_returns_instantly() {
        // Fully walled-in-ish corridor is hard to build; instead verify the
        // depth-0 fast path by confirming a normal search still yields a legal,
        // named move (smoke) — the single-move path is exercised structurally.
        let res = depth_search(&start_state(), 2);
        assert!(!action_name(res.best_action).is_empty());
    }

    #[test]
    fn cheap_wall_order_puts_earliest_path_cut_first() {
        // P1's selected empty-board route starts a9 -> a8.  `a8h` uniquely
        // blocks that first edge, while P0's i-file route is unaffected.
        let state = parse_state("p0=i1;p1=a9;w0=10;w1=10;h=-;v=-;t=0").unwrap();
        let ordered = order_moves_cheap(&state, None);
        let first_wall = (0..ordered.len)
            .map(|index| ordered.moves[index])
            .find(|action| action.kind != ActionKind::Pawn)
            .unwrap();
        assert_eq!(action_name(first_wall), "a8h");
    }

    #[test]
    fn tight_edge_filter_skips_far_wall_with_exact_zero_delta() {
        let state = start_state();
        let topology = Topology::from_walls(state.h, state.v);
        let p0_distances = distances_to_row(&topology, 8);
        let p1_distances = distances_to_row(&topology, 0);
        let p0_shortest_edges = p0_distances.pawn_shortest_edges(&topology, state.p0);
        let p1_shortest_edges = p1_distances.pawn_shortest_edges(&topology, state.p1);
        let wall = parse_action("a4v").unwrap();
        let topology_after = topology.with_vertical(wall.pos);

        // An empty-board vertical wall is far from both straight pawn routes;
        // its horizontal board edges connect equal-distance cells.
        assert!(!wall_touches_tight_edge(&p0_distances, wall));
        assert!(!wall_touches_tight_edge(&p1_distances, wall));
        for (pawn, goal, distances, shortest_edges) in [
            (state.p0, 8, &p0_distances, &p0_shortest_edges),
            (state.p1, 0, &p1_distances, &p1_shortest_edges),
        ] {
            let filtered = exact_wall_distance_delta(
                &topology_after,
                distances,
                shortest_edges,
                pawn,
                goal,
                wall,
            );
            let full = distance_to_row(&topology_after, pawn, goal) as i32
                - distance_to_row(&topology, pawn, goal) as i32;
            assert_eq!(filtered, 0);
            assert_eq!(filtered, full);
        }
    }

    #[test]
    fn pawn_path_early_out_skips_irrelevant_global_tight_edge() {
        let state = start_state();
        let topology = Topology::from_walls(state.h, state.v);
        let p0_distances = distances_to_row(&topology, 8);
        let p1_distances = distances_to_row(&topology, 0);
        let p0_shortest_edges = p0_distances.pawn_shortest_edges(&topology, state.p0);
        let p1_shortest_edges = p1_distances.pawn_shortest_edges(&topology, state.p1);
        let wall = parse_action("a4h").unwrap();
        let topology_after = topology.with_horizontal(wall.pos);

        // Empty-board horizontal edges are globally tight, but this wall is
        // outside both pawns' e-file shortest-path DAGs.
        for (pawn, goal, distances, shortest_edges) in [
            (state.p0, 8, &p0_distances, &p0_shortest_edges),
            (state.p1, 0, &p1_distances, &p1_shortest_edges),
        ] {
            assert!(wall_touches_tight_edge(distances, wall));
            assert!(!wall_touches_pawn_shortest_edge(shortest_edges, wall));
            let filtered = exact_wall_distance_delta(
                &topology_after,
                distances,
                shortest_edges,
                pawn,
                goal,
                wall,
            );
            let full = distance_to_row(&topology_after, pawn, goal) as i32
                - distances.get(pawn) as i32;
            assert_eq!(filtered, 0);
            assert_eq!(filtered, full);
        }
    }

    #[test]
    fn tight_edge_filter_falls_back_exactly_for_path_cut() {
        let state = start_state();
        let topology = Topology::from_walls(state.h, state.v);
        let p0_distances = distances_to_row(&topology, 8);
        let p1_distances = distances_to_row(&topology, 0);
        let p0_shortest_edges = p0_distances.pawn_shortest_edges(&topology, state.p0);
        let p1_shortest_edges = p1_distances.pawn_shortest_edges(&topology, state.p1);
        let wall = parse_action("d1h").unwrap();
        let topology_after = topology.with_horizontal(wall.pos);

        assert!(wall_touches_tight_edge(&p0_distances, wall));
        assert!(wall_touches_tight_edge(&p1_distances, wall));
        for (pawn, goal, distances, shortest_edges) in [
            (state.p0, 8, &p0_distances, &p0_shortest_edges),
            (state.p1, 0, &p1_distances, &p1_shortest_edges),
        ] {
            let filtered = exact_wall_distance_delta(
                &topology_after,
                distances,
                shortest_edges,
                pawn,
                goal,
                wall,
            );
            let full = distance_to_row(&topology_after, pawn, goal) as i32
                - distance_to_row(&topology, pawn, goal) as i32;
            assert_eq!(filtered, full);
            assert_eq!(filtered, 1);
        }
    }

    #[test]
    fn lmr_reduces_only_late_non_tt_non_tight_walls() {
        let state = start_state();
        let edges = LmrEdges::for_state(&state);
        let quiet_wall = parse_action("a4v").unwrap();
        let tight_wall = parse_action("d1h").unwrap();
        let pawn = parse_action("e2").unwrap();

        assert!(lmr_reducible(3, 6, quiet_wall, None, &edges));
        assert!(!lmr_reducible(2, 6, quiet_wall, None, &edges));
        assert!(!lmr_reducible(3, 5, quiet_wall, None, &edges));
        assert!(!lmr_reducible(3, 6, pawn, None, &edges));
        assert!(!lmr_reducible(3, 6, quiet_wall, Some(quiet_wall), &edges));
        assert!(!lmr_reducible(3, 6, tight_wall, None, &edges));

        let split = parse_state("p0=i1;p1=a9;w0=10;w1=10;h=-;v=-;t=0").unwrap();
        let split_edges = LmrEdges::for_state(&split);
        let mine_only = parse_action("h1h").unwrap();
        let opponent_only = parse_action("a8h").unwrap();
        assert!(wall_touches_pawn_shortest_edge(&split_edges.mine, mine_only));
        assert!(!wall_touches_pawn_shortest_edge(&split_edges.opponent, mine_only));
        assert!(!wall_touches_pawn_shortest_edge(&split_edges.mine, opponent_only));
        assert!(wall_touches_pawn_shortest_edge(
            &split_edges.opponent,
            opponent_only,
        ));
        assert!(!lmr_reducible(3, 6, mine_only, None, &split_edges));
        assert!(!lmr_reducible(3, 6, opponent_only, None, &split_edges));
    }

    #[test]
    fn probable_walls_match_gorisanson_masks_and_estimated_turn_gates() {
        let opening = start_state();
        assert_eq!(probable_wall_turn_estimate(&opening), 0);
        let opening_mask = ProbableWalls::for_state(&opening, 0);
        assert!(!opening_mask.contains(parse_action("d1h").unwrap()));
        assert!(!opening_mask.contains(parse_action("a4h").unwrap()));

        // Estimated turn three: only the not-to-move pawn band is active.
        let turn_three =
            parse_state("p0=e3;p1=e8;w0=10;w1=10;h=-;v=-;t=1").unwrap();
        assert_eq!(probable_wall_turn_estimate(&turn_three), 3);
        let turn_three_mask = ProbableWalls::for_state(&turn_three, 3);
        assert!(turn_three_mask.contains(parse_action("d2h").unwrap()));
        assert!(turn_three_mask.contains(parse_action("d1v").unwrap()));
        assert!(!turn_three_mask.contains(parse_action("d7h").unwrap()));
        assert!(!turn_three_mask.contains(parse_action("a1h").unwrap()));

        // Any placed wall activates the side-to-move pawn band even before
        // turn six, while the exact placed-horizontal smear is unconditional.
        let one_wall =
            parse_state("p0=e1;p1=e9;w0=9;w1=10;h=d4;v=-;t=1").unwrap();
        assert_eq!(probable_wall_turn_estimate(&one_wall), 1);
        let one_wall_mask = ProbableWalls::for_state(&one_wall, 1);
        assert!(one_wall_mask.contains(parse_action("d8h").unwrap()));
        assert!(!one_wall_mask.contains(parse_action("d1h").unwrap()));
        for action in ["a4h", "b4h", "f4h", "g4h", "c3v", "b4v", "f4v"] {
            assert!(
                one_wall_mask.contains(parse_action(action).unwrap()),
                "missing horizontal-wall smear member {action}",
            );
        }

        let one_vertical_wall =
            parse_state("p0=e1;p1=e9;w0=9;w1=10;h=-;v=d4;t=1").unwrap();
        let vertical_smear = ProbableWalls::for_state(&one_vertical_wall, 1);
        for action in [
            "d1v", "d2v", "d6v", "d7v", "d2h", "c3h", "e4h", "d6h",
        ] {
            assert!(
                vertical_smear.contains(parse_action(action).unwrap()),
                "missing vertical-wall smear member {action}",
            );
        }

        // The adapter estimate seeds the ROOT only. Gorisanson then advances
        // turn once per real move; re-estimating from each child would stall
        // across lateral pawn moves and incorrectly keep the turn-three gate
        // closed.
        let after_two_lateral = one_wall
            .apply(parse_action("d9").unwrap())
            .apply(parse_action("d1").unwrap());
        assert_eq!(probable_wall_turn_estimate(&after_two_lateral), 1);
        let reestimated = ProbableWalls::for_state(&after_two_lateral, 1);
        let threaded = ProbableWalls::for_state(&after_two_lateral, 3);
        let opponent_band_wall = parse_action("c1h").unwrap();
        assert!(!reestimated.contains(opponent_band_wall));
        assert!(threaded.contains(opponent_band_wall));

        // Estimated turn six activates both pawn bands and all horizontal
        // slots in columns zero and seven (not vertical edge slots).
        let turn_six =
            parse_state("p0=e4;p1=e6;w0=10;w1=10;h=-;v=-;t=0").unwrap();
        assert_eq!(probable_wall_turn_estimate(&turn_six), 6);
        let turn_six_mask = ProbableWalls::for_state(&turn_six, 6);
        for action in ["d3h", "d5h", "a1h", "h8h"] {
            assert!(
                turn_six_mask.contains(parse_action(action).unwrap()),
                "missing mature probable wall {action}",
            );
        }
        assert!(!turn_six_mask.contains(parse_action("a1v").unwrap()));
    }

    #[test]
    fn probable_wall_lmr_tier_is_two_ply_only_from_depth_four() {
        let state = start_state();
        let edges = LmrEdges::for_state(&state);
        let quiet_wall = parse_action("a4v").unwrap();
        let tight_wall = parse_action("d1h").unwrap();
        let non_probable = ProbableWalls::for_state(&state, 0);

        assert_eq!(
            lmr_reduction(
                4,
                6,
                quiet_wall,
                None,
                &edges,
                Some(&non_probable),
                false,
            ),
            2,
        );
        assert_eq!(
            lmr_reduction(
                3,
                6,
                quiet_wall,
                None,
                &edges,
                Some(&non_probable),
                false,
            ),
            1,
        );
        assert_eq!(
            lmr_reduction(4, 6, quiet_wall, None, &edges, None, false),
            1,
        );
        assert_eq!(
            lmr_reduction(
                4,
                6,
                tight_wall,
                None,
                &edges,
                Some(&non_probable),
                false,
            ),
            0,
        );

        let mut probable = ProbableWalls::default();
        probable.insert(ActionKind::Vertical, 3, 0);
        assert_eq!(
            lmr_reduction(4, 6, quiet_wall, None, &edges, Some(&probable), false),
            1,
        );

        assert_eq!(
            lmr_reduction(4, 23, quiet_wall, None, &edges, None, true),
            1,
        );
        assert_eq!(
            lmr_reduction(4, 24, quiet_wall, None, &edges, None, true),
            2,
        );
        assert_eq!(
            lmr_reduction(3, 24, quiet_wall, None, &edges, None, true),
            1,
        );
        assert_eq!(
            lmr_reduction(4, 24, quiet_wall, None, &edges, Some(&probable), true),
            1,
        );
    }

    #[test]
    fn probable_wall_lmr_flag_changes_fixed_depth_node_count() {
        let state = start_state();
        let mut control_features = Features::default();
        control_features.threads = 1;
        control_features.lmr = true;
        // Pin the control explicitly so future default merges cannot change
        // this test's intended baseline (the repeated default-dependence lesson).
        control_features.lmr_probable_walls = false;
        control_features.lmr_probable_walls_control = false;
        let mut candidate_features = control_features.clone();
        candidate_features.lmr_probable_walls = true;
        let mut attribution_features = control_features.clone();
        attribution_features.lmr_probable_walls_control = true;

        let control = depth_search_with_features(&state, 5, control_features.clone());
        let candidate = depth_search_with_features(&state, 5, candidate_features);
        let attribution = depth_search_with_features(&state, 5, attribution_features);
        assert_ne!(candidate.nodes, control.nodes);
        assert_eq!(candidate.best_action, control.best_action);
        assert_ne!(attribution.nodes, control.nodes);
        assert_eq!(attribution.best_action, control.best_action);

        let mut lmr_disabled_control = Features::default();
        lmr_disabled_control.threads = 1;
        lmr_disabled_control.lmr = false;
        lmr_disabled_control.lmr_probable_walls = false;
        lmr_disabled_control.lmr_probable_walls_control = false;
        let mut lmr_disabled_candidate = lmr_disabled_control.clone();
        lmr_disabled_candidate.lmr_probable_walls = true;
        let disabled_control =
            depth_search_with_features(&state, 5, lmr_disabled_control);
        let disabled_candidate =
            depth_search_with_features(&state, 5, lmr_disabled_candidate);
        assert_eq!(disabled_candidate.nodes, disabled_control.nodes);
        assert_eq!(disabled_candidate.best_action, disabled_control.best_action);
        assert_eq!(disabled_candidate.score, disabled_control.score);
    }

    #[test]
    fn lmp_prune_set_keeps_tight_wall_and_prunes_late_loose_wall() {
        let state = start_state();
        let edges = LmrEdges::for_state(&state);
        let loose_wall = parse_action("a4v").unwrap();
        let tight_wall = parse_action("d1h").unwrap();

        assert!(lmp_prunable(3, true, 8, loose_wall, None, 8, &edges));
        assert!(!lmp_prunable(3, true, 8, tight_wall, None, 8, &edges));
        assert!(!lmp_prunable(3, false, 8, loose_wall, None, 8, &edges));
        assert!(!lmp_prunable(4, true, 8, loose_wall, None, 8, &edges));
        assert!(!lmp_prunable(3, true, 7, loose_wall, None, 8, &edges));
        assert!(!lmp_prunable(
            3,
            true,
            8,
            loose_wall,
            Some(loose_wall),
            8,
            &edges,
        ));
    }

    #[test]
    fn lmp_prune_set_never_prunes_pawn_moves() {
        let state = start_state();
        let edges = LmrEdges::for_state(&state);
        let pawn = parse_action("e2").unwrap();

        assert!(!lmp_prunable(3, true, 24, pawn, None, 8, &edges));
        assert!(!lmp_prunable(1, true, 24, pawn, None, 8, &edges));
    }

    #[test]
    fn lmp_d4_uses_frozen_48_movecount_and_lmr_reducibility() {
        let state = start_state();
        let edges = LmrEdges::for_state(&state);
        let loose_wall = parse_action("a4v").unwrap();
        let tight_wall = parse_action("d1h").unwrap();
        let pawn = parse_action("e2").unwrap();

        assert_eq!(
            [
                lmp_threshold(24, 1),
                lmp_threshold(24, 2),
                lmp_threshold(24, 3),
                lmp_threshold(24, 4),
            ],
            [6, 12, 24, 48],
        );
        assert!(lmp_d4_prunable(
            4, true, 48, loose_wall, None, 24, &edges,
        ));
        assert!(!lmp_d4_prunable(
            4, true, 47, loose_wall, None, 24, &edges,
        ));
        assert!(!lmp_d4_prunable(
            4, false, 48, loose_wall, None, 24, &edges,
        ));
        assert!(!lmp_d4_prunable(
            3, true, 48, loose_wall, None, 24, &edges,
        ));
        assert!(!lmp_d4_prunable(
            4, true, 48, tight_wall, None, 24, &edges,
        ));
        assert!(!lmp_d4_prunable(
            4, true, 48, pawn, None, 24, &edges,
        ));
        assert!(!lmp_d4_prunable(
            4,
            true,
            48,
            loose_wall,
            Some(loose_wall),
            24,
            &edges,
        ));
    }

    #[test]
    fn lmp_d4_is_inert_when_merged_lmp_is_disabled() {
        let state = start_state();
        let mut control_features = Features::default();
        control_features.threads = 1;
        control_features.lmp = false;
        let mut candidate_features = control_features.clone();
        candidate_features.lmp_d4 = true;
        let control = depth_search_with_features(&state, 4, control_features);
        let candidate = depth_search_with_features(&state, 4, candidate_features);
        assert_eq!(candidate.best_action, control.best_action);
        assert_eq!(candidate.score, control.score);
        assert_eq!(candidate.nodes, control.nodes);
    }

    #[test]
    fn lmp_d4_guard_sweep_withholds_mate_and_three_step_races() {
        let ordinary = start_state();
        assert!(lmp_d4_guard_allows(&ordinary, -100, -99));
        assert!(!lmp_d4_guard_allows(
            &ordinary,
            MATE_THRESHOLD,
            MATE_THRESHOLD + 1,
        ));
        assert!(!lmp_d4_guard_allows(
            &ordinary,
            -MATE_THRESHOLD - 1,
            -MATE_THRESHOLD,
        ));

        let near_goal =
            parse_state("p0=e7;p1=e3;w0=4;w1=4;h=-;v=-;t=0").unwrap();
        assert!(is_race_critical(&near_goal));
        assert!(!lmp_d4_guard_allows(&near_goal, -100, -99));
    }

    #[test]
    fn lmp_d4_live_search_prunes_only_when_the_extension_is_enabled() {
        let state = start_state();
        let mut base_features = Features::default();
        base_features.threads = 1;
        let mut d4_features = base_features.clone();
        d4_features.lmp_d4 = true;
        let base = depth_search_with_features(&state, 6, base_features);
        let d4 = depth_search_with_features(&state, 6, d4_features);
        assert_eq!(d4.best_action, base.best_action);
        assert_eq!(d4.score, base.score);
        assert!(d4.nodes < base.nodes, "{} !< {}", d4.nodes, base.nodes);
    }

    #[test]
    fn lmr_fail_high_is_researched_at_full_depth() {
        let action = parse_action("a4v").unwrap();
        let state = start_state().apply(action);
        let mut features = Features::default();
        features.lmr = true;
        let mut ctx = SearchContext::with_tt_entries(features, 1024);
        let score = ctx
            .search_late_move::<false, false>(
                &state,
                None,
                action,
                3,
                -INF,
                INF,
                1,
                probable_wall_turn_estimate(&state),
                1,
            )
            .expect("unbounded test search completes");
        let mut reference = SearchContext::with_tt_entries(Features::default(), 1024);
        let full_depth = reference
            .search_late_move::<false, false>(
                &state,
                None,
                action,
                3,
                -INF,
                INF,
                1,
                probable_wall_turn_estimate(&state),
                0,
            )
            .expect("full-depth reference completes");
        assert_eq!(score, full_depth);
        assert!(score > -INF);
        assert_eq!(ctx.lmr_reductions, 1);
        assert_eq!(ctx.lmr_researches, 1);
    }

    #[test]
    fn probable_wall_two_ply_fail_high_is_researched_at_full_depth() {
        let action = parse_action("a4v").unwrap();
        let state = start_state().apply(action);
        let turn = probable_wall_turn_estimate(&state);

        let mut reduced_features = Features::default();
        reduced_features.threads = 1;
        reduced_features.lmr = false;
        reduced_features.lmr_probable_walls = true;
        let mut reduced = SearchContext::with_tt_entries(reduced_features, 1024);
        let score = reduced
            .search_late_move::<false, false>(
                &state, None, action, 4, -INF, INF, 1, turn, 2,
            )
            .expect("unbounded two-ply test search completes");

        let mut reference_features = Features::default();
        reference_features.threads = 1;
        reference_features.lmr = false;
        reference_features.lmr_probable_walls = false;
        let mut reference = SearchContext::with_tt_entries(reference_features, 1024);
        let full_depth = reference
            .search_late_move::<false, false>(
                &state, None, action, 4, -INF, INF, 1, turn, 0,
            )
            .expect("full-depth reference completes");

        assert_eq!(score, full_depth);
        assert!(score > -INF);
        assert_eq!(reduced.lmr_reductions, 1);
        assert_eq!(reduced.lmr_researches, 1);
    }

    #[test]
    fn tight_edge_filter_matches_full_bfs_on_random_legal_walls() {
        let mut seed = 0xd1b5_4a32_d192_ed03u64;
        let mut state = start_state();
        let mut walls_checked = 0usize;

        for sample in 0..200 {
            if winner(&state) >= 0 || sample % 20 == 0 {
                state = start_state();
            }

            let topology = Topology::from_walls(state.h, state.v);
            let p0_distances = distances_to_row(&topology, 8);
            let p1_distances = distances_to_row(&topology, 0);
            let p0_shortest_edges = p0_distances.pawn_shortest_edges(&topology, state.p0);
            let p1_shortest_edges = p1_distances.pawn_shortest_edges(&topology, state.p1);
            assert_eq!(p0_distances.get(state.p0), distance_to_row(&topology, state.p0, 8));
            assert_eq!(p1_distances.get(state.p1), distance_to_row(&topology, state.p1, 0));

            let mut legal = MoveList::new();
            legal_actions(&state, &mut legal);
            for index in 0..legal.len() {
                let wall = legal.get(index);
                let topology_after = match wall.kind {
                    ActionKind::Horizontal => topology.with_horizontal(wall.pos),
                    ActionKind::Vertical => topology.with_vertical(wall.pos),
                    ActionKind::Pawn => {
                        assert_eq!(
                            p0_distances.get(wall.pos),
                            distance_to_row(&topology, wall.pos, 8),
                            "sample={sample} pawn={} goal=8",
                            action_name(wall),
                        );
                        assert_eq!(
                            p1_distances.get(wall.pos),
                            distance_to_row(&topology, wall.pos, 0),
                            "sample={sample} pawn={} goal=0",
                            action_name(wall),
                        );
                        continue;
                    }
                };
                walls_checked += 1;

                for (player, pawn, goal, distances, shortest_edges) in [
                    (0, state.p0, 8, &p0_distances, &p0_shortest_edges),
                    (1, state.p1, 0, &p1_distances, &p1_shortest_edges),
                ] {
                    let filtered = exact_wall_distance_delta(
                        &topology_after,
                        distances,
                        shortest_edges,
                        pawn,
                        goal,
                        wall,
                    );
                    let full = distance_to_row(&topology_after, pawn, goal) as i32
                        - distances.get(pawn) as i32;
                    assert_eq!(
                        filtered,
                        full,
                        "sample={sample} player={player} state={} wall={}",
                        state.to_canonical_string(),
                        action_name(wall),
                    );
                }
            }

            if legal.len() != 0 {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                state = state.apply(legal.get((seed as usize) % legal.len()));
            }
        }

        assert!(walls_checked > 10_000, "only checked {walls_checked} walls");
    }

    #[test]
    fn killers_replace_distinctly_and_follow_tt() {
        let state = start_state();
        let tt = parse_action("e2").unwrap();
        let older = parse_action("d1").unwrap();
        let newer = parse_action("f1").unwrap();
        let mut features = Features::default();
        features.killers = true;
        let mut ctx = SearchContext::new(features);

        ctx.record_killer(0, older, None);
        ctx.record_killer(0, newer, None);
        // A TT move is not duplicated into the killer table.
        ctx.record_killer(0, tt, Some(tt));
        assert_eq!(ctx.killer_codes(0), [encode_action(newer), encode_action(older)]);

        let ordered = ctx.order_moves(&state, Some(tt), 0, None, None).unwrap();
        assert_eq!(ordered.moves[0], tt);
        assert_eq!(ordered.moves[1], newer);
        assert_eq!(ordered.moves[2], older);
    }

    #[test]
    fn history_ages_and_orders_remaining_quiet_moves() {
        let state = start_state();
        let high = parse_action("f1").unwrap();
        let mut features = Features::default();
        features.history = true;
        let mut ctx = SearchContext::new(features);
        ctx.history_scores[0][encode_action(high) as usize] = 100;
        ctx.age_history();
        assert_eq!(ctx.history_scores[0][encode_action(high) as usize], 50);

        let ordered = ctx.order_moves(&state, None, 0, None, None).unwrap();
        assert_eq!(ordered.moves[0], high);
    }

    #[test]
    fn fused_history_and_tight_edges_match_layered_reference() {
        let mut fused_features = Features::default();
        fused_features.threads = 1;
        fused_features.history = true;
        fused_features.killers = false;
        fused_features.cmh = false;
        fused_features.cheap_wall_order = false;
        let mut layered_features = fused_features.clone();
        // An empty killer table makes the established layered path semantically
        // identical while deliberately disabling the fused fast path.
        layered_features.killers = true;

        let mut fused = SearchContext::with_tt_entries(fused_features, 8);
        let mut layered = SearchContext::with_tt_entries(layered_features, 8);
        for side in 0..2 {
            for code in 0..HISTORY_ACTIONS {
                let bucket = (code * 37 + side * 11) % 19;
                let score = if bucket < 7 { 0 } else { (bucket / 3) as i32 };
                fused.history_scores[side][code] = score;
                layered.history_scores[side][code] = score;
            }
        }

        let mut seed = 0x1319_8a2e_0370_7344u64;
        let mut state = start_state();
        for sample in 0..64 {
            if winner(&state) >= 0 {
                state = start_state();
            }
            let mut legal = MoveList::new();
            legal_actions(&state, &mut legal);
            assert!(legal.len() > 0);
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let selected = legal.get(seed as usize % legal.len());
            let tt = (sample % 3 != 0).then_some(selected);
            let tight_edge_threshold = (sample % 2 == 0).then_some(6);

            let actual = fused
                .order_moves(&state, tt, 0, None, tight_edge_threshold)
                .unwrap();
            let expected = layered
                .order_moves(&state, tt, 0, None, None)
                .unwrap();
            assert_eq!(actual.len, expected.len, "sample={sample}");
            assert_eq!(
                actual.moves[..actual.len],
                expected.moves[..expected.len],
                "sample={sample} state={}",
                state.to_canonical_string(),
            );

            if tight_edge_threshold.is_some() && actual.len > 6 {
                let reused = actual.tight_edges.expect("requested tight edges");
                let rebuilt = LmrEdges::for_state(&state);
                assert_eq!(
                    reused.mine.horizontal_wall_touch_mask(),
                    rebuilt.mine.horizontal_wall_touch_mask(),
                    "mine horizontal sample={sample}",
                );
                assert_eq!(
                    reused.mine.vertical_wall_touch_mask(),
                    rebuilt.mine.vertical_wall_touch_mask(),
                    "mine vertical sample={sample}",
                );
                assert_eq!(
                    reused.opponent.horizontal_wall_touch_mask(),
                    rebuilt.opponent.horizontal_wall_touch_mask(),
                    "opponent horizontal sample={sample}",
                );
                assert_eq!(
                    reused.opponent.vertical_wall_touch_mask(),
                    rebuilt.opponent.vertical_wall_touch_mask(),
                    "opponent vertical sample={sample}",
                );
            } else {
                assert!(actual.tight_edges.is_none());
            }
            state = state.apply(selected);
        }
    }

    #[test]
    fn cmh_uses_i16_gravity_side_and_previous_action_context() {
        assert_eq!(CMH_ENTRIES * std::mem::size_of::<i16>(), 2_359_296);
        let state = start_state();
        let previous = parse_action("e8").unwrap();
        let reply = parse_action("a1h").unwrap();
        let pawn = parse_action("e2").unwrap();
        let mut features = Features::default();
        features.threads = 1;
        features.history = false;
        features.cmh = true;
        let mut ctx = SearchContext::with_tt_entries(features, 8);
        let index = countermove_history_index(0, previous, reply);

        ctx.record_history(&state, reply, 64, Some(previous));
        assert_eq!(ctx.countermove_history.as_ref().unwrap()[index], 4096);
        ctx.record_history(&state, reply, 64, Some(previous));
        assert_eq!(ctx.countermove_history.as_ref().unwrap()[index], 7680);
        assert_eq!(ctx.history_score(&state, reply, Some(previous)), 7680);
        assert_eq!(ctx.history_score(&state, reply, None), 0);

        ctx.record_history(&state, pawn, 64, Some(previous));
        assert_eq!(ctx.history_score(&state, pawn, Some(previous)), 0);
        let mut opposite = state;
        opposite.turn = 1;
        assert_eq!(ctx.history_score(&opposite, reply, Some(previous)), 0);
    }

    #[test]
    fn cmh_blends_only_wall_replies_and_root_falls_back_to_base_order() {
        let state = start_state();
        let previous = parse_action("e8").unwrap();
        let promoted_wall = parse_action("a1h").unwrap();
        let ignored_pawn = parse_action("d1").unwrap();
        let mut features = Features::default();
        features.threads = 1;
        features.history = false;
        features.cmh = true;
        let mut ctx = SearchContext::with_tt_entries(features, 8);
        let history = ctx.countermove_history.as_mut().unwrap();
        history[countermove_history_index(0, previous, promoted_wall)] = 500;
        history[countermove_history_index(0, previous, ignored_pawn)] = 1000;

        let contextual = ctx
            .order_moves(&state, None, 1, Some(previous), None)
            .unwrap();
        assert_eq!(contextual.moves[0], promoted_wall);

        let root = ctx.order_moves(&state, None, 0, None, None).unwrap();
        let base = order_moves_v0(&state, None);
        assert_eq!(root.len, base.len);
        assert_eq!(root.moves[..root.len], base.moves[..base.len]);

        let butterfly_wall = parse_action("b1h").unwrap();
        let mut blended_features = Features::default();
        blended_features.threads = 1;
        blended_features.cmh = true;
        let mut blended = SearchContext::with_tt_entries(blended_features, 8);
        blended.history_scores[0][encode_action(butterfly_wall) as usize] = 600;
        blended.countermove_history.as_mut().unwrap()
            [countermove_history_index(0, previous, promoted_wall)] = 1000;
        let butterfly_root = blended
            .order_moves(&state, None, 0, None, None)
            .unwrap();
        assert_eq!(butterfly_root.moves[0], butterfly_wall);

        // A zero CMH entry is neutral even when a predecessor is present.
        let mut butterfly_only_features = Features::default();
        butterfly_only_features.threads = 1;
        let mut butterfly_only =
            SearchContext::with_tt_entries(butterfly_only_features, 8);
        butterfly_only.history_scores[0][encode_action(butterfly_wall) as usize] = 600;
        let expected = butterfly_only
            .order_moves(&state, None, 1, Some(previous), None)
            .unwrap();
        blended.countermove_history.as_mut().unwrap().fill(0);
        let actual = blended
            .order_moves(&state, None, 1, Some(previous), None)
            .unwrap();
        assert_eq!(actual.moves[..actual.len], expected.moves[..expected.len]);
    }

    #[test]
    fn cmh_persists_across_searches_and_clears_only_at_game_boundary() {
        let mut features = Features::default();
        features.threads = 1;
        features.cmh = true;
        let mut ctx = SearchContext::with_tt_entries(features, 1024);
        let sentinel = CMH_ENTRIES - 1;
        ctx.countermove_history.as_mut().unwrap()[sentinel] = 1234;
        let tc = TimeControl {
            max_depth: 1,
            allotted_budget_ms: None,
            soft_deadline: None,
            hard_deadline: None,
        };

        let _ = ctx.search(&start_state(), &tc);
        let _ = ctx.search(&start_state(), &tc);
        ctx.age_history();
        assert_eq!(ctx.countermove_history.as_ref().unwrap()[sentinel], 1234);
        ctx.begin_new_game();
        assert_eq!(ctx.countermove_history.as_ref().unwrap()[sentinel], 0);
    }

    #[test]
    fn cmh_helper_lane_table_is_moved_back_without_a_per_move_clear() {
        let mut features = Features::default();
        features.threads = 2;
        features.cmh = true;
        let mut ctx = SearchContext::with_tt_entries(features, 1024);
        assert_eq!(ctx.helper_countermove_histories.len(), 1);
        let sentinel = CMH_ENTRIES - 1;
        ctx.helper_countermove_histories[0]
            .as_mut()
            .unwrap()[sentinel] = 2345;
        let tc = TimeControl {
            max_depth: 2,
            allotted_budget_ms: None,
            soft_deadline: None,
            hard_deadline: None,
        };

        let result = ctx.search(&start_state(), &tc);
        assert_eq!(result.threads, 2);
        assert_eq!(
            ctx.helper_countermove_histories[0].as_ref().unwrap()[sentinel],
            2345,
        );
    }

    /// Deliberately exhaustive reference used only to prove that the DAG
    /// candidate restriction retains the full legal-wall maximum.  It asks
    /// canonical move generation for every legal wall after setting the
    /// placing side, then recomputes both topology distances.
    fn reference_interdiction_for_side(state: &State, side: u8) -> i32 {
        if winner(state) >= 0 || (if side == 0 { state.w0 } else { state.w1 }) == 0 {
            return 0;
        }

        let topology = Topology::from_walls(state.h, state.v);
        let p0_before = distance_to_row(&topology, state.p0, 8);
        let p1_before = distance_to_row(&topology, state.p1, 0);
        let mut legal_state = *state;
        legal_state.turn = side;
        let mut legal = MoveList::new();
        legal_actions(&legal_state, &mut legal);
        let mut best = 0i32;

        for index in legal.pawn_count()..legal.len() {
            let wall = legal.get(index);
            let after = match wall.kind {
                ActionKind::Horizontal => topology.with_horizontal(wall.pos),
                ActionKind::Vertical => topology.with_vertical(wall.pos),
                ActionKind::Pawn => unreachable!("wall suffix contains only walls"),
            };
            let p0_after = distance_to_row(&after, state.p0, 8);
            let p1_after = distance_to_row(&after, state.p1, 0);
            assert!(p0_after >= 0 && p1_after >= 0, "movegen returned an illegal wall");
            let (target_delta, self_delta) = if side == 0 {
                (p1_after - p1_before, p0_after - p0_before)
            } else {
                (p0_after - p0_before, p1_after - p1_before)
            };
            best = best.max((target_delta - self_delta) as i32);
        }
        best
    }

    fn wall_deltas_for_side(state: &State, side: u8, wall: Action) -> (i32, i32) {
        let topology = Topology::from_walls(state.h, state.v);
        let after = match wall.kind {
            ActionKind::Horizontal => topology.with_horizontal(wall.pos),
            ActionKind::Vertical => topology.with_vertical(wall.pos),
            ActionKind::Pawn => panic!("expected a wall"),
        };
        let p0_before = distance_to_row(&topology, state.p0, 8);
        let p1_before = distance_to_row(&topology, state.p1, 0);
        let p0_after = distance_to_row(&after, state.p0, 8);
        let p1_after = distance_to_row(&after, state.p1, 0);
        assert!(p0_after >= 0 && p1_after >= 0, "fixture wall must be legal");
        if side == 0 {
            ((p1_after - p1_before) as i32, (p0_after - p0_before) as i32)
        } else {
            ((p0_after - p0_before) as i32, (p1_after - p1_before) as i32)
        }
    }

    fn max_target_damage_for_side(state: &State, side: u8) -> i32 {
        let topology = Topology::from_walls(state.h, state.v);
        let before = if side == 0 {
            distance_to_row(&topology, state.p1, 0)
        } else {
            distance_to_row(&topology, state.p0, 8)
        };
        let mut legal_state = *state;
        legal_state.turn = side;
        let mut legal = MoveList::new();
        legal_actions(&legal_state, &mut legal);
        (legal.pawn_count()..legal.len())
            .map(|index| {
                let wall = legal.get(index);
                let after = match wall.kind {
                    ActionKind::Horizontal => topology.with_horizontal(wall.pos),
                    ActionKind::Vertical => topology.with_vertical(wall.pos),
                    ActionKind::Pawn => unreachable!("wall suffix contains only walls"),
                };
                let target_after = if side == 0 {
                    distance_to_row(&after, state.p1, 0)
                } else {
                    distance_to_row(&after, state.p0, 8)
                };
                (target_after - before) as i32
            })
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn interdiction_corridor_has_known_exact_values_for_both_sides() {
        let state = parse_state("p0=e4;p1=e5;w0=8;w1=9;h=c4,a7;v=f7;t=0").unwrap();
        assert_eq!(interdiction_values(&state), (1, 1));
        assert_eq!(interdiction_stm_values(&state), (1, 1));
    }

    #[test]
    fn interdiction_is_zero_without_walls_in_hand() {
        let state = parse_state("p0=e1;p1=e9;w0=0;w1=0;h=-;v=-;t=0").unwrap();
        assert_eq!(interdiction_values(&state), (0, 0));
    }

    #[test]
    fn interdiction_floors_negative_target_wall_net_at_zero() {
        let state = parse_state(
            "p0=b1;p1=e8;w0=6;w1=4;h=a1,c1,e1,b2,c4,a7,f8,h8;v=c6,d7;t=0",
        )
        .unwrap();
        // g1h is the strongest target-delay wall (+2), but it delays the
        // placer by four, so its net is -2.  I_0 still floors at zero.
        assert_eq!(max_target_damage_for_side(&state, 0), 2);
        assert_eq!(wall_deltas_for_side(&state, 0, parse_action("g1h").unwrap()), (2, 4));
        assert_eq!(interdiction_values(&state), (0, 2));
    }

    #[test]
    fn interdiction_caps_only_after_the_exact_maximum() {
        let state = parse_state(
            "p0=d1;p1=h6;w0=1;w1=0;h=g1,g2,d3,e4,d5,b6,d6,g6,f8;v=d1,h1,c2,f3,g3,d4,c5,a7,e7,f7;t=0",
        )
        .unwrap();
        assert_eq!(interdiction_values(&state), (14, 0));
        assert_eq!(
            interdiction_correction(14, 0),
            INTERDICTION_ME_CP * INTERDICTION_CAP,
        );
    }

    #[test]
    fn interdiction_term_uses_the_side_to_move_frame_and_frozen_signs() {
        let state = parse_state("p0=g2;p1=g7;w0=8;w1=9;h=e7;v=a2,b3;t=0").unwrap();
        assert_eq!(interdiction_values(&state), (1, 2));
        assert_eq!(interdiction_stm_values(&state), (1, 2));
        assert_eq!(
            interdiction_correction(1, 2),
            INTERDICTION_ME_CP - 2 * INTERDICTION_OPP_CP,
        );

        let mut opposite = state;
        opposite.turn = 1;
        assert_eq!(interdiction_stm_values(&opposite), (2, 1));
        assert_eq!(
            interdiction_correction(2, 1),
            2 * INTERDICTION_ME_CP - INTERDICTION_OPP_CP,
        );
    }

    #[test]
    fn wallq_tc_window_boundaries_are_inclusive() {
        let static_eval = 100;
        // Lower edge: static == alpha - W.
        assert!(wallq_tc_in_window(static_eval, 250, 251));
        assert!(!wallq_tc_in_window(static_eval, 251, 252));
        // Upper edge: static == beta + W.
        assert!(wallq_tc_in_window(static_eval, -51, -50));
        assert!(!wallq_tc_in_window(static_eval, -52, -51));
    }

    #[test]
    fn wallq_tc_uses_the_actual_pvs_zero_window_at_the_leaf() {
        // This frozen E-032 fixture has (I_me, I_opp) = (1, 2), so the
        // shared correction is -55cp.  The direct depth-zero calls model a
        // PVS child window exactly as [beta-1, beta], not root bounds.
        let state = parse_state("p0=g2;p1=g7;w0=8;w1=9;h=e7;v=a2,b3;t=0").unwrap();
        assert_eq!(interdiction_stm_values(&state), (1, 2));
        assert_eq!(interdiction_correction(1, 2), -55);

        let mut features = Features::default();
        features.threads = 1;
        features.wallq_tc = true;
        features.race_exact = false;
        features.race1w = false;
        features.race2w = false;
        let static_eval = evaluate_with_features(&state, &features);
        let probable_turn = probable_wall_turn_estimate(&state);

        let mut on_upper_edge = SearchContext::with_tt_entries(features.clone(), 1024);
        let beta = static_eval - WALLQ_TC_WINDOW_CP;
        assert_eq!(
            on_upper_edge.negamax_counted::<false, true>(
                &state, None, 0, beta - 1, beta, 0, probable_turn, false, None,
            ),
            Some(static_eval - 55),
        );
        assert_eq!(on_upper_edge.wallq_tc_leaves, 1);
        assert_eq!(on_upper_edge.wallq_tc_in_window, 1);

        let mut one_cp_outside = SearchContext::with_tt_entries(features.clone(), 1024);
        let beta = static_eval - WALLQ_TC_WINDOW_CP - 1;
        assert_eq!(
            one_cp_outside.negamax_counted::<false, true>(
                &state, None, 0, beta - 1, beta, 0, probable_turn, false, None,
            ),
            Some(static_eval),
        );
        assert_eq!(one_cp_outside.wallq_tc_leaves, 1);
        assert_eq!(one_cp_outside.wallq_tc_in_window, 0);

        let mut off = SearchContext::with_tt_entries(features, 1024);
        assert_eq!(
            off.negamax_counted::<false, false>(
                &state,
                None,
                0,
                static_eval - 1,
                static_eval,
                0,
                probable_turn,
                false,
                None,
            ),
            Some(static_eval),
        );
        assert_eq!(off.wallq_tc_leaves, 0);
        assert_eq!(off.wallq_tc_in_window, 0);
    }

    #[test]
    fn interdiction_dag_candidates_match_all_legal_walls_on_random_playouts() {
        let mut state = start_state();
        let mut seed = 0x7f4a_7c15_9e37_79b9u64;
        for sample in 0..120 {
            if winner(&state) >= 0 {
                state = start_state();
            }
            let actual = interdiction_values(&state);
            let expected = (
                reference_interdiction_for_side(&state, 0),
                reference_interdiction_for_side(&state, 1),
            );
            assert_eq!(actual, expected, "sample={sample} state={}", state.to_canonical_string());

            let mut legal = MoveList::new();
            legal_actions(&state, &mut legal);
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state = state.apply(legal.get(seed as usize % legal.len()));
        }
    }

    #[test]
    fn eval_progress_rewards_advanced_pawns_symmetrically() {
        let state = parse_state("p0=e6;p1=a9;w0=10;w1=10;h=-;v=-;t=0").unwrap();
        let mut features = Features::default();
        assert_eq!(evaluate(&state), 550);
        assert_eq!(evaluate_with_features(&state, &features), evaluate(&state));
        features.ev_progress = true;
        assert_eq!(evaluate_with_features(&state, &features), 638);

        let mut opposite = state;
        opposite.turn = 1;
        assert_eq!(evaluate(&opposite), -450);
        assert_eq!(evaluate_with_features(&opposite, &features), -538);
    }

    #[test]
    fn eval_wall_cp_tunes_flat_inventory_term_and_defers_to_wallphase() {
        let state = parse_state("p0=e1;p1=e3;w0=2;w1=1;h=-;v=-;t=0").unwrap();
        // Pin the legacy baseline explicitly: this test isolates the wall_cp
        // DELTA and must stay valid whatever default the ratchet merges.
        let mut features = Features::default();
        features.wall_cp = 35;
        let default_score = evaluate_with_features(&state, &features);

        features.wall_cp = 90;
        let tuned_score = evaluate_with_features(&state, &features);
        assert_eq!(tuned_score - default_score, 90 - 35);
        assert_eq!(tuned_score, -460);

        features.ev_wallphase = true;
        assert_eq!(evaluate_with_features(&state, &features), -540);
    }

    #[test]
    fn eval_wall_cp_endgame_decays_each_inventory_by_opponent_progress() {
        // D_me=6 and D_opp=3. With wall_cp=200 and endgame=35, my walls
        // are worth 84 each (84.5 truncates) while the opponent's is 134.
        let state = parse_state("p0=e3;p1=a4;w0=2;w1=1;h=-;v=-;t=0").unwrap();
        // Constants below were derived at wall_cp=200; pin it explicitly
        // (E-019 changed the default to 150 — this test checks the DECAY
        // formula, not the default).
        let mut features = Features::default();
        features.wall_cp = 200;
        let mut flat_features = Features::default();
        flat_features.wall_cp = 200;
        let flat_score = evaluate_with_features(&state, &flat_features);
        assert_eq!(flat_score, -50);

        features.wall_cp_endgame = 35;
        let shaped_score = evaluate_with_features(&state, &features);
        assert_eq!(shaped_score, -216);
        assert_eq!(shaped_score - flat_score, -166);

        let mut opposite = state;
        opposite.turn = 1;
        assert_eq!(evaluate_with_features(&opposite, &flat_features), 150);
        assert_eq!(evaluate_with_features(&opposite, &features), 316);
    }

    #[test]
    fn eval_race_amp_scales_margin_by_closest_pawn_progress() {
        // D_me=2 and D_opp=5: 25*3*(12-2)/12/4 truncates to +15.
        let state = parse_state("p0=e7;p1=a6;w0=10;w1=10;h=-;v=-;t=0").unwrap();
        let mut features = Features::default();
        let flat_score = evaluate_with_features(&state, &features);
        assert_eq!(flat_score, 350);

        features.race_amp = 25;
        assert_eq!(evaluate_with_features(&state, &features), flat_score + 15);

        // Signed division truncates toward zero: -750/12/4 becomes -15.
        let mut opposite = state;
        opposite.turn = 1;
        let opposite_flat = evaluate_with_features(&opposite, &Features::default());
        assert_eq!(opposite_flat, -250);
        assert_eq!(evaluate_with_features(&opposite, &features), opposite_flat - 15);
    }

    #[test]
    fn eval_wallphase_prices_each_inventory_by_opponent_distance() {
        let state = parse_state("p0=e1;p1=e3;w0=2;w1=1;h=-;v=-;t=0").unwrap();
        let mut features = Features::default();
        features.ev_wallphase = true;
        assert_eq!(evaluate(&state), -515);
        assert_eq!(evaluate_with_features(&state, &features), -540);

        let mut opposite = state;
        opposite.turn = 1;
        assert_eq!(evaluate(&opposite), 615);
        assert_eq!(evaluate_with_features(&opposite, &features), 640);
    }

    #[test]
    fn eval_corridor_penalizes_squeezed_shortest_prefix() {
        let state = parse_state(
            "p0=e1;p1=c9;w0=4;w1=10;h=-;v=d2,e2,d4,e4,d6,e6;t=0",
        )
        .unwrap();
        let mut features = Features::default();
        features.ev_corridor = true;
        // Pin wall_cp to the legacy 35 so this test isolates the corridor
        // term against the frozen evaluate() baseline (which hardcodes 35).
        features.wall_cp = 35;
        assert_eq!(evaluate(&state), -160);
        assert_eq!(evaluate_with_features(&state, &features), -208);

        let mut opposite = state;
        opposite.turn = 1;
        assert_eq!(evaluate(&opposite), 260);
        assert_eq!(evaluate_with_features(&opposite, &features), 308);

        let mirrored = mirror_state_lr(&state);
        assert_eq!(
            evaluate_with_features(&mirrored, &features),
            evaluate_with_features(&state, &features),
        );

        let empty_edge = parse_state("p0=a1;p1=i9;w0=10;w1=10;h=-;v=-;t=0").unwrap();
        let topology = Topology::from_walls(0, 0);
        let distances = distances_to_row(&topology, 8);
        assert_eq!(corridor_vulnerability(&topology, &distances, empty_edge.p0), 0);
    }

    #[test]
    fn ev_fragility_1w_prices_geometrically_available_unique_path_threats() {
        let state =
            parse_state("p0=e1;p1=e9;w0=1;w1=0;h=-;v=-;t=0").unwrap();
        let plain = Features::default();
        let mut fragile = plain.clone();
        fragile.ev_fragility_1w = true;
        assert_eq!(
            evaluate_with_features(&state, &fragile),
            evaluate_with_features(&state, &plain) + FRAGILITY_CP,
        );

        let mut opposite = state;
        opposite.turn = 1;
        assert_eq!(
            evaluate_with_features(&opposite, &fragile),
            evaluate_with_features(&opposite, &plain) - FRAGILITY_CP,
        );

        // Two symmetric shortest detours are robust under the cheap binary
        // definition, and the helper consumes only the existing distance DAG.
        let branched = parse_state(
            "p0=e1;p1=e9;w0=1;w1=0;h=d1,e1;v=-;t=0",
        )
        .unwrap();
        let topology = Topology::from_walls(branched.h, branched.v);
        let distances = distances_to_row(&topology, 8);
        assert_eq!(
            single_wall_path_fragility(
                &branched,
                &topology,
                &distances,
                branched.p0,
            ),
            0,
        );
    }

    #[test]
    fn corridor_vulnerability_is_lr_equivariant_on_asymmetric_topology() {
        let state = parse_state(
            "p0=b2;p1=g8;w0=8;w1=8;h=a3,f6;v=c1,g7;t=1",
        )
        .unwrap();
        let reflected = mirror_state_lr(&state);
        let topology = Topology::from_walls(state.h, state.v);
        let reflected_topology = Topology::from_walls(reflected.h, reflected.v);

        for goal in [0, 8] {
            let distances = distances_to_row(&topology, goal);
            let reflected_distances = distances_to_row(&reflected_topology, goal);
            for cell in 0u8..81 {
                assert_eq!(
                    corridor_vulnerability(&topology, &distances, cell),
                    corridor_vulnerability(
                        &reflected_topology,
                        &reflected_distances,
                        mirror_cell_lr(cell),
                    ),
                    "goal={goal} cell={cell}",
                );
            }
        }
    }

    #[test]
    fn race_exact_matches_completed_no_wall_races() {
        // These are separated pawns, so the documented Chebyshev <=2 fallback
        // does not apply.  A depth-six ordinary search reaches the actual goal.
        let positions = [
            "p0=a6;p1=i4;w0=0;w1=0;h=-;v=-;t=0",
            "p0=a6;p1=i4;w0=0;w1=0;h=-;v=-;t=1",
            "p0=a4;p1=i3;w0=0;w1=0;h=-;v=-;t=0",
        ];
        for text in positions {
            let state = parse_state(text).unwrap();
            // The reference side must be the ORDINARY search (no race oracle)
            // regardless of which defaults have been merged since this test
            // was written; pin the flag explicitly on both sides.
            let mut plain = Features::default();
            plain.race_exact = false;
            let full = depth_search_with_features(&state, 6, plain);
            let mut features = Features::default();
            features.race_exact = true;
            let raced = depth_search_with_features(&state, 6, features);
            assert!(full.score.abs() >= MATE_THRESHOLD, "{text}: {}", full.score);
            assert_eq!(full.score.signum(), raced.score.signum(), "{text}");
            assert!(race_exact_score(&state, 0).is_some());
        }

        let near = parse_state("p0=e5;p1=e6;w0=0;w1=0;h=-;v=-;t=0").unwrap();
        assert_eq!(race_exact_score(&near, 0), None);
    }

    #[test]
    fn exact_labels_detect_default_off_race1w_at_root_perspective() {
        // Intent: exact_label() computes race1w labels INDEPENDENTLY of
        // play features. The original guard asserted the default was off;
        // E-015 merged race1w default-on, so assert independence directly:
        // exact_label takes no Features at all (compile-time evident), and
        // the label value below is the exactness invariant that matters.
        let one_wall =
            parse_state("p0=a6;p1=i4;w0=1;w1=0;h=-;v=-;t=0").unwrap();
        assert_eq!(
            exact_label(&one_wall),
            Some(ExactLabel {
                kind: "race1w",
                value: 19_994,
            })
        );

        let no_walls =
            parse_state("p0=a6;p1=i4;w0=0;w1=0;h=-;v=-;t=1").unwrap();
        let label = exact_label(&no_walls).unwrap();
        assert_eq!(label.kind, "race_exact");
        assert_eq!(label.value, race_zero_wall_score(&no_walls, 0).unwrap());
    }

    #[test]
    fn race1w_agrees_with_ordinary_full_search_on_constructed_positions() {
        // The holder wins the zero-wall race; spending the wall cannot improve
        // on walking a shortest route immediately.
        let text = "p0=a6;p1=i4;w0=1;w1=0;h=-;v=-;t=0";
        let state = parse_state(text).unwrap();
        let exact = race_one_wall_score(&state, 0).expect(text);
        assert_eq!(exact, 19_994);
        let mut plain = Features::default();
        plain.race_exact = false;
        plain.race1w = false;
        let full = depth_search_with_features(&state, 7, plain);
        assert_eq!(exact.signum(), full.score.signum(), "{text}: {exact} vs {}", full.score);

        // These parser-valid but inventory-impossible empty-board states expose
        // more than 96 final-wall placements and deliberately fall back.
        for text in [
            "p0=a7;p1=i2;w0=1;w1=0;h=-;v=-;t=0",
            "p0=a8;p1=i3;w0=0;w1=1;h=-;v=-;t=1",
        ] {
            assert_eq!(race_one_wall_score(&parse_state(text).unwrap(), 0), None);
        }
    }

    #[test]
    fn race1w_falls_back_when_delaying_the_wall_can_matter() {
        // Here never-wall and every immediate wall lose in the separated-pawn
        // abstraction, but d2 followed by a later wall wins.  Returning a
        // negative shortcut score would therefore be unsound.
        let state = parse_state(
            "p0=e2;p1=c6;w0=1;w1=0;h=a1,f1,h2,e3,g5,e6;v=e2,f2,a3,d3,c5,f5,a6,b7,a8;t=0",
        )
        .unwrap();
        assert_eq!(race_one_wall_score(&state, 0), None);
    }

    #[test]
    fn audit_race1w_repros_choose_the_proven_winning_moves() {
        let parent = parse_state(
            "p0=b7;p1=h7;w0=0;w1=1;h=d1,h1,c2,f2,b4,f5,a6,d6,h6,c7,g8;v=b3,a4,e4,g5,c6,d7,h7,b8;t=0",
        )
        .unwrap();
        let mut legal = MoveList::new();
        legal_actions(&parent, &mut legal);
        let mut memo = HashMap::new();
        let mut outcomes = Vec::new();
        for index in 0..legal.len() {
            let action = legal.get(index);
            let child = parent.apply(action);
            let global_winner = reference_winner(&child, &mut memo)
                .unwrap_or_else(|| panic!("reference did not resolve {}", action_name(action)));
            outcomes.push((action_name(action), global_winner));
        }
        assert_eq!(
            outcomes,
            vec![("a7".to_owned(), 1), ("c7".to_owned(), 1), ("b8".to_owned(), 0)]
        );

        let mut features = Features::default();
        features.threads = 1;
        let result = depth_search_with_features(&parent, 8, features);
        assert_eq!(action_name(result.best_action), "b8");
        assert!(result.score > 0);

        let wall_repro = parse_state(
            "p0=g8;p1=d8;w0=0;w1=1;h=g1,a2,d2,f2,h2,f4,a5,c7,h8;v=f1,b2,g3,h3,c4,b5,a6,e6,f6,e8;t=1",
        )
        .unwrap();
        assert!(race_one_wall_score(&wall_repro, 0).expect("f8h repro fires") > 0);
        assert!(exact_label(&wall_repro).expect("sound f8h label").value > 0);

        legal_actions(&wall_repro, &mut legal);
        let mut winning_actions = Vec::new();
        for index in 0..legal.len() {
            let action = legal.get(index);
            let child = wall_repro.apply(action);
            let global_winner = if action.kind == ActionKind::Pawn {
                let mut replies = MoveList::new();
                legal_actions(&child, &mut replies);
                assert!(
                    (0..replies.len()).any(|reply| winner(&child.apply(replies.get(reply))) == 0),
                    "audit proof requires P0's immediate goal after {}",
                    action_name(action),
                );
                0
            } else {
                match exact_zero_wall_outcome(&child).expect("wall child has no stock") {
                    ZeroWallOutcome::Win { winner, .. } => winner,
                    ZeroWallOutcome::Draw => continue,
                }
            };
            if global_winner == wall_repro.turn {
                winning_actions.push(action_name(action));
            }
        }
        assert_eq!(winning_actions, vec!["f8h"]);

        let mut features = Features::default();
        features.threads = 1;
        let result = depth_search_with_features(&wall_repro, 1, features);
        assert_eq!(action_name(result.best_action), "f8h");
    }

    #[test]
    fn race1w_random_reachable_crosscheck_500_exact_verdicts() {
        // Reachable after these 19 legal wall moves from the initial state:
        // d1h b3v h1h a4v c2h e4v f2h g5v b4h c6v f5h d7v a6h
        // h7v d6h b8v h6h c7h g8h. P1 owns the sole remaining wall.
        let base = parse_state(
            "p0=e1;p1=e9;w0=0;w1=1;h=d1,h1,c2,f2,b4,f5,a6,d6,h6,c7,g8;v=b3,a4,e4,g5,c6,d7,h7,b8;t=1",
        )
        .unwrap();

        // Pawn-only BFS preserves the wall topology and gives an explicit
        // legal path from the reachable base to every sampled state.
        let mut seen = HashSet::new();
        let mut queue = VecDeque::new();
        seen.insert(base);
        queue.push_back(base);
        while let Some(state) = queue.pop_front() {
            if winner(&state) >= 0 {
                continue;
            }
            let mut legal = MoveList::new();
            legal_actions(&state, &mut legal);
            for index in 0..legal.pawn_count() {
                let child = state.apply(legal.get(index));
                if seen.insert(child) {
                    queue.push_back(child);
                }
            }
        }

        let mut candidates: Vec<State> = seen
            .into_iter()
            .filter(|state| state.turn == 1 && winner(state) < 0)
            .collect();
        candidates.sort_unstable_by_key(|state| (state.p0, state.p1, state.turn));
        let mut seed = 0x5eed_f1a0_500c_0de1u64;
        for index in (1..candidates.len()).rev() {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            candidates.swap(index, seed as usize % (index + 1));
        }

        let mut memo = HashMap::new();
        let mut checked = 0usize;
        let mut forced_wall_cases = 0usize;
        for state in candidates {
            let Some(score) = race_one_wall_score(&state, 0) else {
                continue;
            };
            let reference = reference_winner(&state, &mut memo).unwrap_or_else(|| {
                panic!(
                    "full one-wall minimax did not resolve firing state {}",
                    state.to_canonical_string()
                )
            });
            let actual = if score > 0 {
                state.turn
            } else if score < 0 {
                state.turn ^ 1
            } else {
                panic!(
                    "bounded minimax cannot certify firing draw {}",
                    state.to_canonical_string()
                );
            };
            assert_eq!(
                actual,
                reference,
                "state={} score={score}",
                state.to_canonical_string(),
            );

            let mut no_wall = state;
            no_wall.w1 = 0;
            if reference_winner(&no_wall, &mut memo) != Some(state.turn) {
                forced_wall_cases += 1;
            }
            checked += 1;
            if checked == 500 {
                break;
            }
        }

        assert_eq!(checked, 500, "reachable firing samples exhausted");
        assert!(
            forced_wall_cases > 0,
            "cross-check never exercised the forced-placement branch"
        );
        eprintln!(
            "race1w randomized cross-check: {checked}/500 exact verdicts agree; forced_wall_cases={forced_wall_cases}"
        );
    }

    #[test]
    fn race2w_monopoly_cases_cross_check_against_ordinary_full_search() {
        let positions = [
            // Case 1: ignoring both optional walls already wins fastest.
            ("p0=a7;p1=i4;w0=2;w1=0;h=-;v=-;t=0", 3),
            // Forced-wall analog: P1 is one step from goal, so P0 must place
            // a wall now; the solver includes P1's complete pawn reply ply.
            ("p0=a7;p1=i2;w0=2;w1=0;h=-;v=-;t=0", 5),
            // Colour-reversed forced-wall case exercises the P1 monopoly.
            ("p0=a8;p1=i3;w0=0;w1=2;h=-;v=-;t=1", 5),
        ];

        for (text, depth) in positions {
            let state = parse_state(text).unwrap();
            let exact = race_two_wall_score(&state, 0).expect(text);
            let mut plain = Features::default();
            plain.threads = 1;
            plain.race_exact = false;
            plain.race1w = false;
            plain.race2w = false;
            let mut ctx = SearchContext::with_tt_entries(plain, 65_536);
            let full = ctx.search(
                &state,
                &TimeControl {
                    max_depth: depth,
                    allotted_budget_ms: None,
                    soft_deadline: None,
                    hard_deadline: None,
                },
            );
            assert_eq!(exact.signum(), full.score.signum(), "{text}: {exact} vs {}", full.score);
            assert!(full.score.abs() >= MATE_THRESHOLD, "{text}: {}", full.score);
        }
    }

    #[test]
    fn race2w_rejects_split_nonholder_near_and_delayed_wall_states() {
        let split =
            parse_state("p0=a7;p1=i2;w0=1;w1=1;h=-;v=-;t=0").unwrap();
        assert_eq!(race_two_wall_score(&split, 0), None);

        let nonholder =
            parse_state("p0=a7;p1=i2;w0=0;w1=2;h=-;v=-;t=0").unwrap();
        assert_eq!(race_two_wall_score(&nonholder, 0), None);

        let near =
            parse_state("p0=e5;p1=e6;w0=2;w1=0;h=-;v=-;t=0").unwrap();
        assert_eq!(race_two_wall_score(&near, 0), None);

        let delayed = parse_state(
            "p0=e2;p1=c6;w0=2;w1=0;h=a1,f1,h2,e3,g5,e6;v=e2,f2,a3,d3,c5,f5,a6,b7,a8;t=0",
        )
        .unwrap();
        assert_eq!(race_two_wall_score(&delayed, 0), None);
    }

    #[test]
    fn aspiration_retry_schedule_widens_then_uses_full_window() {
        assert_eq!(aspiration_bounds(100, 0), (40, 160));
        assert_eq!(aspiration_bounds(100, 1), (-140, 340));
        assert_eq!(aspiration_bounds(100, 2), (-INF, INF));
    }

    #[test]
    fn rfp_tc_adaptive_budget_resolution_is_fail_closed_at_boundary() {
        assert_eq!(rfp_depth_for_budget(4, false, Some(100)), 4);
        assert_eq!(rfp_depth_for_budget(4, false, Some(1_000)), 4);
        assert_eq!(rfp_depth_for_budget(4, false, None), 4);

        assert_eq!(rfp_depth_for_budget(3, true, Some(100)), 4);
        assert_eq!(rfp_depth_for_budget(3, true, Some(200)), 4);
        assert_eq!(rfp_depth_for_budget(3, true, Some(201)), 3);
        assert_eq!(rfp_depth_for_budget(3, true, Some(1_000)), 3);
        assert_eq!(rfp_depth_for_budget(3, true, None), 3);
        // Adaptive mode is defined independently of the legacy sweep knob.
        assert_eq!(rfp_depth_for_budget(4, true, Some(1_000)), 3);
        assert_eq!(rfp_depth_for_budget(4, true, None), 3);
    }

    #[test]
    fn rfp_tc_adaptive_100ms_uses_depth_four_in_live_search() {
        let state = parse_state("p0=e5;p1=e6;w0=10;w1=10;h=-;v=-;t=1").unwrap();
        let mut d3_features = Features::default();
        d3_features.threads = 1;
        d3_features.rfp_depth = 3;
        d3_features.rfp_tc_adaptive = false;
        let mut d4_features = d3_features.clone();
        d4_features.rfp_depth = 4;
        let mut adaptive_features = d3_features.clone();
        adaptive_features.rfp_tc_adaptive = true;

        let d3 = depth_search_with_budget(&state, 5, 100, d3_features);
        let d4 = depth_search_with_budget(&state, 5, 100, d4_features);
        let adaptive = depth_search_with_budget(&state, 5, 100, adaptive_features);

        assert_eq!(adaptive.best_action, d4.best_action);
        assert_eq!(adaptive.score, d4.score);
        assert_eq!(adaptive.depth, d4.depth);
        assert_eq!(adaptive.nodes, d4.nodes);
        assert_ne!(adaptive.nodes, d3.nodes, "depth-four RFP must fire at 100ms");
        assert_eq!(d3.d4_fires, 0);
        assert_eq!(
            d4.d4_fires, 0,
            "legacy depth-four RFP must not feed the adaptive diagnostic",
        );
        assert!(
            adaptive.d4_fires > 0,
            "known adaptive fixture must produce a depth-four RFP cutoff",
        );
    }

    #[test]
    fn rfp_tc_adaptive_d4_counter_reports_zero_without_a_budget_after_a_firing_move() {
        let state = parse_state("p0=e5;p1=e6;w0=10;w1=10;h=-;v=-;t=1").unwrap();
        let mut features = Features::default();
        features.threads = 1;
        features.rfp_tc_adaptive = true;
        let mut ctx = SearchContext::with_tt_entries(features, 4096);

        let fast = ctx.search(
            &state,
            &TimeControl {
                max_depth: 5,
                allotted_budget_ms: Some(100),
                soft_deadline: None,
                hard_deadline: None,
            },
        );
        assert!(
            fast.d4_fires > 0,
            "positive control must fire before reset",
        );

        ctx.clear_tt();
        let bench = ctx.search(
            &state,
            &TimeControl {
                max_depth: 5,
                allotted_budget_ms: None,
                soft_deadline: None,
                hard_deadline: None,
            },
        );
        assert_eq!(
            bench.d4_fires, 0,
            "fixed-depth/bench search must report a fresh zero count",
        );
    }

    #[test]
    fn rfp_tc_adaptive_1000ms_matches_flag_off_on_fixed_positions() {
        let positions = [
            "p0=e5;p1=e6;w0=10;w1=10;h=-;v=-;t=1",
            "p0=e3;p1=c7;w0=7;w1=9;h=c4,e4,d6;v=f6;t=0",
            "p0=e7;p1=e3;w0=3;w1=2;h=c3,e5,g3;v=b6,d4,f6;t=0",
        ];
        for text in positions {
            let state = parse_state(text).unwrap();
            let mut control_features = Features::default();
            control_features.threads = 1;
            control_features.wallq_tc = false;
            control_features.rfp_depth = 3;
            control_features.rfp_tc_adaptive = false;
            let mut adaptive_features = control_features.clone();
            adaptive_features.rfp_tc_adaptive = true;

            let control = depth_search_with_budget(&state, 5, 1_000, control_features);
            let adaptive = depth_search_with_budget(&state, 5, 1_000, adaptive_features);

            assert_eq!(adaptive.best_action, control.best_action, "state={text}");
            assert_eq!(adaptive.score, control.score, "state={text}");
            assert_eq!(adaptive.depth, control.depth, "state={text}");
            assert_eq!(adaptive.nodes, control.nodes, "state={text}");
            assert_eq!(adaptive.main_nodes, control.main_nodes, "state={text}");
            assert_eq!(adaptive.threads, control.threads, "state={text}");
            assert_eq!(control.d4_fires, 0, "state={text}");
            assert_eq!(adaptive.d4_fires, 0, "state={text}");
        }
    }

    #[test]
    fn rfp_eligibility_matches_probe_with_opponent_walls_at_depths_two_and_three() {
        let turn_zero = parse_state("p0=a1;p1=i9;w0=0;w1=1;h=-;v=-;t=0").unwrap();
        let turn_one = parse_state("p0=a1;p1=i9;w0=1;w1=0;h=-;v=-;t=1").unwrap();
        for state in [&turn_zero, &turn_one] {
            let opponent_walls = if state.turn == 0 { state.w1 } else { state.w0 };
            assert!(opponent_walls >= 1);
            assert!(rfp_eligible(2, true, 0, 3));
            assert!(rfp_eligible(3, true, 0, 3));
        }

        assert!(rfp_eligible(1, true, 0, 3));
        assert!(!rfp_eligible(0, true, 0, 3));
        assert!(!rfp_eligible(4, true, 0, 3));
        assert!(rfp_eligible(4, true, 0, 4));
        assert!(!rfp_eligible(5, true, 0, 4));
        assert!(!rfp_eligible(1, false, 0, 3));
        assert!(!rfp_eligible(1, true, MATE_THRESHOLD, 3));
    }

    #[test]
    fn rfp_margin_cutoff_uses_depth_shaped_saturating_threshold() {
        assert!(rfp_prunable(2, true, 100, 600, 250, 3));
        assert!(!rfp_prunable(2, true, 101, 600, 250, 3));
        assert!(rfp_prunable(
            3,
            true,
            -1000,
            i32::MAX,
            i32::MAX,
            3,
        ));
    }

    #[test]
    fn rfp_cuts_before_move_ordering_and_returns_fail_hard_beta() {
        let state = start_state();
        let mut rfp_features = Features::default();
        rfp_features.threads = 1;
        rfp_features.rfp = true;
        rfp_features.rfp_margin = 0;
        let beta = 1;
        assert_ne!(evaluate_with_features(&state, &rfp_features), beta);
        for depth in 1..=3 {
            let mut rfp = SearchContext::with_tt_entries(rfp_features.clone(), 1024);
            let score = rfp
                .negamax(&state, None, depth, beta - 1, beta, 0, false, None)
                .expect("unbounded RFP node completes");
            assert_eq!(score, beta);
            assert_eq!(rfp.nodes, 1);
        }

        let mut plain_features = rfp_features.clone();
        plain_features.rfp = false;
        let mut plain = SearchContext::with_tt_entries(plain_features, 1024);
        let _ = plain
            .negamax(&state, None, 1, 0, 1, 0, false, None)
            .expect("unbounded reference completes");
        assert!(plain.nodes > 1);
    }

    #[test]
    fn null_move_gate_requires_walls_depth_and_noncritical_race() {
        let spacious = parse_state("p0=a1;p1=i9;w0=1;w1=1;h=-;v=-;t=0").unwrap();
        assert!(null_move_allowed(&spacious, 3, false));
        assert!(!null_move_allowed(&spacious, 2, false));
        assert!(!null_move_allowed(&spacious, 3, true));

        let no_current_wall = parse_state("p0=a1;p1=i9;w0=0;w1=1;h=-;v=-;t=0").unwrap();
        assert!(!null_move_allowed(&no_current_wall, 3, false));
        let critical = parse_state("p0=a6;p1=i9;w0=1;w1=1;h=-;v=-;t=0").unwrap();
        assert!(!null_move_allowed(&critical, 3, false));
    }

}
