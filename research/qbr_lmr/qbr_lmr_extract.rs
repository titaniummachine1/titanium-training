// QBR LMR — exact production code, extracted from rust/src/search.rs @ master 70dab79
// Merged E-011: SPRT(0,+5) H1, n=1028, 57.9%, Elo +55.2 [34.2, 76.7] @50ms
// Production config: lmr=true, lmr_probable_walls=false (the Q4 two-ply tier is
// a dormant experiment flag) => uniform 1-ply reductions in practice.

// ============ 1. ELIGIBILITY + REDUCTION (search.rs:9330-9372) ============
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

// ============ 2. THE TIGHT-WALL CLASSIFIER (search.rs:9151-9174) ============
// The domain-specific heart: a wall is "tight" if it touches either pawn's
// BFS shortest-path edge set. Tight walls are NEVER reduced.
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


// ============ 3. THE REDUCED SEARCH + VERIFIED RE-SEARCH (search.rs:6761-6841) ============

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

// ============ 4. ACTIVATION SITE (search.rs:7002-7003) ============
// LMR arms only when: feature on, node depth >= 3, and more than 6 ordered moves.
        let lmr_edges = (self.features.lmr && depth >= 3 && ordered.len > 6)
            .then(|| ordered.tight_edges.unwrap_or_else(|| LmrEdges::for_state(state)));