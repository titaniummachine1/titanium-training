# DIVERSITY SPEC V1 — fused self-play distribution design (pre-registered)

Provenance: blind two-party derivation 2026-07-11. Binding pre-registration for
frontier corpus generation. Supersedes NET_PLAN_V7 §8 opening mechanism upon
frontier open.

## Principle

Temperature's role in AlphaZero-style self-play (move-selection noise) is
replaced by **start/population diversity**. Every recorded move is played at
full strength; labels stay clean.

## The five mechanisms (ranked, binding)

1. **QUOTA-CONSTRAINED ERR-MAP ALLOCATION** — 30% closed-loop population, 15%
   behavioral cross-play, 10% forks, 10% solver seam, 5% exact anchors; 30%
   adaptive residual only after ERR-MAP validates on held-out regret.
2. **PAIRED FORKS 5%+5%** — regret-mined + plausible-deviation (2nd/3rd PV).
3. **POPULATION + BEHAVIORAL CROSS-PLAY** — current/frontier/two-history +
   three style treatments.
4. **SOLVER SEAM + EXACT ANCHORS** — 10% seam + 5% exact anchors.
5. **TRAINING-ONLY REGIME OPENINGS + HARD COLLAPSE CERTIFICATE** — theory-24
   and frozen gate battery are **evaluation-only**. Training openings = disjoint
   centroids with top-5 candidate support per **first two plies**.

### Collapse certificate (BLOCK on failure)

- max two-ply prefix mass ≤ 10% (game-weighted)
- N_eff(2) ≥ 16; N_eff(4) ≥ 64
- zero duplicate canonical states; row caps per source game / fork lineage
- ≥ 25% states absent from both preceding training corpora
- strata shares within quota; STM 45–55%

## Kills

- Random-move openings
- **Temperature-only diversity**
- Diversity via weakened engines (label corruption)
- Training on any evaluation battery
- **Forcing a single four-ply trunk (e2 e8 e3 e7) as a training corpus filter**

## Code mapping (2026-07-14)

| Concern                             | Module                                                                 | Notes                                                              |
| ----------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Training opening gate (min e2 e8)   | `training/game_opening_gate.py`                                        | Rejects wall-first garbage only                                    |
| Deploy collapse check (e2 e8 e3 e7) | `training/titanium_training/validation/opening_sanity.py`              | Eval/deploy BLOCK, not training filter                             |
| Collapse certificate                | `training/diversity_spec.py`                                           | Logged every epoch; set `DIVERSITY_CERTIFICATE_ENFORCE=1` to BLOCK |
| Generation (no move temperature)    | `training/generation_matchup.py`, `start_local_game_pool_detached.ps1` | Full-strength self-play                                            |

## Preparation phase (2026-07-14)

- `TRAINING_PREP_ONLY=1` is the default. All real-work entry points exit 2.
- Dry-run: `python training/prepare_diversity_plan.py --rows 100000 --dry-run`
- Future real run requires `TRAINING_PREP_ONLY=0`, frozen semantics, manifest, and `training/APPROVE_GENERATION.json` (not created).
- `DIVERSITY_CERTIFICATE_ENFORCE` has no effect while prep-only is on.

Forks, solver seam, ERR-MAP adaptive residual, style cross-play, and seeded
opening centroids (required for N_eff(2)≥16 from a standard board — only nine
central two-ply keys exist at ply 0). Enable hard BLOCK with
`DIVERSITY_CERTIFICATE_ENFORCE=1` once those lanes ship.
