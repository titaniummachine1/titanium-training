# Architecture handoff — Titanium Quoridor NN + minimax

**Read this first.** Assumes no prior conversation context.

**Doc map**

| Doc                                                  | When to read                                    |
| ---------------------------------------------------- | ----------------------------------------------- |
| This file                                            | Architecture philosophy, phases, do-not-do list |
| [`README.md`](README.md)                             | Operational pipeline, scripts, guards, commands |
| [`zero_teacher/HANDOFF.md`](zero_teacher/HANDOFF.md) | quoridor-zero.ink MCTS label mining only        |

---

## Project goal

Build the best **practical** neural network architecture for a Quoridor minimax/alpha-beta engine.

**Core philosophy**

- **NN** = fast geometric prior + search-risk estimator
- **Minimax** = tactical proof
- Search must **never** blindly trust the net for legality or hard pruning

The network should make the engine spend **less time understanding the board** and **more time searching strong lines**.

---

## High-level decision

**Do NOT** build a giant Ka / Quoridor-Zero-style ResNet for live minimax.

**Best target**

- Shared fast HalfPW / NNUE-style trunk
- Value head
- Conservative search-pressure / uncertainty head
- Optional move-ordering policy head **later**

**Why:** In minimax, NPS matters. A network 5% more accurate but halving nodes/sec may lose. The net must be small enough to run at every leaf or many internal nodes.

---

## Engine architecture

### Layer A — hard engine

- Legal movegen
- Wall legality (binary / bitboard flood fill — see **Terminology** below)
- BFS path checks
- Transposition table
- Alpha-beta / minimax search

### Terminology — `pbff_*` / wall legality / flood fill

In this repo, **`pbff_*`** names a family of **binary flood fill** (also **bitboard flood fill**, **BFF**) helpers in `engine/src/path/parallel.rs`. They answer the standard Quoridor question after a tentative wall placement: _do both players still have a path to their goal?_

- **Not** a separate neural or search architecture, and **not** a proprietary algorithm — it is ordinary flood fill over compact `u128` reachability masks.
- **`pbff_to_goal`** — BFF from a start square to a goal row; returns visited bits for reuse.
- **`pbff_wall_legal`** — two-player wall trial: flood player 1, then player 2 with **visited-bit reuse** / **cached reachable-mask splice** (informal: “bit theft”).
- **`expand_wave`** — one dilation step of that flood (four directional shifts on a bitboard).
- **SIMD / shift tricks** (`expand_wave`, optional Kogge–Stone variants in benches) are **implementation accelerators**, not a different legality rule.

ACE distance fields (`acev13/dist.rs`) use the same **bitboard flood** idea via `expand_frontier` + `DirMasks`; Titanium wall movegen uses `pbff_wall_legal` on `WallGrids`. Same graph question, two coordinate layouts — function names stay `pbff_*` for historical reasons.

### Engine IDs (site + CLI)

| ID                                     | Meaning                                                                               |
| -------------------------------------- | ------------------------------------------------------------------------------------- |
| `titanium-v15`                         | Current strongest **live** minimax engine (latest trained NNUE)                       |
| `titanium-v15-frozen`                  | Same search — **pinned baseline** NNUE blob for A/B                                   |
| `ace-v13-js`                           | JS ACE baseline / comparison tier                                                     |
| `ace-v13-ti`                           | ACE v13 with Titanium O1 movegen (MoveGen+)                                           |
| `session_v15` / engine infinite search | **Disabled** — not default; `run_infinite_benchmark.py` = repeated match batches only |

### Layer B — geometric feature extractor

Precompute strong Quoridor-specific geometry:

- Shortest-path fields
- Pawn-forward distance fields
- `corridor_delta`
- `path_cross`
- Choke / forcedness
- Contested shortest-route cells
- Placed wall sparse slots
- Pawn positions
- Wall counts
- `legal_wall_count`

### Layer C — neural evaluator

Small shared trunk: `hidden_features[H]`

**Heads**

1. **Value head** — leaf eval (live today)
2. **Search-pressure head** — conservative compute scheduler (train only; not in engine yet)
3. **Optional policy head** — move ordering only, later

---

## Current best input set

### Keep

**Pawn embeddings**

- `po[81 × H]`
- `px[81 × H]`

**Sparse wall embeddings**

- `w1c[9 buckets × 128 wall slots × H]`

**11 BFS / geometry planes**

- `goal_inv_p0`, `goal_inv_p1`
- `pawn_fwd_p0`, `pawn_fwd_p1`
- `corridor_delta_p0`, `corridor_delta_p1`
- `path_cross_p0`, `path_cross_p1`
- `choke_p0`, `choke_p1`
- `contested`

**Scalar skip features** `ws[0..15]`

### Important scalar: `ws[14]`

```text
ws[14] = legal_wall_count / 128.0
```

`legal_wall_count` = number of **path-valid** wall slots: each slot is a tentative placement where **both players still have a path to goal** (checked via binary/bitboard flood fill / `pbff_wall_legal`).

- BFS planes = current **realized** geometry
- `legal_wall_count` = **unrealized** future wall capacity

---

## Do NOT steal from Ka CNN

**Reject**

- 18-layer ResNet
- Self-attention body
- 8×8 wall grids
- 8×8 placable planes _(rejected Ka artifact — not used in this engine)_
- `cross_arr` / DirMasks as raw planes
- Global distance broadcast planes
- One-hot pawn planes
- Side-to-move full-board plane
- Ka weights

**Reason:** Ka is raw-board CNN / MCTS-style. This engine is minimax + handcrafted geometry + NNUE-like eval. Copying Ka channels bloats the net and duplicates BFS work.

**quoridor-zero.ink:** Same rule for **runtime** — never ship their ResNet in browser/engine. OK to mine **MCTS attention labels** offline for the pressure head (see `zero_teacher/HANDOFF.md`).

---

## Value head

**Current style**

```text
eval = scalar_ws_terms + dot(w2, hidden_features)
```

**Use for**

- Leaf evaluation
- Shallow node evaluation
- Training target from WDL / game result / deeper search

**Do NOT** make the value head too large.

| `H`   | Status                           |
| ----- | -------------------------------- |
| 32    | **Current**                      |
| 48–64 | Only if NPS survives measurement |

**Rule:** Do not increase `H` until eval accuracy gain is proven, NPS loss is measured, and Elo or fixed-suite improvement survives.

---

## Search-pressure / uncertainty head

**Goal:** Predict how dangerous it is to reduce / search this position shallowly.

This head does **not** need to be perfect. It must be **conservative**.

| Pressure | Behavior                                            |
| -------- | --------------------------------------------------- |
| High     | Search more / reduce less / maybe extend            |
| Medium   | Normal search                                       |
| Low      | Allow **mild** reduction only under safe conditions |

**Bad behavior:** low pressure → hard prune. **NEVER.**

### Architecture (implemented in `train_search_importance.py`)

```text
shared hidden_features[H]
    -> Linear(H, 1)
    -> tanh  ->  pressure in [-1, +1]
```

**Cost:** ~`H` multiply-adds + activation. Negligible if trunk `hid` is reused.

**Training:** Freeze trunk; train only pressure head.

**Labels**

- Deeper search changed best move
- Deeper search changed eval a lot
- PV instability
- Shallow vs deep disagreement (`collect_search_importance.py`)
- Quoridor-zero / stronger teacher importance (`zero_teacher/collect_budget.py`)

**Safe integration**

- Pressure can reduce LMR reductions
- Pressure can add tiny extension
- Pressure can bias ordering
- Pressure **cannot** skip moves alone

### Conservative search integration (not wired yet)

```text
pressure = pressure_head(hidden)

reduction = base_lmr(depth, move_index)

if pressure > 0.75:
    reduction = max(0, reduction - 1)

if pressure < 0.15
   and depth >= 5
   and move_is_late
   and node_not_tactical:
    reduction = min(reduction + 1, base_reduction + 1)

never prune solely because pressure is low
```

**Initial clamps**

- Max extra reduction from pressure: **+1 ply**
- Max extension from pressure: **−1 reduction / +1 ply**
- No effect at shallow depth unless tested

---

## Optional policy / move-ordering head

**Not first priority.** Only after value + pressure are stable.

**Purpose:** Order moves, not choose moves.

**Possible targets**

- Deeper search best move
- MCTS / Quoridor-zero policy (`visitFraction`)
- PV move from stronger engine
- Wall moves that frequently survive deep search

**Architecture options**

- **A.** Move-class head: pawn / wall / tactical-wall scores
- **B.** Sparse move head: 128 wall-slot logits + small pawn logits

**Use only for** move ordering, history prior, root sorting.

**Do NOT** use policy head to delete legal moves.

---

## Training phases

### Phase 1 — value only

Train HalfPW trunk + value head (`train.py`).

**Targets:** WDL, game outcome, deeper-search eval (mixed if stable).

### Phase 2 — frozen trunk pressure head

Load trained value net. Freeze all trunk/value weights. Train `Linear(H,1)` only (`train_search_importance.py`).

**Targets:** shallow-vs-deep instability, best-move change, eval delta, teacher/MCTS importance.

### Phase 3 — export unified runtime

Export:

- Existing `net_weights.bin`
- Pressure head weights (today: separate `search_pressure_head.pt` — **fuse into schema next**)
- Schema version bump

Rust `evaluate()` should compute once:

```text
hidden = trunk(position)
value  = value_head(hidden, ws)
pressure = pressure_head(hidden)
```

**Status:** Phase 3 **not done** — engine returns value only.

### Phase 4 — search integration

1. Pressure affects ordering / LMR clamps only
2. Later, if proven: selective extensions
3. **Never:** pressure hard-prunes legal moves

### Phase 5 — optional policy head

Only if pressure/value stable and search still wastes nodes.

---

## Training data pipeline

```text
SQLite DB (move sequences only)
    ↓
eval-batch with current titanium.exe
    ↓
expand_games() generates features live
    ↓
train.py trains net
```

**Important**

- DB does **not** store features
- Old games are safe — replayed through current engine
- Purity depends on eval-batch binary correctness

**Hard failures required**

- Missing `legal_wall_count` → abort
- Parity mismatch → abort
- Wrong schema → abort
- Stale checkpoint across schema change → avoid / reset

---

## Critical current contract: `ws[14]`

```text
ws[14] = legal_wall_count / 128.0
```

**Everywhere:** engine `evaluate()`, eval JSON, `halfpw.py`, `train.py`, `datagen.py`.

**Must NOT mean**

- `corridor_width_me`
- Fallback `128`
- Raw wall count
- p0−p1 legal advantage

**`ws[15]`** remains opponent corridor width.

Schema: `halfpw-field11-ws14-legal-wall-v1`

---

## Build / deploy safety

Before training:

1. Native rebuild: `RUSTFLAGS="-C target-cpu=native"` + `cargo build --release`
2. `parity_check.py` must pass 6/6
3. `eval-batch` must emit `legal_wall_count`
4. Do not resume old checkpoint if `ws[14]` schema changed (mismatch → fresh init from `net_weights.bin`; see `train.py`)
5. Restart self-play pool after engine rebuild

**Micro-train resume:** `run_nnue_cycle.py` passes `--resume` to `train.py`. Safe only for checkpoints stamped `halfpw-field11-ws14-legal-wall-v1`. Pre-ws[14] optimizer state (`ws[14]=corridor_width_me` era) must not be resumed — schema mismatch re-inits model weights from `net_weights.bin` and drops optimizer state.

**Main footgun:** stale `titanium.exe` at eval-batch time.

**Optional guard:** build stamp (git hash, timestamp, `feature_schema_version`); `datagen.py` rejects wrong binary. (`engine_identity.py` partially covers this today.)

---

## Why engine got weaker after infinite-search attempt

Likely **not** an NN problem.

**Most likely causes**

- Search policy changed
- TT reuse became unsafe
- Root stability worse
- Move ordering collapsed diversity
- Always-restart / infinite daemon changed position distribution
- Ponder / `session_v15` partially bypassed normal search assumptions

**Correct action**

- Keep infinite search **disabled by default**
- Train only on stable standard session unless explicitly testing
- A/B: standard warm session vs `session_v15` / infinite — same eval, depth, time, positions

Do **not** train on experimental search mode until parity passes, fixed-suite improves, no Elo regression.

---

## Regression tests before trusting engine

Must pass:

- `cargo build --release` native
- `parity_check.py` 6/6
- eval-batch schema probe
- Fixed tactical / wall-heavy suite
- `plateau_probe`
- NPS check
- Standard session vs experimental session A/B

**If engine got weaker:** suspect search / session / TT first, not value net.

---

## Best final target architecture

**Titanium Quoridor minimax network**

- Shared HalfPW trunk, **H = 32** initially
- **Inputs:** pawns, sparse walls, 11 BFS planes, scalar features, `legal_wall_count`
- **Head 1 — Value:** leaf eval
- **Head 2 — Search-pressure:** frozen-trunk, conservative scheduler
- **Head 3 — Policy (optional):** ordering only, no hard pruning

**Search:** alpha-beta remains authority. NN guides eval, ordering, reductions/extensions. NN never replaces legality, proof, or full move consideration.

---

## Success metrics

**Primary**

- Fixed-suite eval/search agreement
- Wall-heavy position accuracy
- Shallow-vs-deep correlation
- PV stability
- NPS retained

**Secondary**

- Elo

**Bad signs**

- Big Elo jump without explanation
- Huge eval drift after tiny feature change
- Pressure head causing tactical misses
- Stronger NPS but weaker fixed suite

**Expected:** small stable improvement, not magic.

---

## Absolute do-not-do list

- Train on stale binary
- Resume old `ws[14]` checkpoint blindly
- Enable infinite search by default
- Hard prune with pressure head
- Add Ka CNN planes
- Add giant ResNet for live search
- Expand net size before NPS/Elo proof
- Trust Elo before fixed-suite sanity
- Distill zero-ink / Ka **eval** into main WDL (`train.py`)

---

## One-line handoff

Build a tiny shared HalfPW trunk that sees already-solved Quoridor geometry, use it for value, add a conservative frozen-trunk search-pressure head to spend compute where deeper search matters, keep minimax as the tactical authority, and never train from an unverified or experimental engine binary.
