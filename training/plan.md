Good — now we compress everything into a **single execution-grade spec**. No discussion, no interpretation layer. This is what you give to a stronger coding agent.

---

> **HISTORICAL PLAN — SUPERSEDED.**  
> **Current master spec:** [`ARCHITECTURE_HANDOFF.md`](ARCHITECTURE_HANDOFF.md)  
> **Current `ws[14]` contract:** `legal_wall_count / 128.0`  
> Old `ws[14]=corridor_width_me` notes below are **historical only**.

---

# 🧠 ACE v10 — HalfPW + ws[14] LEGAL WALL COUNT SYSTEM (FINAL SPEC)

## 0. System definition (unchanged architecture)

This is a **value-only NNUE-style evaluator + BFS geometry stack + full search engine**.

### Evaluation model

```
score =
  BFS geometry (11 planes)
+ sparse wall embedding (w1c)
+ pawn embeddings (po, px)
+ scalar interactions (ws[0–15])
```

---

## 1. Frozen architecture (DO NOT CHANGE)

### Input layers

- BFS field planes (11 total)
  - goal_inv_p0 / p1
  - pawn_fwd_p0 / p1
  - corridor_delta_p0 / p1
  - path_cross_p0 / p1
  - choke_p0 / p1
  - contested

- Sparse wall state
  - `w1c` = placed walls (9 × 128 × H)

- Pawns
  - `po`, `px`

---

### Scalar features (ws)

| Index    | Meaning                      | Status             |
| -------- | ---------------------------- | ------------------ |
| ws[0–12] | learned interaction features | **FROZEN**         |
| ws[13]   | `pd × w_opp / 10`            | **FROZEN FORMULA** |
| ws[14]   | ⚠️ NEW: legal wall capacity  | **ACTIVE CHANGE**  |
| ws[15]   | opponent corridor width      | **KEEP**           |

---

## 2. The ONLY architectural change

### ws[14] definition (FINAL)

Replace old feature (corridor_width_me) entirely.

```
ws[14] input =
    legal_wall_count / 128.0
```

### legal_wall_count definition

Computed in **engine only**:

- Count of **path-valid** wall slots (normalized as `legal_wall_count / 128.0` for `ws[14]`)
- **Path-valid** = tentative placement keeps **both players connected to goal**
- Counted via binary/bitboard flood fill / `pbff_wall_legal` (no separate heuristic)

---

## 3. Where this must be wired (CRITICAL CONSISTENCY RULE)

This value must come from the SAME engine logic everywhere:

### MUST MATCH ACROSS:

```
engine/src/acev13/search.rs   (evaluate)
training/halfpw.py            (forward)
training/train.py             (feature ingestion)
training/datagen.py           (eval-batch schema)
```

### Rule:

> If any of these disagree → system is invalid.

---

## 4. Data pipeline (ABSOLUTE TRUTH PATH)

```
SQLite DB
  (moves only)
      ↓
titanium.exe (eval-batch)
  → reconstruct positions
  → compute BFS + ws + legal_wall_count
      ↓
expand_games
  → training records
      ↓
train.py
```

### Important invariants:

- DB NEVER stores features
- Features are NEVER reused across engine versions
- eval-batch ALWAYS runs current binary

---

## 5. Critical failure rules (HARD FAILS)

Training must crash if:

### ❌ Missing feature

```
legal_wall_count not present → abort
```

### ❌ Wrong binary

```
eval-batch binary != expected build → abort (if stamp enabled)
```

### ❌ Schema mismatch

```
ws[14] != legal_wall_count/128 across components → abort
```

---

## 6. Checkpoint rules (VERY IMPORTANT)

### Allowed:

- resume ONLY if:
  - ws[14] already = legal_wall_count semantics
  - optimizer state matches same schema generation

### Forbidden:

- resuming checkpoint trained under old:

  ```
  ws[14] = corridor_width_me
  ```

### Safe rule:

> First ws[14] run must use a **fresh checkpoint OR reset ws[14] weight state**

---

## 7. Build / runtime requirement

### Always required:

```
RUSTFLAGS="-C target-cpu=native"
cargo build --release
```

### Pool rule:

- pool MUST use same binary as eval-batch
- after rebuild → restart pool

---

## 8. What was explicitly rejected (DO NOT IMPLEMENT)

- CNN / ResNet / attention
- 8×8 wall grids
- cross_arr / DirMasks planes
- global distance broadcasts
- per-player legal wall advantage (p0 − p1) — **not** `ws[14]`; rejected experimental scalar
- dataset versioning systems
- ML orchestration frameworks
- SQLite feature storage
- action-space heads

---

## 9. Expected effect (DO NOT OVERRUN EXPECTATIONS)

- small evaluation calibration improvement
- no guaranteed Elo jump
- improves:
  - midgame evaluation stability
  - latent wall-availability understanding

- NOT a tactical revolution

---

## 10. Minimal execution checklist

### Phase 0 — Build correctness

```
cargo build --release (native)
parity_check.py → 6/6
```

---

### Phase 1 — Deployment correctness

- restart pool with new binary
- ensure eval-batch uses same executable
- verify `legal_wall_count` exists in JSON output

---

### Phase 2 — Training run

```
run_nnue_cycle.py
→ expand_games (eval-batch)
→ train.py
```

- no old JSONL allowed unless schema-complete
- no old checkpoint unless ws[14 compatible

---

### Phase 3 — validation

- plateau_probe.py
- probe_legal_wall_signal.py
- optional Elo comparison

---

## 11. Optional safety upgrade (NOT REQUIRED)

If desired:

### Add binary stamp check

```
engine/build_info.json
{
  git_hash,
  build_time
}
```

Assert in:

- datagen
- expand_games

---

## 12. One-line system contract

> All training data is a deterministic replay of moves through a single verified engine build where ws[14] = legal_wall_count / 128.

---

# THAT IS THE ENTIRE SYSTEM

No missing pieces for implementation.

If you hand this to a strong coding agent, their job is only:

- wire ws[14]
- enforce consistency
- ensure single-binary eval-batch
- prevent stale checkpoint reuse

Everything else is already decided and frozen.
