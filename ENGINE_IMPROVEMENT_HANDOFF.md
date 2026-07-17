# Titanium engine improvement handoff — priority queue

**Snapshot commit:** broke-side race bounds landed (see engine submodule + this doc).  
**Purpose:** ordered work to gain strength without another pile of wall-count special cases.

---

## 0) Immediate validation (do before more features)

**Goal:** confirm broke-side did not regress and may gain Elo.

1. Build native release both sides from the snapshot commit:
   ```powershell
   $env:RUSTFLAGS = '-C target-cpu=native'
   Remove-Item Env:TITANIUM_ALLOW_SUBOPTIMAL -ErrorAction SilentlyContinue
   cargo build --release -p titanium --manifest-path engine\Cargo.toml
   ```
2. A/B match **parent of broke-side commit** vs **this commit** (same weights, same clock, same threads).
3. Watch `broke_*` stats in think JSON if available; expect cuts mainly when `min(wl)==0`.
4. Gate: no clear Elo loss; ideally ≥0 with hint of +.

Corpus hint from profiling: ~14% of thinks had one side broke; decisive ~2% of in-tree calls; real fail-high/low cuts observed.

---

## Priority ranking (oracle-first)

### P0 — Race / oracle infrastructure (highest leverage)

The exact race system is the most important engine asset. Everything else should feed on it or stay out of its way.

| ID | Task | Why |
|----|------|-----|
| O1 | **Warm / share `race_tbl` graph** across probes (broke, 0-wall, 1w/2w) | Cuts cost of exact proofs; more nodes can afford table hits |
| O2 | **Exact ETA feature when one side broke** (eval + RFP margins, not hard cuts) | Sound arrival floor; better than constant ±1800 soft eval |
| O3 | **Preserve / surface DTM in UI & TT hygiene** | Already done for broke bounds; audit other `RACE_WIN_FLOOR`-only paths |
| O4 | **Gate1 + Service B coherence** | Cheap bound then exact; avoid duplicate work; measure hit rates |
| O5 | **Oracle correctness regression pack** | Empty + walled topologies, jump-heavy; never ship race changes without it |

### P1 — Generalize certificates (replace future N-wall layers)

| ID | Task | Why |
|----|------|-----|
| C1 | **`certify()` → typed `RaceBound` + budget** (`Lower`/`Upper`/`Unknown`) | One mechanism for both-armed positions; kills monopoly-3 temptation |
| C2 | Measure / decide **`wall_ignore` loss cert** | Already coded, default off — enable only if stats + Elo say so |
| C3 | Cert memo (`cw_cache`) hit-rate + budget tuning | Eval already uses ~2500 on hit; make it fire more when cheap |

**Do not:** add `three_wall_monopoly_bound` or more hand-written sum==N solvers.

### P2 — Search uses race info better

| ID | Task | Why |
|----|------|-----|
| S1 | **Move ordering** from exact/approx race ETA & tempo margin | More Elo than rare hard cuts |
| S2 | **LMR / extensions** conditioned on race criticality (corridor / jump squares) | Spend nodes where race flips |
| S3 | Aspiration / window from known Lower/Upper | Broke-side DTM bounds already help when they cut |

### P3 — NNUE / features from oracle

| ID | Task | Why |
|----|------|-----|
| N1 | Train/input **exact optimistic ETA** (esp. broke-side) | Replace weak BFF proxies where safe |
| N2 | Race margin / inevitability / corridor width planes | Oracle-derived features beat hand heuristics |
| N3 | Keep CAT heat optimization separate but profiled | Flamegraphs: CAT was top exclusive hotspot (~11–17%); race ≈ 0% |

### P4 — Performance (only with profiles)

| ID | Task | Why |
|----|------|-----|
| F1 | Finish **whole-game** symbolicated flamegraphs (Workers=1) | Earlier coverage incomplete |
| F2 | Optimize **`add_catv5_propagated_heat`** / evaluate / BFF | Real NPS bottleneck (~200k 1t today vs remembered 1M) |
| F3 | Never ship suboptimal CPU builds for Elo | `RUSTFLAGS=-C target-cpu=native` mandatory |

---

## Explicit non-goals (for now)

- Dedicated 3+/N-wall monopoly solvers  
- Using BFF alone to prove wins/losses  
- Converting “wallless side wins pure race while opp has walls” into Lower  
- Elo claims without A/B on the snapshot commit  

---

## Suggested order of attack

```text
1. A/B match snapshot (this commit) vs parent          ← gate
2. O1 shared/warm race_tbl                             ← oracle cost
3. O2 exact ETA into eval/RFP when one side broke      ← soft floor/roof
4. C1 certify → RaceBound API                           ← general proofs
5. S1 race-aware move ordering                         ← Elo compounding
6. N1/N2 NNUE features from oracle                     ← long cycle
7. F1/F2 CAT/eval NPS only with flamegraph evidence
```

---

## Current certified race coverage (after snapshot)

```text
both hands 0     → Gate1 + exact race_tbl DTM
one side broke   → refuse-to-place Lower/Upper with exact DTM (new)
1 wall           → sound subset
2 walls          → monopoly subset (often PV-only on v17)
both armed >2    → search / optional certify / wall_ignore (off)
```

---

## Files / tools

| Path | Role |
|------|------|
| `engine/src/titanium/search.rs` | `one_side_broke_race_bound`, ab hooks, tests |
| `engine/src/titanium/race.rs` | `RaceBound`, stats incl. `broke_*` |
| `HANDOFF_BROKE_SIDE_RACE.md` | broke-side design + GPT iteration |
| `HANDOFF_REMAINING_WALL_RACE_LAYERS.md` | 0/1/2 wall layer map |
| `tools/profile_tt_per_think/` | thinks collect, flamegraph, broke stats |

Measure broke triggers:

```powershell
python tools\profile_tt_per_think\measure_broke_side_stats.py `
  training\data\profiles\tt_per_think_w4\thinks.jsonl `
  engine\target\release\search_bench.exe `
  training\data\profiles\tt_per_think_w4\broke_side_stats.json `
  500
```
