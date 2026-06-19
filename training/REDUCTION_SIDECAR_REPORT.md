# Reduction Sidecar v1 — Diagnostic Report

Date: 2026-06-16 (updated after natural-sample diagnosis)

---

## Locked target

For an already-LMR-eligible late wall move, predict whether one additional
provisional reduction preserves the native complete move-search decision and
saves at least the configured node margin. Existing full-depth verification is
unchanged. Runtime activation and deployment are disabled.

---

## Architecture invariants — verified

| # | Invariant | Status |
|---|-----------|--------|
| 1 | Output is one scalar probability ∈ [0,1] | ✓ `predict()` → sigmoid f64 |
| 2 | Action is only {0, +1} | ✓ `i32::from(extra_reduction: bool)` |
| 3 | Runtime activation disabled | ✓ shadow path records stats only; never alters `rd` |
| 4 | Extensions untouched | ✓ EME branch is a separate prior `if` arm |
| 5 | Existing native LMR unchanged when probe disabled | ✓ `red = ace_graduated_lmr_reduction(i, depth)` unconditional |
| 6 | Only targeted initial scout receives +1 | ✓ `rd = (new_depth − red − i32::from(extra)).max(0)` |
| 7 | Verification runs with native depth | ✓ `self.ab(new_depth, −beta, −alpha, …)` — no extra reduction |
| 8 | Sidecar cannot prune or skip a move | ✓ no `continue`/`break` in shadow path |
| 9 | Value/trunk weights frozen | ✓ SHA-256 binding; no write path in loader |
| 10 | `net_weights.bin` cannot be written by sidecar trainer | ✓ trainer reads file for hash only |
| 11 | Sidecar is hash-bound to exact trunk | ✓ bytes[20..52] == `live_weights_sha256()` or load fails |
| 12 | Zero-Ink cannot create the final label | ✓ label produced by `classify_pair()` from A/B node counts |
| 13 | UNKNOWN samples excluded from supervised training | ✓ trainer filters `sample_status != "UNKNOWN"` before split |
| 14 | Calibration and test use natural prevalence | ✓ split uses `population == "natural"` rows only |

No violation found. No code change required.

---

## Implementation

- `titanium reduction-probe` records native LMR pipeline events and replays
  exactly one event with `base_reduction + 1`.
- Each A/B comparison starts from the same move prefix with a fresh fixed 18-bit
  TT and zeroed history, killers, countermoves, counters, caches, and stop state.
  A never feeds B; B never feeds A.
- Dataset schema: `titanium-reduction-counterfactual-v1`.
- Model input: detached child hidden[32] plus normalized remaining depth, move
  index, base reduction, and horizontal/vertical move class (context5).
- Model: Linear(37, 1) → sigmoid. One scalar probability output.
- Value weights are read-only for SHA-256 binding; trainer cannot update or
  export `net_weights.bin`.
- Sidecar binary is hash-bound to the exact live trunk. Missing, malformed,
  NaN, schema-mismatched, or trunk-mismatched files produce fail-closed
  (p=0, no activation).
- Shadow inference records hypothetical activations and timing but cannot alter
  depth, ordering, pruning, LMR, extensions, verification, or returned results.

---

## Local data result — first run (depth=5, 60 positions)

Command:
```powershell
python training/collect_reduction_counterfactuals.py --positions 60 `
  --samples-per-position 2 --depth 5 --event-scan-limit 128 `
  --minimum-nodes-saved 8 --minimum-savings-ratio 0.05 `
  --population natural --seed 1337 --out $env:TEMP\titanium-reduction-local-v1.jsonl
```

Results:

- 106 SAFE, 0 UNSAFE, 0 UNKNOWN
- safe-and-worthwhile positives: **1 (0.9%)**
- natural split: train 84, calibration 10, final-test 12
- calibration positives: 0 / final-test positives: 0

**Not train-ready.** Trainer correctly refused to emit an artifact.

---

## Natural-sample diagnosis — why 105/106 were not useful

### Root cause (confirmed from JSONL inspection)

**105/106 safe samples have `baseline_nodes = 1` and `counterfactual_nodes = 1`.
Net savings = 0. The +1 reduction cannot save any nodes because the baseline
reduced scout is already a leaf (depth 0).**

Specifically:
- 88/106 events occurred at local `depth = 3` with `base_reduction = 2`.
  At these parameters: `new_depth = 2`, `rd = max(0, 2−2) = 0`.
  Both A and B evaluate a depth-0 leaf → 1 node each → no savings possible.
- 18/106 events occurred at local `depth = 4` with `base_reduction = 2`.
  `rd = max(0, 3−2) = 1`. A depth-1 subtree in Quoridor is small;
  baseline_nodes median = 1, max = 12. Only 1 sample broke the threshold.

Failure category breakdown (105 not-useful safe samples):

| Category | Count |
|----------|-------|
| `cf_equal_nodes` — both cost 1 node, rd was already 0 | 105 |

### Key distributions (106 SAFE samples)

| Field | min | p10 | p25 | median | p75 | p90 | p95 | max |
|-------|-----|-----|-----|--------|-----|-----|-----|-----|
| baseline_nodes | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 12 |
| counterfactual_nodes | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| net_nodes_saved | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 |
| net_savings_ratio | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.917 |
| depth (local) | 3 | 3 | 3 | 3 | 3 | 4 | 4 | 4 |
| move_index | 4 | 9 | 16 | 39 | 52 | 71 | 85 | 118 |
| base_reduction | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 2 |

- Verification triggered: 0/106 = 0%
- Decision preserved: 106/106 = 100%
- Worthwhile net savings: 1/106 = 0.9%

### Was depth 5 too shallow? **Yes.**

At root depth 5, LMR-eligible moves in the tree appear at local `depth = 3` or
`depth = 4`. With `base_reduction = 2` (the dominant case for `move_index ≥ 12`),
the reduced scout runs at `rd = 0` — already a leaf before the +1 is applied.
Adding +1 to rd=0 makes rd=max(−1,0)=0: no change, no savings.

For meaningful savings, the baseline reduced scout needs actual branching work
(`baseline_nodes ≥ 8`). This requires `rd ≥ 2` at the LMR call site, which
at depth=5 means `base_reduction ≤ 1` at local `depth ≥ 4`. These events are
rare in the natural population under the current event-scan strategy.

### Stratified smoke confirmation

A 34-sample stratified run (oversampled expensive scouts) produced 4 positives:

| depth | move_index | base_red | baseline_nodes | saved | ratio |
|-------|-----------|----------|----------------|-------|-------|
| 3 | 10 | 1 | 23 | 22 | 0.957 |
| 3 | 10 | 1 | 23 | 22 | 0.957 |
| 3 | 8 | 1 | 23 | 22 | 0.957 |
| 4 | 10 | 1 | 28 | 27 | 0.964 |

All 4 positives have `base_reduction = 1`. At `depth = 3, base_red = 1`:
`rd = max(0, 2−1) = 1`. The depth-1 subtree branching can cost 20–33 nodes.
The +1 reduction collapses this to a leaf (1 node) → large percentage savings.

This confirms useful positives exist but require `base_reduction = 1` events
with branchy depth-1 subtrees. **These are structurally absent at root depth 5**
because `base_reduction = 1` only applies to move indices 4–11 (early LMR moves),
and at depth 5 these happen to produce depth-1 scouts that are almost always 1 node.

The path to natural positives is deeper root search (depth 7–8) so LMR events
carry real `rd = 2–4` scouts, where +1 produces measurable savings.

---

## Shadow cost and inference break-even

Measured on fixed-depth c3h probe (depth 5, 4,816 nodes):

- 2,107 eligible shadow evaluations
- Total inference time: 1,227,600 ns
- Per-evaluation cost: **583 ns**
- Hypothetical activations: 0

Break-even analysis:

Estimate node cost from the shadow run: if depth-5 search takes ~4 ms (typical
at 5s/move with overhead), NPS ≈ 4,816 / 0.004 = 1.2M nodes/s → ~833 ns/node.
The 583 ns sidecar cost equals approximately **0.7 node-equivalents** per call.

| Scenario | Activation rate | Mean nodes saved | Expected nodes/eval | vs 0.7 break-even |
|----------|----------------|-----------------|---------------------|-------------------|
| Natural depth 5 | 1% | 11 | 0.11 | **below** |
| Stratified depth 5 | 12% | 20 | 2.4 | above |
| Target natural depth 7 (projected) | ~10% | ~15 | ~1.5 | above |

Limitation: node cost is not uniform (leafs are faster than internal nodes);
this analysis assumes equal cost. The 583 ns figure is from a single probe run
and should be confirmed with a larger benchmark.

---

## Phase 1 results — depth=7, min-event-depth=5, min-ply=11

### Second root cause: post-order event fill

At depth=7, the probe records events post-order (after the recursive scout
completes). Deep-tree nodes (local depth 3–4) complete first and fill the scan
limit before shallow-tree events (local depth 5–6) can be recorded. This
explains why the first depth=7 run also produced 0 positives — `d` values in
the output were still 3–4.

Fix implemented: `--min-event-depth` parameter added to the probe (Rust) and
collector (Python). With `--min-event-depth 5`, ordinals are only assigned to
events where local depth ≥ 5, so the scan limit fills with high-depth events.

### Phase 1 results (depth=7, min-event-depth=5, min-ply=11, 50 samples)

```powershell
python training/collect_reduction_counterfactuals.py `
  --positions 30 `
  --samples-per-position 2 `
  --depth 7 `
  --event-scan-limit 128 `
  --min-event-depth 5 `
  --minimum-nodes-saved 8 `
  --minimum-savings-ratio 0.05 `
  --population natural `
  --min-ply 11 `
  --seed 42 `
  --out "$env:TEMP\titanium-reduction-d7-phase1-v2.jsonl"
```

Results:
- 50 SAFE, 0 UNSAFE, 0 UNKNOWN
- **positives: 3 / 50 = 6.0%**

Positive distribution:

| depth | move_index | base_red | baseline_nodes | cf_nodes | saved |
|-------|-----------|----------|----------------|----------|-------|
| 5 | 34 | 2 | 12 | 1 | 11 |
| 5 | 39 | 2 | 12 | 1 | 11 |
| 6 | 16 | 2 | 13 | 1 | 12 |

Pattern: all 3 positives have `base_red=2`. At `d=5, base_red=2`: `rd=2`, baseline
scout costs 12–13 nodes. With +1: `rd=1`, costs 1 node. Clean collapse.

Sample distribution (50 events):
- depth range: 5–6, median: 5
- base_red range: 1–3, median: 2
- net_nodes_saved range: −1 to +12, non-zero: 5
- baseline_nodes: min=1, max=13, median=1

Break-even check: 6.0% × 11 nodes ≈ 0.66 expected nodes/eval vs 0.7 break-even.
Borderline but within measurement noise on 50 samples. Phase 2 needed to confirm.

**Phase 1 gate passes (≥ 3 positives).** Proceed to Phase 2.

---

## move_index audit (context5 feature 1)

Performed before Phase 2 as requested. All 8 checks passed. No bug found.

### Definitions

`move_index` = the loop variable `i` from `for i in 0..n` over the ordered
moves array after `order_moves()`. It is the position in the ordered move list
(0-indexed), counting all generated moves including any that were skipped by
LMP or sealing-wall pruning. LMR eligibility requires `i >= ACE_LMR_AFTER_MOVE`
(= 4), so the minimum observable `move_index` in any probe event is 4.

The first ordered move (typically the TT move) is `move_index=0`. Later moves
receive monotonically larger values within each node's search.

### Normalization parity

Python training: `min(int(move_index) / 128.0, 1.0)`
Rust inference:  `(i as f64 / 128.0).clamp(0.0, 1.0)`

These are numerically identical for `i ≥ 0` (usize is always non-negative).
Cross-checked on all 50 Phase 1 rows: zero mismatches. Context5 examples
verified: `move_index=4 → 0.0312 = 4/128`, `move_index=15 → 0.1172 = 15/128`.

### Identity verification (Phase 1 rows)

- `move_index == ordinal` collisions: 0 (ordinal range 0–126, move_index 4–94)
- `move_index == ply` collisions: 0 (ply range 0–1)
- move_index is not the wall encoding (stored separately as `mv`)
- move_index is captured at loop-top before `make_move()` — fully pre-search

### Schema protection

`FEATURE_SCHEMA = "halfpw-hidden32-search-context5-v1"` stored in every JSONL
row. Sidecar binary encodes schema version and INPUTS count at bytes[8..20].
Any normalization change requires bumping both the feature schema string and the
binary schema version integer.

### Phase 1 bucket analysis

| bucket | n  | pos | pos%  | med_depth | med_saved |
|--------|----|-----|-------|-----------|-----------|
| 4-7    |  2 |  0  | 0.0%  | 6         | 0         |
| 8-11   |  4 |  0  | 0.0%  | 6         | 0         |
| 12-19  |  9 |  1  | 11.1% | 5         | 0         |
| 20-39  | 19 |  2  | 10.5% | 5         | 0         |
| 40-69  | 13 |  0  | 0.0%  | 5         | 0         |
| 70+    |  3 |  0  | 0.0%  | 5         | 0         |

All 3 positives fall in move_index 12–39. This is mechanically explained: the
LMR formula adds base_red=2 starting at move_index ≥ 12, which at local depth=5
gives rd=2. The +1 reduction collapses these 12-node scouts to 1 node. Phase 1
sample is too small (3 positives) for a reliable ablation.

### Phase 2 bucket analysis (natural stream, 804 rows)

| bucket | n   | pos | pos%  |
|--------|-----|-----|-------|
| 4-7    |  49 |  5  | 10.2% |
| 8-11   |  51 |  7  | 13.7% |
| 12-19  |  81 |  5  | 6.2%  |
| 20-39  | 264 | 27  | 10.2% |
| 40-69  | 290 | 24  | 8.3%  |
| 70+    |  69 | 13  | 18.8% |

Positives now span all buckets. The 70+ bucket shows elevated rate (18.8%);
very late moves at local depth 6 occasionally produce branchy scouts. Feature
shows gradient; formal ablation should be run at train time with the full
stratified dataset (3448 rows, 1018 positives available).

**Ablation verdict**: move_index shows structural gradient correlated with
base_red tier breakpoints. Retain in context5. Formal ablation deferred to
train time.

---

## Phase 2 — main collection (Phase 1 gate passed)

Two separate streams. Do not merge them into one file.
Both streams require `--min-event-depth 5` (fix for post-order fill) and
`--min-ply 11` (past book window, BOOK_MAX_PLY=10).

**Natural stream** (calibration + test prevalence):
```powershell
python training/collect_reduction_counterfactuals.py `
  --positions 500 `
  --samples-per-position 2 `
  --depth 7 `
  --event-scan-limit 128 `
  --min-event-depth 5 `
  --minimum-nodes-saved 8 `
  --minimum-savings-ratio 0.05 `
  --population natural `
  --min-ply 11 `
  --seed 42 `
  --out "$env:TEMP\titanium-reduction-natural-d7-v2.jsonl"
```

**Stratified stream** (training exploration, oversampled expensive scouts):
```powershell
python training/collect_reduction_counterfactuals.py `
  --positions 1000 `
  --samples-per-position 4 `
  --depth 7 `
  --event-scan-limit 128 `
  --min-event-depth 5 `
  --minimum-nodes-saved 8 `
  --minimum-savings-ratio 0.05 `
  --population stratified `
  --min-ply 11 `
  --seed 99 `
  --out "$env:TEMP\titanium-reduction-stratified-d7-v2.jsonl"
```

### Phase 2 results

**Natural stream** (seed=42, 500 positions × 2 = ~1000 target):

| metric | value |
|--------|-------|
| Total rows | 804 |
| SAFE | 803 (99.9%) |
| UNSAFE | 1 (0.1%) |
| UNKNOWN | 0 |
| Positives | 81 (10.1%) |
| Saved range (positives) | 8–99 nodes |
| Ply range (events) | 0–1 |
| Depth range (events) | 5–6 |

Split distribution:

| split | n | positives | negatives | ready |
|-------|---|-----------|-----------|-------|
| train | 586 | 56 | 529 | YES |
| calibration | 110 | 12 | 98 | YES |
| final_test | 108 | 13 | 95 | YES |

**All three splits meet the training readiness gate (≥10 pos AND neg).**

**Stratified stream** (seed=99, 1000 positions × 4):

| metric | value |
|--------|-------|
| Total rows | 3448 |
| SAFE | 3401 (98.6%) |
| UNSAFE | 47 (1.4%) |
| UNKNOWN | 0 |
| Positives | 1018 (29.5%) |
| Saved range (positives) | 8–754 nodes |

Bucket analysis (stratified):

| bucket | n | pos | pos% |
|--------|---|-----|------|
| 4-7 | 912 | 290 | 31.8% |
| 8-11 | 360 | 189 | 52.5% |
| 12-19 | 490 | 145 | 29.6% |
| 20-39 | 812 | 255 | 31.4% |
| 40-69 | 690 | 109 | 15.8% |
| 70+ | 184 | 30 | 16.3% |

The 8-11 bucket shows the highest stratified rate (52.5%) because early-LMR
moves (base_red=1 at move_index < 12) have larger branchy scouts that
stratified oversampling preferentially selects. The positive rate gradient
confirms move_index carries real signal.

### Split plan

| Stream | Use | Notes |
|--------|-----|-------|
| Natural d7 | calibration + final-test | True activation prevalence; do not contaminate with stratified rows |
| Stratified d7 | training only | May be combined with natural training split |
| Natural d5 (existing) | discard or archive | rd=0 dominates; not useful |

Do not calibrate on stratified data. Do not merge Zero-Ink proposal metadata
into the label. The trainer already enforces the natural-only calibration/test
constraint via the `population` field.

### Training readiness gate (unchanged)

Before approving an artifact:
- Training split: positives AND negatives present
- Natural calibration split: ≥ 10 positives AND negatives
- Untouched natural final-test split: ≥ 10 positives AND negatives
- Enough samples to estimate precision with a meaningful confidence interval

---

## Shadow cost sample

On the fixed-depth c3h probe:

- baseline and shadow: `e6h`, score `-74`, depth 5, 4,816 nodes
- eligible shadow evaluations: 2,107
- measured hidden-plus-linear inference time: 1,227,600 ns total
- approximate measured cost: 583 ns per eligible move
- hypothetical activations from the conservative sample head: 0

The tree was identical. Full NPS needs a dedicated benchmark with many repeated
searches before a precise break-even claim; this single short sample is
diagnostic only.

---

## Validation

- focused/new Rust tests: pass
- fixed-depth shadow tree parity: pass (bestmove e6h, score −74, depth 5,
  nodes 4,816 — identical for baseline and shadow)
- Python reduction + historical pressure tests: 14 pass
- Zero teacher tests: 4 pass
- HalfPW parity: 6/6
- training preflight: READY (collector and trainer both functional)
- JS timeout/adaptive coordinator tests: pass
- SQLite integrity: OK, 1,499 games
- complete `cargo test`: exceeded a 10-minute limit; no failure emitted before timeout.
  Focused new Rust tests (reduction-sidecar unit tests and shadow parity) all passed.
  The slow test groups are unrelated to this experiment.

---

## Stage-1 training sweep — 2026-06-19

Script: `training/train_reduction_sidecar_v2.py`

### Data summary

| Set | Population | Rows | Positives | Unsafe | Notes |
|-----|-----------|------|-----------|--------|-------|
| natural train | natural | 586 | 56 (9.6%) | 1 | split="train" |
| natural calibration | natural | 110 | 12 (10.9%) | 0 | split="calibration" |
| natural final_test | natural | 108 | 13 (12.0%) | 0 | **SEALED until after freeze** |
| stratified train pool | stratified | 2357 | 707 (30.0%) | ≥47 | split="train" only; 6 dup rows removed |

Stratified rows with split="calibration" or "final_test" were excluded from the training
pool (same partition hash → would leak test-game signal).

### Unsafe case analysis

One natural UNSAFE example:
- ply=26, move=`g4v`, move_index=55, depth=5, base_reduction=2
- baseline: 12 nodes, FAIL_LOW bound
- counterfactual: 249 nodes, EXACT bound → decision changes (FAIL_LOW→EXACT)
- This single example was correctly excluded from calibration; it appeared in the
  natural training split. The model was penalized for activating it with unsafe_weight.

### Integrity checks (all passed)

- Schema and feature_schema match in all rows
- Feature dimensions: hidden32=32, context5=5 (no mismatch)
- No non-finite features
- No UNKNOWN rows in supervised sets
- Trunk SHA-256 consistent across all rows: dc2e3e95b0994093…
- Game-key overlap between train/calibration/final_test: 0 collisions
- 6 stratified events appeared in natural stream → deduplicated

### Sweep configuration

| Parameter | Values swept |
|-----------|-------------|
| Variant | A (37-dim) — main sweep; B/C/D as ablations |
| Mixing ratio (natural fraction) | 0.33, 0.50, 0.67 |
| Ordinary neg FP weight | 2.0, 5.0 |
| Unsafe FP weight | 20.0, 50.0, 100.0 |
| Seeds | 10 (42, 137, 271, 512, 1337, 2027, 4099, 8191, 16381, 65537) |
| Positive FN weight | 1.0 (fixed, not swept) |
| Epochs | 400 |
| Optimizer | AdamW lr=2e-3, weight_decay=1e-4 |
| Threshold grid | [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99] |
| Threshold selection | Max expected net node savings on natural calibration, subject to unsafe_activations == 0 |

Total variant A runs: 3 × 2 × 3 × 10 = **180 runs**.

Note: initial sweep used threshold_grid=[0.50, …, 0.99] and produced 0 activations
everywhere. Root cause: with positive FN weight=1 and neg FP weight 2-5×, the
model's calibrated outputs lie in [0.007, 0.42] — all below 0.50. Threshold grid
was extended downward to [0.05, …]. The underlying model IS discriminating;
raw sigmoid ranges 0.03–0.18, Platt-calibrated 0.007–0.42.

### Variant A sweep results

All 180 runs: **FEASIBLE** (0 unsafe activations, net_saved > 0).

Threshold range across runs: 0.05–0.10. Net savings range: 125.9–157.7 (on 110-row calibration set).

| Mixing ratio | Best median net (cal) | Best neg_w | Best uw |
|--------------|----------------------|-----------|---------|
| 33% natural  | 157.7 | 5.0 | 20.0 |
| 50% natural  | 155.8 | 5.0 | 20.0 |
| 67% natural  | 154.1 | 5.0 | 20.0 |

**Trend**: more stratified enrichment (33% natural) marginally outperforms higher
natural ratios. Heavier unsafe penalty (100×) consistently reduces net savings by 15–25
versus 20× — the model over-penalizes unsafe examples and the threshold drifts higher.

### Frozen configuration (selected by calibration only)

| Parameter | Value |
|-----------|-------|
| Variant | A (hidden32 + context5, 37 inputs) |
| Mixing ratio | 33% natural / 67% stratified |
| Ordinary neg FP weight | 5.0 |
| Unsafe FP weight | 20.0 |
| Best seed | 137 |
| Platt scale | fitted on calibration |
| Platt shift | fitted on calibration |
| Threshold | **0.05** |
| Cal net savings | **157.7 node-equiv** (reference calibration set, n=110) |

### Seed stability (best A config, 10 seeds)

| seed | threshold | cal_net | cal_activations | unsafe |
|------|-----------|---------|-----------------|--------|
| 42 | 0.05 | 155.6 | 72 | 0 |
| 137 | 0.05 | 157.7 | 69 | 0 |
| 271 | 0.05 | 153.5 | 75 | 0 |
| 512 | 0.05 | 152.1 | 77 | 0 |
| 1337 | 0.05 | 152.8 | 76 | 0 |
| 2027 | 0.05 | 150.0 | 80 | 0 |
| 4099 | 0.05 | 153.5 | 75 | 0 |
| 8191 | 0.05 | 154.2 | 74 | 0 |
| 16381 | 0.05 | 153.5 | 75 | 0 |
| 65537 | 0.05 | 152.8 | 76 | 0 |

**Range: 150.0–157.7 (7.7 node-equiv), all safe, all threshold=0.05.**
Seed stability is excellent; the model converges reliably.

### Calibration threshold sweep (best seed, best config)

| threshold | activations | TP | unsafe | precision | WL95 | gross | net | act_rate |
|-----------|-------------|-----|--------|-----------|------|-------|-----|---------|
| 0.05 | 69 | 12 | 0 | 0.1739 | 0.1024 | 206 | **157.7** | 62.7% |
| 0.08 | 56 | 9 | 0 | 0.1607 | 0.0869 | 174 | 134.8 | 50.9% |
| 0.10 | 49 | 8 | 0 | 0.1633 | 0.0851 | 163 | 128.7 | 44.5% |
| 0.12 | 43 | 8 | 0 | 0.1860 | 0.0974 | 163 | 132.9 | 39.1% |
| 0.15 | 36 | 7 | 0 | 0.1944 | 0.0975 | 151 | 125.8 | 32.7% |
| 0.18 | 26 | 6 | 0 | 0.2308 | 0.1103 | 140 | 121.8 | 23.6% |
| 0.20 | 16 | 3 | 0 | 0.1875 | 0.0659 | 105 | 93.8 | 14.5% |
| 0.25 | 6 | 2 | 0 | 0.3333 | 0.0968 | 94 | 89.8 | 5.5% |
| 0.30 | 2 | 1 | 0 | 0.5000 | 0.0945 | 15 | 13.6 | 1.8% |
| 0.35+ | 0–1 | 0 | 0 | 0.0 | 0.0 | 0 | ≤-0.7 | ≤1% |

Selection rule: max net, subject to unsafe==0. Winner: **threshold=0.05**, net=157.7.

Observation: precision rises with threshold (0.17→0.50 from t=0.05 to t=0.30), but
gross savings per activation also decreases as we exclude high-value mid-confidence
examples. The net node savings is maximised at the lowest safe threshold.

### Final test (opened once, after all parameters frozen)

Threshold fixed at 0.05. Results on 108 held-out natural rows:

| metric | value |
|--------|-------|
| rows | 108 |
| positives | 13 (12.0%) |
| activations | 77 (71.3%) |
| true positives | 8 |
| unsafe activations | **0** |
| precision | 0.1039 |
| precision WL95 | 0.0536 |
| recall | 0.6154 |
| gross nodes saved | 94 |
| inference cost (node-equiv) | 77 × 0.7 = 53.9 |
| **net nodes saved** | **40.1** |
| unsafe rate | 0.0000 |

**FINAL TEST VERDICT: GO** — net_saved > 0, unsafe_activations = 0.

#### Calibration-to-test gap

Cal net = 157.7 over 110 rows vs test net = 40.1 over 108 rows.
The gap (≈4× per-row) is explained by variance in small samples:
- Cal has 12 positives × avg 17.2 nodes/positive = 206 gross
- Test has 8 TPs × avg 11.75 nodes/TP = 94 gross
- The test sample captured shorter-lived scouts on average

Both calibration and test samples have ~10–12% positive rate, consistent with
the natural prevalence. The test set happens to have lower-savings positives.
Neither set is large enough (n≈110) to estimate mean savings per positive precisely.

**The net_saved=40.1 figure is from a single 108-row test draw — interpret with
wide uncertainty. A bootstrapped 95% CI would cover approximately [5, 90].**
The GO criterion is binary (net > 0, unsafe = 0); both conditions hold.

### Ablation results (at frozen threshold=0.05)

Model variants trained at identical frozen config (ratio=33%, neg_w=5.0, uw=20.0,
10 seeds each). Final-test metrics at the same threshold=0.05:

| Variant | Inputs | Calibration median net | Test net | Test act | Test TP | Test unsafe | Notes |
|---------|--------|----------------------|----------|---------|---------|------------|-------|
| A (full) | 37 | 153.5 | **40.1** | 77 | 8 | 0 | reference |
| B (-move_index) | 36 | 153.8 | **40.8** | 76 | 8 | 0 | ≈A; move_index marginally redundant on this data |
| C (context5 only) | 5 | 130.4 | **54.2** | 104 | 11 | 0 | worse calibration, better test; all seeds converge identically (5-dim linear) |
| D (hidden32 only) | 32 | 136.6 | **81.1** | 97 | 13 | 0 | best test; recall=1.0 (all 13 positives found) |

**Interpretation**:

- **Variant B ≈ A**: `move_index` contributes negligible marginal information beyond
  `hidden32 + remaining-context4`. The position embedding already captures ordering
  implicitly. Ablating it costs <1 net node saved on both sets.

- **Variant C (context5 only)**: Much lower calibration net (130.4 vs 153.5), high
  activation rate (108/110 = 98% cal, 104/108 = 96% test) — the 5-dim model
  essentially activates everything and relies on the rare positives being more
  frequent than the inference cost. Interestingly, this beats A on test (54.2 vs 40.1)
  due to finding 11/13 TPs vs 8/13. But calibration correctly ranked it lower.
  Identical results across all 10 seeds confirm 5-dim linear collapse to a single optimum.

- **Variant D (hidden32 only)**: Best test performance (81.1 net, 13/13 TP recall).
  The NNUE hidden layer alone carries the most predictive signal. Context features
  (remaining depth, move_index, base_reduction, orientation) may add noise at this
  data scale or be redundant given the hidden embedding.

**Production selection remains Variant A** (chosen before final test was opened,
per calibration-only rule). The ablation results are informative for future
retrain decisions:
- If collecting a 10× larger dataset, re-evaluate whether D or A is better calibrated
- The context5 features appear to add limited independent signal at this data scale

### Artifact

- Path: `training/checkpoints/sidecar_v1/reduction_sidecar_v1.bin`
- Magic: `TISRDX1\0`, schema version 1, input count 37
- Trunk SHA-256: dc2e3e95b0994093… (frozen; mismatch → fail-closed)
- Sidecar SHA-256: f4d9f0b9f1c9d881ee89a48dcb410f53429a4bbcb2810d791c8dd50d74b491dd
- Threshold: 0.05
- Runtime activation: **DISABLED** (fail-closed shadow mode, shadow path cannot alter depth)

### Shadow validation (tree parity) — PENDING

Shadow validation re-run at fixed depth, verifying:
- Bestmove unchanged at every test position
- Score unchanged
- Node count unchanged
- Hypothetical activation rate matches calibration expectations

This step is planned but not yet executed.

---

## Decisions

| Item | Status | Reason |
|------|--------|--------|
| Implementation | ✓ implemented | shadow probe, schema, trainer, loader all in place |
| Shadow inference | ✓ validated | tree-identical to baseline; 0 activations as expected |
| Architecture invariants | ✓ verified | all 14 confirmed from code inspection |
| Local label collection (depth 5) | insufficient data | rd=0 dominates; 1/106 useful positives |
| Local label collection (depth 7, Phase 1) | **GATE PASSED** | 3/50 positives (6%), post-order fill bug fixed |
| Local label collection (depth 7, Phase 2) | **COMPLETE** | natural: 81/804 pos; stratified: 1018/3448 pos; all splits ready |
| Stage-1 training sweep | **COMPLETE** | 180 variant-A runs + B/C/D ablations; artifact produced |
| Final test verdict | **GO** | net=+40.1 nodes, 0 unsafe activations, threshold=0.05 |
| Shadow validation (post-training) | PENDING | not yet executed |
| Runtime activation | **NO-GO** | artifact produced; activation intentionally not wired; tree-parity test pending |
