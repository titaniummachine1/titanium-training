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

## Decisions

| Item | Status | Reason |
|------|--------|--------|
| Implementation | ✓ implemented | shadow probe, schema, trainer, loader all in place |
| Shadow inference | ✓ validated | tree-identical to baseline; 0 activations as expected |
| Architecture invariants | ✓ verified | all 14 confirmed from code inspection |
| Local label collection (depth 5) | insufficient data | rd=0 dominates; 1/106 useful positives |
| Local label collection (depth 7, Phase 1) | **GATE PASSED** | 3/50 positives (6%), post-order fill bug fixed |
| Local label collection (depth 7, Phase 2) | **proposed** | awaiting explicit approval to run |
| Offline cluster training | **NO-GO** | natural calibration/test have no useful positives yet |
| Runtime activation | **NO-GO** | no calibrated model; activation intentionally not wired |
