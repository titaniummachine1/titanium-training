# Reduction Sidecar v1 - Local Shakedown

Date: 2026-06-19

## Locked target

For an already-LMR-eligible late wall move, predict whether one additional
provisional reduction preserves the native complete move-search decision and
saves at least the configured node margin. Existing full-depth verification is
unchanged. Runtime activation and deployment are disabled.

## Implementation

- `titanium reduction-probe` records native LMR pipeline events and can replay
  exactly one event with `base_reduction + 1`.
- Each A/B invocation starts from the same move prefix with a fresh fixed 18-bit
  TT and zeroed history, killers, countermoves, counters, caches, and stop state.
- Dataset schema: `titanium-reduction-counterfactual-v1`.
- Model input: detached child hidden[32] plus normalized remaining depth, move
  index, base reduction, and horizontal/vertical move class.
- Model: one linear sigmoid output. Value weights are read only for their SHA-256;
  the trainer cannot update or export `net_weights.bin`.
- Sidecar binary is separately hashed and bound to the exact live trunk hash.
  Missing, malformed, NaN, schema-mismatched, or trunk-mismatched files disable it.
- Shadow inference records hypothetical activations and timing but cannot alter
  depth, ordering, pruning, LMR, extensions, or verification.

## Local data result

Command:

```powershell
python training/collect_reduction_counterfactuals.py --positions 60 `
  --samples-per-position 2 --depth 5 --event-scan-limit 128 `
  --minimum-nodes-saved 8 --minimum-savings-ratio 0.05 `
  --population natural --seed 1337 --out $env:TEMP\titanium-reduction-local-v1.jsonl
```

Result:

- 106 comparable samples
- SAFE: 106
- UNSAFE: 0
- UNKNOWN: 0
- safe and worthwhile positives: 1
- natural split: train 84, calibration 10, final test 12
- calibration positives: 0
- final-test positives: 0

This is **not train-ready**. The trainer now refuses to emit an artifact unless
train, natural calibration, and untouched final test all contain positive and
negative support. The likely next data improvement is targeted sampling of
events whose baseline scout actually costs more than one node, while retaining
an untouched natural population for calibration.

A follow-up 20-position stratified smoke test oversampled expensive native
scouts and produced 34 SAFE samples, including 4 useful positives and 88 total
nodes saved. This confirms the collector can find positives, but stratified
rows remain training-only and cannot replace natural calibration/test data.

## Shadow cost sample

On the fixed-depth c3h probe:

- baseline and shadow: `e6h`, score `-74`, depth 5, 4,816 nodes
- eligible shadow evaluations: 2,107
- measured hidden-plus-linear inference time: 1,227,600 ns total
- approximate measured cost: 583 ns per eligible move
- hypothetical activations from the conservative sample head: 0

The tree was identical. Timed NPS needs a larger repeated benchmark before a
break-even claim; this single short sample is diagnostic only.

## Validation

- focused/new Rust tests: pass
- fixed-depth shadow tree parity: pass
- Python reduction + historical pressure tests: 14 pass
- Zero teacher tests: 4 pass
- HalfPW parity: 6/6
- training preflight: READY
- JS timeout/adaptive coordinator tests: pass
- SQLite integrity: OK, 1,499 games
- complete `cargo test`: exceeded a 10-minute limit; no failure was emitted

## Decision

Offline implementation: ready for more local label collection.

Cluster data/training: **NO-GO** until natural calibration/test contain enough
useful positives and the full Rust suite is either completed or its known slow
tests are split into an explicit long-test job.

Runtime activation: **NO-GO**. It remains disabled by design.
